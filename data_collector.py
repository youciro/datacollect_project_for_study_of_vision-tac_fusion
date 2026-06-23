# -*- coding: utf-8 -*-
"""
视触觉融合抓取 —— 主数据采集脚本
=====================================
功能:
- 整合 CameraReader / TactileReader / ArmController 三个模块
- Tkinter GUI 操作界面 + 5 类结果标注 + 硬件状态栏
- 相机预览窗口 + 触觉热力图窗口 (独立 OpenCV 窗口, 同一显示线程驱动)
- 硬件可单独连接/断开, 任一硬件未连接也能启动 GUI
- 5 阶段状态机流程
- 每次抓取保存一个独立文件夹 (rgb/depth/tactile序列/meta)
- 断点续传

操作流程:
  S0 就绪   -> 放好物体, 在GUI选网格点/角度/物体类别, 点[开始采集]
  S1 运动   -> 程序自动移动机械臂到目标位置上方并下降
  S2 闭合   -> 提示"按遥控器闭合", 你按遥控器, 程序监测触觉接触
  S3 提起   -> 程序自动提起并稳定保持, 同步记录触觉序列
  S4 标注   -> GUI弹出5个结果按钮, 你点击标注
  S5 复位   -> 提示"按遥控器张开", 程序回初始位
"""

import os
import json
import time
import threading
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import shutil

import numpy as np
import cv2

import config
from camera_reader import CameraReader
from tactile_reader import TactileReader

# 机械臂模块 (你的文件名是 armconctrl.py)
try:
    from armconctrl import ArmController
except ImportError:
    try:
        from arm_controller import ArmController
    except ImportError:
        print("⚠️  找不到机械臂模块 (armconctrl.py / arm_controller.py)")
        ArmController = None


# ====================================================================== #
#  触觉热力图绘制 (OpenCV 自绘, vertical 方向, 对齐之前 matplotlib 效果)
# ====================================================================== #

def render_tactile_heatmap(tactile, noise_high, per_finger_force,
                           total_force, freq):
    """
    tactile: np.ndarray (3, 15, 5)
    返回一张 BGR 图, 三指竖条横排, 上方标注指号和指力
    """
    # --- 单指色块参数 ---
    cell = 26                 # 每个传感点的像素边长
    finger_w = 5 * cell       # 单指宽 (5列)
    finger_h = 15 * cell      # 单指高 (15行)
    gap = 40                  # 指间距
    margin_top = 70           # 顶部留白(写标题)
    margin_side = 30
    margin_bot = 30

    canvas_w = margin_side * 2 + finger_w * 3 + gap * 2
    canvas_h = margin_top + finger_h + margin_bot
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)  # 深灰底

    # 色阶上限: 动态, 至少 noise_high 的 10%
    vmax = max(noise_high * 0.1, float(tactile.max()))

    for i in range(3):
        finger = tactile[i]  # (15, 5)

        # 归一化到 0-255
        norm = np.clip(finger / vmax * 255.0, 0, 255).astype(np.uint8)
        # 放大到色块尺寸 (最近邻, 保留格子感)
        big = cv2.resize(norm, (finger_w, finger_h),
                         interpolation=cv2.INTER_NEAREST)
        # 伪彩色 HOT
        colored = cv2.applyColorMap(big, cv2.COLORMAP_HOT)

        # 贴到画布
        x0 = margin_side + i * (finger_w + gap)
        y0 = margin_top
        canvas[y0:y0 + finger_h, x0:x0 + finger_w] = colored

        # 网格线
        for r in range(16):
            y = y0 + r * cell
            cv2.line(canvas, (x0, y), (x0 + finger_w, y), (60, 60, 60), 1)
        for c in range(6):
            x = x0 + c * cell
            cv2.line(canvas, (x, y0), (x, y0 + finger_h), (60, 60, 60), 1)

        # 指号 + 指力标注
        label = "Finger {}".format(i + 1)
        cv2.putText(canvas, label, (x0, y0 - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        force_txt = "F={:.0f}".format(per_finger_force[i])
        cv2.putText(canvas, force_txt, (x0, y0 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    # 顶部总信息
    title = "Total={:.0f}   {:.1f}Hz   vmax={:.0f}".format(
        total_force, freq, vmax)
    cv2.putText(canvas, title, (margin_side, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)

    return canvas


# ====================================================================== #
#  数据采集器主类
# ====================================================================== #

class GraspDataCollector:

    # 状态机阶段
    S_IDLE = 'IDLE'
    S_MOVING = 'MOVING'
    S_CLOSING = 'CLOSING'
    S_LIFTING = 'LIFTING'
    S_LABELING = 'LABELING'
    S_RESETTING = 'RESETTING'

    def __init__(self, save_root):
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok=True)

        # --- 三大硬件模块 ---
        self.cam = None
        self.tactile = None
        self.arm = None

        # --- 硬件连接状态 ---
        self.cam_connected = False
        self.tactile_connected = False
        self.arm_connected = False

        # --- 状态机 ---
        self.state = self.S_IDLE
        self.sample_id = self._load_progress()
        self.current_sample = {}

        # --- 显示窗口控制 ---
        self.show_camera = False
        self.show_heatmap = False
        self._display_running = False
        self._display_thread = None

        # --- 手动触发接触事件 (抓空时用) ---
        self._manual_contact_event = threading.Event()

        # --- GUI 引用 ---
        self.gui = None

    # ================================================================== #
    #  硬件连接 (每个设备独立, 互不影响)
    # ================================================================== #

    def connect_camera(self):
        """连接相机, 返回 True/False"""
        if self.cam_connected:
            self._log("相机已连接")
            return True
        self._log("正在连接相机...")
        try:
            self.cam = CameraReader()
            if not self.cam.start():
                self._log("❌ 相机启动失败")
                self.cam = None
                return False
            for _ in range(50):
                if self.cam.is_ready():
                    break
                time.sleep(0.1)
            if self.cam.is_ready():
                self.cam_connected = True
                self._log("✅ 相机就绪")
                return True
            else:
                self._log("⚠️ 相机未出图")
                self.cam.stop()
                self.cam = None
                return False
        except Exception as e:
            self._log("❌ 相机连接异常: {}".format(e))
            self.cam = None
            return False

    def disconnect_camera(self):
        if not self.cam_connected:
            return
        self.show_camera = False
        try:
            self.cam.stop()
        except Exception:
            pass
        self.cam = None
        self.cam_connected = False
        self._log("相机已断开")

    def connect_tactile(self):
        """连接触觉传感器"""
        if self.tactile_connected:
            self._log("触觉已连接")
            return True
        self._log("正在连接触觉传感器...")
        try:
            self.tactile = TactileReader(
                port=config.TACTILE_PORT,
                baudrate=config.TACTILE_BAUDRATE,
                freq=config.TACTILE_FREQ,
                buffer_seconds=max(8.0, config.TACTILE_PRE_SECONDS +
                                   config.TACTILE_POST_SECONDS + 4.0),
                endian_big=config.TACTILE_ENDIAN_BIG,
                noise_high=config.TACTILE_NOISE_HIGH,
                noise_low=config.TACTILE_NOISE_LOW,
                finger_order=config.TACTILE_FINGER_ORDER,
            )
            self.tactile.start()
            time.sleep(1.5)
            self.tactile_connected = True
            self._log("✅ 触觉就绪")
            return True
        except Exception as e:
            self._log("❌ 触觉连接失败: {}".format(e))
            self.tactile = None
            return False

    def disconnect_tactile(self):
        if not self.tactile_connected:
            return
        self.show_heatmap = False
        try:
            self.tactile.stop()
        except Exception:
            pass
        self.tactile = None
        self.tactile_connected = False
        self._log("触觉已断开")

    def connect_arm(self):
        """连接机械臂"""
        if self.arm_connected:
            self._log("机械臂已连接")
            self.arm.go_home()
            self._log("✅ 机械臂已回到初始位置")
            return True
        if ArmController is None:
            self._log("❌ 机械臂模块未导入")
            return False
        self._log("正在连接机械臂...")
        try:
            self.arm = ArmController(ip=config.ARM_IP)
            self.arm.start()
            self.arm.set_speed(config.ARM_SPEED)
            time.sleep(1.0)
            self.arm_connected = True
            self._log("✅ 机械臂就绪")
            return True
        except Exception as e:
            self._log("❌ 机械臂连接失败: {}".format(e))
            self.arm = None
            return False

    def disconnect_arm(self):
        if not self.arm_connected:
            return
        try:
            self.arm.stop()
        except Exception:
            pass
        self.arm = None
        self.arm_connected = False
        self._log("机械臂已断开")

    def connect_all(self):
        """尝试连接全部硬件 (启动时调用, 失败不阻塞)"""
        self.connect_camera()
        self.connect_tactile()
        self.connect_arm()

    def shutdown(self):
        """安全关闭所有硬件和显示线程"""
        self._log("正在关闭...")
        self._display_running = False
        if self._display_thread:
            self._display_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        self.disconnect_arm()
        self.disconnect_camera()
        self.disconnect_tactile()
        self._log("已全部关闭")

    # ================================================================== #
    #  显示线程 (相机预览 + 触觉热力图, 同一线程统一刷新)
    # ================================================================== #

    def start_display_thread(self):
        if self._display_running:
            return
        self._display_running = True
        self._display_thread = threading.Thread(
            target=self._display_loop, daemon=True, name="DisplayThread")
        self._display_thread.start()

    def _display_loop(self):
        """统一显示循环, 约15Hz; 两个窗口共享一个 waitKey"""
        CAM_WIN = "Camera Preview"
        HEAT_WIN = "Tactile Heatmap"
        cam_win_open = False
        heat_win_open = False

        while self._display_running:
            # --- 相机窗口 (RGB 左 + Depth伪彩色 右 拼合) ---
            if self.show_camera and self.cam_connected:
                rgb, depth, ts = self.cam.get_latest_frame()
                if rgb is not None:
                    left = rgb.copy()
                    rgb_h, rgb_w = left.shape[:2]
                    if depth is not None:
                        # 深度伪彩色 (COLORMAP_JET)
                        depth_vis = cv2.normalize(
                            depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        right = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        # 强制和 RGB 同高 (防止分辨率不一致导致 hstack 崩溃)
                        if right.shape[0] != rgb_h:
                            right = cv2.resize(right, (rgb_w, rgb_h))
                        # 中心点距离标注
                        h, w = depth.shape
                        cd = depth[h // 2, w // 2]
                        cv2.putText(right, "Center: {:.0f}mm".format(cd),
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255), 2)
                        disp = np.hstack([left, right])
                    else:
                        disp = left
                    # 缩放到 75% 适配 1920x1080 屏幕
                    dh, dw = disp.shape[:2]
                    disp = cv2.resize(disp, (int(dw * 0.75), int(dh * 0.75)))
                    cv2.imshow(CAM_WIN, disp)
                    cam_win_open = True
            else:
                if cam_win_open:
                    cv2.destroyWindow(CAM_WIN)
                    cam_win_open = False

            # --- 热力图窗口 ---
            if self.show_heatmap and self.tactile_connected:
                tactile = self.tactile.get_latest()
                per_finger = self.tactile.get_per_finger_force()
                total = self.tactile.get_total_force()
                freq = self.tactile.get_actual_freq()
                heat = render_tactile_heatmap(
                    tactile, config.TACTILE_NOISE_HIGH,
                    per_finger, total, freq)
                cv2.imshow(HEAT_WIN, heat)
                heat_win_open = True
            else:
                if heat_win_open:
                    cv2.destroyWindow(HEAT_WIN)
                    heat_win_open = False

            # 共享 waitKey, 约15Hz
            cv2.waitKey(1)
            time.sleep(0.066)

        # 退出清理
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ================================================================== #
    #  断点续传
    # ================================================================== #

    def _load_progress(self):
        progress_path = os.path.join(self.save_root, config.PROGRESS_FILE)
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('next_sample_id', 1)
            except Exception:
                pass
        return 1

    def _save_progress(self):
        progress_path = os.path.join(self.save_root, config.PROGRESS_FILE)
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump({
                'next_sample_id': self.sample_id,
                'updated_at': datetime.now().isoformat(),
            }, f, indent=2, ensure_ascii=False)

    # ================================================================== #
    #  状态机: 5 阶段流程
    # ================================================================== #

    def start_collection(self, grid_idx, angle, object_class):
        """S0->S1"""
        if self.state != self.S_IDLE:
            self._log("⚠️ 当前不是空闲状态, 忽略")
            return

        # 采集前检查硬件
        missing = []
        if not self.cam_connected:
            missing.append("相机")
        if not self.tactile_connected:
            missing.append("触觉")
        if not self.arm_connected:
            missing.append("机械臂")
        if missing:
            self._log("❌ 无法开始: {} 未连接".format("/".join(missing)))
            if self.gui:
                messagebox.showwarning(
                    "硬件未就绪",
                    "以下硬件未连接:\n{}\n\n请先连接全部硬件".format(
                        "、".join(missing)))
            return

        x, y = config.GRID_POINTS[grid_idx]
        self.current_sample = {
            'sample_id': self.sample_id,
            'grid_idx': grid_idx,
            'object_xy': (x, y),
            'grasp_angle': angle,
            'object_class': object_class,
            't_start': time.time(),
            'timestamps': {},
        }

        t = threading.Thread(target=self._run_grasp_sequence, daemon=True)
        t.start()

    def _run_grasp_sequence(self):
        """S1->S2->S3"""
        try:
            # ===== S1: 机械臂运动 =====
            self.state = self.S_MOVING
            self._update_gui_state()
            self._log("[S1] 机械臂运动到目标位置...")

            x, y = self.current_sample['object_xy']
            grasp_z = config.OBJECT_HEIGHTS.get(self.current_sample['object_class'], 0.13)
            # 安全限制，不允许低于最低Z值
            if grasp_z < config.Z_MIN_LIMIT:
                self._log("⚠️ 目标Z={:.3f} 低于安全限制 Z_MIN={:.3f}，已自动修正".format(
                    grasp_z, config.Z_MIN_LIMIT))
                grasp_z = config.Z_MIN_LIMIT
            z_approach = grasp_z + 0.15
            z_grasp = grasp_z

            rgb0, depth0, ts0 = self.cam.get_latest_frame()
            self.current_sample['rgb_before'] = rgb0
            self.current_sample['depth_before'] = depth0

            pose, _ = self.arm.get_current_pose()
            ori = pose['ori']

            self.arm.move_to_pose(x, y, z_approach, ori)
            time.sleep(0.5)
            self.arm.move_to_pose(x, y, z_grasp, ori)
            self.current_sample['timestamps']['arm_in_place'] = time.time()
            self._log("[S1] 机械臂到位")

            # ===== S2: 等待闭合 =====
            self.state = self.S_CLOSING
            self._update_gui_state()
            self._log("[S2] 请按遥控器闭合夹爪...")

            t_contact = self._wait_for_contact(timeout=30.0)
            if t_contact is None:
                self._log("[S2] ⚠️ 超时未检测到接触, 本次取消")
                self._abort_to_idle()
                return
            self.current_sample['timestamps']['contact'] = t_contact
            self._log("[S2] ✅ 检测到接触")

            # ===== S3: 提起 + 采集 =====
            self.state = self.S_LIFTING
            self._update_gui_state()
            self._log("[S3] 提起并采集...")

            time.sleep(0.3)

            pose, _ = self.arm.get_current_pose()
            self.arm.move_to_pose(
                pose['pos'][0], pose['pos'][1],
                pose['pos'][2] + config.LIFT_HEIGHT, pose['ori'])
            self.current_sample['timestamps']['lifted'] = time.time()

            time.sleep(config.LIFT_STABLE_SECONDS)
            self.current_sample['timestamps']['t_end'] = time.time()

            total_span = (config.TACTILE_PRE_SECONDS +
                          config.TACTILE_POST_SECONDS +
                          config.LIFT_STABLE_SECONDS + 2.0)
            tac_data, tac_ts = self.tactile.get_history(seconds=total_span)
            self.current_sample['tactile_seq'] = tac_data
            self.current_sample['tactile_ts'] = tac_ts

            rgb1, depth1, _ = self.cam.get_latest_frame()
            self.current_sample['rgb_after'] = rgb1
            self.current_sample['depth_after'] = depth1

            self._log("[S3] 采集完成, 触觉序列 {} 帧".format(
                0 if tac_data is None else len(tac_data)))

            # ===== S4: 标注 =====
            self.state = self.S_LABELING
            self._update_gui_state()
            self._log("[S4] 请在界面上标注本次结果")

        except Exception as e:
            self._log("❌ 采集流程出错: {}".format(e))
            self._abort_to_idle()

    def _wait_for_contact(self, timeout=30.0):
        """监测触觉总力跃变 或 手动触发"""
        self._manual_contact_event.clear()
        deadline = time.time() + timeout
        baseline = self.tactile.get_total_force()
        while time.time() < deadline:
            # 手动触发（抓空时使用）
            if self._manual_contact_event.is_set():
                self._log('[S2] 手动确认接触')
                return time.time()
            force = self.tactile.get_total_force()
            if force - baseline > config.CONTACT_FORCE_THRESHOLD:
                return time.time()
            baseline = 0.95 * baseline + 0.05 * force
            time.sleep(0.02)
        return None

    def manual_contact_trigger(self):
        """GUI 按钮调用: 手动触发接触确认"""
        if self.state == self.S_CLOSING:
            self._manual_contact_event.set()
        else:
            self._log('⚠️ 当前不是等待闭合状态')

    def submit_label(self, result_label):
        """S4->S5"""
        if self.state != self.S_LABELING:
            self._log("⚠️ 当前不是标注状态")
            return
        self.current_sample['result_label'] = result_label
        self._log("[S4] 标注: {}".format(result_label))
        self._save_sample()
        t = threading.Thread(target=self._run_reset, daemon=True)
        t.start()

    def _run_reset(self):
        self.state = self.S_RESETTING
        self._update_gui_state()
        self._log("[S5] 先下降放回物体...")
        try:
            # 下降回抓取高度放回物体
            x, y = self.current_sample['object_xy']
            object_class = self.current_sample['object_class']
            grasp_z = config.OBJECT_HEIGHTS.get(object_class, 0.13)
            if grasp_z < config.Z_MIN_LIMIT:
                grasp_z = config.Z_MIN_LIMIT
            pose, _ = self.arm.get_current_pose()
            self.arm.move_to_pose(x, y, grasp_z, pose['ori'])
            time.sleep(0.5)
        except Exception as e:
            self._log("[S5] ⚠️ 下降失败: {}".format(e))

        self._log("[S5] 请按遥控器张开夹爪...")
        time.sleep(2.0)

        try:
            # 张开后先抬升再回home，避免碰到物体
            pose, _ = self.arm.get_current_pose()
            self.arm.move_to_pose(
                pose['pos'][0], pose['pos'][1],
                pose['pos'][2] + 0.15, pose['ori'])
            time.sleep(0.5)
            self.arm.go_home()
        except Exception as e:
            self._log("[S5] ⚠️ 回原位失败: {}".format(e))

        self.sample_id += 1
        self._save_progress()
        self.state = self.S_IDLE
        self._update_gui_state()
        self._log("[S5] ✅ 复位完成, 下一个ID={}".format(self.sample_id))

    def _abort_to_idle(self):
        try:
            if self.arm_connected:
                self.arm.go_home()
        except Exception:
            pass
        self.state = self.S_IDLE
        self._update_gui_state()

    def delete_last_sample(self):
        """删除上一个样本，sample_id 减一"""
        if self.state != self.S_IDLE:
            self._log("⚠️ 采集进行中，不能删除")
            return False

        last_id = self.sample_id - 1
        if last_id < 1:
            self._log("⚠️ 没有可删除的样本")
            return False

        sample_dir = os.path.join(
            self.save_root, config.SAMPLE_DIR_FORMAT.format(last_id))

        if not os.path.exists(sample_dir):
            self._log("⚠️ 样本文件夹不存在: {}".format(sample_dir))
            # 即使文件夹不存在也回退id
            self.sample_id = last_id
            self._save_progress()
            return False

        import shutil
        shutil.rmtree(sample_dir)
        self.sample_id = last_id
        self._save_progress()
        self._log("已删除样本 {} ，当前ID={}".format(last_id, self.sample_id))
        return True

    # ================================================================== #
    #  数据保存
    # ================================================================== #

    def _save_sample(self):
        s = self.current_sample
        sample_dir = os.path.join(
            self.save_root, config.SAMPLE_DIR_FORMAT.format(s['sample_id']))
        os.makedirs(sample_dir, exist_ok=True)

        if s.get('rgb_before') is not None:
            cv2.imwrite(os.path.join(sample_dir, 'rgb_before.png'), s['rgb_before'])
        if s.get('depth_before') is not None:
            np.save(os.path.join(sample_dir, 'depth_before.npy'), s['depth_before'])
        if s.get('rgb_after') is not None:
            cv2.imwrite(os.path.join(sample_dir, 'rgb_after.png'), s['rgb_after'])
        if s.get('tactile_seq') is not None:
            np.save(os.path.join(sample_dir, 'tactile_seq.npy'), s['tactile_seq'])
        if s.get('tactile_ts') is not None:
            np.save(os.path.join(sample_dir, 'tactile_ts.npy'), s['tactile_ts'])
        if s.get('depth_after') is not None:
            np.save(os.path.join(sample_dir, 'depth_after.npy'), s['depth_after'])

        meta = {
            'sample_id': s['sample_id'],
            'object_class': s['object_class'],
            'grid_idx': s['grid_idx'],
            'object_xy': list(s['object_xy']),
            'grasp_angle': s['grasp_angle'],
            'result_label': s['result_label'],
            'timestamps': s['timestamps'],
            'collected_at': datetime.now().isoformat(),
            'tactile_frames': 0 if s.get('tactile_seq') is None else int(len(s['tactile_seq'])),
        }
        with open(os.path.join(sample_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        self._log("💾 已保存 {}".format(sample_dir))

    # ================================================================== #
    #  GUI 回调辅助
    # ================================================================== #

    def _log(self, msg):
        print(msg)
        if self.gui:
            self.gui.append_log(msg)

    def _update_gui_state(self):
        if self.gui:
            self.gui.on_state_changed(self.state)


# ====================================================================== #
#  Tkinter GUI
# ====================================================================== #

class CollectionGUI:
    def __init__(self, collector):
        self.collector = collector
        collector.gui = self

        self.root = tk.Tk()
        self.root.title(config.GUI_WINDOW_TITLE)
        self.root.geometry('580x780')

        self._build_widgets()
        self._refresh_loop()

    def _build_widgets(self):
        pad = {'padx': 6, 'pady': 3}
        row = 0

        # ===== 硬件状态栏 =====
        hw_frame = tk.LabelFrame(self.root, text='硬件连接', font=('', 10))
        hw_frame.grid(row=row, column=0, columnspan=2, sticky='we', padx=8, pady=6)

        # 相机
        self.cam_status = tk.StringVar(value='❌ 未连接')
        tk.Label(hw_frame, text='相机:', width=8, anchor='e').grid(row=0, column=0, **pad)
        tk.Label(hw_frame, textvariable=self.cam_status, width=12,
                 anchor='w').grid(row=0, column=1, **pad)
        self.cam_btn = tk.Button(hw_frame, text='连接', width=8,
                                 command=self._toggle_camera)
        self.cam_btn.grid(row=0, column=2, **pad)

        # 触觉
        self.tac_status = tk.StringVar(value='❌ 未连接')
        tk.Label(hw_frame, text='触觉:', width=8, anchor='e').grid(row=1, column=0, **pad)
        tk.Label(hw_frame, textvariable=self.tac_status, width=12,
                 anchor='w').grid(row=1, column=1, **pad)
        self.tac_btn = tk.Button(hw_frame, text='连接', width=8,
                                 command=self._toggle_tactile)
        self.tac_btn.grid(row=1, column=2, **pad)

        # 机械臂
        self.arm_status = tk.StringVar(value='❌ 未连接')
        tk.Label(hw_frame, text='机械臂:', width=8, anchor='e').grid(row=2, column=0, **pad)
        tk.Label(hw_frame, textvariable=self.arm_status, width=12,
                 anchor='w').grid(row=2, column=1, **pad)
        self.arm_btn = tk.Button(hw_frame, text='重新连接', width=8,
                                 command=self._reconnect_arm)
        self.arm_btn.grid(row=2, column=2, **pad)
        row += 1

        # ===== 显示窗口开关 =====
        view_frame = tk.LabelFrame(self.root, text='预览窗口', font=('', 10))
        view_frame.grid(row=row, column=0, columnspan=2, sticky='we', padx=8, pady=6)
        self.cam_view_btn = tk.Button(view_frame, text='显示相机画面', width=16,
                                      command=self._toggle_cam_view)
        self.cam_view_btn.grid(row=0, column=0, padx=8, pady=4)
        self.heat_view_btn = tk.Button(view_frame, text='显示触觉热力图', width=16,
                                       command=self._toggle_heat_view)
        self.heat_view_btn.grid(row=0, column=1, padx=8, pady=4)
        row += 1

        # ===== 状态机状态 =====
        self.state_var = tk.StringVar(value='IDLE')
        tk.Label(self.root, text='当前状态:', font=('', 11)).grid(
            row=row, column=0, sticky='e', **pad)
        self.state_label = tk.Label(self.root, textvariable=self.state_var,
                                    font=('', 14, 'bold'), fg='blue')
        self.state_label.grid(row=row, column=1, sticky='w', **pad)
        row += 1

        self.id_var = tk.StringVar(value=str(self.collector.sample_id))
        tk.Label(self.root, text='下一个ID:', font=('', 11)).grid(
            row=row, column=0, sticky='e', **pad)
        tk.Label(self.root, textvariable=self.id_var, font=('', 12)).grid(
            row=row, column=1, sticky='w', **pad)
        row += 1

        # ===== 采集参数 =====
        frame = tk.LabelFrame(self.root, text='本次采集参数', font=('', 10))
        frame.grid(row=row, column=0, columnspan=2, sticky='we', padx=8, pady=6)

        tk.Label(frame, text='网格点:').grid(row=0, column=0, sticky='e', **pad)
        self.grid_var = tk.IntVar(value=4)
        tk.OptionMenu(frame, self.grid_var,
                      *range(len(config.GRID_POINTS))).grid(
            row=0, column=1, sticky='w', **pad)

        tk.Label(frame, text='抓取角度:').grid(row=1, column=0, sticky='e', **pad)
        self.angle_var = tk.IntVar(value=config.GRASP_ANGLES[0])
        tk.OptionMenu(frame, self.angle_var,
                      *config.GRASP_ANGLES).grid(row=1, column=1, sticky='w', **pad)

        tk.Label(frame, text='物体类别:').grid(row=2, column=0, sticky='e', **pad)
        self.obj_var = tk.StringVar(value=config.OBJECT_CLASSES[0])
        tk.OptionMenu(frame, self.obj_var,
                      *config.OBJECT_CLASSES).grid(row=2, column=1, sticky='w', **pad)
        row += 1

        # ===== 开始采集 =====
        self.start_btn = tk.Button(self.root, text='▶ 开始采集',
                                   font=('', 13, 'bold'), bg='#4CAF50', fg='white',
                                   command=self._on_start)
        self.start_btn.grid(row=row, column=0, columnspan=2, sticky='we',
                            padx=8, pady=6)
        row += 1

        # ===== 手动确认接触按钮 (S2阶段抓空时用) =====
        self.manual_contact_btn = tk.Button(
            self.root, text='手动确认接触 (抓空/弱信号时用)',
            font=('', 11), bg='#FF9800', fg='white',
            state='disabled',
            command=self._on_manual_contact)
        self.manual_contact_btn.grid(row=row, column=0, columnspan=2,
                                     sticky='we', padx=8, pady=2)
        row += 1

        # ===== 删除样本按钮 =====
        self.delete_btn = tk.Button(self.root, text='删除上一个样本',
                                    font=('', 11), bg='#F44336', fg='white',
                                    command=self._on_delete)
        self.delete_btn.grid(row=row, column=0, columnspan=2, sticky='we',
                             padx=8, pady=2)
        row += 1

        # ===== 标注按钮 =====
        self.label_frame = tk.LabelFrame(self.root, text='结果标注 (采集完成后点击)',
                                         font=('', 10))
        self.label_frame.grid(row=row, column=0, columnspan=2, sticky='we',
                              padx=8, pady=6)
        colors = {'success': '#4CAF50', 'slip': '#FF9800', 'miss': '#9E9E9E',
                  'pose_error': '#F44336', 'deform': '#9C27B0'}
        self.label_btns = []
        for i, lbl in enumerate(config.RESULT_LABELS):
            b = tk.Button(self.label_frame, text=lbl, width=12,
                          bg=colors.get(lbl, '#607D8B'), fg='white',
                          state='disabled',
                          command=lambda l=lbl: self._on_label(l))
            b.grid(row=i // 3, column=i % 3, padx=4, pady=4)
            self.label_btns.append(b)
        row += 1

        # ===== 实时状态 =====
        info_frame = tk.LabelFrame(self.root, text='实时状态', font=('', 10))
        info_frame.grid(row=row, column=0, columnspan=2, sticky='we', padx=8, pady=4)
        self.force_var = tk.StringVar(value='触觉总力: --')
        tk.Label(info_frame, textvariable=self.force_var).grid(
            row=0, column=0, sticky='w', **pad)
        self.cam_info = tk.StringVar(value='相机: --')
        tk.Label(info_frame, textvariable=self.cam_info).grid(
            row=0, column=1, sticky='w', **pad)
        row += 1

        # ===== 日志 =====
        tk.Label(self.root, text='日志:').grid(row=row, column=0, sticky='w', padx=8)
        row += 1
        self.log_text = tk.Text(self.root, height=9, width=68)
        self.log_text.grid(row=row, column=0, columnspan=2, padx=8, pady=4)

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ================== 硬件按钮回调 ==================

    def _toggle_camera(self):
        if self.collector.cam_connected:
            self.collector.disconnect_camera()
        else:
            self.collector.connect_camera()
        self._refresh_hw_status()

    def _toggle_tactile(self):
        if self.collector.tactile_connected:
            self.collector.disconnect_tactile()
        else:
            self.collector.connect_tactile()
        self._refresh_hw_status()

    def _reconnect_arm(self):
        # 采集中不允许操作机械臂
        if self.collector.state != GraspDataCollector.S_IDLE:
            messagebox.showwarning("提示", "采集进行中, 不能操作机械臂")
            return
        if self.collector.arm_connected:
            self.collector.disconnect_arm()
        self.collector.connect_arm()
        self._refresh_hw_status()

    def _refresh_hw_status(self):
        # 相机
        if self.collector.cam_connected:
            self.cam_status.set('✅ 已连接')
            self.cam_btn.config(text='断开')
        else:
            self.cam_status.set('❌ 未连接')
            self.cam_btn.config(text='连接')
        # 触觉
        if self.collector.tactile_connected:
            self.tac_status.set('✅ 已连接')
            self.tac_btn.config(text='断开')
        else:
            self.tac_status.set('❌ 未连接')
            self.tac_btn.config(text='连接')
        # 机械臂
        if self.collector.arm_connected:
            self.arm_status.set('✅ 已连接')
        else:
            self.arm_status.set('❌ 未连接')

    # ================== 预览窗口回调 ==================

    def _toggle_cam_view(self):
        if not self.collector.cam_connected:
            messagebox.showinfo("提示", "请先连接相机")
            return
        self.collector.show_camera = not self.collector.show_camera
        self.cam_view_btn.config(
            text='关闭相机画面' if self.collector.show_camera else '显示相机画面')

    def _toggle_heat_view(self):
        if not self.collector.tactile_connected:
            messagebox.showinfo("提示", "请先连接触觉传感器")
            return
        self.collector.show_heatmap = not self.collector.show_heatmap
        self.heat_view_btn.config(
            text='关闭触觉热力图' if self.collector.show_heatmap else '显示触觉热力图')

    # ================== 采集 / 标注回调 ==================

    def _on_start(self):
        self.collector.start_collection(
            self.grid_var.get(), self.angle_var.get(), self.obj_var.get())

    def _on_delete(self):
        if self.collector.state != GraspDataCollector.S_IDLE:
            messagebox.showwarning("提示", "采集进行中，不能删除")
            return
        last_id = self.collector.sample_id - 1
        if last_id < 1:
            messagebox.showinfo("提示", "没有可删除的样本")
            return
        if messagebox.askokcancel("确认删除",
                                  "确定要删除样本 {} 吗？此操作不可撤销".format(last_id)):
            self.collector.delete_last_sample()
            self.on_state_changed(self.collector.state)

    def _on_manual_contact(self):
        self.collector.manual_contact_trigger()

    def _on_label(self, label):
        self.collector.submit_label(label)

    def _on_close(self):
        if messagebox.askokcancel('退出', '确定要退出并关闭硬件吗?'):
            self.collector.shutdown()
            self.root.destroy()

    # ================== 状态更新 ==================

    def on_state_changed(self, state):
        self.state_var.set(state)
        self.id_var.set(str(self.collector.sample_id))
        is_idle = (state == GraspDataCollector.S_IDLE)
        is_labeling = (state == GraspDataCollector.S_LABELING)
        is_closing = (state == GraspDataCollector.S_CLOSING)
        self.start_btn.config(state='normal' if is_idle else 'disabled')
        self.manual_contact_btn.config(
            state='normal' if is_closing else 'disabled')
        for b in self.label_btns:
            b.config(state='normal' if is_labeling else 'disabled')

    def append_log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        try:
            self.log_text.insert('end', '[{}] {}\n'.format(ts, msg))
            self.log_text.see('end')
        except Exception:
            pass

    def _refresh_loop(self):
        try:
            if self.collector.tactile_connected:
                f = self.collector.tactile.get_total_force()
                freq = self.collector.tactile.get_actual_freq()
                self.force_var.set('触觉总力: {:.0f}  ({:.0f}Hz)'.format(f, freq))
            else:
                self.force_var.set('触觉总力: 未连接')
            if self.collector.cam_connected:
                ready = self.collector.cam.is_ready()
                self.cam_info.set('相机: {}'.format('OK' if ready else '无信号'))
            else:
                self.cam_info.set('相机: 未连接')
        except Exception:
            pass
        self.root.after(config.GUI_REFRESH_MS, self._refresh_loop)

    def run(self):
        self.root.mainloop()


# ====================================================================== #
#  主入口
# ====================================================================== #

def main():
    save_root = config.SAVE_ROOT
    print("数据保存目录: {}".format(save_root))

    collector = GraspDataCollector(save_root)

    print("=" * 50)
    print("尝试连接硬件 (失败不影响GUI启动)...")
    collector.connect_all()
    print("=" * 50)

    # 启动显示线程 (窗口默认隐藏, 由按钮控制显示)
    collector.start_display_thread()

    # 启动 GUI
    gui = CollectionGUI(collector)
    gui._refresh_hw_status()
    gui.on_state_changed(GraspDataCollector.S_IDLE)
    gui.run()


if __name__ == '__main__':
    main()