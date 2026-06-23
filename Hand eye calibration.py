# -*- coding: utf-8 -*-
"""
手眼标定脚本 (eye-on-base)
============================
棋盘格固定在机械臂末端，相机固定在支架上。

操作流程:
1. 运行脚本，相机画面弹出
2. 用示教器手动移动机械臂到一个姿态（相机能看到完整棋盘格）
3. 按空格拍照（自动记录棋盘格位姿 + 机械臂位姿）
4. 换姿态，重复步骤 2-3，至少 15 张
5. 按 Q 结束，自动计算并保存变换矩阵
s
棋盘格参数:
- GP290-20-12×9: 12列×9行方格 = 11×8 内角点，格子边长 20mm
"""

import time
import json
import numpy as np
import cv2
import os
from datetime import datetime

# 导入硬件模块
from camera_reader import CameraReader

try:
    from armconctrl import ArmController
except ImportError:
    from arm_controller import ArmController

import config


# ====================== 棋盘格参数 ======================
BOARD_COLS = 11          # 内角点列数 (方格列数12 - 1)
BOARD_ROWS = 8           # 内角点行数 (方格行数9 - 1)
SQUARE_SIZE = 0.020      # 格子边长 (米)

# 最少采集张数
MIN_CAPTURES = 10

# 保存目录
CALIB_DIR = os.path.join(config.SAVE_ROOT, 'calibration')


def create_object_points():
    """棋盘格的3D世界坐标 (Z=0平面)"""
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def detect_chessboard(gray):
    """检测棋盘格角点，返回 (found, corners_refined)"""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return found, corners


def pose_dict_to_matrix(pose):
    """机械臂 pose dict → 4x4 齐次变换矩阵
    pose: {'pos': [x,y,z], 'ori': [w,x,y,z]}
    """
    pos = pose['pos']
    ori = pose['ori']  # AUBO 返回 (w, x, y, z)

    # 四元数 → 旋转矩阵
    w, qx, qy, qz = ori[0], ori[1], ori[2], ori[3]
    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*w),       2*(qx*qz + qy*w)],
        [2*(qx*qy + qz*w),        1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*w)],
        [2*(qx*qz - qy*w),        2*(qy*qz + qx*w),        1 - 2*(qx**2 + qy**2)]
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def main():
    os.makedirs(CALIB_DIR, exist_ok=True)

    # 加载相机内参
    intrinsics_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'camera_intrinsics.npz')
    if not os.path.exists(intrinsics_path):
        intrinsics_path = 'camera_intrinsics.npz'
    if not os.path.exists(intrinsics_path):
        print("❌ 找不到 camera_intrinsics.npz，请先运行 save_intrinsics.py")
        return

    data = np.load(intrinsics_path)
    K = data['color_K']
    dist = data['color_dist']
    print("相机内参 K:\n", K)

    # 启动相机
    print("正在启动相机...")
    cam = CameraReader()
    if not cam.start():
        print("❌ 相机启动失败")
        return
    for _ in range(50):
        if cam.is_ready():
            break
        time.sleep(0.1)
    if not cam.is_ready():
        print("❌ 相机未就绪")
        cam.stop()
        return
    print("✅ 相机就绪")

    # 启动机械臂
    print("正在连接机械臂...")
    arm = ArmController(ip=config.ARM_IP)
    arm.start()
    arm.set_speed(0.3)
    time.sleep(1.0)
    print("✅ 机械臂就绪")

    # 3D 世界点
    objp = create_object_points()

    # 采集数据存储
    R_gripper2base_list = []   # 机械臂末端→基座 旋转
    t_gripper2base_list = []   # 机械臂末端→基座 平移
    R_target2cam_list = []     # 棋盘格→相机 旋转
    t_target2cam_list = []     # 棋盘格→相机 平移
    capture_count = 0

    print("\n" + "=" * 60)
    print("手眼标定采集")
    print("=" * 60)
    print("操作说明:")
    print("  1. 用示教器移动机械臂，让相机看到完整棋盘格")
    print("  2. 按 [空格] 拍照记录")
    print("  3. 换姿态，重复上述步骤，至少 {} 张".format(MIN_CAPTURES))
    print("  4. 按 [Q] 结束并计算")
    print("=" * 60)

    try:
        while True:
            rgb, depth, ts = cam.get_latest_frame()
            if rgb is None:
                time.sleep(0.03)
                continue

            display = rgb.copy()
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

            # 实时检测棋盘格
            found, corners = detect_chessboard(gray)

            if found:
                cv2.drawChessboardCorners(display, (BOARD_COLS, BOARD_ROWS),
                                          corners, found)
                status_text = "Chessboard FOUND - Press SPACE to capture"
                status_color = (0, 255, 0)
            else:
                status_text = "Chessboard NOT found"
                status_color = (0, 0, 255)

            # 状态栏
            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display, "Captured: {}/{}".format(capture_count, MIN_CAPTURES),
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 缩放显示
            h, w = display.shape[:2]
            display = cv2.resize(display, (int(w * 0.75), int(h * 0.75)))
            cv2.imshow("Hand-Eye Calibration", display)

            key = cv2.waitKey(30)

            # 空格: 拍照
            if key == 32 and found:
                # solvePnP: 棋盘格→相机
                ret, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
                if not ret:
                    print("  ⚠️ solvePnP 失败，跳过")
                    continue

                R_cam, _ = cv2.Rodrigues(rvec)

                # 读取机械臂末端位姿
                arm_pose, _ = arm.get_current_pose()
                if arm_pose is None:
                    print("  ⚠️ 机械臂位姿读取失败，跳过")
                    continue

                T_gripper2base = pose_dict_to_matrix(arm_pose)

                # 存入列表
                R_gripper2base_list.append(T_gripper2base[:3, :3])
                t_gripper2base_list.append(T_gripper2base[:3, 3].reshape(3, 1))
                R_target2cam_list.append(R_cam)
                t_target2cam_list.append(tvec.reshape(3, 1))

                capture_count += 1
                print("  ✅ 第 {} 张采集成功  arm_pos=({:.4f}, {:.4f}, {:.4f})".format(
                    capture_count,
                    arm_pose['pos'][0], arm_pose['pos'][1], arm_pose['pos'][2]))

                # 保存图片
                cv2.imwrite(os.path.join(CALIB_DIR,
                            'calib_{:03d}.png'.format(capture_count)), rgb)

            # Q: 退出
            elif key == ord('q') or key == 27:
                break

    finally:
        cv2.destroyAllWindows()

    # ================================================================== #
    #  计算手眼标定
    # ================================================================== #

    if capture_count < 3:
        print("\n❌ 采集数量不足 (最少3张)，无法标定")
        arm.stop()
        cam.stop()
        return

    if capture_count < MIN_CAPTURES:
        print("\n⚠️ 采集 {} 张，少于建议的 {} 张，结果可能不够精确".format(
            capture_count, MIN_CAPTURES))

    print("\n计算手眼标定 ({} 组数据)...".format(capture_count))

    # OpenCV calibrateHandEye
    # eye-on-base: 相机固定, 棋盘格在机械臂末端
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base_list,
        t_gripper2base=t_gripper2base_list,
        R_target2cam=R_target2cam_list,
        t_target2cam=t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # 组装 4x4 变换矩阵
    T_cam2base = np.eye(4, dtype=np.float64)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()

    print("\n相机→机器人基座 变换矩阵:")
    print(T_cam2base)

    # 验证: 旋转矩阵行列式应接近1
    det = np.linalg.det(R_cam2base)
    print("\n旋转矩阵行列式: {:.6f} (应接近1.0)".format(det))

    # 保存
    save_path = os.path.join(CALIB_DIR, 'hand_eye_transform.npz')
    np.savez(save_path,
             T_cam2base=T_cam2base,
             R_cam2base=R_cam2base,
             t_cam2base=t_cam2base,
             capture_count=capture_count,
             calibrated_at=str(datetime.now()))

    # 同时保存一份到项目根目录
    np.savez('hand_eye_transform.npz',
             T_cam2base=T_cam2base,
             R_cam2base=R_cam2base,
             t_cam2base=t_cam2base,
             capture_count=capture_count,
             calibrated_at=str(datetime.now()))

    print("\n✅ 标定结果已保存:")
    print("  - {}".format(save_path))
    print("  - hand_eye_transform.npz")

    # 保存详细日志
    log = {
        'board': {'cols': BOARD_COLS, 'rows': BOARD_ROWS, 'square_size': SQUARE_SIZE},
        'capture_count': capture_count,
        'det_R': float(det),
        'T_cam2base': T_cam2base.tolist(),
        'calibrated_at': datetime.now().isoformat(),
    }
    with open(os.path.join(CALIB_DIR, 'calibration_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    # 清理
    arm.stop()
    cam.stop()
    print("\n标定完成！")


if __name__ == '__main__':
    main()