# -*- coding: utf-8 -*-
"""
Gemini 335L 相机读取器
- 适配自编译 pyorbbecsdk (cp37 v1系列)
- eye-on-base 固定俯拍
- 整合 utils.py 的 frame_to_bgr_image，无需外部依赖
- 提供 get_latest_frame() → (rgb_bgr, depth_mm, timestamp)
"""
import time
import threading
import numpy as np
import cv2

from pyorbbecsdk import (
    Pipeline, Config,
    OBSensorType, OBFormat, OBConvertFormat,
    FormatConvertFilter, VideoFrame
)

MIN_DEPTH = 20      # mm
MAX_DEPTH = 10000   # mm


# ====================================================================== #
#  frame_to_bgr_image (整合自 utils.py，无需外部文件)
# ====================================================================== #

def _i420_to_bgr(data, width, height):
    y = data[0:height, :]
    u = data[height:height + height // 4].reshape(height // 2, width // 2)
    v = data[height + height // 4:].reshape(height // 2, width // 2)
    yuv = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


def _nv12_to_bgr(data, width, height):
    y = data[0:height, :]
    uv = data[height:height + height // 2].reshape(height // 2, width)
    yuv = cv2.merge([y, uv])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


def _nv21_to_bgr(data, width, height):
    y = data[0:height, :]
    uv = data[height:height + height // 2].reshape(height // 2, width)
    yuv = cv2.merge([y, uv])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)


def frame_to_bgr_image(frame):
    """VideoFrame → BGR ndarray，支持 RGB/BGR/YUYV/MJPG/I420/NV12/NV21/UYVY"""
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if fmt == OBFormat.RGB:
        img = np.resize(data, (height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        img = np.resize(data, (height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif fmt == OBFormat.YUYV:
        img = np.resize(data, (height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    elif fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.I420:
        return _i420_to_bgr(data, width, height)
    elif fmt == OBFormat.NV12:
        return _nv12_to_bgr(data, width, height)
    elif fmt == OBFormat.NV21:
        return _nv21_to_bgr(data, width, height)
    elif fmt == OBFormat.UYVY:
        img = np.resize(data, (height, width, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("[相机] 不支持的彩色格式: {}".format(fmt))
        return None


# ====================================================================== #
#  CameraReader
# ====================================================================== #

class CameraReader:
    def __init__(self):
        self.latest_rgb = None      # np.ndarray (H, W, 3) uint8 BGR
        self.latest_depth = None    # np.ndarray (H, W) float32，单位 mm
        self.latest_ts = 0.0        # time.time() 时间戳

        self._lock = threading.Lock()
        self._pipeline = None
        self._running = False
        self._thread = None

    # ------------------------------------------------------------------ #
    #  公开接口
    # ------------------------------------------------------------------ #

    def start(self):
        """启动相机，开启彩色+深度双流，帧同步"""
        self._pipeline = Pipeline()
        config = Config()

        # ---- 彩色流（默认配置，兼容性最好）----
        try:
            profile_list = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                color_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
            except Exception:
                color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            print("[相机] 彩色流: {}x{}@{} fmt={}".format(
                color_profile.get_width(), color_profile.get_height(),
                color_profile.get_fps(), color_profile.get_format()))
        except Exception as e:
            print("[相机] ❌ 彩色流配置失败: {}".format(e))
            return False

        # ---- 深度流（默认配置）----
        try:
            profile_list = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)
            print("[相机] 深度流: {}x{}@{} fmt={}".format(
                depth_profile.get_width(), depth_profile.get_height(),
                depth_profile.get_fps(), depth_profile.get_format()))
        except Exception as e:
            print("[相机] ❌ 深度流配置失败: {}".format(e))
            return False

        # ---- 帧同步（彩色+深度时间戳对齐）----
        try:
            self._pipeline.enable_frame_sync()
            print("[相机] 帧同步已启用")
        except Exception as e:
            print("[相机] ⚠️  帧同步启用失败(不影响使用): {}".format(e))

        self._pipeline.start(config)
        time.sleep(0.8)

        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="CameraReader"
        )
        self._thread.start()
        print("[相机] ✅ 启动成功")
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        print("[相机] 已停止")

    def get_latest_frame(self):
        """
        返回 (rgb, depth, timestamp)
          rgb   : np.ndarray (H, W, 3) uint8 BGR
          depth : np.ndarray (H, W) float32，单位 mm，无效点为 0
          ts    : float，time.time() 时间戳
        任意一路未就绪则返回 (None, None, None)
        """
        with self._lock:
            if self.latest_rgb is None or self.latest_depth is None:
                return None, None, None
            return self.latest_rgb.copy(), self.latest_depth.copy(), self.latest_ts

    def is_ready(self):
        with self._lock:
            return self.latest_rgb is not None and self.latest_depth is not None

    # ------------------------------------------------------------------ #
    #  内部采集循环
    # ------------------------------------------------------------------ #

    def _read_loop(self):
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                ts = time.time()

                # ---- 彩色帧 ----
                color_frame = frames.get_color_frame()
                rgb = None
                if color_frame is not None:
                    rgb = frame_to_bgr_image(color_frame)

                # ---- 深度帧 ----
                depth_frame = frames.get_depth_frame()
                depth = None
                if depth_frame is not None:
                    w = depth_frame.get_width()
                    h = depth_frame.get_height()
                    scale = depth_frame.get_depth_scale()
                    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    raw = raw.reshape((h, w)).astype(np.float32) * scale
                    # 过滤无效距离
                    depth = np.where((raw > MIN_DEPTH) & (raw < MAX_DEPTH), raw, 0).astype(np.float32)

                # ---- 更新共享变量 ----
                with self._lock:
                    if rgb is not None:
                        self.latest_rgb = rgb
                    if depth is not None:
                        self.latest_depth = depth
                    if rgb is not None or depth is not None:
                        self.latest_ts = ts

            except Exception as e:
                print("[相机] 读取错误: {}".format(e))
                time.sleep(0.05)


# ====================================================================== #
#  独立测试：按 Q 退出，显示彩色 + 深度伪彩 + 中心距离
# ====================================================================== #
if __name__ == '__main__':
    cam = CameraReader()
    if not cam.start():
        print("相机启动失败")
        exit(1)

    print("等待相机就绪...")
    for _ in range(50):
        if cam.is_ready():
            break
        time.sleep(0.1)

    if not cam.is_ready():
        print("❌ 相机未就绪，请检查USB连接和驱动")
        cam.stop()
        exit(1)

    print("✅ 相机就绪，按 Q 退出")

    try:
        while True:
            rgb, depth, ts = cam.get_latest_frame()
            if rgb is None:
                time.sleep(0.03)
                continue

            # 深度伪彩色
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # 显示中心点距离
            h, w = depth.shape
            center_dist = depth[h // 2, w // 2]
            cv2.putText(rgb, "Center: {:.0f}mm".format(center_dist),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("RGB", rgb)
            cv2.imshow("Depth", depth_color)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()