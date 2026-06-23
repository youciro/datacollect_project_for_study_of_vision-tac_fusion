# -*- coding: utf-8 -*-
"""
柔触三指夹爪触觉读取器
========================
硬件协议:
- 帧头: 0xFFFF (两字节)
- 数据: 3指 × (15行×5列×2字节) = 450字节
- 无校验位
- 波特率: 921600
- 采样频率: 50Hz (硬件配置)

数据排列:
- 三指顺序: 指1全部 → 指2全部 → 指3全部
- 单指内: 行优先, 第0行5列 → 第1行5列 → ... → 第14行5列
- 单点: 16位整数, 默认大端序
"""

import serial
import serial.tools.list_ports
import threading
import time
import numpy as np


class TactileReader:
    # ========== 硬件协议常量 ==========
    HEADER = b'\xff\xff'
    FINGERS = 3
    ROWS_PER_FINGER = 15
    COLS_PER_FINGER = 5
    POINTS_PER_FINGER = ROWS_PER_FINGER * COLS_PER_FINGER   # 75
    BYTES_PER_POINT = 2
    BYTES_PER_FINGER = POINTS_PER_FINGER * BYTES_PER_POINT  # 150
    TOTAL_DATA_BYTES = FINGERS * BYTES_PER_FINGER           # 450

    def __init__(self,
                 port=None,
                 baudrate=921600,
                 freq=50,
                 buffer_seconds=5.0,
                 endian_big=True,
                 noise_high=5000,
                 noise_low=10,
                 finger_order=(0, 1, 2)):
        """
        参数说明:
        ----------
        port: 串口号 (Windows 上是 'COM3' 等), None=自动选第一个
        baudrate: 波特率, 默认 921600
        freq: 采样频率Hz, 50Hz (硬件配置后的预期频率)
        buffer_seconds: 环形缓冲保留多少秒数据
        endian_big: True=大端序(高位在前), False=小端序
        noise_high: 噪声上限, 值 >= 此值置0 (默认 3000, 过滤高位噪点)
        noise_low: 噪声下限, 值 <= 此值置0 (默认 50, 过滤本底)
        finger_order: 物理指顺序映射, 默认数据顺序与物理指顺序一致(0,1,2)
                      如发现指1按压时数组指2响应, 改为 (1,0,2) 等
        """
        # 基本配置
        self.port = port
        self.baudrate = baudrate
        self.freq = freq
        self.endian_big = endian_big
        self.noise_high = noise_high
        self.noise_low = noise_low
        self.finger_order = finger_order

        # 环形缓冲 (capacity, 3, 15, 5)
        self.capacity = int(buffer_seconds * freq)
        self.data_buffer = np.zeros(
            (self.capacity, self.FINGERS, self.ROWS_PER_FINGER, self.COLS_PER_FINGER),
            dtype=np.float32
        )
        self.ts_buffer = np.zeros(self.capacity, dtype=np.float64)
        self.write_idx = 0
        self.count = 0
        self.buf_lock = threading.Lock()

        # 最新值缓存(快速访问,不加锁)
        self.latest = np.zeros(
            (self.FINGERS, self.ROWS_PER_FINGER, self.COLS_PER_FINGER),
            dtype=np.float32
        )
        self.latest_total_force = 0.0
        self.latest_ts = 0.0

        # 零偏 (预留接口, 默认不启用)
        self.zero_bias = None  # 设置后会自动用于校准

        # 统计
        self.frame_count = 0
        self.start_time = 0.0

        # 运行状态
        self.ser = None
        self.running = False
        self.thread = None

    # ========================================================
    # 串口管理
    # ========================================================

    @staticmethod
    def list_available_ports():
        """列出所有可用串口"""
        return [p.device for p in serial.tools.list_ports.comports()]

    def _open_serial(self):
        if self.port is None:
            ports = self.list_available_ports()
            if not ports:
                raise RuntimeError("找不到可用串口")
            self.port = ports[0]
            print(f"[触觉] 自动选择串口 {self.port}")

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            parity='N',
            bytesize=8,
            stopbits=1,
            timeout=0.5,
        )
        if not self.ser.isOpen():
            raise RuntimeError(f"串口 {self.port} 打开失败")
        print(f"[触觉] 串口 {self.port}@{self.baudrate} 打开成功")

        # 清缓冲, 等待硬件稳定
        time.sleep(0.5)
        self.ser.reset_input_buffer()

    # ========================================================
    # 启动 / 停止
    # ========================================================

    def start(self):
        """启动后台读取线程"""
        self._open_serial()
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print("[触觉] 后台线程已启动")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.ser and self.ser.isOpen():
            self.ser.close()
        print("[触觉] 已停止")

    # ========================================================
    # 核心: 读取与解析
    # ========================================================

    def _sync_header(self, timeout_sec=2.0):
        """字节级同步: 找到连续两个 0xFF
        成功返回True, 超时返回False"""
        last_byte = b''
        timeout_time = time.time() + timeout_sec

        while time.time() < timeout_time:
            b = self.ser.read(1)
            if len(b) == 0:
                continue
            if last_byte == b'\xff' and b == b'\xff':
                return True
            last_byte = b
        return False

    def _read_one_frame(self):
        """读一帧, 返回 shape=(3, 15, 5) 的np.float32, 失败返回None"""
        # 1. 找帧头
        if not self._sync_header():
            return None

        # 2. 读 450 字节数据
        raw = self.ser.read(self.TOTAL_DATA_BYTES)
        if len(raw) != self.TOTAL_DATA_BYTES:
            return None

        # 3. 字节 → 16位整数数组
        dtype = '>u2' if self.endian_big else '<u2'
        values = np.frombuffer(raw, dtype=dtype).astype(np.float32)

        # 4. 双阈值过滤
        values[values >= self.noise_high] = 0
        values[values <= self.noise_low] = 0

        # 5. 重塑为 (3指, 15行, 5列) — 行优先reshape
        tactile = values.reshape(self.FINGERS, self.ROWS_PER_FINGER, self.COLS_PER_FINGER)

        # 6. 三指顺序映射
        if self.finger_order != (0, 1, 2):
            tactile = tactile[list(self.finger_order)]

        # 7. (可选) 零偏校准 — 现在 zero_bias 为 None, 跳过
        if self.zero_bias is not None:
            tactile = tactile - self.zero_bias
            tactile = np.clip(tactile, 0, None)  # 不允许负值

        return tactile

    def _read_loop(self):
        """后台线程主循环"""
        while self.running:
            try:
                frame = self._read_one_frame()
                if frame is None:
                    continue

                ts = time.time()

                # 写入环形缓冲
                with self.buf_lock:
                    self.data_buffer[self.write_idx] = frame
                    self.ts_buffer[self.write_idx] = ts
                    self.write_idx = (self.write_idx + 1) % self.capacity
                    self.count = min(self.count + 1, self.capacity)

                # 更新最新值
                self.latest = frame
                self.latest_total_force = float(frame.sum())
                self.latest_ts = ts
                self.frame_count += 1

            except Exception as e:
                print(f"[触觉] 读取异常: {e}")
                time.sleep(0.01)

    # ========================================================
    # 零偏校准 (现在不用, 接口预留)
    # ========================================================

    def calibrate_zero_bias(self, num_frames=50, wait_seconds=1.0):
        """夹爪空载时调用, 采集多帧求平均作为零偏
        之后所有get_latest/get_history返回值都会自动减去零偏并clip到0
        """
        print(f"[触觉] 开始零偏校准, 请保持夹爪空载...")
        time.sleep(wait_seconds)

        frames = []
        for _ in range(num_frames):
            frames.append(self.latest.copy())
            time.sleep(1.0 / self.freq)

        self.zero_bias = np.mean(frames, axis=0)
        print(f"[触觉] 零偏校准完成, 平均零偏={self.zero_bias.mean():.2f}")
        return self.zero_bias

    def reset_zero_bias(self):
        """取消零偏校准, 恢复原始输出"""
        self.zero_bias = None
        print("[触觉] 已取消零偏校准")

    # ========================================================
    # 数据访问 API
    # ========================================================

    def get_latest(self):
        """最新一帧 shape=(3,15,5)"""
        return self.latest.copy()

    def get_total_force(self):
        """当前所有225点的总力"""
        return self.latest_total_force

    def get_per_finger_force(self):
        """每指总力 shape=(3,)"""
        #return self.latest.sum(axis=(1, 2))
        """每指平均压力"""
        return self.latest.mean(axis=(1, 2))

    def get_history(self, seconds=3.0):
        """最近N秒数据, 返回 (data, timestamps) 按时间顺序"""
        with self.buf_lock:
            if self.count == 0:
                return None, None

            now = time.time()
            if self.count < self.capacity:
                data = self.data_buffer[:self.count].copy()
                ts = self.ts_buffer[:self.count].copy()
            else:
                data = np.concatenate([
                    self.data_buffer[self.write_idx:],
                    self.data_buffer[:self.write_idx]
                ], axis=0)
                ts = np.concatenate([
                    self.ts_buffer[self.write_idx:],
                    self.ts_buffer[:self.write_idx]
                ])

            mask = ts > (now - seconds)
            return data[mask], ts[mask]

    def get_actual_freq(self):
        """实际采样频率"""
        elapsed = time.time() - self.start_time
        if elapsed < 1:
            return 0
        return self.frame_count / elapsed


# ============================================================
# 独立测试: 文字打印 + 热力图可视化
# ============================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # ========== 配置 ==========
    # 热力图朝向: 'vertical' = 15高×5宽 (贴合手指实际形状)
    #            'horizontal' = 5高×15宽 (横置, 屏幕窄)
    HEATMAP_ORIENTATION = 'vertical'   # 改这里切换显示朝向

    print("可用串口:", TactileReader.list_available_ports())

    reader = TactileReader(
        port='COM14',             # None=自动选第一个
        baudrate=1000000,
        freq=20,               # 期望50Hz
        endian_big=True,
        noise_high=5000,
        noise_low=0,
        finger_order=(0, 1, 2),
    )

    reader.start()
    print("等待数据流稳定...")
    time.sleep(2)

    # === 第一阶段: 文字打印10秒 ===
    print("\n[阶段1] 文字打印10秒")
    print("=" * 70)
    for i in range(20):
        time.sleep(0.5)
        total = reader.get_total_force()
        per_finger = reader.get_per_finger_force()
        freq = reader.get_actual_freq()
        print(f"[{i:2d}] 频率={freq:5.1f}Hz  总力={total:9.1f}  "
              f"指1={per_finger[0]:7.1f}  指2={per_finger[1]:7.1f}  指3={per_finger[2]:7.1f}")

    # === 第二阶段: 实时热力图 (Ctrl+C 退出) ===
    print(f"\n[阶段2] 实时热力图 (方向: {HEATMAP_ORIENTATION}, Ctrl+C 退出)")
    print("=" * 70)

    # 自适应布局
    if HEATMAP_ORIENTATION == 'vertical':
        # 15高×5宽, 三个垂直长条, 横排
        fig, axes = plt.subplots(1, 3, figsize=(9, 9))
        display_shape = (15, 5)
    else:  # horizontal
        # 5高×15宽, 三个横长条, 竖排
        fig, axes = plt.subplots(3, 1, figsize=(12, 6))
        display_shape = (5, 15)

    images = []
    for i, ax in enumerate(axes):
        # 初始数据: 全0
        data_init = np.zeros(display_shape)
        img = ax.imshow(data_init, cmap='hot', vmin=0, vmax=reader.noise_high, # ← 这里修改热力图范围
                       aspect='auto', origin='upper')
        ax.set_title(f'指 {i + 1}')
        if HEATMAP_ORIENTATION == 'vertical':
            ax.set_xlabel('列 (5)')
            ax.set_ylabel('行 (15, 0=根部, 14=指尖)')
        else:
            ax.set_xlabel('行 (15, 0=根部 → 14=指尖)')
            ax.set_ylabel('列 (5)')
        images.append(img)
        plt.colorbar(img, ax=ax)

    plt.tight_layout()
    plt.ion()
    plt.show()

    try:
        while True:
            tactile = reader.get_latest()  # (3, 15, 5)
            # 在while循环里, 计算所有手指数据的最大值
            max_val = max(reader.noise_high * 0.1, tactile.max())  # 至少noise_high的10%作为下限避免全黑
            print(f"当前最大值: {tactile.max():.0f}")

            for i in range(3):
                if HEATMAP_ORIENTATION == 'vertical':
                    # 直接显示 (15, 5)
                    images[i].set_data(tactile[i])
                else:  # horizontal
                    # 转置成 (5, 15)
                    images[i].set_data(tactile[i].T)
                images[i].set_clim(vmin=0, vmax=max_val)  # 动态调整色阶

            total = reader.get_total_force()
            per_finger = reader.get_per_finger_force()
            fig.suptitle(
                f'总力={total:.0f}  '
                f'指力=[{per_finger[0]:.0f}, {per_finger[1]:.0f}, {per_finger[2]:.0f}]  '
                f'频率={reader.get_actual_freq():.1f}Hz'
            )
            plt.pause(0.05)

    except KeyboardInterrupt:
        print("\n用户终止")
    finally:
        plt.ioff()
        reader.stop()