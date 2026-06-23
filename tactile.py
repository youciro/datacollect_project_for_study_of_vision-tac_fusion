import serial
import serial.tools.list_ports
import numpy as np
import time

# ==================== 核心配置 ====================
ROWS = 32  # 传感器行数
COLS = 32  # 传感器列数
BAUDRATE = 921600  # 波特率
TIMEOUT = 0.1  # 串口超时（秒）

# ✅ 大小端序控制（True=大端序，False=小端序）
ENDIAN_MODE = True  # 默认大端序（高位在前）


# ==================== 工具函数 ====================
def get_available_serial_ports():
    """获取所有可用串口"""
    return [port.device for port in serial.tools.list_ports.comports()]


def open_serial(port):
    """打开串口并返回串口对象"""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = BAUDRATE
    ser.parity = 'N'
    ser.bytesize = 8
    ser.stopbits = 1
    ser.timeout = TIMEOUT

    try:
        ser.open()
        if ser.isOpen():
            print(f"✅ 串口 {port} 打开成功")
            return ser
        print(f"❌ 串口 {port} 打开失败")
        return None
    except Exception as e:
        print(f"❌ 打开串口出错: {e}")
        return None


def parse_raw_data(raw_bytes):
    """
    解析原始字节流 → 十进制数组
    ENDIAN_MODE=True  → 大端序（高位在前，低位在后）
    ENDIAN_MODE=False → 小端序（低位在前，高位在后）
    """
    raw_data = []

    # 每2个字节（4个十六进制字符）解析为一个16位整数
    for i in range(0, len(raw_bytes), 2):
        high_byte = raw_bytes[i]
        low_byte = raw_bytes[i + 1] if i + 1 < len(raw_bytes) else 0

        if ENDIAN_MODE:
            # 大端序：高位在前，低位在后
            value = (high_byte << 8) | low_byte
        else:
            # 小端序：低位在前，高位在后
            value = (low_byte << 8) | high_byte

        # 原始代码中的噪声过滤逻辑
        raw_data.append(0 if value >= 3000 else value)

    return raw_data


def read_and_parse_frame(ser):
    """从串口读取一帧数据，解析为ROWS×COLS的十进制数组"""
    # 1. 找帧头（0xFF 0xFF）
    while True:
        if ser.read() == b'\xff' and ser.read() == b'\xff':
            break
        time.sleep(0.001)

    # 2. 读取一帧原始数据（每个采样值占2字节）
    raw_bytes = ser.read(ROWS * COLS * 2)
    if len(raw_bytes) < ROWS * COLS * 2:
        print("⚠️ 数据不足，跳过当前帧")
        return None

    # 3. 解析字节 → 十进制数组
    raw_data = parse_raw_data(raw_bytes)

    # 4. 修复帧同步（将帧头后的数据移到前面）
    frame_start_idx = 0
    for idx, val in enumerate(raw_data):
        if val == 0xFFFF:  # 找到帧头位置
            frame_start_idx = idx
            break

    reordered = raw_data[frame_start_idx:] + raw_data[:frame_start_idx]

    # 5. 转为标准二维数组
    return np.array(reordered, dtype=np.float32).reshape(ROWS, COLS)


# ==================== 主流程 ====================
if __name__ == "__main__":
    # 1. 选择串口
    ports = get_available_serial_ports()
    if not ports:
        print("❌ 未找到可用串口")
        exit()
    print(f"可用串口: {ports}")
    target_port = ports[0]

    # 2. 打开串口
    ser = open_serial(target_port)
    if not ser:
        exit()

    # 3. 持续采集数据
    try:
        while True:
            frame = read_and_parse_frame(ser)
            if frame is not None:
                print(f"📊 帧数据 | 形状: {frame.shape} | "
                      f"最大值: {frame.max():.2f} | "
                      f"端序: {'大端' if ENDIAN_MODE else '小端'}")
    except KeyboardInterrupt:
        print("\n🛑 程序退出")
    finally:
        ser.close()
        print("✅ 串口已关闭")