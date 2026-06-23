# -*- coding: utf-8 -*-
"""
AUBO i5机械臂控制器
- 封装常用操作: connect/move_to_pose/get_pose
- 后台线程持续获取关节状态
"""
import time
import threading
import numpy as np

from robotcontrol import (
    Auboi5Robot, RobotErrorType, RobotError,
    logger, logger_init
)


class ArmController:
    def __init__(self, ip='192.168.100.1', port=8899):
        self.ip = ip
        self.port = port
        self.robot = None
        self.connected = False

        # 后台获取最新位姿
        self.latest_pose = None  # {'joint':..., 'pos':..., 'ori':...}
        self.latest_ts = 0
        self.lock = threading.Lock()

        self.running = False
        self.thread = None

        # 预设的Home位姿(改成你实际的)弧度值
        self.home_joint = (1.833858, -0.044208, -1.540935, 0.071174, -1.582123, 1.107987)

    def start(self):
        logger_init()
        Auboi5Robot.initialize()
        self.robot = Auboi5Robot()
        self.robot.create_context()

        result = self.robot.connect(self.ip, self.port)
        if result != RobotErrorType.RobotError_SUCC:
            raise RuntimeError(f"AUBO连接失败,错误码{result}")

        self.robot.robot_startup()
        time.sleep(2)

        self.robot.set_collision_class(7)
        self.robot.init_profile()

        # 默认中速
        self.set_speed(0.5)

        self.connected = True

        # 启动后台获取位姿
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

        print("[机械臂] 启动成功")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

        if self.connected:
            self.robot.robot_shutdown()
            self.robot.disconnect()
            self.connected = False
        Auboi5Robot.uninitialize()
        print("[机械臂] 已停止")

    def set_speed(self, scale=0.5):
        """speed scale: 0.1-1.0,小心从0.3开始测"""
        acc = (scale * 1.5,) * 6
        vel = (scale * 1.5,) * 6
        self.robot.set_joint_maxacc(acc)
        self.robot.set_joint_maxvelc(vel)

    def _update_loop(self):
        """后台10Hz更新位姿"""
        while self.running:
            try:
                wp = self.robot.get_current_waypoint()
                with self.lock:
                    self.latest_pose = wp
                    self.latest_ts = time.time()
                time.sleep(0.05)
            except Exception as e:
                print(f"[机械臂] 状态获取错误: {e}")
                time.sleep(0.1)

    def get_current_pose(self):
        """获取最新位姿"""
        with self.lock:
            if self.latest_pose is None:
                return None
            return dict(self.latest_pose), self.latest_ts

    def move_to_joint(self, joint_radian, wait=True):
        """运动到关节角"""
        if not self.connected:
            return False
        result = self.robot.move_joint(tuple(joint_radian))
        return result == RobotErrorType.RobotError_SUCC

    def move_to_pose(self, x, y, z, ori=None):
        """运动到笛卡尔位置(用逆解)
        x, y, z: 位置(米)
        ori: 姿态(四元数w,x,y,z) 默认末端朝下
        """
        if ori is None:
            # 默认: 末端Z轴朝下(夹爪向下)
            ori = (0, 0, 1, 0)  # 你需要根据实际工具坐标系调整

        # 当前关节角作为seed
        current = self.get_current_pose()[0]
        seed = current['joint']

        # 逆解
        ik_result = self.robot.inverse_kin(seed, (x, y, z), ori)
        if ik_result is None:
            print(f"[机械臂] 逆解失败 ({x},{y},{z})")
            return False

        return self.move_to_joint(ik_result['joint'])

    def move_relative_z(self, dz):
        """末端相对运动Z方向(用于提起)"""
        current = self.get_current_pose()[0]
        new_z = current['pos'][2] + dz
        return self.move_to_pose(current['pos'][0], current['pos'][1], new_z, current['ori'])

    def go_home(self):
        return self.move_to_joint(self.home_joint)


# ====== 单独测试 ======
if __name__ == '__main__':
    arm = ArmController(ip='192.168.100.1')
    arm.start()

    try:
        arm.set_speed(0.1)

        # 读当前位姿
        pose, ts = arm.get_current_pose()
        print(f"当前: {pose}")

        # 测试: Z方向上移50cm再下来
        #print("Z上移50cm...")
        #arm.move_relative_z(0.5)
        #time.sleep(2)

        print("Z上移2cm...")
        arm.move_relative_z(0.02)
        time.sleep(2)

    finally:
        arm.stop()