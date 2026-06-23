import numpy as np

# Gemini 335L 出厂内参 (从SDK日志读取)

# 彩色相机 640x480
color_K = np.array([
    [366.298,   0.0,   319.701],
    [  0.0,   366.423, 241.678],
    [  0.0,     0.0,     1.0  ]
], dtype=np.float64)

# 深度相机 640x400
depth_K = np.array([
    [305.249,   0.0,   319.917],
    [  0.0,   305.353, 201.398],
    [  0.0,     0.0,     1.0  ]
], dtype=np.float64)

# 畸变系数（出厂标定通常畸变已校正，设为0）
color_dist = np.zeros(5, dtype=np.float64)
depth_dist = np.zeros(5, dtype=np.float64)

np.savez('camera_intrinsics.npz',
         color_K=color_K, color_dist=color_dist,
         depth_K=depth_K, depth_dist=depth_dist,
         color_resolution=(640, 480),
         depth_resolution=(640, 400))

print("已保存 camera_intrinsics.npz")
print("彩色内参 K:\n", color_K)
print("深度内参 K:\n", depth_K)