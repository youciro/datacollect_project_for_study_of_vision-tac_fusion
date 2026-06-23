import numpy as np
import cv2

# 改成你要查看的样本路径
sample_dir = 'D:/grasp_data/sample_00087'

# 读取深度图
depth = np.load(sample_dir + '/depth_before.npy')

# 伪彩色显示
depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# 中心点距离
h, w = depth.shape
print('深度图尺寸:', depth.shape)
print('中心点距离: {:.0f}mm'.format(depth[h//2, w//2]))
print('最小值: {:.0f}mm  最大值: {:.0f}mm'.format(depth.min(), depth.max()))

# 显示
cv2.imshow('depth_before', depth_color)

# 如果有 depth_after 也显示
import os
if os.path.exists(sample_dir + '/depth_after.npy'):
    depth_after = np.load(sample_dir + '/depth_after.npy')
    depth_after_vis = cv2.normalize(depth_after, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_after_color = cv2.applyColorMap(depth_after_vis, cv2.COLORMAP_JET)
    cv2.imshow('depth_after', depth_after_color)

cv2.waitKey(0)
cv2.destroyAllWindows()