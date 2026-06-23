import numpy as np
import cv2

# 改成你要查看的样本路径
sample_dir = 'D:/grasp_data/sample_00087'

# 读取触觉序列
tactile_seq = np.load(sample_dir + '/tactile_seq.npy')  # shape: (N, 3, 15, 5)
tactile_ts = np.load(sample_dir + '/tactile_ts.npy')

print('触觉序列形状:', tactile_seq.shape)
print('帧数: {}  时长: {:.1f}s'.format(
    len(tactile_seq), tactile_ts[-1] - tactile_ts[0]))

# 参数
NOISE_HIGH = 10000
cell = 26
finger_w = 5 * cell
finger_h = 15 * cell
gap = 40
margin_top = 70
margin_side = 30
margin_bot = 30

def render_frame(tactile):
    canvas_w = margin_side * 2 + finger_w * 3 + gap * 2
    canvas_h = margin_top + finger_h + margin_bot
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)
    vmax = max(NOISE_HIGH * 0.1, float(tactile.max()))
    for i in range(3):
        finger = tactile[i]
        norm = np.clip(finger / vmax * 255.0, 0, 255).astype(np.uint8)
        big = cv2.resize(norm, (finger_w, finger_h), interpolation=cv2.INTER_NEAREST)
        colored = cv2.applyColorMap(big, cv2.COLORMAP_HOT)
        x0 = margin_side + i * (finger_w + gap)
        y0 = margin_top
        canvas[y0:y0+finger_h, x0:x0+finger_w] = colored
        for r in range(16):
            cv2.line(canvas, (x0, y0+r*cell), (x0+finger_w, y0+r*cell), (60,60,60), 1)
        for c in range(6):
            cv2.line(canvas, (x0+c*cell, y0), (x0+c*cell, y0+finger_h), (60,60,60), 1)
        cv2.putText(canvas, 'F{}'.format(i+1), (x0, y0-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    total = float(tactile.sum())
    cv2.putText(canvas, 'Total={:.0f}  vmax={:.0f}'.format(total, vmax),
                (margin_side, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 1)
    return canvas

# 播放模式: 按空格暂停/继续, 按Q退出, 左右方向键逐帧
print('操作: 空格=暂停/继续  Q=退出  左右方向键=逐帧')
paused = False
idx = 0
while True:
    frame = render_frame(tactile_seq[idx])

    # 进度条
    progress = int(idx / len(tactile_seq) * frame.shape[1])
    cv2.rectangle(frame, (0, frame.shape[0]-8),
                  (progress, frame.shape[0]), (0, 200, 100), -1)
    cv2.putText(frame, 'Frame {}/{}'.format(idx+1, len(tactile_seq)),
                (margin_side, frame.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow('Tactile Sequence', frame)
    key = cv2.waitKey(50 if not paused else 0)

    if key == ord('q') or key == 27:
        break
    elif key == 32:  # 空格
        paused = not paused
    elif key == 83 or key == 3:  # 右方向键
        idx = min(idx + 1, len(tactile_seq) - 1)
        paused = True
    elif key == 81 or key == 2:  # 左方向键
        idx = max(idx - 1, 0)
        paused = True
    elif not paused:
        idx = (idx + 1) % len(tactile_seq)

cv2.destroyAllWindows()