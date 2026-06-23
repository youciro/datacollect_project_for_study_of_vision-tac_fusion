# from tactile_reader import TactileReader
# import time
# reader = TactileReader(port='COM14', baudrate=1000000, noise_low=0, noise_high=5000)
# reader.start()
# time.sleep(2)
# data = reader.get_latest()
# per_finger = data.sum(axis=(1,2))
# print('三指空载总力:', per_finger)
# print('均衡度(标准差/均值):', per_finger.std()/per_finger.mean())
# reader.stop()


import numpy as np
import json
import os

# 空载基线
BASELINE_PER_FINGER = np.array([5988., 5994., 5992.])
BASELINE_PER_POINT = (BASELINE_PER_FINGER / 75)[:, np.newaxis, np.newaxis]  # (3,1,1)

def compute_force_center(tactile):
    finger_pos = np.array([
        [-50.0, 0.0],  # 指1：九点钟
        [25.0, -43.3],  # 指2：五点钟
        [25.0, 43.3],  # 指3：一点钟
    ])
    finger_forces = tactile.sum(axis=(1, 2))
    total = finger_forces.sum()
    if total < 1e-6:
        return 0.0, 0.0
    cx = (finger_forces * finger_pos[:, 0]).sum() / total
    cy = (finger_forces * finger_pos[:, 1]).sum() / total
    return cx, cy

data_root = 'D:/grasp_data'

results = {'success': [], 'radial_offset': [], 'grasp_miss': [],
           'too_shallow': [], 'too_deep': []}

for folder in sorted(os.listdir(data_root)):
    sample_dir = os.path.join(data_root, folder)
    meta_path = os.path.join(sample_dir, 'meta.json')
    tac_path = os.path.join(sample_dir, 'tactile_seq.npy')
    if not os.path.exists(meta_path) or not os.path.exists(tac_path):
        continue

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    label = meta['result_label']
    if label not in results:
        continue

    tactile_seq = np.load(tac_path)  # (N, 3, 15, 5)

    # 把脚本里取帧的部分改成这样
    ts = np.load(tac_path.replace('tactile_seq', 'tactile_ts'))
    t_contact = meta['timestamps']['contact']
    t_lifted = meta['timestamps']['lifted']

    # 取接触后0.3秒到提起前0.3秒的稳定段
    mask = (ts >= t_contact + 0.3) & (ts <= t_lifted - 0.3)
    if mask.sum() < 3:
        # 稳定段太短，取接触后的所有帧
        mask = ts >= t_contact + 0.3


    tactile_mean = tactile_seq[mask].mean(axis=0)

    # 减去空载基线
    tactile_cal = tactile_mean - BASELINE_PER_POINT
    tactile_cal = np.clip(tactile_cal, 0, None)

    total_force = float(tactile_cal.sum())
    cx, cy = compute_force_center(tactile_cal)
    magnitude = np.sqrt(cx**2 + cy**2)

    results[label].append({
        'id': meta['sample_id'],
        'total': total_force,
        'cx': cx, 'cy': cy,
        'magnitude': magnitude
    })

# 打印统计
for label, samples in results.items():
    if not samples:
        continue
    totals = [s['total'] for s in samples]
    mags = [s['magnitude'] for s in samples]
    print('{} ({} 个样本):'.format(label, len(samples)))
    print('  校准后总力: 均值={:.0f}  std={:.0f}'.format(
        np.mean(totals), np.std(totals)))
    print('  力心偏移: 均值={:.1f}mm  std={:.1f}mm'.format(
        np.mean(mags), np.std(mags)))
    print()