import pathlib
import numpy as np
from BCQ import ReplayBuffer

DREAMER_TRAINDIR = '/home/jiangbo.chai/DATACENTER4/SWJTU_OOPAO/tutorials/scao_system/myAoSystem/RL4AO/agent/dreamerv3/logdir/AO1_R1DISC_lightRatio0.30/train_eps'

# 建议你先用一个文件打印shape，确认下面的维度
sample_file = next(pathlib.Path(DREAMER_TRAINDIR).glob("*.npz"))
sample = dict(np.load(sample_file, allow_pickle=True))
img_shape = sample['image'][0].flatten().shape[0]
dm_shape = sample['dmCoefs'][0].flatten().shape[0]
wfs_shape = sample['wfsSingnal'][0].flatten().shape[0]
turb_shape = sample['turb_features'][0].flatten().shape[0]
STATE_DIM = img_shape + dm_shape + wfs_shape + turb_shape
ACTION_DIM = 19+20+19+20  # MultiDiscrete总动作数

offline_buffer = ReplayBuffer(STATE_DIM, ACTION_DIM)

npz_files = list(pathlib.Path(DREAMER_TRAINDIR).glob("*.npz"))
print(f"找到 {len(npz_files)} 个 npz 文件")
for npz_file in npz_files:
    episode = dict(np.load(npz_file, allow_pickle=True))
    images = episode['image']
    dmCoefs = episode['dmCoefs']
    wfsSingnal = episode['wfsSingnal']
    turb_features = episode['turb_features']
    actions_onehot = episode['action']
    rewards = episode['reward']
    dones = episode.get('is_terminal', episode.get('is_last'))

    states = [
        np.concatenate([
            img.flatten(),
            dm.flatten(),
            wfs.flatten(),
            turb.flatten()
        ])
        for img, dm, wfs, turb in zip(images, dmCoefs, wfsSingnal, turb_features)
    ]
    actions = np.argmax(actions_onehot, axis=1)

    for i in range(len(states)-1):
        offline_buffer.add(states[i], actions[i], states[i+1], rewards[i+1], dones[i+1])
        # 可选：打印进度
        # print(f"Added transition {i+1}/{len(states)-1}: action={actions[i]}, reward={rewards[i+1]}, done={dones[i+1]}")
print(f"总共装载了 {offline_buffer.size} 条数据到ReplayBuffer。")