#聚合前k个时刻的状态作为当前状态

from typing import Optional
import sys
sys.path.append('/DATACENTER4/jiangbo.chai/SWJTU_OOPAO')
import gym
from gym import spaces
import numpy as np
from collections import deque  # 添加deque导入
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.calibration.ao_calibration import ao_calibration
from tutorials.scao_system.myAoSystem.Imager import Imager
from tutorials.scao_system.parameter_files.parameterFile_VLT_I_Band_SHWFS import initializeParameterFile

class AOEnv(gym.Env):
    def __init__(self, args):
        super(AOEnv, self).__init__()
        self.param = initializeParameterFile()
        
        # 保持原始参数命名
        self.max_step = args.max_step
        self.init_gainCL = args.gainCL
        self.gainCL = self.init_gainCL
        self.sampling_rate = args.sampling_rate
        self.exposure_time = args.exposure_time
        self.clock_rate = args.clock_rate
        self.lightRatio = args.lightRatio
        
        # 初始化光学组件（保持原始结构）
        self.tel = Telescope(resolution=120, diameter=8, 
                           samplingTime=1/self.sampling_rate, 
                           centralObstruction=0)
        self.ngs = Source(magnitude=5, optBand='H')
        self.ngs * self.tel
        self.atm = Atmosphere(telescope=self.tel, r0=0.15, L0=30,
                            windSpeed=[5,10,20], fractionalR0=[0.7,0.25,0.05],
                            windDirection=[0,90,360], altitude=[0,4000,10000])
        self.dm = DeformableMirror(telescope=self.tel, nSubap=20, mechCoupling=0.1)
        self.wfs = ShackHartmann(nSubap=20, telescope=self.tel, 
                               lightRatio=0.3, is_geometric=False, 
                               threshold_cog=0.01)
        self.camH = Imager(exposure_time=self.exposure_time, 
                         clock_rate=self.clock_rate)

        # 校准矩阵（原始名称保留）
        self.ao_calib = ao_calibration(
            param=self.param, ngs=self.ngs, tel=self.tel,
            atm=self.atm, dm=self.dm, wfs=self.wfs
        )
        self.calib_CL = self.ao_calib.calib
        self.M2C_CL = self.ao_calib.M2C

        # 时间窗口参数（新增）
        self.time_window = 3  # 时间平均窗口大小
        
        # 初始化数据缓冲区（使用deque）
        self.wfs_buffer = deque(maxlen=self.time_window)  # 波前传感器图像
        self.far_buffer = deque(maxlen=self.time_window)  # 远场图像
        self.slope_buffer = deque(maxlen=self.time_window)  # 斜率信号
        self.dm_buffer = deque(maxlen=self.time_window)    # 变形镜电压

        # 保持原始状态空间维度计算
        obs_dim = (20*20 + 120*120 + 140 + 280) * self.time_window
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

        # 保持原始动作空间定义
        self.action_space = spaces.Box(
            low=np.array([0.1, 100, 0.1, 100], dtype=np.float32),
            high=np.array([1.0, 2000, 1.0, 2000], dtype=np.float32),
            dtype=np.float32
        )

        # 其他原始参数初始化
        self.current_step = 0
        self.SR = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)

    def _get_state(self):
        """构建时间平均状态向量（参考专利CN114488518B的光强处理）"""
        # 动态归一化（保持原始处理逻辑）
        wfs_frames = [self._normalize_wfs(f) for f in self.wfs_buffer]
        far_frames = [self._normalize_far(f) for f in self.far_buffer]
        
        # 时间维度拼接（参考网页6的波前重构思想）
        state = np.concatenate([
            np.array(wfs_frames).flatten(),  # 20x20 x time_window
            np.array(far_frames).flatten(), # 120x120 x time_window 
            np.array(self.slope_buffer).flatten(),  # 280 x time_window
            np.array(self.dm_buffer).flatten()      # 140 x time_window
        ])
        return state.astype(np.float32)

    def _normalize_wfs(self, frame):
        """波前传感器动态归一化（保持原始处理逻辑）"""
        return (frame - np.percentile(frame, 1)) / \
              (np.percentile(frame, 99) - np.percentile(frame, 1) + 1e-6)

    def _normalize_far(self, frame):
        """远场对数压缩（参考专利CN114488518B）"""
        return np.log1p(frame) / (np.log1p(frame.max()) + 1e-6)

    def step(self, action):
        """执行步骤（保持原始控制流程）"""
        # 动作参数处理（保持原始顺序）
        gainCL, tel_freq, exposure, cam_freq = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        
        # 更新系统参数（保持原始参数命名）
        self.gainCL = gainCL
        self.tel.samplingTime = 1 / tel_freq
        self.camH.exposure_time = exposure
        self.camH.clock_rate = cam_freq

        # 大气扰动更新（保持原始方法）
        self.atm.update()
        self.tel - self.atm
        self.camH.relay(self.tel.src)
        self.tel + self.atm

        # 波前校正（保持原始控制算法）
        self.tel * self.dm * self.wfs
        self.dm.coefs -= self.gainCL * self.M2C_CL @ self.calib_CL.M @ self.wfs.signal
        
        # 更新缓冲区（新增）
        self._update_buffers()

        # 计算奖励（保持原始逻辑）
        reward = self._calculate_reward(action)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_step
        
        return self._get_state(), reward, terminated, False, {}

    def _update_buffers(self):
        """更新数据缓冲区（参考网页1的状态空间构建方法）"""
        self.wfs_buffer.append(self.wfs.cam.frame.copy())
        self.far_buffer.append(self.camH.frame.copy())
        self.slope_buffer.append(self.wfs.signal.copy())
        self.dm_buffer.append(self.dm.coefs.copy())

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """环境重置（保持原始初始化流程）"""
        self.current_step = 0
        self.dm.coefs = np.zeros(140)
        self._clear_buffers()
        self._initialize_optical_system()
        return self._get_state(), {}

    def _clear_buffers(self):
        """清空缓冲区（新增）"""
        self.wfs_buffer.clear()
        self.far_buffer.clear()
        self.slope_buffer.clear()
        self.dm_buffer.clear()
        # 初始化填充空数据
        for _ in range(self.time_window):
            self.wfs_buffer.append(np.zeros((20,20)))
            self.far_buffer.append(np.zeros((120,120)))
            self.slope_buffer.append(np.zeros(280))
            self.dm_buffer.append(np.zeros(140))

    def _initialize_optical_system(self):
        """光学系统初始化（保持原始方法）"""
        self.atm.initializeAtmosphere(self.tel)
        self.tel - self.atm
        self.ngs * self.tel * self.dm * self.wfs
        self.camH.relay(self.tel.src)
        self.camH.reference_frame = self.camH.frame.copy()
        self.tel + self.atm