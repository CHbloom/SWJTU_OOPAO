#聚合前k个时刻的状态作为当前状态

from typing import Optional
import sys
sys.path.append('/DATACENTER5/jicheng.liu/SWJTU_OOPAO/SWJTU_OOPAO')
import gym
from gym import spaces
import numpy as np
import time
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.calibration.ao_calibration import ao_calibration
from tutorials.scao_system.myAoSystem.Imager import Imager
import matplotlib.pyplot as plt
from collections import deque 
# 核心新增：导入 resize 函数
from skimage.transform import resize

import importlib

def load_parameter_file(file_name):
    """动态加载指定参数文件"""
    try:
        module = importlib.import_module(f'envs.paramsettings.{file_name}')
        return module.initializeParameterFile()
    except ImportError:
        raise ValueError(f"参数文件 {file_name} 不存在")

class AOEnv(gym.Env):
    def __init__(self, args):
        self.param = load_parameter_file(args.paramFile)
        # 获取命令行参数
        self.args = args
        self.max_step = args.max_step
        self.init_gainCL = args.gainCL
        self.gainCL = self.init_gainCL
        self.sampling_rate = args.sampling_rate
        self.exposure_time = args.exposure_time
        self.clock_rate = args.clock_rate
        # args.lightRatio 默认0.3，如果命令行未输入lightRatio，就使用param中的值
        # if args.lightRatio==0.3:
        #     self.lightRatio = self.param['lightRatio']
        # else:
        #     self.lightRatio = args.lightRatio
        # self.max_step = 200
        # self.init_gainCL = 0.6
        # self.gainCL = self.init_gainCL
        # self.sampling_rate = 1000
        # self.exposure_time = 1
        # self.clock_rate = 1000
        # self.lightRatio = 0.3
        # 初始化望远镜、光源、大气、变形镜、波前传感器和科学相机
        self.tel = Telescope(resolution=self.param['resolution'], 
                             diameter=self.param['diameter'], 
                             samplingTime=1/self.sampling_rate, 
                             centralObstruction=self.param['centralObstruction'])
        self.ngs = Source(magnitude=self.param['magnitude'],optBand=self.param['opticalBand'])
        self.ngs * self.tel
        self.atm = Atmosphere(telescope=self.tel, 
                              r0=self.param['r0'], 
                              L0=self.param['L0'], 
                              windSpeed=self.param['windSpeed'],
                              fractionalR0=self.param['fractionnalR0'], 
                              windDirection=self.param['windDirection'],
                              altitude=self.param['altitude'])

        self.dm = DeformableMirror(telescope=self.tel, 
                                   nSubap=self.param['nSubap'], 
                                   mechCoupling=self.param['mechCoupling'])
        self.wfs = ShackHartmann(nSubap=self.param['nSubap'], 
                                 telescope=self.tel, 
                                 lightRatio=self.param['lightRatio'], 
                                 is_geometric=self.param['is_geometric'], 
                                 threshold_cog=self.param['threshold_cog'])
        self.camH = Imager(exposure_time=self.exposure_time, 
                         clock_rate=self.clock_rate)
        
        self.wfs.cam.photonNoise = self.param['photonNoise']

        # 核心新增：固定图像分辨率
        self.image_resolution = 64

        # 校准矩阵（原始名称保留）
        self.ao_calib = ao_calibration(
            param=self.param, ngs=self.ngs, tel=self.tel,
            atm=self.atm, dm=self.dm, wfs=self.wfs
        )
        self.calib_CL = self.ao_calib.calib
        self.M2C_CL = self.ao_calib.M2C

        self.time_window = 3
        
        # --- 核心修改 1: 移除 wfs_buffer ---
        self.far_buffer = deque(maxlen=self.time_window)
        self.slope_buffer = deque(maxlen=self.time_window)
        self.dm_buffer = deque(maxlen=self.time_window)

        # --- 核心修改 2: 在 __init__ 中就使用动态维度定义 observation_space ---
        self.observation_space = spaces.Dict({
            "far_img": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.time_window, self.image_resolution, self.image_resolution), 
                dtype=np.float32
            ),
            "slope_vec": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.time_window, self.wfs.nSignal), 
                dtype=np.float32
            ),
            "dm_vec": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.time_window, self.dm.nValidAct), 
                dtype=np.float32
            )
        })

        # 离散化动作空间的离散值数量
        self.gain_dim = 19  # 从0.1到1.0，间隔0.05
        self.sampling_rate_dim = 20  # 从100到2000，间隔100
        self.exposure_time_dim = 19  # 从0.1到1.0，间隔0.05
        self.clock_rate_dim = 20  # 从100到2000，间隔100

       # 定义离散的动作空间
        self.action_space = spaces.MultiDiscrete([
            self.gain_dim,  # 增益离散值
            self.sampling_rate_dim,  # 采样频率离散值
            self.exposure_time_dim,  # 曝光时间离散值
            self.clock_rate_dim  # 时钟频率离散值
        ])

        # 其他原始参数初始化
        self.current_step = 0
        self.SR = np.zeros(self.max_step)
        self.SR_old = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)

    def _get_state(self):
        """ --- 核心修改 3: 构建并返回一个不含 wfs_img 的状态字典 --- """
        # 核心修正：在归一化和堆叠前，对图像进行 resize
        resized_frames = [resize(f, (self.image_resolution, self.image_resolution), anti_aliasing=True) for f in self.far_buffer]
        far_frames = np.array([self._normalize_far(f) for f in resized_frames])
        
        state_dict = {
            "far_img": far_frames.astype(np.float32),
            "slope_vec": np.array(self.slope_buffer).astype(np.float32),
            "dm_vec": np.array(self.dm_buffer).astype(np.float32)
        }
        return state_dict

    def _normalize_wfs(self, frame):
        """波前传感器动态归一化（保持原始处理逻辑）"""
        return (frame - np.percentile(frame, 1)) / \
              (np.percentile(frame, 99) - np.percentile(frame, 1) + 1e-6)

    def _normalize_far(self, frame):
        """远场对数压缩（参考专利CN114488518B）"""
        return np.log1p(frame) / (np.log1p(frame.max()) + 1e-6)
    
    def get_action_from_discrete(self, discrete_action):
        """
        将离散动作转化为实际的控制值。
        """
        gain_values = np.arange(0.1, 1.05, 0.05)  # [0.1, 1.0]，步长0.05
        sampling_rate_values = np.arange(100, 2100, 100)  # [100, 2000]，步长100
        exposure_time_values = np.arange(0.1, 1.05, 0.05)  # [0.1, 1.0]，步长0.05
        clock_rate_values = np.arange(100, 2100, 100)  # [100, 2000]，步长100

        # 使用离散的动作索引来获取实际值
        gain = gain_values[discrete_action[0]]
        sampling_rate = sampling_rate_values[discrete_action[1]]
        exposure_time = exposure_time_values[discrete_action[2]]
        clock_rate = clock_rate_values[discrete_action[3]]

        return np.array([gain, sampling_rate, exposure_time, clock_rate])

    def step(self, action):
        """执行步骤（保持原始控制流程）"""
        # 动作参数处理（保持原有参数顺序）
        # 将离散动作转换为实际的控制值
        discrete_action = self.get_action_from_discrete(action)
        # print("discrete_action:", discrete_action)
        # 更新动作参数
        self.gainCL = discrete_action[0]  # 波前传感器增益
        self.tel.samplingTime = 1/discrete_action[1]  # 波前传感器采样频率
        self.camH.exposure_time = discrete_action[2]  # 成像相机曝光时间
        self.camH.clock_rate = round(discrete_action[3])  # 成像相机时钟频率
    

        # 大气扰动更新（保持原始方法）
        self.atm.update()
        self.tel - self.atm
        self.camH.relay(self.tel.src)
        self.camH.reference_frame = self.camH.frame
        self.tel + self.atm

        # 波前校正（保持原始控制算法）
        self.tel * self.dm * self.wfs
        self.dm.coefs -= self.gainCL * self.M2C_CL @ self.calib_CL.M @ self.wfs.signal
        
        # 本次波前误差
        current_wfe = np.std(self.tel.OPD[self.tel.pupil > 0]) * 1e9
        # 更新缓冲区（新增）
        self._update_buffers()

        # # 光波传播：自然引导星 -> 望远镜 -> 变形镜 -> 科学相机
        # self.ngs * self.tel * self.dm * self.camH
        self.camH.relay(self.tel.src)

        # 计算斯特列尔比（基于科学相机的PSF）
        self.SR[self.current_step] = self.camH.strehl
        self.SR_old[self.current_step] = np.exp(-np.var(self.tel.src.phase[np.where(self.tel.pupil==1)]))
        # 记录残余波前误差
        self.residual[self.current_step] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9

        # 计算奖励（保持原始逻辑）
        reward = self._calculate_reward(discrete_action, current_wfe)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_step
        
        return self._get_state(), reward, terminated, False, {}
    

    def _calculate_reward(self, action, current_wfe):
        w = {
            "base_reward_scale": 1.0,       # 基础奖励放大倍数
            "turb_penalty_scale": 0.1,       # 湍流惩罚系数
            "gain_penalty_scale": 0.1,       # 增益约束惩罚系数
            "freq_gain_penalty_scale": 0.1,  # 采样频率与增益约束惩罚系数
            "freq_expo_penalty_scale": 0.2,  # 曝光时间与采样频率乘积约束惩罚系数
            "clock_penalty_scale": 0.1,      # 时钟频率与曝光时间约束惩罚系数
            "volt_penalty_scale": 0.1        # 电压波动惩罚系数
        }
        """优化奖励函数（新增湍流惩罚）"""
        # 解包动作参数
        gain, sample_freq, exposure, clock_freq = action
        # print(gain, sample_freq, exposure, clock_freq, current_wfe)
        
        # 基础奖励（保持原有计算方式）
        normalized_wfe = 0.5 * (np.tanh((current_wfe - 100)/300) + 1)
        if current_wfe < 100:
            reward = self.SR[self.current_step] * w["base_reward_scale"] + (1 - normalized_wfe)
        else:
            reward = self.SR[self.current_step] * w["base_reward_scale"] - 0.3 * normalized_wfe
        


        # 约束1: 曝光时间↑ → 增益↓ (反向关系约束)
        # 理想比例：曝光时间占最大值时，增益应接近最小值
        gain_penalty = np.abs(gain - (1.0 - 0.9 * (exposure - 0.1) / 0.9))  # 线性惩罚项
        reward -= w["gain_penalty_scale"] * gain_penalty  # 比例系数可调

        # print("reward---3:", reward)

        # 约束2: 采样频率↑ → 增益↑ (正向关系约束)
        # 理想比例：采样频率2000Hz时增益应接近1.0，100Hz时接近0.1
        desired_gain = 0.1 + 0.9 * (sample_freq - 100) / 1900
        freq_gain_penalty = np.abs(gain - desired_gain)
        reward -= w["freq_gain_penalty_scale"] * freq_gain_penalty

        # print("reward---4:", reward)

        # 约束3: 曝光时间↑ → 采样频率↓ (反向关系约束)
        # 限制曝光时间×采样频率的乘积不超过阈值（如500）
        freq_exposure_penalty = max(0, (exposure * sample_freq - 500) / 500)
        reward -= w["freq_expo_penalty_scale"] * freq_exposure_penalty  # 强惩罚越界行为

        # print("reward---5:", reward)

        # 约束4: 时钟频率↑ → 曝光时间↓ (反向关系约束)
        # 理想关系：曝光时间 ∝ 1/clock_freq
        desired_exposure = 1.0 / (clock_freq / 500)  # 基准值500Hz时曝光1.0
        clock_penalty = np.abs(exposure - desired_exposure)
        reward -= w["clock_penalty_scale"] * clock_penalty   

        # print("reward---6:", reward)     
        
        # # 新增时序同步惩罚（保持原有逻辑）
        # tel_freq, cam_freq = action[1], action[3]
        # if abs(tel_freq - cam_freq) > 100:
        #     reward -= 1
        
        # 新增电压波动惩罚
        # volt_diff = 0.0
        # if self.dm_coefs_history is not None and len(self.dm_coefs_history) >= 2:
        #     diffs = np.diff(self.dm_coefs_history, axis=0)
        #     volt_diff = np.mean(diffs**2)
        #     reward -= w["volt_penalty_scale"] * volt_diff / self.dm.nActAlongDiameter**2
        # else:
        #     print("Skipping voltage penalty: not enough history.")

        # print(f"[Reward Debug] SR={self.SR[self.current_step]:.4f}, SR_old={self.SR_old[self.current_step]} WFE={current_wfe:.2f}, normWFE={normalized_wfe:.2f}, GainPenalty={gain_penalty:.2f}, FreqGainPenalty={freq_gain_penalty:.2f}, FreqExpoPenalty={freq_exposure_penalty:.2f}, ClockPenalty={clock_penalty:.2f}, Reward={reward:.2f}")

        return reward

    def _update_buffers(self):
        # self.wfs_buffer.append(self.wfs.cam.frame.copy()) # 已移除
        self.far_buffer.append(self.camH.frame.copy())
        self.slope_buffer.append(self.wfs.signal.copy())
        self.dm_buffer.append(self.dm.coefs.copy())

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """环境重置"""
        # 核心修正 1: 调用父类的 reset 方法，这会创建 self.np_random
        super().reset(seed=seed)
        self.current_step = 0
        
        # 使用正确的DM维度
        self.dm.coefs = np.zeros(self.dm.nValidAct)
        self.SR = np.zeros(self.max_step)
        self.SR_old = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)
        
        self._initialize_optical_system()
        self._clear_buffers()

        # --- 核心修改 5: reset() 中不再需要重新定义 observation_space ---
        # self.observation_space 已在 __init__ 中正确定义
        
        return self._get_state(), {}

    def _clear_buffers(self):
        # self.wfs_buffer.clear() # 已移除
        self.far_buffer.clear()
        self.slope_buffer.clear()
        self.dm_buffer.clear()
        
        slope_dim = self.wfs.nSignal
        dm_dim = self.dm.nValidAct
        
        # 初始化填充空数据
        for _ in range(self.time_window):
            # self.wfs_buffer.append(np.zeros((self.wfs.cam.frame.shape[0], self.wfs.cam.frame.shape[1]))) # 已移除
            # 核心修正：使用固定的分辨率来创建占位符
            self.far_buffer.append(np.zeros((self.image_resolution, self.image_resolution)))
            self.slope_buffer.append(np.zeros(slope_dim))
            self.dm_buffer.append(np.zeros(dm_dim))

    def _initialize_optical_system(self):
        # 初始值
        self.tel.samplingTime = 1/self.sampling_rate
        self.camH.exposure_time = self.exposure_time
        self.camH.clock_rate = self.clock_rate
        self.gainCL = self.init_gainCL
        """光学系统初始化（保持原始方法）"""
        self.atm.initializeAtmosphere(self.tel)
        self.tel - self.atm
        self.ngs * self.tel * self.dm * self.wfs
        self.camH.relay(self.tel.src)
        self.camH.reference_frame = self.camH.frame.copy()
        self.tel + self.atm