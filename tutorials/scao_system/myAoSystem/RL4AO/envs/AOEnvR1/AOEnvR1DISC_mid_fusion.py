#这个文件的主要目的是将湍流强度整合进了state中

#湍流特征整合：
# 残差波前误差滑动平均（wfe_history）
# 斜率信号方差（slope_var_history）
# 电压变化率（dm_diff_history）


# 奖励函数增强
# Reward = 基础奖励 - 湍流波动惩罚 - 时序失配惩罚 - 电压波动惩罚

from typing import Optional
import sys
sys.path.append('/DATACENTER4/jiangbo.chai/SWJTU_OOPAO')
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
        super(AOEnv, self).__init__()
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
        if args.lightRatio==0.3:
            self.lightRatio = self.param['lightRatio']
        else:
            self.lightRatio = args.lightRatio
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

        # 校准矩阵（保持原有名称）
        self.ao_calib = ao_calibration(
            param=self.param, ngs=self.ngs, tel=self.tel,
            atm=self.atm, dm=self.dm, wfs=self.wfs
        )
        self.calib_CL = self.ao_calib.calib
        self.M2C_CL = self.ao_calib.M2C

        # 观测空间将在第一次reset时动态定义
        self.observation_space = None
        self._dimensions_initialized = False

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

        # 湍流特征历史（新增）
        self.wfe_history = deque(maxlen=5)
        self.slope_var_history = deque(maxlen=3)
        self.dm_diff_history = deque(maxlen=5)

        # 运行参数（保持原有名称）
        self.current_step = 0
        self.SR = np.zeros(self.max_step)
        self.SR_old = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)
        self.dm_coefs_history = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境，并在第一次调用时动态设置观测空间"""
        super().reset(seed=seed)
        self.current_step = 0
        self.dm.coefs = 0
        self.SR = np.zeros(self.max_step)
        self.SR_old = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)
        self.dm_coefs_history = []
        
        # 硬件参数初始化（保持原有设置）
        self.tel.samplingTime = 1/self.sampling_rate
        self.camH.exposure_time = self.exposure_time
        self.camH.clock_rate = self.clock_rate
        self.gainCL = self.init_gainCL
        
        # 大气湍流初始化（新增相位屏重置）
        self.atm.initializeAtmosphere(self.tel)
        self.tel - self.atm
        self.ngs * self.tel * self.dm * self.wfs
        self.camH.relay(self.tel.src)
        self.camH.reference_frame = self.camH.frame.copy()
        self.tel + self.atm

        # 在第一次reset时，动态获取维度并定义观测空间
        if not self._dimensions_initialized:
            # 获取图像维度
            wfs_img_shape = self.wfs.cam.frame.shape
            far_img_shape = self.camH.frame.shape
            
            # 获取向量维度
            dm_voltage_dim = self.dm.nValidAct
            slopes_dim = self.wfs.nSignal
            turb_features_dim = 4  # 来自4个历史统计量
            params_dim = 3         # 来自3个系统参数
            vec_dim = dm_voltage_dim + slopes_dim + turb_features_dim + params_dim

            self.observation_space = spaces.Dict({
                "img_wfs": spaces.Box(low=0, high=1, shape=wfs_img_shape, dtype=np.float32),
                "img_far": spaces.Box(low=0, high=1, shape=far_img_shape, dtype=np.float32),
                "vec": spaces.Box(low=-np.inf, high=np.inf, shape=(vec_dim,), dtype=np.float32)
            })
            self._dimensions_initialized = True
            print(f"Observation space initialized dynamically: {self.observation_space}")
        
        return self._get_state(), {}

    def step(self, action):
        """执行步骤（优化控制流程）"""
        # 动作参数处理（保持原有参数顺序）
        # 将离散动作转换为实际的控制值
        discrete_action = self.get_action_from_discrete(action)
        # 更新动作参数
        self.gainCL = discrete_action[0]  # 波前传感器增益
        self.tel.samplingTime = 1/discrete_action[1]  # 波前传感器采样频率
        self.camH.exposure_time = discrete_action[2]  # 成像相机曝光时间
        self.camH.clock_rate = round(discrete_action[3])  # 成像相机时钟频率

        # 大气扰动更新（优化相位屏更新）
        self.atm.update()
        self.tel - self.atm
        self.camH.relay(self.tel.src)
        self.camH.reference_frame = self.camH.frame
        self.tel + self.atm

        # 波前校正（保持原有控制算法）
        self.tel * self.dm * self.wfs
        self.dm.coefs -= self.gainCL * self.M2C_CL @ self.calib_CL.M @ self.wfs.signal
        self.dm_coefs_history.append(self.dm.coefs.copy())

        # 记录湍流特征（新增）
        current_wfe = np.std(self.tel.OPD[self.tel.pupil > 0]) * 1e9
        self.wfe_history.append(current_wfe)
        self.slope_var_history.append(np.var(self.wfs.signal))
        
        # 电压变化率计算（新增）
        if len(self.dm_coefs_history) >= 2:
            diff = np.linalg.norm(self.dm_coefs_history[-1] - self.dm_coefs_history[-2])
            self.dm_diff_history.append(diff)

        # 性能指标计算（保持原有参数）
        self.residual[self.current_step] = current_wfe
        self.camH.relay(self.tel.src)
        self.SR[self.current_step] = self.camH.strehl
        self.SR_old[self.current_step] = np.exp(-np.var(self.tel.src.phase[np.where(self.tel.pupil==1)]))

        # 计算奖励（优化奖励函数）
        reward = self._calculate_reward(discrete_action, current_wfe)

        self.current_step += 1
        terminated = self.current_step >= self.max_step

        # print(reward)
        
        return self._get_state(), reward, terminated, False, {}
    
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

    def _get_state(self):
        """构建包含多个图像和向量的状态字典"""
        # 图像数据归一化
        wfs_image = (self.wfs.cam.frame - self.wfs.cam.frame.min()) / \
                   (self.wfs.cam.frame.max() - self.wfs.cam.frame.min() + 1e-6)
        far_field = np.log1p(self.camH.frame) / (np.log1p(self.camH.frame.max()) + 1e-6)
        
        # 控制参数归一化（新增动态范围）
        dm_voltage = self.dm.coefs / 50.0  # 假设±50V工作范围
        slopes = self.wfs.signal / (np.std(self.wfs.signal) + 1e-6)
        
        # 湍流特征（新增）
        turb_features = [
            np.mean(self.wfe_history) if self.wfe_history else 0,
            np.std(self.wfe_history) if self.wfe_history else 0,
            np.mean(self.slope_var_history) if self.slope_var_history else 0,
            np.mean(self.dm_diff_history) if self.dm_diff_history else 0
        ]
        
        # 系统参数（保持原有归一化方式）
        params = [
            self.gainCL,
            (1/self.tel.samplingTime - 100)/1900,
            (self.camH.clock_rate - 100)/1900
        ]
        
        # 将所有向量部分拼接成一个向量
        vec_state = np.concatenate([
            dm_voltage,
            slopes,
            turb_features,
            params
        ]).astype(np.float32)

        # 返回字典形式的状态
        return {
            "img_wfs": wfs_image.astype(np.float32),
            "img_far": far_field.astype(np.float32),
            "vec": vec_state
        }

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
        

        # print("reward---1:", reward)
        
        # 新增湍流波动惩罚
        window_size = 5
        recent_wfe = list(self.wfe_history)[-window_size:]
        if len(recent_wfe) < window_size or self.current_step < 10:
            turb_penalty = 0.0
        else:
            turb_penalty = w["turb_penalty_scale"] * np.std(recent_wfe)
        reward -= turb_penalty

        # print("reward---2:", reward)

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
        volt_diff = 0.0
        if self.dm_coefs_history is not None and len(self.dm_coefs_history) >= 2:
            diffs = np.diff(self.dm_coefs_history, axis=0)
            volt_diff = np.mean(diffs**2)
            reward -= w["volt_penalty_scale"] * volt_diff / self.dm.nActAlongDiameter**2
        else:
            print("Skipping voltage penalty: not enough history.")

        # print("reward---7:", reward)
        # print(f"[Reward Debug] SR={self.SR[self.current_step]:.4f}, SR_old={self.SR_old[self.current_step]} WFE={current_wfe:.2f}, normWFE={normalized_wfe:.2f}, TurbPenalty={turb_penalty:.2f}, GainPenalty={gain_penalty:.2f}, FreqGainPenalty={freq_gain_penalty:.2f}, FreqExpoPenalty={freq_exposure_penalty:.2f}, ClockPenalty={clock_penalty:.2f}, VoltPenalty={volt_diff:.4f}, Reward={reward:.2f}")

        return reward

    def render(self, mode='human'):
        """可视化（新增湍流特征显示）"""
        plt.figure(figsize=(15,8))
        
        # 原有可视化内容
        plt.subplot(231)
        plt.imshow(self.wfs.cam.frame, cmap='gray')
        plt.title("WFS Image")
        
        plt.subplot(232)
        plt.imshow(self.camH.frame, cmap='hot')
        plt.title("Far Field")
        
        plt.subplot(233)
        plt.plot(self.dm.coefs)
        plt.title("DM Voltages")
        
        # 新增湍流特征显示
        plt.subplot(234)
        plt.plot(list(self.wfe_history))
        plt.title("Turbulence Features")
        
        plt.subplot(235)
        plt.plot(self.SR[:self.current_step])
        plt.title("Strehl Ratio")
        
        plt.subplot(236)
        plt.plot(self.residual[:self.current_step])
        plt.title("Residual WFE (nm)")
        
        plt.tight_layout()
        plt.show()