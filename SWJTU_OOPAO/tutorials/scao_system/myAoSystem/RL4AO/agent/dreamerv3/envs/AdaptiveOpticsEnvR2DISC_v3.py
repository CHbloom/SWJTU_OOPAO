
from typing import Optional
import sys
sys.path.append('/DATACENTER4/jiangbo.chai/SWJTU_OOPAO')
sys.path.append('/DATACENTER4/jiangbo.chai/SWJTU_OOPAO/tutorials/scao_system/myAoSystem')
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
from Imager_new import Imager_new
import matplotlib.pyplot as plt
import importlib
from collections import deque
from skimage.transform import resize


def load_parameter_file(file_name):
    """动态加载指定参数文件"""
    try:
        module = importlib.import_module(f'envs.paramsettings.{file_name}')
        return module.initializeParameterFile()
    except ImportError:
        raise ValueError(f"参数文件 {file_name} 不存在")


def get_param(args):
    '''
    param没有的则添加，有的则用args的覆盖，无论是否为默认值
    后续调用统一用param
    '''
    param = load_parameter_file(args.paramFile)
    for k, v in vars(args).items():
        if k == "paramFile":  
            continue
        param[k] = v
    return param



class AOEnv(gym.Env):
    def __init__(self, args):
        super(AOEnv, self).__init__()

        '''
        AO参数设置
        '''
        self.param = get_param(args)
        self.current_step = 0
        self.max_step = self.param['maxStep']
    
        self.tel = Telescope(resolution         =   self.param['resolution'], 
                             diameter           =   self.param['diameter'], 
                             samplingTime       =   1/self.param['samplingRate'], 
                             centralObstruction =   self.param['centralObstruction'])
        
        self.ngs = Source(magnitude             =   self.param['magnitude'],
                          optBand               =   self.param['opticalBand'])
        self.ngs * self.tel
        self.atm = Atmosphere(telescope         =   self.tel, 
                              r0                =   self.param['r0'], 
                              L0                =   self.param['L0'], 
                              windSpeed         =   self.param['windSpeed'],
                              fractionalR0      =   self.param['fractionnalR0'], 
                              windDirection     =   self.param['windDirection'],
                              altitude          =   self.param['altitude'])

        self.dm = DeformableMirror(telescope    =   self.tel, 
                                   nSubap       =   self.param['nSubap'], 
                                   mechCoupling =   self.param['mechCoupling'])
        
        self.wfs = ShackHartmann(nSubap         =   self.param['nSubap'], 
                                 telescope      =   self.tel, 
                                 lightRatio     =   self.param['lightRatio'], 
                                 is_geometric   =   self.param['is_geometric'], 
                                 threshold_cog  =   self.param['threshold_cog'])
        
        self.wfs.cam.photonNoise                =   self.param['photonNoise']
        self.wfs.cam.readoutNoise               =   self.param['readoutNoise']
        self.wfs.cam.darkCurrent                =   self.param['darkCurrent']

        self.camH = Imager_new(exposure_time    =   self.param['exposureTime'], 
                               clock_rate       =   self.param['exposureTime'],
                               nyquist_sampling =   self.param['nyquist_sampling'],
                               field_stop_size  =   self.param['field_stop_size'])
        
        self.ao_calib = ao_calibration(param=self.param, \
                                  ngs=self.ngs, \
                                  tel=self.tel, \
                                  atm=self.atm, \
                                  dm=self.dm, \
                                  wfs=self.wfs, \
                                  nameFolderIntMat=None, \
                                  nameIntMat=None, \
                                  nameFolderBasis=None, \
                                  nameBasis=None, \
                                  nMeasurements=100)
        self.calib_CL = self.ao_calib.calib  # 交互矩阵
        self.M2C_CL = self.ao_calib.M2C

        self.image_resolution = 128 # 固定图像分辨率
        '''
        RL动作、状态设置
        '''
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
        self.gain_values = np.round(np.linspace(0.1, 1.0, self.gain_dim), 2)
        self.sampling_rate_values = np.arange(100, 2100, 100)
        self.exposure_time_values = np.round(np.linspace(0.1, 1.0, self.exposure_time_dim), 2)
        self.clock_rate_values = np.arange(100, 2100, 100)


        # 观测空间定义
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=(self.image_resolution,self.image_resolution,1),dtype=np.float32),  # 科学相机图像
            "image_max": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            # "wfsFrame": spaces.Box(low=0, high=1, shape=(self.tel.resolution,self.tel.resolution,1)), # 波前传感器图像
            "wfs_stats": spaces.Box(low=-1, high=1, shape=(self.wfs.signal.shape[0]*2,),dtype=np.float32), # 电压均值方差
            "wfs_max": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "dm_stats": spaces.Box(low=-1, high=1, shape=(self.dm.coefs.shape[0]*2,),dtype=np.float32),  # 波前信号（斜率）均值方差
            "dm_max": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            "image_updated": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "is_terminal": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            "is_first": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
        })

        # 初始化参数
        self.total = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)
        self.coefs_list = np.zeros(self.max_step)
        self.fixed_step_time = 0.05 # 每个step经过的时间，固定50ms

        # 用于存储每个 RL 步中所有 WFS 迭代的瞬时 SR
        # 这是一个变长的列表，用于训练后的数据分析和可视化
        self.inst_SR = [] 
        
        # 用于存储每个 RL 步中最终计算的 SR
        # 这是一个固定长度的数组，与 RL 步数对应
        self.actual_SR = np.zeros(self.max_step)

        self.image = np.zeros([self.image_resolution,self.image_resolution]) # 存储图像

        # self.obs_buffer = deque(maxlen=self.param['k'])

        # 信号延迟队列
        self.wfs_buffer = deque([np.zeros(self.wfs.nSignal)] * (self.param['delay']+1), maxlen=self.param['delay']+1)

    def get_action_from_discrete(self, discrete_action):
        """
        将离散动作转化为实际的控制值。
        """
        gain_idx, sampling_idx, exposure_idx, clock_idx = discrete_action
        gain = self.gain_values[gain_idx]
        sampling_rate = self.sampling_rate_values[sampling_idx]
        exposure_time = self.exposure_time_values[exposure_idx]
        clock_rate = self.clock_rate_values[clock_idx]

        return np.array([gain, sampling_rate, exposure_time, clock_rate])

    def step(self, action):
        """
        执行动作并返回新的状态、奖励和是否结束。

        思路：一个step模拟固定的物理时间，如50ms
        """
        # 1.解析动作参数
        discrete_action = self.get_action_from_discrete(action)
        self.gainCL = discrete_action[0]  # 波前传感器增益
        self.tel.samplingTime = 1/discrete_action[1]  # 波前传感器采样频率
        self.camH.exposureTime = discrete_action[2]  # 成像相机曝光时间
        self.camH.clockRate = round(discrete_action[3])  # 成像相机时钟频率

        wfs_history = []
        dm_history = []
        # 2. 确定内部循环次数
        num_iterations = int(np.ceil(self.fixed_step_time / self.tel.samplingTime))
        # 3. 运行内部子循环：模拟固定的 RL 决策周期
        image_updated = False # 是否更新了图像
        for _ in range(num_iterations):
            self.atm.update()
            self.tel * self.dm * self.wfs 
            self.wfsSignal = self.wfs.signal.copy()
            
            # 存入队列，取出延迟信号
            self.wfs_buffer.append(self.wfsSignal)
            delayed_signal = self.wfs_buffer[0]

            wfs_history.append(self.wfsSignal)
            dm_history.append(self.dm.coefs.copy())

            self.dm.coefs = self.dm.coefs - self.gainCL * self.M2C_CL @ self.calib_CL.M @ delayed_signal

            self.camH.relay(self.tel.src)
            self.inst_SR.append(np.exp(-np.var(self.tel.OPD[np.where(self.tel.pupil>0)])))
            if self.camH.frame is not None:
                image_updated = True
                self.image = resize(self.camH.frame, (self.image_resolution,self.image_resolution), mode='reflect', anti_aliasing=True)

        # 计算当前step斜率和电压的均值、方差
        wfs_history_np = np.array(wfs_history)
        dm_history_np = np.array(dm_history)
        if wfs_history_np.size > 0:
            wfs_mean = np.mean(wfs_history_np, axis=0)
            wfs_var = np.var(wfs_history_np, axis=0)
            dm_mean = np.mean(dm_history_np, axis=0)
            dm_var = np.var(dm_history_np, axis=0)
        else:
            wfs_mean = np.zeros_like(self.wfsSignal)
            wfs_var = np.zeros_like(self.wfsSignal)  
            dm_mean = np.zeros_like(self.dm.coefs)
            dm_var = np.zeros_like(self.dm.coefs)
        # 计算当前step估算的SR的均值
        SR_proxy_mean = np.mean(self.inst_SR[-num_iterations:]) if num_iterations>0 else 0.0
        # 记录实际用图像计算的SR，如果当前step没有新的图像则SR不会变化    
        self.actual_SR[self.current_step] = self.camH.strehl[0][0]

        self.residual[self.current_step] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9

        
        reward = self._calculate_reward(discrete_action, image_updated, SR_proxy_mean)
        self.current_step += 1
        terminated = self.current_step >= self.max_step
        truncated = False

        return self._get_state(reward, terminated, truncated,image_updated,
                               wfs_stats={'mean': wfs_mean, 'var': wfs_var}, 
                               dm_stats={'mean': dm_mean, 'var': dm_var}
                               )
    

    def reset(self, seed: Optional[int] = None,
        options: Optional[dict] = None,):
        """重置环境到初始状态"""
        
        # 1.重置初始化参数
        self.current_step = 0

        self.total = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)
        self.coefs_list = np.zeros(self.max_step)
        self.fixed_step_time = 0.05 # 每个step经过的时间，固定50ms

        # 用于存储每个 RL 步中所有 WFS 迭代的瞬时 SR
        # 这是一个变长的列表，用于训练后的数据分析和可视化
        self.inst_SR = [] 
        
        # 用于存储每个 RL 步中最终计算的 SR
        # 这是一个固定长度的数组，与 RL 步数对应
        self.actual_SR = np.zeros(self.max_step)

        # self.obs_buffer = deque(maxlen=self.param['k'])

        # 信号延迟队列
        self.wfs_buffer = deque([np.zeros(self.wfs.nSignal)] * (self.param['delay']+1), maxlen=self.param['delay']+1)

        # 2.重置初始动作
        self.tel.samplingTime = 1/self.param['samplingRate']
        self.camH.exposureTime = self.param['exposureTime']
        self.camH.clockRate = self.param['clockRate']
        self.gainCL = self.param['gainCL']

        # 3.重置AO状态
        self.atm.initializeAtmosphere(telescope=self.tel)

        self.tel-self.atm
        self.ngs * self.tel * self.dm * self.wfs
        self.camH.relay(self.tel.src)
        self.camH.referenceFrame = self.camH.frame
        self.camH.frame = None
        
        self.dm.coefs = 0
        self.tel+self.atm
        obs, reward, is_terminal, _ = self._get_state(0.0, is_first=True)
        return obs
    
    def _get_state(self, reward, terminated, truncated, image_updated, wfs_stats, dm_stats):
        """获取当前状态"""
        # 归一化科学相机图像
        image_max = self.image.max() + 1e-6
        image_norm = self.image / image_max

        # 归一化统计特征
        # 归一化范围可根据实际数据调整
        wfs_combined = np.concatenate([wfs_stats['mean'], wfs_stats['var']])
        wfs_max = np.abs(wfs_combined).max() + 1e-6
        wfs_stats_norm = wfs_combined / wfs_max

        dm_combined = np.concatenate([dm_stats['mean'], dm_stats['var']])
        dm_max = np.abs(dm_combined).max() + 1e-6
        dm_stats_norm = dm_combined / dm_max

        observation = {
            "image": image_norm[..., None], # 添加通道维度
            "image_max": np.array([image_max]).astype(np.float32),
            "wfs_stats": wfs_stats_norm.astype(np.float32),
            "wfs_max": np.array([wfs_max]).astype(np.float32),
            "dm_stats": dm_stats_norm.astype(np.float32),
            "dm_max": np.array([dm_max]).astype(np.float32),
            "image_updated": np.array([image_updated]).astype(np.uint8), #图像是否更新
            "is_terminal": np.array([terminated]).astype(np.uint8),
            "is_first": np.array([self.current_step == 0]).astype(np.uint8)
        }

        return (observation,
                reward,
                terminated,
                {"truncated": truncated})
    
    def _calculate_reward(self, action, image_updated, SR_proxy_mean):
        """优化奖励函数（新增湍流惩罚）"""
        
        # 解包动作参数
        gain, sample_freq, exposure, clock_freq = action

        # 1.基础奖励
        """将WFE压缩到[0,1]范围，100nm为0.5，两端渐进饱和"""
        normalized_wfe = 0.5 * (np.tanh((self.residual[self.current_step] - 100) / 300) + 1)  # 300控制斜率
        if self.residual[self.current_step] < 100:     # 波前误差<100:超高性能奖励
            reward = SR_proxy_mean + (1 - normalized_wfe)
        else:
            reward = SR_proxy_mean - 0.3 * normalized_wfe
        

        # 2.动作约束惩罚
        # 约束1: 曝光时间↑ → 增益↓ (反向关系约束)
        # 理想比例：曝光时间占最大值时，增益应接近最小值
        gain_penalty = np.abs(gain - (1.0 - 0.9 * (exposure - 0.1) / 0.9))  # 线性惩罚项
        reward -= 0.2 * gain_penalty  # 比例系数可调

        # 约束2: 采样频率↑ → 增益↑ (正向关系约束)
        # 理想比例：采样频率2000Hz时增益应接近1.0，100Hz时接近0.1
        desired_gain = 0.1 + 0.9 * (sample_freq - 100) / 1900
        freq_gain_penalty = np.abs(gain - desired_gain)
        reward -= 0.2 * freq_gain_penalty

        # 约束3: 曝光时间↑ → 采样频率↓ (反向关系约束)
        # 限制曝光时间×采样频率的乘积不超过阈值（如500）
        freq_exposure_penalty = max(0, (exposure * sample_freq - 500) / 500)
        reward -= 0.5 * freq_exposure_penalty  # 强惩罚越界行为

        # 约束4: 时钟频率↑ → 曝光时间↓ (反向关系约束)
        # 理想关系：曝光时间 ∝ 1/clock_freq
        desired_exposure = 1.0 / (clock_freq / 500)  # 基准值500Hz时曝光1.0
        clock_penalty = np.abs(exposure - desired_exposure)
        reward -= 0.3 * clock_penalty

        # TODO 3.成像质量奖励
    