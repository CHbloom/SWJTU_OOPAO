
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

        self.wfs_stats_dim = self.wfs.nSignal * 2 # 均值 + 方差
        self.dm_stats_dim = self.dm.nValidAct * 2 # 均值 + 方差

        # 观测空间定义
        self.observation_space = spaces.Dict({
            # "image": spaces.Box(low=0, high=1, shape=(args.size[0],args.size[1],1)),  # 科学相机图像
            # "wfsFrame": spaces.Box(low=0, high=1, shape=(args.size[0],args.size[1],1)), # 波前传感器图像
            "image": spaces.Box(low=0, high=1, shape=(self.tel.resolution,self.tel.resolution,1)),  # 科学相机图像
            # "wfsFrame": spaces.Box(low=0, high=1, shape=(self.tel.resolution,self.tel.resolution,1)), # 波前传感器图像
            "wfs_stats": spaces.Box(low=-1, high=1, shape=(self.wfs_stats_dim,)), # 电压
            "dm_stats": spaces.Box(low=-1, high=1, shape=(self.dm_stats_dim,)),  # 波前信号（斜率）
            "is_terminal": spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_first": spaces.Box(0, 1, (1,), dtype=np.uint8),
        })

        # 初始化参数
        self.SR = np.zeros(self.max_step)
        self.SR_old = np.zeros(self.max_step)
        self.total = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)
        self.coefs_list = np.zeros(self.max_step)

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

        思路：一次曝光作为一个step，过程中会有多次矫正，期间动作参数不变，
        step的最后得到图像和SR
            状态包括 1.当前的图像（不像之前R2堆叠）     
                    2.电压和斜率：一个step有多个，可以选开头、中间、结尾堆叠（类似R2），再计算方差、均值等（类似R1）   
        """
        discrete_action = self.get_action_from_discrete(action)
        # 更新动作参数
        self.gainCL = discrete_action[0]  # 波前传感器增益
        self.tel.samplingTime = 1/discrete_action[1]  # 波前传感器采样频率
        self.camH.exposureTime = discrete_action[2]  # 成像相机曝光时间
        self.camH.clockRate = round(discrete_action[3])  # 成像相机时钟频率

        wfs_history = []
        dm_history = []
        self.camH.frame = None 

        while self.camH.frame is None:
            self.atm.update()
            self.tel * self.dm * self.wfs 
            self.wfsSignal = self.wfs.signal.copy()
            
            # 存入队列，取出延迟信号
            self.wfs_buffer.append(self.wfsSignal)
            delayed_signal = self.wfs_buffer[0]

            wfs_history.append(delayed_signal)
            dm_history.append(self.dm.coefs.copy())

            self.dm.coefs = self.dm.coefs - self.gainCL * self.M2C_CL @ self.calib_CL.M @ delayed_signal

            self.camH.relay(self.tel.src)

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
        self.SR[self.current_step] = self.camH.strehl[0][0]
        self.residual[self.current_step] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9
        reward = self._calculate_reward(discrete_action)
        self.current_step += 1
        terminated = self.current_step >= self.max_step
        truncated = False

        return self._get_state(reward, terminated, truncated, 
                               wfs_stats={'mean': wfs_mean, 'var': wfs_var}, 
                               dm_stats={'mean': dm_mean, 'var': dm_var})

    def reset(self, seed: Optional[int] = None,
        options: Optional[dict] = None,):
        """重置环境到初始状态"""
        self.current_step = 0
        self.dm.coefs = 0
        self.SR = np.zeros(self.max_step)
        self.SR_old = np.zeros(self.max_step)
        self.total = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)
        self.coefs_list = np.zeros(self.max_step)

        # self.obs_buffer = deque(maxlen=self.param['k'])

        self.wfs_buffer = deque([np.zeros(self.wfs.nSignal)] * (self.param['delay']+1), maxlen=self.param['delay']+1)
        # 初始值
        self.tel.samplingTime = 1/self.param['samplingRate']
        self.camH.exposureTime = self.param['exposureTime']
        self.camH.clockRate = self.param['clockRate']
        self.gainCL = self.param['gainCL']
        self.atm.initializeAtmosphere(telescope=self.tel)
        self.tel-self.atm
        self.ngs * self.tel * self.dm * self.wfs
        self.camH.relay(self.tel.src)
        self.camH.referenceFrame = self.camH.frame
        self.camH.frame = None
        self.tel+self.atm
        obs, reward, is_terminal, _ = self._get_state(0.0, is_first=True)
        return obs
    
    def _get_state(self, reward, terminated, truncated, wfs_stats, dm_stats):
        """获取当前状态"""
        # 归一化科学相机图像
        image_norm = self.camH.frame / (self.camH.frame.max() + 1e-6)

        # 归一化统计特征
        # 归一化范围可根据实际数据调整
        wfs_stats_norm = np.concatenate([
            wfs_stats['mean'] / (self.wfs.nSignal * 1e-6),  # 示例归一化
            wfs_stats['var'] / (self.wfs.nSignal * 1e-6)
        ])
        dm_stats_norm = np.concatenate([
            dm_stats['mean'] / (self.dm.nValidAct * 1e-6), # 示例归一化
            dm_stats['var'] / (self.dm.nValidAct * 1e-6)
        ])

        return ({
            "image": image_norm[..., None], # 添加通道维度
            "wfs_stats": wfs_stats_norm.astype(np.float32),
            "dm_stats": dm_stats_norm.astype(np.float32),
            "is_terminal": terminated,
            "is_first": self.current_step == 0
        },
        reward,
        terminated,
        {"truncated": truncated})
    