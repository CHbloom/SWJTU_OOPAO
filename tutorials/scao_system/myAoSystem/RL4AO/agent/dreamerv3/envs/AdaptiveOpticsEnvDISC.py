'''
修改了state为:
波前传感器图像
斜率
波前校正器电压
远场图像
5.15 state去除波前传感器图像
'''


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
        self.wfs.cam.photonNoise = True

        self.camH = Imager(exposure_time=self.exposure_time, clock_rate=self.clock_rate)

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
        # 定义动作空间和状态空间
        
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

        # 观测空间定义,图像reshpe为(args.size[0],args.size[1],1)
        self.observation_space = spaces.Dict({
            # "image": spaces.Box(low=0, high=1, shape=(args.size[0],args.size[1],1)),  # 科学相机图像
            # "wfsFrame": spaces.Box(low=0, high=1, shape=(args.size[0],args.size[1],1)), # 波前传感器图像
            "image": spaces.Box(low=0, high=1, shape=(self.tel.resolution,self.tel.resolution,1)),  # 科学相机图像
            # "wfsFrame": spaces.Box(low=0, high=1, shape=(self.tel.resolution,self.tel.resolution,1)), # 波前传感器图像
            "dmCoefs": spaces.Box(low=-1, high=1, shape=(self.dm.nValidAct,)), # 电压
            "wfsSingnal": spaces.Box(low=-1, high=1, shape=(self.wfs.nSignal,)),  # 波前信号（斜率）
            "is_terminal": spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_first": spaces.Box(0, 1, (1,), dtype=np.uint8),
        })

        # 初始化参数
        self.current_step = 0
        self.wfs.cam.photonNoise = True
        self.SR = np.zeros(self.max_step)
        self.SR_old = np.zeros(self.max_step)
        self.total = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)

    def get_action_from_discrete(self, discrete_action):
        """
        将离散动作转化为实际的控制值。
        """
        gain_idx, sampling_idx, exposure_idx, clock_idx = discrete_action  # 解包
        gain = self.gain_values[gain_idx]
        sampling_rate = self.sampling_rate_values[sampling_idx]
        exposure_time = self.exposure_time_values[exposure_idx]
        clock_rate = self.clock_rate_values[clock_idx]

        return np.array([gain, sampling_rate, exposure_time, clock_rate])

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
        # 初始值
        self.tel.samplingTime = 1/self.sampling_rate
        self.camH.exposure_time = self.exposure_time
        self.camH.clock_rate = self.clock_rate
        self.gainCL = self.init_gainCL
        self.atm.initializeAtmosphere(telescope=self.tel)
        self.tel-self.atm
        self.ngs * self.tel * self.dm * self.wfs
        self.camH.relay(self.tel.src)
        self.camH.reference_frame = self.camH.frame
        self.camH.frame = np.zeros([self.tel.resolution, self.tel.resolution])
        self.tel+self.atm
        obs, reward, is_terminal, _ = self._get_state(0.0, is_first=True)
        return obs

    def step(self, action):
        """
        执行动作并返回新的状态、奖励和是否结束。

        参数：
        - action: 动作向量

        返回：
        - state: 新状态
        - reward: 奖励
        - done: 是否结束
        - info: 附加信息
        """
        # 将离散动作转换为实际的控制值
        discrete_action = self.get_action_from_discrete(action)
        # 更新动作参数
        self.gainCL = discrete_action[0]  # 波前传感器增益
        self.tel.samplingTime = 1/discrete_action[1]  # 波前传感器采样频率
        self.camH.exposure_time = discrete_action[2]  # 成像相机曝光时间
        self.camH.clock_rate = round(discrete_action[3])  # 成像相机时钟频率

        # 更新大气湍流相位屏
        self.atm.update()
        self.tel-self.atm
        self.camH.relay(self.tel.src)
        self.camH.reference_frame = self.camH.frame
        self.tel+self.atm
        # 记录总的波前误差
        self.total[self.current_step] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9

        # 光波传播：自然引导星 -> 望远镜 -> 变形镜 -> 波前传感器
        self.tel * self.dm * self.wfs

        # 更新变形镜控制信号
        self.dm.coefs = self.dm.coefs - self.gainCL * self.M2C_CL @ self.calib_CL.M @ self.wfsSignal

        # 更新波前传感器信号
        self.wfsSignal = self.wfs.signal

        # # 光波传播：自然引导星 -> 望远镜 -> 变形镜 -> 科学相机
        # self.ngs * self.tel * self.dm * self.camH
        self.camH.relay(self.tel.src)

        # 计算斯特列尔比（基于科学相机的PSF）
        self.SR[self.current_step] = self.camH.strehl
        self.SR_old[self.current_step] = np.exp(-np.var(self.tel.src.phase[np.where(self.tel.pupil==1)]))
        # 记录残余波前误差
        self.residual[self.current_step] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9

        # 计算奖励
        """将WFE压缩到[0,1]范围，100nm为0.5，两端渐进饱和"""
        normalized_wfe = 0.5 * (np.tanh((self.residual[self.current_step] - 100) / 300) + 1)  # 300控制斜率
        if self.residual[self.current_step] < 100:     # 波前误差<100:超高性能奖励
            reward = self.SR[self.current_step] + (1 - normalized_wfe)
        else:
            reward = self.SR[self.current_step] - 0.3 * normalized_wfe
        # if abs(discrete_action[1] - discrete_action[3]) > 100: # 望远镜帧频和成像帧频不匹配
        #     reward -= 1  # 惩罚时序失配

        # 更新步数
        self.current_step += 1

        # 检查是否结束
        terminated = self.current_step >= self.max_step
        truncated = False
        # 返回新状态、奖励和是否结束
        return self._get_state(
            reward,
            is_last=terminated,
            is_terminal=False
        )
    
    def calculate_reward(self, action, residual_wfe, SR):
        '''计算奖励（初步方案，暂未采纳）'''
        # 解包动作参数
        gain, sample_freq, exposure, clock_freq = action

        # 基础奖励：基于波前误差和斯特列尔比（原逻辑）
        normalized_wfe = 0.5 * (np.tanh((residual_wfe - 100) / 300) + 1)
        if residual_wfe < 100:
            reward = SR + (1 - normalized_wfe)  # 超高性能奖励
        else:
            reward = SR - 0.3 * normalized_wfe

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

        # 原有约束：望远镜帧频和成像帧频失配惩罚（保留）
        if abs(sample_freq - clock_freq) > 100:
            reward -= 1  # 时序失配惩罚

        return reward

    def _get_state(self, reward, is_first=False, is_last=False, is_terminal=False):
        """获取当前状态"""
        # 归一化
        image_norm = self.camH.frame / (self.camH.frame.max() + 1e-6)
        wfsFrame_norm = self.wfs.cam.frame / (self.wfs.cam.frame.max() + 1e-6)
        # # 裁剪
        # start_row = (self.tel.resolution - self.args.size[0]) // 2
        # start_col = (self.tel.resolution - self.args.size[1]) // 2
        # image_norm = image_norm[start_row:start_row+self.args.size[0], start_col:self.args.size[1]]
        # wfsFrame_norm = wfsFrame_norm[start_row:start_row+self.args.size[0], start_col:self.args.size[1]]

        dmCoefs_norm = self.dm.coefs / (self.dm.coefs.max() + 1e-6)
        wfsSingnal_norm = self.wfsSignal / (self.wfsSignal.max() + 1e-6)
        return ({
                    "image": image_norm[...,None]*255, 
                    # "wfsFrame": wfsFrame_norm[...,None]*255,
                    "dmCoefs" : dmCoefs_norm,
                    "wfsSingnal": wfsSingnal_norm,
                    "is_terminal": is_terminal, 
                    "is_first": is_first
                },
                reward,
                is_last,
                {})

    def render(self, mode='human'):
        """显示当前状态"""
        if mode == 'human':
            plt.figure()
            plt.imshow(self.camH.frame, cmap='gray')
            plt.title('Science Camera Frame')
            plt.colorbar(label='Intensity')
            plt.show()