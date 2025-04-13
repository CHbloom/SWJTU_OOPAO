from typing import Optional
import sys
sys.path.append('/DATACENTER4/jiangbo.chai/OOPAO_TRWFS-master')
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
from tutorials.scao_system.myAoSystem.Imager2 import Imager
import matplotlib.pyplot as plt
from tutorials.scao_system.parameter_files.parameterFile_VLT_I_Band_SHWFS import initializeParameterFile
class AOEnv(gym.Env):
    def __init__(self):
        super(AOEnv, self).__init__()
        self.param = initializeParameterFile()
        # 初始化望远镜、光源、大气、变形镜、波前传感器和科学相机
        self.tel = Telescope(resolution=120, diameter=8, samplingTime=1/1000, centralObstruction=0)
        self.ngs = Source(magnitude=5,optBand='H')
        self.ngs * self.tel
        self.atm = Atmosphere(telescope=self.tel, r0=0.15, L0=30, windSpeed=[5,10,20],
                              fractionalR0=[0.7,0.25,0.05], windDirection=[0,90,360],
                              altitude=[0,4000,10000])

        self.dm = DeformableMirror(telescope=self.tel, nSubap=20, mechCoupling=0.1)
        self.wfs = ShackHartmann(nSubap=20, telescope=self.tel, lightRatio=0.3, is_geometric=False, threshold_cog=0.01)
        self.wfs.cam.photonNoise = True

        self.camH = Imager()

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
        
        self.action_space = spaces.Box(
            low=np.array([0.1, 100, 0.1, 100],dtype=np.float32),  # 最小增益、采样频率、成像相机曝光时间、时钟频率
            high=np.array([1.0, 2000, 1.0, 2000],dtype=np.float32),  # 最大增益、采样频率、成像相机曝光时间、时钟频率
            dtype=np.float32
        )
        '''self.observation_space = spaces.Dict({
            "wfs_slopes": spaces.Box(low=-1, high=1, shape=(self.wfs.nSignal,)),
            "science_image": spaces.Box(low=0, high=1, shape=(self.tel.resolution, self.tel.resolution)),
            "parameters": spaces.Box(low=0, high=1, shape=(3,))
        })'''
        # 当前的low，high定义不太合理---------------------------------------------
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.wfs.nSignal + self.tel.resolution ** 2 + 3,), dtype=np.float32
        )

        # 初始化参数
        self.current_step = 0
        self.max_step = 200
        self.gainCL = 0.6
        self.wfs.cam.photonNoise = True
        self.SR = np.zeros(self.max_step)
        self.total = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)

    def reset(self, seed: Optional[int] = None,
        options: Optional[dict] = None,):
        """重置环境到初始状态"""
        self.current_step = 0
        self.dm.coefs = 0
        self.SR = np.zeros(self.max_step)
        self.total = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)
        # 初始值
        self.tel.samplingTime = 1/1000 
        self.camH.exposure_time = 1
        self.camH.clock_rate = 1000
        
        self.atm.initializeAtmosphere(telescope=self.tel)
        self.tel-self.atm
        self.ngs * self.tel * self.dm * self.wfs
        self.camH.relay(self.tel.src)
        # self.camH.display()#---------------
        self.camH.reference_frame = self.camH.frame
        self.camH.frame = np.zeros([self.tel.resolution, self.tel.resolution])
        self.tel+self.atm
        info = ""
        return self._get_state(), info

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
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        # 更新动作参数
        self.gainCL = clipped_action[0]  # 波前传感器增益
        self.tel.samplingTime = 1/clipped_action[1]  # 波前传感器采样频率
        self.camH.exposure_time = clipped_action[2]  # 成像相机曝光时间
        self.camH.clock_rate = round(clipped_action[3])  # 成像相机时钟频率

        # 更新大气湍流相位屏
        self.atm.update()

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
        # self.camH.display()#---------------

        # 计算斯特列尔比（基于科学相机的PSF）
        self.SR[self.current_step] = self.camH.strehl
        # 记录残余波前误差
        self.residual[self.current_step] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9

        # 计算奖励
        """将WFE压缩到[0,1]范围，100nm为0.5，两端渐进饱和"""
        normalized_wfe = 0.5 * (np.tanh((self.residual[self.current_step] - 100) / 300) + 1)  # 300控制斜率
        if self.residual[self.current_step] < 100:     # 波前误差<100:超高性能奖励
            reward = self.SR[self.current_step] + 0.2 * (1 - normalized_wfe)
        else:
            reward = self.SR[self.current_step] - 0.3 * normalized_wfe
        if abs(action[1] - action[3]) > 100: # 望远镜帧频和成像帧频不匹配
            reward -= 0.1  # 轻微惩罚时序失配

        # 更新步数
        self.current_step += 1

        # 检查是否结束
        terminated = self.current_step >= self.max_step
        truncated = False
        # 返回新状态、奖励和是否结束
        return self._get_state(), reward, terminated, truncated, {}

    def _get_state(self):
        """获取当前状态"""
        # 归一化
        # samplingTime = 1/action[1]，而 action[1] ∈ [100,2000] Hz,因此 samplingTime ∈ [0.0005, 0.01] 秒
        samplingTime_norm = (self.tel.samplingTime - 0.0005) / (0.01 - 0.0005)  
        clock_rate_norm = (self.camH.clock_rate - 100) / 1900

        wfs_norm = self.wfsSignal / (self.wfsSignal.max() + 1e-6)
        frame_norm = self.camH.frame / (self.camH.frame.max() + 1e-6)
        state = np.concatenate([
            wfs_norm.flatten(),  # 波前传感器斜率
            frame_norm.flatten(),  # 远场图像
            np.array([self.gainCL, samplingTime_norm, clock_rate_norm])  # 其他状态
        ])
        return state

    def render(self, mode='human'):
        """显示当前状态"""
        if mode == 'human':
            plt.figure()
            plt.imshow(self.camH.frame, cmap='gray')
            plt.title('Science Camera Frame')
            plt.colorbar(label='Intensity')
            plt.show()