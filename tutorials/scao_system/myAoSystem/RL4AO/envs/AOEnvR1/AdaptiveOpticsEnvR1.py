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
from tutorials.scao_system.parameter_files.parameterFile_VLT_I_Band_SHWFS import initializeParameterFile
from collections import deque 

class AOEnv(gym.Env):
    def __init__(self, args):
        super(AOEnv, self).__init__()
        self.param = initializeParameterFile()
        
        # 系统参数（保持原有参数命名）
        self.max_step = args.max_step
        self.init_gainCL = args.gainCL
        self.gainCL = self.init_gainCL
        self.sampling_rate = args.sampling_rate
        self.exposure_time = args.exposure_time
        self.clock_rate = args.clock_rate
        self.lightRatio = args.lightRatio
        
        # 光学组件初始化（保持原有结构）
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

        # 校准矩阵（保持原有名称）
        self.ao_calib = ao_calibration(
            param=self.param, ngs=self.ngs, tel=self.tel,
            atm=self.atm, dm=self.dm, wfs=self.wfs
        )
        self.calib_CL = self.ao_calib.calib
        self.M2C_CL = self.ao_calib.M2C

        # 状态空间优化（向量形式）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(20*20 + 120*120 + 140 + 280 + 4 + 3,), dtype=np.float32
        )

        # 动作空间（保持原有参数范围和顺序）
        self.action_space = spaces.Box(
            low=np.array([0.1, 100, 0.1, 100], dtype=np.float32),
            high=np.array([1.0, 2000, 1.0, 2000], dtype=np.float32),
            dtype=np.float32
        )

        # 湍流特征历史（新增）
        self.wfe_history = deque(maxlen=5)
        self.slope_var_history = deque(maxlen=3)
        self.dm_diff_history = deque(maxlen=5)

        # 运行参数（保持原有名称）
        self.current_step = 0
        self.SR = np.zeros(self.max_step)
        self.residual = np.zeros(self.max_step)
        self.wfsSignal = np.zeros(self.wfs.nSignal)
        self.dm_coefs_history = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境（保持原有结构）"""
        self.current_step = 0
        self.dm.coefs = 0
        self.SR = np.zeros(self.max_step)
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
        
        return self._get_state(), {}

    def step(self, action):
        """执行步骤（优化控制流程）"""
        # 动作参数处理（保持原有参数顺序）
        gainCL, tel_freq, exposure, cam_freq = np.clip(
            action, self.action_space.low, self.action_space.high
        )
        
        # 更新系统参数（保持原有参数命名）
        self.gainCL = gainCL
        self.tel.samplingTime = 1 / tel_freq
        self.camH.exposure_time = exposure
        self.camH.clock_rate = cam_freq

        # 大气扰动更新（优化相位屏更新）
        self.atm.update()
        self.tel - self.atm
        self.camH.relay(self.tel.src)
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

        # 计算奖励（优化奖励函数）
        reward = self._calculate_reward(action, current_wfe)

        self.current_step += 1
        terminated = self.current_step >= self.max_step
        
        return self._get_state(), reward, terminated, False, {}

    def _get_state(self):
        """构建聚合状态向量（整合湍流特征）"""
        # 传感器数据归一化（保持原有处理方式）
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
        
        return np.concatenate([
            wfs_image.flatten(),    # 400
            far_field.flatten(),    # 14400
            dm_voltage,             # 140
            slopes,                 # 280
            turb_features,          # 4
            params                  # 3
        ], dtype=np.float32)

    def _calculate_reward(self, action, current_wfe):
        """优化奖励函数（新增湍流惩罚）"""
        # 基础奖励（保持原有计算方式）
        normalized_wfe = 0.5 * (np.tanh((current_wfe - 100)/300) + 1)
        if current_wfe < 100:
            reward = self.SR[self.current_step] + (1 - normalized_wfe)
        else:
            reward = self.SR[self.current_step] - 0.3 * normalized_wfe
        
        # 新增湍流波动惩罚
        turb_penalty = 0.3 * np.std(list(self.wfe_history)[-3:])
        
        # 新增时序同步惩罚（保持原有逻辑）
        tel_freq, cam_freq = action[1], action[3]
        if abs(tel_freq - cam_freq) > 100:
            reward -= 1
        
        # 新增电压波动惩罚
        if self.dm_coefs_history:
            volt_diff = np.mean(np.diff(self.dm_coefs_history, axis=0)**2)
            reward -= 0.1 * volt_diff / self.dm.nSubap**2
            
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