#创建科学成像相机的版本
import time
import numpy as np
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.ShackHartmann import ShackHartmann
from tutorials.scao_system.myAoSystem.Imager import Imager  # 导入科学相机模块

# 创建望远镜
tel = Telescope(resolution=120,  # 分辨率
                diameter=8,  # 直径
                samplingTime=1/1000,  # 采样时间
                centralObstruction=0)  # 中心遮挡比例

# 创建自然引导星
ngs = Source(optBand='I',  # 光波段
             magnitude=8)  # 星等，表示光源的亮度
ngs * tel

# 创建大气湍流
atm = Atmosphere(telescope=tel,
                 r0=0.15,  # 大气湍流的弗里德参数（Fried parameter），表示大气相干长度，单位为米
                 L0=30,  # 大气湍流的外尺度（Outer scale），单位为米
                 windSpeed=[10, 12, 11, 15, 20],  # 每个湍流层的风速，单位为米/秒
                 fractionalR0=[0.45, 0.1, 0.1, 0.25, 0.1],  # 每个湍流层的 Cn² 剖面，表示每个层对总湍流的贡献比例
                 windDirection=[0, 72, 144, 216, 288],  # 每个湍流层的风向，单位为度
                 altitude=[0, 1000, 5000, 10000, 12000])  # 每个湍流层的海拔高度，单位为米
atm.initializeAtmosphere(telescope=tel)  # 初始化

# 创建变形镜
dm = DeformableMirror(telescope=tel,
                      nSubap=20,  # 沿望远镜直径的变形镜驱动器数，即总驱动数=nSubap*nSubap
                      mechCoupling=0.45)  # 驱动器之间的机械耦合系数

# 创建SH-WFS
wfs = ShackHartmann(nSubap=20,  # 沿望远镜直径的子孔径数
                    telescope=tel,
                    lightRatio=0.3,  # 用于筛选有效子孔径
                    is_geometric=False,  # 是否启用几何模式（直接测量梯度）
                    threshold_cog=0.01)  # 计算光斑质心时的阈值

# 创建科学相机
camH = Imager(telescope=tel,  # 望远镜对象
              nPix=128,  # 相机分辨率
              fieldStopSize=10,  # 视场光阑大小（单位：角秒）
              wavelength=ngs.wavelength)  # 波长

# 初始化参数
param = {'nLoop': 20}  # 循环次数
gainCL = 0.6  # 闭环控制增益
wfs.cam.photonNoise = True  # 启用光子噪声

# 分配内存以保存数据
SR = np.zeros(param['nLoop'])  # 斯特列尔比
total = np.zeros(param['nLoop'])  # 总波前误差
residual = np.zeros(param['nLoop'])  # 残余波前误差
wfsSignal = np.zeros(wfs.nSignal)  # 波前传感器信号

# 初始化变形镜控制信号
dm.coefs = 0

# 主循环
for i in range(param['nLoop']):
    start_time = time.time()

    # 更新大气湍流相位屏
    atm.update()

    # 记录总的波前误差
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9

    # 光波传播：自然引导星 -> 望远镜 -> 变形镜 -> 波前传感器
    tel * dm * wfs

    # 更新变形镜控制信号
    dm.coefs = dm.coefs - gainCL * M2C_CL @ calib_CL.M @ wfsSignal

    # 更新波前传感器信号
    wfsSignal = wfs.signal

    # 光波传播：自然引导星 -> 望远镜 -> 变形镜 -> 科学相机
    ngs * tel * dm * camH

    # 计算斯特列尔比（基于科学相机的PSF）
    SR[i] = camH.strehl  # 科学相机计算的斯特列尔比

    # 记录残余波前误差
    residual[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9

    # 打印结果
    elapsed_time = time.time() - start_time
    print(f'Loop {i+1}/{param["nLoop"]} - Elapsed time: {elapsed_time:.2f} s')
    print(f'Turbulence: {total[i]:.2f} nm -- Residual: {residual[i]:.2f} nm -- SR: {SR[i]:.2f}\n')

# 绘制斯特列尔比随时间的变化
import matplotlib.pyplot as plt
plt.figure()
plt.plot(np.arange(param['nLoop']), SR, label='Strehl Ratio')
plt.xlabel('Loop Index')
plt.ylabel('Strehl Ratio')
plt.title('Strehl Ratio over Time')
plt.legend()
plt.grid()
plt.show()