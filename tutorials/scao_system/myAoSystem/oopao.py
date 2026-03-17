import time

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Pyramid import Pyramid
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.calibration.ao_calibration import ao_calibration
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.tools.displayTools import displayMap
# 创建望远镜
tel = Telescope(resolution          = 120, # 分辨率
                diameter            = 8, # 直径
                samplingTime        = 1/1000, # 采样时间
                centralObstruction  = 0) # 中心遮挡比例
# 创建自然引导星
ngs=Source(optBand   = 'I', # 光波段
           magnitude = 8) # 星等，表示光源的亮度
ngs*tel
atm=Atmosphere(telescope     = tel,
               r0            = 0.15, # 大气湍流的弗里德参数（Fried parameter），表示大气相干长度，单位为米
               L0            = 30, # 大气湍流的外尺度（Outer scale），单位为米
               windSpeed     = [10,12,11,15,20] , # 每个湍流层的风速，单位为米/秒
               fractionalR0  = [0.45,0.1,0.1,0.25,0.1], # 每个湍流层的 Cn² 剖面，表示每个层对总湍流的贡献比例
               windDirection = [0,72,144,216,288], # 每个湍流层的风向，单位为度
               altitude      = [0, 1000,5000,10000,12000 ]) # 每个湍流层的海拔高度，单位为米
atm.initializeAtmosphere(telescope=tel) # 初始化
atm.display_atm_layers() # 可视化//第一个可以修改的参数，大气环境我们要怎么检测得到

atm.print_properties()

print(tel.isPaired)
tel+atm # 望远镜与大气结合
print(tel.isPaired)

# 创建变形镜
dm=DeformableMirror(telescope    = tel,
                    nSubap       = 20, # 沿望远镜直径的变形镜驱动器数，即总驱动数=nSubap*nSubap
                    mechCoupling = 0.45) # 驱动器之间的机械耦合系数,用于描述变形镜的各个驱动器之间的机械相互作用程度

# 创建SH-WFS

wfs = ShackHartmann(nSubap=20,  # 沿望远镜直径的子孔径数
                    telescope=tel,
                    lightRatio=0.3,  # 用于筛选有效子孔径
                    is_geometric=False,  # 是否启用几何模式（直接测量梯度）
                    threshold_cog=0.01)  # 计算光斑质心时的阈值

from parameter_files.parameterFile_VLT_I_Band_SHWFS import initializeParameterFile

param = initializeParameterFile()

'''
加载或生成模态基,即模式到控制矩阵（M2C）用于将模式映射到变形镜（DM）的控制信号
计算或加载交互矩阵
'''
ao_calib =  ao_calibration(param            = param,\
                           ngs              = ngs,\
                           tel              = tel,\
                           atm              = atm,\
                           dm               = dm,\
                           wfs              = wfs,\
                           nameFolderIntMat = None,\
                           nameIntMat       = None,\
                           nameFolderBasis  = None,\
                           nameBasis        = None,\
                           nMeasurements    = 100)

# 闭环控制
calib_CL = ao_calib.calib  # 交互矩阵
M2C_CL = ao_calib.M2C

tel + atm
ngs * tel * dm * wfs
# initialize DM commands
dm.coefs = 0

param['nLoop'] = 20
# allocate memory to save data
SR = np.zeros(param['nLoop'])  # 斯特列尔比
total = np.zeros(param['nLoop'])
residual = np.zeros(param['nLoop'])
wfsSignal = np.arange(0, wfs.nSignal) * 0
# loop parameters
gainCL = 0.6  # 传感增益
wfs.cam.photonNoise = True
display = True

for i in range(param['nLoop']):
    a = time.time()
    # 更新所有湍流层的相位屏，模拟时间演化
    atm.update()
    # 记录总的波前误差
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9

    tel * dm * wfs

    '''
    更新DM的控制信号
    dm.coefs：变形镜的控制系数，表示每个驱动器的位移量。 11111
    gainCL：闭环控制增益，用于调节校正的强度    11111波前传感器的增益 而采样频率通常是和望远镜的采样频率同步的
    M2C_CL：模式到控制矩阵，用于将模式映射到变形镜的控制信号。
    calib_CL.M：交互矩阵的伪逆矩阵，用于将波前传感器的信号转换为模式系数。
    wfsSignal:shwfs测量的斜率，形状：（2*nValidSubaperture,）
    '''
    dm.coefs = dm.coefs - gainCL * M2C_CL @ calib_CL.M @ wfsSignal

    wfsSignal = wfs.signal

    b = time.time()
    print('Elapsed time: ' + str(b - a) + ' s')

    SR[i] = np.exp(-np.var(
        tel.src.phase[np.where(tel.pupil == 1)]))  # 实际代码的SR比不通过图像的比来得到，而使用近似的相位差距来得到，后续需要修改 改成用科学成像相机的方式来测量斯特列尔比
    residual[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9

    print('Loop' + str(i) + '/' + str(param['nLoop']) + ' Turbulence: ' + str(total[i]) + ' -- Residual:' + str(
        residual[i]) + '\n')





