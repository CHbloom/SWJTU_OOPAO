'''
设置3种不同风速，三种不同风向排列组合
'''

def initializeParameterFile():
    param = dict()

    # Atmosphere
    param['r0'] = 0.16                          # 大气湍流的弗里德参数（Fried parameter），表示大气相干长度，单位为米
    param['L0'] = 50                            # 大气湍流的外尺度（Outer scale），单位为米
    param['fractionnalR0'] = [0.6,0.25,0.15]    # 每个湍流层的 Cn² 剖面，表示每个层对总湍流的贡献比例
    param['windSpeed'] = [15,20,30]                 # 每个湍流层的风速，单位为米/秒
    param['windDirection'] = [0,30,60]            # 每个湍流层的风向，单位为度
    param['altitude'] = [0,5000,12000]          # 每个湍流层的海拔高度，单位为米

    # Telescope
    param['resolution'] = 192                   # 分辨率
    param['diameter'] = 4                       # 直径
    param['sampling_rate'] = 1000               # 采样帧频
    param['centralObstruction'] = 0.2             # 中心遮挡比例

    # Source
    param['magnitude'] = 0                      # 星等
    param['opticalBand'] = 'I'                  # 光波段

    # DeformableMirror
    param['nSubap'] = 32                        # 沿望远镜直径的变形镜驱动器数，即总驱动数=nSubap*nSubap
    param['mechCoupling'] = 0.1                 # 驱动器之间的机械耦合系数,用于描述变形镜的各个驱动器之间的机械相互作用程度

    # ShackHartmann                                                                                
    param['lightRatio'] = 0.6                   # 用于筛选有效子孔径                     
    param['is_geometric'] = False               # 是否启用几何模式（直接测量梯度
    param['threshold_cog'] = 0.01               # 计算光斑质心时的阈值

    param['field_stop_size'] = 32
    param['nyquist_sampling'] = 2

    # 噪声
    param['photonNoise'] = True                 # 光子噪声
    param['readoutNoise'] = 1                  # 读出噪声
    param['darkCurrent'] = 1e-4                  # 暗电流

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['nModes'                ] = 200                                          # number of KL modes controlled 
    # name of the system
    param['name'] = 'VLT_' +  param['opticalBand'] +'_band_'+ str(param['nSubap'])+'x'+ str(param['nSubap'])  
    
    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = 'data_cl/'

    return param
