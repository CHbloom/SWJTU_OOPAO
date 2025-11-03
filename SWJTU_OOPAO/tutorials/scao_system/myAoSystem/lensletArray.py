import numpy as np
import logging
from scipy.ndimage import rotate
from matplotlib import pyplot as plt

class lensletArray:
    """
    一个透镜阵列（lenslet array）对象。

    Args:
        nLenslet (int): 透镜阵列单边上的透镜数量。

    属性:
        nLenslet (int): 透镜阵列单边上的透镜数量。
        pitch (float): 透镜尺寸。
        minLightRatio (float): 每个透镜的最小光线比率。
        conjugationAltitude (float): 透镜的共轭高度。
        wave (object): 输入波对象。
        sumStack (bool): 堆叠图像的总和。
        nArray (int): 透镜阵列的数量。
        throughput (float): 光学吞吐量。
        tag (str): 透镜阵列标签。
        elongatedFieldStopSize (float): 额外的视场，单位是衍射 fwhm。
        convKernel (bool): 卷积核标志。
        opticalAberration (object): 光学像差对象。
    """

    def __init__(self, nLenslet, nyquist_sampling=4, field_stop_size=10):
        # 构造函数，初始化所有属性，并设置合理的默认值
        self.nLenslet = nLenslet
        self.minLightRatio = 0.0
        self.throughput = 1.0
        self.nArray = 1
        self.tag = 'LENSLET ARRAY'
        self.conjugationAltitude = np.inf
        self.wave = None
        self.sumStack = False
        self.elongatedFieldStopSize = None
        self.convKernel = False
        self.opticalAberration = None
        self.pitch = None

        # 内部属性，将 _fft_pad 和 _n_lenslet_wave_px 初始化为非 None 值
        self._imagelets = None
        self._offset = np.zeros((3, 1))

        # 配置日志
        self.log = logging.getLogger(self.tag)

        # 依赖属性的初始化顺序至关重要
        # 首先设置 nyquistSampling，因为它会计算 _fft_pad
        self._nyquist_sampling = nyquist_sampling
        self._fft_pad = np.ceil(self._nyquist_sampling * 2)

        # 接着设置 _n_lenslet_wave_px 和 _field_stop_size
        self._field_stop_size = field_stop_size
        # 这里的计算确保了 nLensletWavePx 始终有值
        self._n_lenslet_wave_px = int(np.ceil(2 * self._nyquist_sampling * self._field_stop_size))

    # --- Dependent 属性使用 @property 装饰器 ---
    @property
    def nLensletWavePx(self):
        return self._n_lenslet_wave_px

    @nLensletWavePx.setter
    def nLensletWavePx(self, val):
        self._n_lenslet_wave_px = val
        if self._field_stop_size is None:
            self.log.info('Setting the lenslet field stop size!')
            self.fieldStopSize = self._n_lenslet_wave_px / self.nyquistSampling / 2
        self._fft_pad = np.ceil(self.nyquistSampling * 2)
        self._set_phasor()

    @property
    def nyquistSampling(self):
        return self._nyquist_sampling

    @nyquistSampling.setter
    def nyquistSampling(self, val):
        self._nyquist_sampling = val
        self._fft_pad = np.ceil(self._nyquist_sampling * 2)
        self._n_lenslet_wave_px = int(np.ceil(2 * self._nyquist_sampling * self.fieldStopSize))

    @property
    def fieldStopSize(self):
        return self._field_stop_size

    @fieldStopSize.setter
    def fieldStopSize(self, val):
        self._field_stop_size = val

    @property
    def offset(self):
        return self._offset[0:2, :]

    @offset.setter
    def offset(self, val):
        # 简化版，仅用于示例
        if isinstance(val, (list, np.ndarray)):
            val = np.array(val).reshape(2, -1)
            self._offset[0:2, :val.shape[1]] = val
        else:
            self._offset[0:2, :] = val

    @property
    def rotation(self):
        return self._offset[2, :]

    @rotation.setter
    def rotation(self, val):
        # 简化版，仅用于示例
        self._offset[2, :] = val

    @property
    def nLensletImagePx(self):
        # 计算每个像点（imagelet）的像素数
        return int(np.ceil(self.fieldStopSize * self.nyquistSampling * 2))

    @property
    def nLensletsImagePx(self):
        # 计算整个透镜阵列图像的像素数
        return self.nLenslet * self.nLensletImagePx

    @property
    def imagelets(self):
        return self._imagelets

    # setter 方法用于设置 imagelets，可以用于触发更新
    @imagelets.setter
    def imagelets(self, value):
        self._imagelets = value
        # 这里可以放置一个事件或回调函数，来模拟 MATLAB 的监听器功能
        # 例如：self.imagesc()

    # --- 方法 ---
    def relay(self, src, clockRate):
        """
        继电器，将光源传递给 propagate_through 方法并计算 imagelets。
        """
        wave_prgted = self.propagate_through(clockRate, src=src)
        # 计算强度
        self.imagelets = np.abs(wave_prgted)**2 * self.throughput

    # def propagate_through(self, src):
    #     """
    #     在透镜阵列焦平面上模拟 Fraunhoffer 波传播。
    #     """
    #     # 获取波形数据
    #     # val = src.catWave() if hasattr(src, 'catWave') else src #-----matlab
    #     # 获取瞬时光子通量图
    #     flux_map = src.fluxMap
    #     # 计算振幅 A(x, y) = sqrt(flux_map)
    #     # fluxMap本身就是瞬时通量，不需要乘以时间
    #     amplitude = np.sqrt(flux_map)
    #     # 生成复数波前 U(x, y) = A * exp(i * phase)
    #     val = amplitude * np.exp(1j * src.phase)


    #     # 应用光学像差
    #     if self.opticalAberration is not None:
    #         val = self.opticalAberration * val
    #     print('-----',self.nLensletWavePx,self._fft_pad)
    #     # 获取像点图像的大小
    #     n_out_wave_px = int(self.nLensletWavePx * self._fft_pad)

    #     # 执行傅里叶变换
    #     # 使用 np.fft.fft2 进行二维FFT

    #     wave_prgted = np.fft.fft2(val, s=(n_out_wave_px, n_out_wave_px))
    #     wave_prgted = np.fft.fftshift(wave_prgted)
        
    #     # 裁剪到指定的视场大小
    #     center = np.array(wave_prgted.shape) // 2
    #     half_length = self.nLensletImagePx // 2
    #     wave_prgted = wave_prgted[
    #         center[0] - half_length : center[0] + half_length,
    #         center[1] - half_length : center[1] + half_length
    #     ]

    #     # 扩展透镜视场（如果有需要）
    #     if self.elongatedFieldStopSize is not None:
    #         # 这里的填充逻辑会很复杂，需要根据 MATLAB 代码的细节实现
    #         # 使用 np.pad 或其他方法
    #         pass

    #     # 堆叠求和
    #     if self.sumStack:
    #         wave_prgted = np.mean(wave_prgted, axis=-1)

    #     self.sumStack = False
    #     return wave_prgted
    
    def propagate_through(self, clockRate, src=None, data=None):
        """
        根据 Source 对象的属性或直接提供的图像数据来计算传播后的波前。
        
        Args:
            src (Source): 包含相位信息的 Source 对象。
            data (numpy.ndarray): 直接提供的图像数据，通常用于分析。
            
        Returns:
            numpy.ndarray: 代表传播后波前的复数数组。
        """
        if data is not None:
            # 如果提供了数据，直接将其作为波前幅度，相位为0
            val = np.sqrt(data)
            # 在没有相位信息时，波前为实数
            # 如果需要模拟相位，这部分逻辑需要更复杂
            # 这里的简化是基于 flush 方法中不需要相位信息的假设
        elif src is not None:
            # 如果提供了 Source 对象，使用其相位和通量来构建波前
            if not hasattr(src, 'phase') or not hasattr(src, 'fluxMap'):
                print("警告：Source 对象缺少所需的相位或通量属性。")
                return np.ones((self.resolution, self.resolution), dtype=complex)
            print("fluxMap:",src.fluxMap.shape)
            print("src.phase:",src.phase.shape)
            amplitude = np.sqrt(src.fluxMap/clockRate)
            val = amplitude * np.exp(1j * src.phase)
        else:
            print("错误：必须提供一个 Source 对象或一个数据数组。")
            return np.ones((self.resolution, self.resolution), dtype=complex)
        
        
        if self.opticalAberration is not None:
            val = self.opticalAberration * val

        n_out_wave_px = int(self.nLensletWavePx * self._fft_pad)
        wave_prgted = np.fft.fft2(val, s=(n_out_wave_px, n_out_wave_px))
        wave_prgted = np.fft.fftshift(wave_prgted)
        # 裁剪到指定的视场大小
        center = np.array(wave_prgted.shape) // 2
        half_length = self.nLensletImagePx // 2
        wave_prgted = wave_prgted[
            center[0] - half_length : center[0] + half_length,
            center[1] - half_length : center[1] + half_length
        ]

        # 扩展透镜视场（如果有需要）
        if self.elongatedFieldStopSize is not None:
            # 这里的填充逻辑会很复杂，需要根据 MATLAB 代码的细节实现
            # 使用 np.pad 或其他方法
            pass

        # 堆叠求和
        if self.sumStack:
            wave_prgted = np.mean(wave_prgted, axis=-1)

        self.sumStack = False
        return wave_prgted

    def imagesc(self, **kwargs):
        """
        使用 Matplotlib 显示透镜图像。
        
        Args:
            **kwargs: 传递给 matplotlib.pyplot.imshow 的额外参数。
        """
        if self.imagelets is None:
            print("没有图像可以显示。请先运行 'relay' 方法。")
            return

        plt.imshow(self.imagelets, cmap='pink', origin='lower', **kwargs)
        plt.colorbar(label='Intensity')
        plt.title(f"{self.tag} Imagelets")
        plt.xlabel('X Pixels')
        plt.ylabel('Y Pixels')
        plt.show()

    def _set_phasor(self):
        """
        私有方法，设置用于傅里叶变换的相位因子。
        """
        # 此方法旨在优化 FFT，以确保像点居中对齐
        self._fft_phasor = None
        if self.nLensletImagePx % 2 == 0:
            self.log.info("设置相位因子以进行偶数像素采样")
            # 在这里计算并设置 self._fft_phasor 矩阵
            # ...
        else:
            self.log.info("重置相位因子")

    def display(self):
        """
        打印对象信息。
        """
        print(f"___ {self.tag} ___")
        if self.nArray > 1:
            print(f" {self.nArray} {self.nLenslet}x{self.nLenslet} lenslet array: ")
        else:
            print(f" {self.nLenslet}x{self.nLenslet} lenslet array: ")
        print(f"  . {self.nyquistSampling * 2:.1f} 像素/衍射极限 fwhm")
        print(f"  . {self.fieldStopSize * self.nyquistSampling * 2} 像素/方形透镜视场光阑大小")
        print(f"  . 光学吞吐量系数: {self.throughput:.1f}")
        print("----------------------------------------------------")