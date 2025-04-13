import numpy as np
from numpy.fft import fft2, fftshift
from tutorials.scao_system.myAoSystem.LensletArray import LensletArray
class Imager:
    def __init__(self, nyquist_sampling=4, field_stop_size=10, exposure_time=1, clock_rate=1, diameter=1):
        """
        初始化成像相机。

        参数：
        - nyquist_sampling: 奈奎斯特采样率
        - field_stop_size: 视场光阑大小（单位：角秒）
        - exposure_time: 曝光时间（单位：秒）
        - clock_rate: 时钟频率
        - diameter: 望远镜直径（单位：米）
        """
        self.nyquist_sampling = nyquist_sampling
        self.field_stop_size = field_stop_size
        self.exposure_time = exposure_time
        self.clock_rate = clock_rate
        self.diameter = diameter

        # 计算分辨率
        self.resolution = 2 * nyquist_sampling * field_stop_size

        # 初始化帧数据
        self.frame = np.zeros((self.resolution, self.resolution))
        self.reference_frame = None
        self.strehl = 0.0
        self.ee = None  # 捕获能量
        self.ee_width = None  # 捕获能量滤波宽度

        # 初始化透镜阵列（假设有一个透镜阵列类 LensletArray）
        self.img_lens = LensletArray(nyquist_sampling=nyquist_sampling, field_stop_size=field_stop_size)

    def relay(self, source):
        """
        将光源传播到成像相机。

        参数：
        - source: 光源对象
        """
        self.img_lens.propagate(source)  # 传播到透镜阵列
        self.frame = self.img_lens.imagelets  # 获取图像数据

    def compute_strehl(self, phase):
        """
        计算斯特列尔比。

        参数：
        - phase: 相位分布

        返回：
        - strehl: 斯特列尔比
        """
        phase_variance = np.var(phase)
        strehl = np.exp(-phase_variance)
        return strehl

    def compute_psf(self, phase):
        """
        计算点扩散函数（PSF）。

        参数：
        - phase: 相位分布

        返回：
        - psf: 点扩散函数（光强分布）
        """
        complex_field = np.exp(1j * phase)  # 复振幅
        psf = np.abs(fftshift(fft2(complex_field))) ** 2  # 光强分布
        psf /= psf.max()  # 归一化
        return psf

    def compute_ee(self, psf, ee_width):
        """
        计算捕获能量。

        参数：
        - psf: 点扩散函数
        - ee_width: 捕获能量滤波宽度

        返回：
        - ee: 捕获能量
        """
        n = len(psf)
        u = np.linspace(-1, 1, n) / self.nyquist_sampling
        x, y = np.meshgrid(u, u)

        ee_filter = (ee_width ** 2) * (np.sinc(np.pi * x * ee_width) * np.sinc(np.pi * y * ee_width))
        ee = np.trapz(np.trapz(psf * ee_filter, u), u)
        return ee

    def display_image(self):
        """
        显示成像相机的帧数据。
        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(self.frame, cmap='gray')
        plt.title('Imager Frame')
        plt.colorbar(label='Intensity')
        plt.show()