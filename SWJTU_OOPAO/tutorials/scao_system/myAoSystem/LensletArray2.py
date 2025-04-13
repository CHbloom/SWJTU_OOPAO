import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

class LensletArray:
    def __init__(self, n_lenslet):
        """
        初始化 LensletArray 类
        :param n_lenslet: 微透镜阵列一侧的透镜数量
        """
        self.n_lenslet = n_lenslet  # 微透镜阵列规模
        self.pitch = 1.0  # 微透镜间距
        self.min_light_ratio = 0.0  # 最小光强比
        self.conjugation_altitude = float('inf')  # 共轭高度
        self.wave = None  # 输入光波
        self.imagelets = None  # 微透镜焦面上的图像
        self.n_array = 1  # 微透镜阵列的数量
        self.throughput = 1.0  # 光学透过率
        self.offset = np.zeros((2,))  # 偏移量
        self.rotation = 0.0  # 旋转角

        # 采样相关参数
        self.nyquist_sampling = 1.0
        self.field_stop_size = 1.0
        self.n_lenslet_wave_px = 32

    @property
    def n_lenslet_image_px(self):
        """
        计算每个微透镜焦平面的像素数
        """
        return int(np.ceil(self.field_stop_size * self.nyquist_sampling * 2))

    @property
    def n_lenslets_image_px(self):
        """
        计算整个微透镜阵列的像素总数
        """
        return self.n_lenslet * self.n_lenslet_image_px

    def propagate_through(self, wave):
        """
        模拟光波通过微透镜阵列的传播过程
        :param wave: 输入的光波复数阵列
        """
        self.wave = wave

        n = wave.shape[0]  # 输入光波的尺寸

        # 应用空间偏移
        if np.any(self.offset != 0):
            wave = self.apply_offset(wave)

        # 傅里叶变换模拟衍射
        wave_propagated = fftshift(fft(wave, n=n, axis=0), axes=0)

        # 计算光斑强度
        self.imagelets = np.abs(wave_propagated) ** 2

    def apply_offset(self, wave):
        """
        根据 offset 调整输入波的相位
        :param wave: 输入的光波
        :return: 调整后的光波
        """
        nx, ny = wave.shape
        x, y = np.meshgrid(np.linspace(-0.5, 0.5, nx), np.linspace(-0.5, 0.5, ny))

        phase_shift = np.exp(2j * np.pi * (self.offset[0] * x + self.offset[1] * y))
        return wave * phase_shift

    def plot_imagelets(self):
        """
        可视化微透镜焦平面的光斑图像
        """
        if self.imagelets is None:
            raise ValueError("请先使用 propagate_through 方法计算光斑。")

        plt.imshow(self.imagelets, cmap='pink')
        plt.colorbar()
        plt.title("Lenslet Array Imagelets")
        plt.xlabel("Pixels")
        plt.ylabel("Pixels")
        plt.show()

    def display(self):
        """
        输出 LensletArray 对象的基本信息
        """
        print("--- Lenslet Array ---")
        print(f"  尺寸: {self.n_lenslet}x{self.n_lenslet}")
        print(f"  像素/微透镜: {self.n_lenslet_image_px}")
        print(f"  透过率: {self.throughput}")
        print(f"  偏移量: {self.offset}")
        print(f"  旋转角: {self.rotation}")
        print("----------------------")

# 示例代码
if __name__ == '__main__':
    # 创建一个 8x8 的微透镜阵列
    lenslet_array = LensletArray(8)

    # 生成一个模拟的输入光波
    wave_input = np.ones((256, 256), dtype=complex)

    # 设置偏移量和旋转角
    lenslet_array.offset = np.array([0.1, -0.1])
    lenslet_array.rotation = np.pi / 4

    # 执行光波传播
    lenslet_array.propagate_through(wave_input)

    # 显示微透镜阵列信息
    lenslet_array.display()

    # 可视化焦面光斑
    lenslet_array.plot_imagelets()
