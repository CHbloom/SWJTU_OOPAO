import math

import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt

class LensletArray:
    def __init__(self, n_lenslet, nyquist_sampling=1, min_light_ratio=0, conjugation_altitude=np.inf, throughput=1):
        """
        初始化透镜阵列。

        参数：
        - n_lenslet: 透镜阵列每边的透镜数量
        - nyquist_sampling: 奈奎斯特采样率
        - min_light_ratio: 每个透镜的最小光量比
        - conjugation_altitude: 透镜阵列的共轭高度
        - throughput: 光学透过率
        """
        self.n_lenslet = n_lenslet
        self.nyquist_sampling = nyquist_sampling
        self.min_light_ratio = min_light_ratio
        self.conjugation_altitude = conjugation_altitude
        self.throughput = throughput
        self.imagelets = None
        self.fft_pad = math.ceil(self.nyquist_sampling*2)
        self.fft_phasor = None
        self.optical_aberration = None
        self.elongated_field_stop_size = None
        self.conv_kernel = False

    @property
    def n_lenslet_wave_px(self):
        """每个透镜的波前像素数"""
        return self._n_lenslet_wave_px

    @n_lenslet_wave_px.setter
    def n_lenslet_wave_px(self, value):
        self._n_lenslet_wave_px = value
        self.set_phasor()

    @property
    def field_stop_size(self):
        """透镜视场光阑大小"""
        return self._field_stop_size

    @field_stop_size.setter
    def field_stop_size(self, value):
        self._field_stop_size = value
        self.set_phasor()

    def set_phasor(self):
        """设置用于傅里叶变换的相位因子"""
        n_out_wave_px = self.n_lenslet_wave_px * self.fft_pad
        if n_out_wave_px % 2 == 0 and self.n_lenslet_wave_px % 2 != 0:
            n_out_wave_px += 1

        if self.n_lenslet_wave_px % 2 == 0:
            u, v = np.meshgrid(
                np.arange(self.n_lenslet_wave_px) * (1 - n_out_wave_px) / n_out_wave_px,
                np.arange(self.n_lenslet_wave_px) * (1 - n_out_wave_px) / n_out_wave_px
            )
            self.fft_phasor = np.tile(np.exp(-1j * np.pi * (u + v)), (self.n_lenslet, self.n_lenslet))
        else:
            self.fft_phasor = None

    def propagate_through(self, src):
        """
        将光源传播到透镜阵列的焦平面。

        参数：
        - src: 光源对象

        返回：
        - wave_propagated: 传播后的波前
        """
        val = src.phase  # 获取复振幅
        n_lenslet_wave_px = val.shape[0] // self.n_lenslet
        self.n_lenslet_wave_px = n_lenslet_wave_px

        if self.optical_aberration is not None:
            val = self.optical_aberration * val

        n_out_wave_px = self.n_lenslet_wave_px * self.fft_pad
        wave_propagated = np.zeros((n_out_wave_px, n_out_wave_px * self.n_lenslet ** 2), dtype=complex)

        if self.fft_phasor is None:
            val = val.reshape(self.n_lenslet_wave_px, -1)
            wave_propagated[:, :] = fftshift(fft2(val, (n_out_wave_px, n_out_wave_px)), axes=(0, 1))
        else:
            val = val * self.fft_phasor
            val = val.reshape(self.n_lenslet_wave_px, -1)
            wave_propagated[:, :] = fft2(val, (n_out_wave_px, n_out_wave_px))

        # 裁剪视场
        if n_out_wave_px > self.n_lenslet_wave_px:
            center_index = n_out_wave_px // 2
            half_length = self.n_lenslet_wave_px // 2
            wave_propagated = wave_propagated[
                center_index - half_length:center_index + half_length + 1,
                center_index - half_length:center_index + half_length + 1
            ]

        # 计算光强
        self.imagelets = np.abs(wave_propagated) ** 2 * self.throughput
        return wave_propagated

    def relay(self, src):
        """
        将光源传播到透镜阵列并生成图像。

        参数：
        - src: 光源对象
        """
        wave_propagated = self.propagate_through(src)
        self.imagelets = np.abs(wave_propagated) ** 2 * self.throughput

    def display_imagelets(self):
        """显示透镜阵列的图像"""
        if self.imagelets is not None:
            plt.figure()
            plt.imshow(self.imagelets, cmap='gray')
            plt.title('Lenslet Array Imagelets')
            plt.colorbar(label='Intensity')
            plt.show()
        else:
            print("No imagelets to display.")


# 创建光源
# class Source:
#     def __init__(self, wave):
#         self.wave = wave
#
# # 创建透镜阵列
# lenslet_array = LensletArray(n_lenslet=10, nyquist_sampling=2)
#
# # 创建光源的复振幅
# n_pixels = lenslet_array.n_lenslet * 20  # 假设每个透镜有 20 个像素
# wave = np.random.rand(n_pixels, n_pixels) * np.exp(1j * np.random.rand(n_pixels, n_pixels))
# src = Source(wave)
#
# # 传播光源并显示图像
# lenslet_array.relay(src)
# lenslet_array.display_imagelets()