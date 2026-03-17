# 目前使用的成像相机，LensletArray在此文件中

import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fft2
from lensletArray import lensletArray


class Imager2:
    def __init__(self, nyquist_sampling=4, field_stop_size=10, exposure_time=1, clock_rate=1, diameter=1):
        """
        初始化成像相机对象
        :param nyquist_sampling: 奈奎斯特采样率，控制每个透镜的分辨率
        :param field_stop_size: 视场大小，定义图像的空间范围
        :param exposure_time: 曝光时间
        :param clock_rate: 时钟速率，控制图像更新速度
        :param diameter: 望远镜的直径，默认为 1 米
        """
        self.nyquist_sampling = nyquist_sampling
        self.field_stop_size = field_stop_size
        self.exposure_time = exposure_time
        self.clock_rate = clock_rate
        self.diameter = diameter

        # 初始化图像和相关参数
        self.frame = None
        self.reference_frame = None
        self.strehl = None  # 斯特列尔比
        self.ee = None      # 包裹能量 (Encircled Energy)
        self.ee_width = None

        # 生成微透镜阵列
        self.resolution =int(np.ceil(2 * nyquist_sampling * field_stop_size)) #计算分辨率
        self.img_lens = LensletArray(1, nyquist_sampling, field_stop_size)

    def relay(self, source):
        """
        进行波前的传输，计算通过微透镜阵列后的图像
        :param wavefront: 输入波前的复数形式
        """
        # 根据曝光时间和时钟速率计算每帧的光子数
        # fluxMap (list)：单位时间内每像素的光子数分布
        n_photon_total = source.fluxMap * self.exposure_time  # 在曝光时间内累积光子数
        n_photon_per_frame = n_photon_total / self.clock_rate  # 每帧的光子数

        # 计算振幅 A(x, y) = sqrt(n_photon_per_frame)
        amplitude = np.sqrt(n_photon_per_frame)
        # 生成复数波前 U(x, y) = A * exp(i * phase)
        wavefront = amplitude * np.exp(1j * source.phase)
        propagated_wave = self.img_lens.propagate_through(wavefront)# 傅里叶变换
        # 读取并累加图像帧
        if self.frame is None:
            self.frame = np.abs(propagated_wave) ** 2
        else:
            self.frame = np.abs(propagated_wave) ** 2
        self.flush()

    def flush(self):
        """
        读取和清空缓冲区，计算斯特列尔比和包裹能量
        """
        # wavePrgted = propagateThrough(obj.imgLens, src_);
        # otf = abs(wavePrgted);
        # otf = otf / max(otf(:));
        # otf = mat2cell(otf, size(otf, 1), size(otf, 2) / nSrc * ones(1, nSrc));
        if self.reference_frame is not None and self.frame is not None:
            # 计算斯特列尔比
            # otf = self.reference_frame
            # otf = otf / np.max(otf)
            #
            # otfAO = self.frame
            # otfAO = otfAO / np.max(otf)
            # self.strehl = np.sum(otfAO) / np.sum(otf)
            self.strehl = np.max(self.frame) / np.max(self.reference_frame)
            # 计算包裹能量
            if self.ee_width:
                self.ee = self.calculate_encircled_energy(self.ee_width)

            # 清空图像缓冲区
            # self.frame = None

    def calculate_encircled_energy(self, width):
        """
        计算指定像素范围内的包裹能量
        :param width: 包裹能量计算的窗口大小
        :return: 包裹能量值
        """
        center_x, center_y = np.array(self.frame.shape) // 2
        mask = self.create_circular_mask(center_x, center_y, width)
        return np.sum(self.frame[mask]) / np.sum(self.frame)

    @staticmethod
    def create_circular_mask(x, y, radius):
        """
        创建一个圆形掩码，用于包裹能量计算
        :param x: 圆心的 x 坐标
        :param y: 圆心的 y 坐标
        :param radius: 圆的半径
        :return: 圆形掩码
        """
        Y, X = np.ogrid[:2 * x, :2 * y]
        dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
        return dist <= radius

    def display(self):
        """
        显示当前的成像结果
        """
        if self.frame is None:
            print("当前没有成像数据。")
            return

        # plt.imshow(self.frame, cmap='gray')


        plt.imshow(self.frame, cmap='pink', interpolation='bilinear',origin='upper')
        plt.colorbar()
        plt.title("Imager Output")
        plt.show()

class LensletArray:
    def __init__(self, n_lenslet, nyquist_sampling=1, field_stop_size=1):
        """
        微透镜阵列初始化
        :param n_lenslet: 微透镜数量
        :param nyquist_sampling: 奈奎斯特采样率
        :param field_stop_size: 视场大小
        """
        self.n_lenslet = n_lenslet
        self.nyquist_sampling = nyquist_sampling
        self.field_stop_size = field_stop_size

    def propagate_through(self, wavefront):
        """
        模拟光波通过微透镜阵列的传播
        :param wavefront: 输入光波
        :return: 传播后的光波
        """
        
        return fftshift(fft2(wavefront))
