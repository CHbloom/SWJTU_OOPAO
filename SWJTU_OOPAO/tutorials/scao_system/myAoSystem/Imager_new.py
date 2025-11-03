import numpy as np 
from OOPAO import Detector # 假设 detector 是一个父类，需要从同一包中导入
from lensletArray import lensletArray
from scipy.signal import convolve2d
from scipy.ndimage import convolve, fourier_shift
import numpy as np
from matplotlib import pyplot as plt


class tools:
    @staticmethod
    def rearrange(shape, tile_shape):
        n_rows, n_cols = shape
        tile_rows, tile_cols = tile_shape
        row_indices = np.arange(n_rows).reshape(-1, 1) // tile_rows
        col_indices = np.arange(n_cols).reshape(1, -1) // tile_cols
        return np.ravel_multi_index((row_indices.T, col_indices.T), (n_rows//tile_rows, n_cols//tile_cols), order='F').T

    @staticmethod
    def crop(image, size):
        s = image.shape
        start_x = (s[0] - size) // 2
        start_y = (s[1] - size) // 2
        return image[start_x:start_x + size, start_y:start_y + size]
        
    @staticmethod
    def binning(array, new_shape):
        shape = (new_shape[0], array.shape[0] // new_shape[0],
                 new_shape[1], array.shape[1] // new_shape[1])
        return (array.reshape(shape).mean(-1).mean(1)) * \
               (array.shape[0] // new_shape[0]) * (array.shape[1] // new_shape[1])
               
    @staticmethod
    def padArray2(arr, pad_h, pad_w):
        return np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)))
# 假设一个简单的 source 类用于演示
class source:
    def __init__(self, data=None):
        self.data = data
    def catWave(self):
        return self.data

class Imager_new():
    """
    一个成像相机对象，用于模拟图像采集和光学性能分析。
    
    该类通过一个透镜阵列（lenslet array）处理光源数据，并计算光学指标
    如 Strehl 比和陷落能量。
    """
    def __init__(self, nyquist_sampling=4, field_stop_size=10, exposure_time=1, clock_rate=1, diameter=1):
        """
        初始化 Imager 对象。

        Args:
            nyquist_sampling (float): 奈奎斯特采样率，默认值 4。
            field_stop_size (float): 视场光阑大小，默认值 10。
            exposure_time (float): 曝光时间，默认值 1。
            clock_rate (float): 时钟速率，默认值 1。
            diameter (float): 望远镜直径，默认值 1。
        """
        # 计算分辨率
        self.resolution = int(np.ceil(2 * nyquist_sampling * field_stop_size))

        # 初始化 Imager 的公共属性
        self.referenceFrame = None
        self.frame = None
        self.imgLens = lensletArray(1,nyquist_sampling=nyquist_sampling,field_stop_size=field_stop_size)  # 初始化一个 lensletArray 实例
        self.imgLens.nyquistSampling = nyquist_sampling
        self.imgLens.fieldStopSize = field_stop_size
        self.strehl = None
        self.ee = None
        self.eeWidth = None
        self.diameter = diameter
        
        self.tel = None
        self.exposureTime = exposure_time
        self.clockRate = clock_rate
        self.frameCount = 0
        self.frameBuffer = 0

        # 对应 MATLAB 代码中新增的属性
        self.spotsLgsSrcKernel = None
        self.fftSpotsLgsSrcKernel = None
        self.offSetSpotsLgsSrcKernel = None
        self.binningFactor = 1
        self.photonNoise = False
        self.nPhotonBackground = 0
        self.quantumEfficiency = 1
        self.readOutNoise = 0
        self.darkBackground = 0
        self.pixelGains = 0
        self.startDelay = 0
        
        if diameter is None:
            print('Caution, a default 1m telescope is associated to the class\n')
            
    def relay(self, src):
        """
        将光源通过镜头阵列传递，并触发图像数据处理。
        
        Args:
            src (object): 源对象。
        """
        self.imgLens.relay(clockRate=self.clockRate, src=src)
        # plt.imshow(self.imgLens.imagelets)
        # 将 imgLens 的图像数据传递给 readout 方法
        self.readOut(self.imgLens.imagelets)
        
        # 调用 flush 方法来处理缓冲区，如果条件满足
        if self.frameCount == 0 and self.startDelay == 0:
            # self.flush(len(src)) # --------matlab代码的逻辑，src可能是一个光源列表
            self.flush(1) # ----------我们是单光源，直接传入1，方法输入src对象

    def flush(self, nSrc=1):
        """
        处理已捕获的帧，并计算光学性能指标。
        """
        if self.referenceFrame is not None and self.frame is not None:
            
            self.imgLens.fieldStopSize *= 2
            
            # ref_source = source(data=self.referenceFrame) ----------matlab：计算光学传递函数otf
            # wave_prgted_ref = self.imgLens.propagate_through(ref_source)
            wave_prgted_ref = self.imgLens.propagate_through(self.clockRate, data=self.referenceFrame)
            otf_ref = np.abs(wave_prgted_ref)
            otf_ref /= np.max(otf_ref)
            
            m_frame = self.frame / (self.exposureTime * self.clockRate)
            
            n1, n2 = self.referenceFrame.shape
            num_frames = m_frame.shape[1] // n2
            
            self.strehl = np.zeros((nSrc, num_frames))
            if self.eeWidth is not None:
                self.ee = np.zeros((num_frames, len(self.eeWidth)))

            for kFrame in range(num_frames):
                current_frame_data = m_frame[:, kFrame * n2 : (kFrame + 1) * n2]
                # current_source = source(data=current_frame_data) ----------matlab
                # wave_prgted_ao = self.imgLens.propagate_through(current_source)
                wave_prgted_ao = self.imgLens.propagate_through(self.clockRate, data=current_frame_data)
                otf_ao = np.abs(wave_prgted_ao)
                otf_ao /= np.max(otf_ao)
                
                otf_ao_list = np.split(otf_ao, nSrc, axis=1)
                otf_ref_list = np.split(otf_ref, nSrc, axis=1)

                for kobj in range(nSrc):
                    strehl_val = np.sum(otf_ao_list[kobj]) / np.sum(otf_ref_list[kobj])
                    self.strehl[kobj, kFrame] = strehl_val
                    
                    if self.eeWidth is not None:
                        n_otf = otf_ao_list[kobj].shape[0]
                        u = np.linspace(-1, 1, n_otf) / self.imgLens.nyquistSampling
                        x, y = np.meshgrid(u, u)
                        
                        for kIntegBoxSize, a in enumerate(self.eeWidth):
                            ee_filter = a**2 * (np.sinc(x * a)) * (np.sinc(y * a))
                            entrapped_energy = np.sum(otf_ao_list[kobj] * ee_filter)
                            self.ee[kFrame, kIntegBoxSize] = entrapped_energy
                            
            self.imgLens.fieldStopSize /= 2
            self.frameCount = 0

    def readOut(self, image):
        """
        根据 MATLAB 代码实现探测器读出，包括图像处理和噪声添加。
        """
        # --- 1. 图像处理（如果 spotsLgsSrcKernel 存在） ---
        if self.spotsLgsSrcKernel is not None or self.fftSpotsLgsSrcKernel is not None:
            n, m = image.shape[:2]
            nLenslet = int(np.sqrt(self.spotsLgsSrcKernel.shape[2]))
            nPxDetector = self.resolution // nLenslet
            
            # 简化版逻辑，处理 2D 和 3D 图像
            if image.ndim == 2:
                nArray = m // n
            else:
                nArray = image.shape[2]
            
            output = np.zeros_like(image)
            
            for iArray in range(nArray):
                if image.ndim == 2:
                    tmp = image[:, iArray * n : (iArray + 1) * n]
                else:
                    tmp = image[:, :, iArray]
                
                nPx, mPx = tmp.shape
                nPxLenslet = nPx // nLenslet
                mPxLenslet = mPx // nLenslet
                
                index_raster_lenslet = tools.rearrange(tmp.shape, (nPxLenslet, mPxLenslet))
                buffer = tmp[index_raster_lenslet].T.reshape(nPxLenslet, mPxLenslet, -1)
                
                buffer_detect = np.zeros((nPxDetector, nPxDetector, nLenslet**2))
                
                for ii in range(nLenslet**2):
                    subap = buffer[:, :, ii]
                    
                    binned_kernel = tools.padArray2(self.spotsLgsSrcKernel[:, :, ii, iArray], self.binningFactor, self.binningFactor) / self.binningFactor**2
                    
                    if subap.shape[0] >= binned_kernel.shape[0]:
                        conv_kernel = convolve2d(subap, binned_kernel, mode='same')
                    else:
                        conv_kernel = convolve2d(binned_kernel, subap, mode='same')
                    
                    n_crop = nPxDetector * self.binningFactor
                    conv_kernel_cropped = tools.crop(conv_kernel, n_crop)
                    
                    buffer_detect[:, :, ii] = tools.crop(tools.binning(conv_kernel_cropped, conv_kernel_cropped.shape[0] // self.binningFactor), nPxDetector)

                index_raster_detector = tools.rearrange(np.array([self.resolution, self.resolution]), (nPxDetector, nPxDetector))
                tmp_detector = np.zeros((self.resolution * self.resolution, nLenslet**2))
                tmp_detector[index_raster_detector.flatten()] = buffer_detect.reshape(nPxDetector * nPxDetector, -1)
                tmp_detector = tmp_detector.reshape(self.resolution, self.resolution, -1)
                output[:, iArray * self.resolution : (iArray + 1) * self.resolution] = tmp_detector.squeeze()
                
            image = output
        
        # --- 2. 图像 binning (如果需要) ---
        if any(np.array(image.shape[:2]) > np.array([self.resolution, self.resolution])):
            image = tools.binning(image, np.array([self.resolution, self.resolution]))
            
        # --- 3. 帧缓冲和噪声处理 ---
        self.frameCount += 1
        
        if self.frameCount < self.exposureTime * self.clockRate and self.frameCount > self.startDelay * self.clockRate:
            self.frameCount -= self.startDelay * self.clockRate
            self.startDelay = 0
            self.frameBuffer += image
            self.frame = None
        else:
            image += self.frameBuffer
            
            if self.photonNoise:
                image = np.random.poisson(image + self.nPhotonBackground) - self.nPhotonBackground
            
            image = self.quantumEfficiency * image
            
            if self.readOutNoise > 0:
                image += np.random.randn(*image.shape) * self.readOutNoise
            
            if self.darkBackground:
                image += self.darkBackground
            
            if self.pixelGains:
                image *= self.pixelGains
            
            if self.frameCount > self.startDelay * self.clockRate:
                self.frameCount = 0
            
            self.frame = image
            self.frameBuffer = 0

    def imagesc(self, **kwargs):
        """
        使用 Matplotlib 显示 Imager 帧图像。
        """
        if self.frame is None:
            print("没有图像可以显示。请先运行 'relay' 方法。")
            return
            
        plt.imshow(self.frame, cmap='pink', origin='lower', **kwargs)
        plt.colorbar(label='Intensity')
        plt.title('Imager Frame')
        plt.xlabel('X Pixels')
        plt.ylabel('Y Pixels')
        plt.show()