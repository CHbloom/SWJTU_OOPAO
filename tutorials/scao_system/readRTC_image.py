import numpy as np
import struct
import os

def ReadRTC(FilePath, SoftORHard, DataFrm, HasHSImg, ShowRange):
    # 初始化返回变量
    Stamp = None
    Slp = None
    Vol = None
    HSImage = None
    
    with open(FilePath, 'rb') as fid:
        # 读取PkgInfo: 7个int32
        PkgInfo = np.frombuffer(fid.read(7 * 4), dtype=np.int32)
        
        # 读取PkgSize: 1个uint64
        PkgSize = np.frombuffer(fid.read(8), dtype=np.uint64)[0]
        
        # 读取PkgTime: 7个int32
        PkgTime = np.frombuffer(fid.read(7 * 4), dtype=np.int32)
        
        # 计算各数据维度
        HSWins = PkgInfo[0]  # 窗口数量
        GSNums = PkgInfo[1]  # 梯度数量
        SlpSize = HSWins * 2 * GSNums
        VolSize = PkgInfo[4]  # 电压数据长度
        
        if DataFrm == 0:
            DataFrm = PkgInfo[6]  # 总帧数
        
        # 设置窗口尺寸
        if SoftORHard == 2:
            HSWinWid = PkgInfo[2]  # 窗口宽度
            HSWinHig = PkgInfo[3]  # 窗口高度
        elif SoftORHard == 1:
            HSWinWid = 24
            HSWinHig = 20
        
        # 图像相关初始化
        if HasHSImg:
            DataSize = HSWinHig * HSWinWid * HSWins
            HSImgHig, HSImgWid = 519, 528
            
            # 加载窗口位置数据
            script_dir = os.path.dirname(os.path.realpath(__file__))
            winlt_path = os.path.join(script_dir, 'HSWinLT_396.txt')
            HSWinLT = np.loadtxt(winlt_path, dtype=np.int32)
            
            # 初始化图像存储空间
            show_count = max(1, len(ShowRange)) if ShowRange is not None else 0
            if show_count > 0:
                HSImage = np.zeros((HSImgHig, HSImgWid, show_count), dtype=np.uint8)
            count = 0  # 图像存储计数器
        else:
            HSImage = None
        
        # 预分配数据存储空间
        Stamp = np.zeros((2, DataFrm), dtype=np.int32)
        Slp = np.zeros((SlpSize, DataFrm), dtype=np.float32)
        Vol = np.zeros((VolSize, DataFrm), dtype=np.float32)
        
        # 逐帧读取数据
        for i in range(DataFrm):
            if HasHSImg:
                # 读取图像原始数据
                raw_data = fid.read(DataSize)
                if len(raw_data) != DataSize:
                    raise EOFError(f"Unexpected end of file at frame {i}")
                
                # 转换为numpy数组并重塑维度
                RawData = np.frombuffer(raw_data, dtype=np.uint8)
                RawData = RawData.reshape(HSWinHig, HSWinWid, HSWins, order='F')
                
                # 如果当前帧需要保存图像
                if ShowRange is not None and i in ShowRange:
                    for j in range(HSWins):
                        x_start = HSWinLT[j, 0]
                        y_start = HSWinLT[j, 1]
                        HSImage[y_start:y_start+HSWinHig, 
                                x_start:x_start+HSWinWid, 
                                count] = RawData[:, :, j]
                    count += 1
            
            # 读取时间戳 (2个int32)
            stamp_data = fid.read(8)
            if len(stamp_data) < 8:
                raise EOFError(f"Unexpected end of file at frame {i} timestamp")
            Stamp[:, i] = np.frombuffer(stamp_data, dtype=np.int32)
            
            # 读取斜率数据
            slp_data = fid.read(SlpSize * 4)
            if len(slp_data) < SlpSize * 4:
                raise EOFError(f"Unexpected end of file at frame {i} slope data")
            Slp[:, i] = np.frombuffer(slp_data, dtype=np.float32)
            
            # 读取电压数据
            vol_data = fid.read(VolSize * 4)
            if len(vol_data) < VolSize * 4:
                raise EOFError(f"Unexpected end of file at frame {i} voltage data")
            Vol[:, i] = np.frombuffer(vol_data, dtype=np.float32)
    
    return Stamp, Slp, Vol, HSImage

# 使用示例
if __name__ == "__main__":
    Stamp, Slp, Vol, HSImage = ReadRTC(
        FilePath="1600PPS开环.dat",
        SoftORHard=2,
        DataFrm=0,
        HasHSImg=True,
        ShowRange=[0, 3, 12]  # Python使用0-based索引
    )