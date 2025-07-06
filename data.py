import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import cv2  # 添加cv2导入

# 读取数据文件
# 读取第一个数据文件
data = np.load('李龙_24_4_20250629_165210.npy')


# 分离标签和EMG数据
labels = data[:, 0]  # 第一列是标签
emg_data = data[:, 1:17]  # 第二到十七列是EMG数据

# 计算组数
num_groups = len(emg_data) // 5000

# 重塑EMG数据为 (组数, 5000, 16)
emg_reshaped = emg_data[:num_groups * 5000].reshape(num_groups, 5000, 16)

# 重塑标签为 (组数,)
labels_reshaped = labels[:num_groups * 5000:5000]

# 创建列表存储所有切割后的数据和对应的标签
all_data = []
all_labels = []

# 对所有组进行滑动窗口切割
window_size = 2000  # 窗口大小
step_size = 1000    # 步长
start_frame = 1000 # 起始帧
end_frame = 4000   # 结束帧

print("开始处理数据...")
print(f"总组数: {num_groups}")

for group_idx in range(len(emg_reshaped)):
    group_data = emg_reshaped[group_idx]  # 获取当前组的数据 (5000, 16)
    current_label = labels_reshaped[group_idx]  # 获取当前组的标签
    
    # 提取1000-4000帧的数据 (3000帧)
    extracted_data = group_data[start_frame:end_frame]  # (3000, 16)
    
    # 计算可以提取的窗口数量
    num_windows = (len(extracted_data) - window_size) // step_size + 1
    
    # 使用滑动窗口切割数据
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_data = extracted_data[start_idx:end_idx]  # (2000, 16)
        all_data.append(window_data)
        all_labels.append(current_label)
    
    if (group_idx + 1) % 10 == 0:
        print(f"已处理 {group_idx + 1}/{num_groups} 组")

# 将所有切割后的数据和标签转换为numpy数组
all_data = np.array(all_data)  # shape: (总窗口数, 2000, 16)
all_labels = np.array(all_labels)  # shape: (总窗口数,)

print("\n数据处理完成！")
print(f"处理后数据形状: {all_data.shape}")
print(f"处理后标签形状: {all_labels.shape}")

# 创建零均值数据
all_data_zero_mean = np.zeros_like(all_data)

# 对每个窗口的每个通道进行去均值处理
for i in range(len(all_data)):
    for channel in range(all_data.shape[2]):
        channel_mean = np.mean(all_data[i, :, channel])
        all_data_zero_mean[i, :, channel] = all_data[i, :, channel] - channel_mean

print("零均值处理完成！")


# 对每一组的每一个通道进行带通滤波
from scipy import signal

# 设置滤波器参数
fs = 1000  # 采样频率
low_freq = 20  # 低频截止
high_freq = 450  # 高频截止

# 计算归一化频率
nyquist_freq = fs / 2
low = low_freq / nyquist_freq
high = high_freq / nyquist_freq

# 设计4阶巴特沃斯带通滤波器
b, a = signal.butter(4, [low, high], btype='band')

# 创建滤波后的数据数组
all_label3_data_filtered = np.zeros_like(all_data_zero_mean)

# 对每一组的每一个通道进行滤波
for group_idx in range(all_data_zero_mean.shape[0]):  # 遍历每一组
    for channel in range(all_data_zero_mean.shape[2]):  # 遍历每一个通道
        # 应用零相位滤波
        all_label3_data_filtered[group_idx, :, channel] = signal.filtfilt(b, a, all_data_zero_mean[group_idx, :, channel])
# 移除异常值（大于20000和小于-20000的值）并替换为局部均值
print("开始移除异常值并替换为局部均值...")

# 创建掩码，标记需要移除的异常值
outlier_mask = (all_label3_data_filtered > 20000) | (all_label3_data_filtered < -20000)

# 统计异常值数量
num_outliers = np.sum(outlier_mask)
total_values = all_label3_data_filtered.size
outlier_percentage = (num_outliers / total_values) * 100

# 对每个异常值，计算其局部均值进行替换
window_size = 1000  # 局部窗口大小

for group_idx in range(all_label3_data_filtered.shape[0]):
    for channel in range(all_label3_data_filtered.shape[2]):
        for time_idx in range(all_label3_data_filtered.shape[1]):
            if outlier_mask[group_idx, time_idx, channel]:
                # 计算局部均值
                start_idx = max(0, time_idx - window_size)
                end_idx = min(all_label3_data_filtered.shape[1], time_idx + window_size + 1)
                
                # 获取局部窗口内的有效值（非异常值）
                local_window = all_label3_data_filtered[group_idx, start_idx:end_idx, channel]
                valid_mask = ~outlier_mask[group_idx, start_idx:end_idx, channel]
                
                if np.sum(valid_mask) > 0:
                    # 使用有效值的均值替换异常值
                    local_mean = np.mean(local_window[valid_mask])
                    all_label3_data_filtered[group_idx, time_idx, channel] = local_mean
                else:
                    # 如果没有有效值，使用0替换
                    all_label3_data_filtered[group_idx, time_idx, channel] = 0

# 计算每一组的标准差


window_size = 200

num_groups = all_label3_data_filtered.shape[0]
num_points = all_label3_data_filtered.shape[1]
num_channels = all_label3_data_filtered.shape[2]

stride = 50
num_windows = (num_points - window_size) // stride + 1

# 新 shape: (组数, 窗口数, 通道数)
all_label3_data_rms = np.zeros((num_groups, num_windows, num_channels))

for group_idx in range(num_groups):  
    for channel in range(num_channels):  
        channel_data =  all_label3_data_filtered[group_idx, :, channel]

        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            window = channel_data[start_idx:end_idx]
            rms_value = np.sqrt(np.mean(window**2))
            all_label3_data_rms[group_idx, i, channel] = rms_value

# 计算WL（波形长度），滑动窗口为200，步长为2
window_size = 200
stride = 50

num_groups = all_label3_data_filtered.shape[0]
num_points = all_label3_data_filtered.shape[1]
num_channels = all_label3_data_filtered.shape[2]

# 计算窗口数量
num_windows = (num_points - window_size) // stride + 1

# 初始化WL特征数组
all_label3_data_wl = np.zeros((num_groups, num_windows, num_channels))

for group_idx in range(num_groups):
    for channel in range(num_channels):
        channel_data = all_label3_data_filtered[group_idx, :, channel]
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            window = channel_data[start_idx:end_idx]
            
            # 计算WL（波形长度）- 相邻点之间的绝对差值之和
            wl_value = np.sum(np.abs(np.diff(window)))
            all_label3_data_wl[group_idx, i, channel] = wl_value

# 计算MNF特征
window_size = 200
stride = 50

num_groups = all_label3_data_filtered.shape[0]
num_points = all_label3_data_filtered.shape[1]
num_channels = all_label3_data_filtered.shape[2]

# 计算窗口数量
num_windows = (num_points - window_size) // stride + 1

# 初始化MNF特征数组
all_label3_data_mnf = np.zeros((num_groups, num_windows, num_channels))

for group_idx in range(num_groups):
    for channel in range(num_channels):
        channel_data = all_label3_data_filtered[group_idx, :, channel]
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            window = channel_data[start_idx:end_idx]
            
            # 计算MNF (Mean Frequency)
            # 1. 计算功率谱
            freqs, psd = signal.welch(window, fs=1000, nperseg=min(window_size, len(window)))
            # 2. 计算MNF (频率的加权平均值，权重为功率谱密度)
            mnf_value = np.sum(freqs * psd) / np.sum(psd)
            all_label3_data_mnf[group_idx, i, channel] = mnf_value

# 归一化MNF特征到0-255并转换为uint8
all_label3_data_mnf_normalized = np.zeros_like(all_label3_data_mnf, dtype=np.uint8)
for group_idx in range(all_label3_data_mnf.shape[0]):
    for channel in range(all_label3_data_mnf.shape[2]):
        channel_data = all_label3_data_mnf[group_idx, :, channel]
        # 去除零值（无效数据）
        valid_data = channel_data[channel_data != 0]
        if len(valid_data) > 0:
            # 归一化到0-255
            normalized_data = ((channel_data - np.min(valid_data)) / (np.max(valid_data) - np.min(valid_data)) * 255).astype(np.uint8)
            all_label3_data_mnf_normalized[group_idx, :, channel] = normalized_data

# 归一化WL特征到0-255并转换为uint8
all_label3_data_wl_normalized = np.zeros_like(all_label3_data_wl, dtype=np.uint8)
for group_idx in range(all_label3_data_wl.shape[0]):
    for channel in range(all_label3_data_wl.shape[2]):
        channel_data = all_label3_data_wl[group_idx, :, channel]
        # 去除零值（无效数据）
        valid_data = channel_data[channel_data != 0]
        if len(valid_data) > 0:
            # 归一化到0-255
            normalized_data = ((channel_data - np.min(valid_data)) / (np.max(valid_data) - np.min(valid_data)) * 255).astype(np.uint8)
            all_label3_data_wl_normalized[group_idx, :, channel] = normalized_data

# 归一化RMS特征到0-255并转换为uint8
all_label3_data_rms_normalized = np.zeros_like(all_label3_data_rms, dtype=np.uint8)
for group_idx in range(all_label3_data_rms.shape[0]):
    for channel in range(all_label3_data_rms.shape[2]):
        channel_data = all_label3_data_rms[group_idx, :, channel]
        # 去除零值（无效数据）
        valid_data = channel_data[channel_data != 0]
        if len(valid_data) > 0:
            # 归一化到0-255
            normalized_data = ((channel_data - np.min(valid_data)) / (np.max(valid_data) - np.min(valid_data)) * 255).astype(np.uint8)
            all_label3_data_rms_normalized[group_idx, :, channel] = normalized_data

# 在计算完特征后，打印形状信息
print("\n数据形状信息:")
print(f"标签形状 (labels_reshaped): {labels_reshaped.shape}")
print(f"RMS特征形状 (all_label3_data_rms_normalized): {all_label3_data_rms_normalized.shape}")
print(f"WL特征形状 (all_label3_data_wl_normalized): {all_label3_data_wl_normalized.shape}")
print(f"MNF特征形状 (all_label3_data_mnf_normalized): {all_label3_data_mnf_normalized.shape}")

# 创建images文件夹结构
if not os.path.exists('images'):
    os.makedirs('images')

# 获取所有唯一的标签值
unique_labels = np.unique(labels_reshaped)
print(f"\n所有标签值: {unique_labels}")

# 为每个标签创建对应的文件夹
for label in unique_labels:
    label_folder = os.path.join('images', f'label_{int(label)}')
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
        print(f"创建文件夹: {label_folder}")

# 为每组数据生成RGB图像并保存
print("\n开始保存图像...")
for group_idx in range(len(all_labels)):
    current_label = int(all_labels[group_idx])
    
    # 创建16*47*3的RGB图像数组
    rgb_image = np.zeros((16, 37, 3), dtype=np.uint8)
    
    # 将MNF特征作为红色通道 (R)
    rgb_image[:, :, 0] = all_label3_data_mnf_normalized[group_idx, :, :].T
    
    # 将WL特征作为绿色通道 (G) 
    rgb_image[:, :, 1] = all_label3_data_wl_normalized[group_idx, :, :].T
    
    # 将RMS特征作为蓝色通道 (B)
    rgb_image[:, :, 2] = all_label3_data_rms_normalized[group_idx, :, :].T
    # 将图像调整为64x64大小
    rgb_image_resized = cv2.resize(rgb_image, (37, 37), interpolation=cv2.INTER_LINEAR)
    # 保存图像到对应标签文件夹
    image_filename = f'group_{group_idx:03d}.png'
    image_path = os.path.join('images', f'label_{current_label}', image_filename)
    plt.imsave(image_path, rgb_image_resized)
    
    if (group_idx + 1) % 10 == 0:
        print(f"已保存 {group_idx + 1}/{len(all_labels)} 张图像")

print(f"\n所有图像保存完成！共保存了 {len(all_labels)} 张图像")
print("图像保存在各自的标签文件夹中: images/label_X/group_XXX.png")

