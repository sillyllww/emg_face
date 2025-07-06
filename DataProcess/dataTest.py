import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import cv2  # 添加cv2导入
import tensorflow as tf
from tensorflow import keras
import random

# 读取数据文件
# 读取第一个数据文件
data1 = np.load('孙慧玲_25_30_20250630_151253.npy')
# 读取第二个数据文件
data2 = np.load('李龙_24_30_20250629_185426.npy')
# 合并两个数据文件
data = np.vstack((data1, data2))

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

# 随机抽取一组的中间2000帧
window_size = 2000  # 窗口大小

print("开始处理数据...")
print(f"总组数: {num_groups}")

# 随机选择一组数据
random_group_idx = random.randint(0, len(emg_reshaped) - 1)
group_data = emg_reshaped[random_group_idx]  # 获取随机选择组的数据 (5000, 16)
current_label = labels_reshaped[random_group_idx]  # 获取随机选择组的标签

print(f"随机选择的组索引: {random_group_idx}")

# 计算中间2000帧的起始位置
start_frame = (len(group_data) - window_size) // 2
end_frame = start_frame + window_size

# 提取中间2000帧的数据
window_data = group_data[start_frame:end_frame]  # (2000, 16)
all_data.append(window_data)
all_labels.append(current_label)

# 将所有切割后的数据和标签转换为numpy数组
all_data = np.array(all_data)  # shape: (1, 2000, 16)
all_labels = np.array(all_labels)  # shape: (1,)

print("\n数据处理完成！")
print(f"处理后数据形状: {all_data.shape}")
print(f"处理后标签形状: {all_labels.shape}")

def process_and_predict(all_data):
    """
    处理EMG数据并进行预测
    
    参数:
    all_data: shape (1, 2000, 16) 的numpy数组
    
    返回:
    prediction: 模型的预测结果
    """
    # 创建零均值数据
    all_data_zero_mean = np.zeros_like(all_data)

    # 对每个窗口的每个通道进行去均值处理
    for i in range(len(all_data)):
        for channel in range(all_data.shape[2]):
            channel_mean = np.mean(all_data[i, :, channel])
            all_data_zero_mean[i, :, channel] = all_data[i, :, channel] - channel_mean

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
    for group_idx in range(all_data_zero_mean.shape[0]):
        for channel in range(all_data_zero_mean.shape[2]):
            all_label3_data_filtered[group_idx, :, channel] = signal.filtfilt(b, a, all_data_zero_mean[group_idx, :, channel])

    # 移除异常值
    outlier_mask = (all_label3_data_filtered > 20000) | (all_label3_data_filtered < -20000)
    window_size = 1000

    for group_idx in range(all_label3_data_filtered.shape[0]):
        for channel in range(all_label3_data_filtered.shape[2]):
            for time_idx in range(all_label3_data_filtered.shape[1]):
                if outlier_mask[group_idx, time_idx, channel]:
                    start_idx = max(0, time_idx - window_size)
                    end_idx = min(all_label3_data_filtered.shape[1], time_idx + window_size + 1)
                    local_window = all_label3_data_filtered[group_idx, start_idx:end_idx, channel]
                    valid_mask = ~outlier_mask[group_idx, start_idx:end_idx, channel]
                    
                    if np.sum(valid_mask) > 0:
                        local_mean = np.mean(local_window[valid_mask])
                        all_label3_data_filtered[group_idx, time_idx, channel] = local_mean
                    else:
                        all_label3_data_filtered[group_idx, time_idx, channel] = 0

    # 计算RMS特征
    window_size = 200
    stride = 50
    num_groups = all_label3_data_filtered.shape[0]
    num_points = all_label3_data_filtered.shape[1]
    num_channels = all_label3_data_filtered.shape[2]
    num_windows = (num_points - window_size) // stride + 1

    all_label3_data_rms = np.zeros((num_groups, num_windows, num_channels))
    all_label3_data_wl = np.zeros((num_groups, num_windows, num_channels))
    all_label3_data_mnf = np.zeros((num_groups, num_windows, num_channels))

    for group_idx in range(num_groups):
        for channel in range(num_channels):
            channel_data = all_label3_data_filtered[group_idx, :, channel]
            
            # 计算RMS
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = start_idx + window_size
                window = channel_data[start_idx:end_idx]
                rms_value = np.sqrt(np.mean(window**2))
                all_label3_data_rms[group_idx, i, channel] = rms_value
                
                # 计算WL
                wl_value = np.sum(np.abs(np.diff(window)))
                all_label3_data_wl[group_idx, i, channel] = wl_value
                
                # 计算MNF
                freqs, psd = signal.welch(window, fs=1000, nperseg=min(window_size, len(window)))
                mnf_value = np.sum(freqs * psd) / np.sum(psd)
                all_label3_data_mnf[group_idx, i, channel] = mnf_value

    # 归一化特征
    def normalize_feature(feature_data):
        normalized = np.zeros_like(feature_data, dtype=np.uint8)
        for group_idx in range(feature_data.shape[0]):
            for channel in range(feature_data.shape[2]):
                channel_data = feature_data[group_idx, :, channel]
                valid_data = channel_data[channel_data != 0]
                if len(valid_data) > 0:
                    normalized_data = ((channel_data - np.min(valid_data)) / 
                                    (np.max(valid_data) - np.min(valid_data)) * 255).astype(np.uint8)
                    normalized[group_idx, :, channel] = normalized_data
        return normalized

    all_label3_data_mnf_normalized = normalize_feature(all_label3_data_mnf)
    all_label3_data_wl_normalized = normalize_feature(all_label3_data_wl)
    all_label3_data_rms_normalized = normalize_feature(all_label3_data_rms)

    # 创建RGB图像
    rgb_image = np.zeros((16, 37, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = all_label3_data_mnf_normalized[0, :, :].T
    rgb_image[:, :, 1] = all_label3_data_wl_normalized[0, :, :].T
    rgb_image[:, :, 2] = all_label3_data_rms_normalized[0, :, :].T

    # 调整图像大小
    rgb_image_resized = cv2.resize(rgb_image, (37, 37), interpolation=cv2.INTER_LINEAR)

    # 加载模型
    model_path = 'models/CNNAttention2D_dl2_si0_ex1.0_dw1_oc1.0_0.966.h5'
    model = keras.models.load_model(model_path)

    # 准备输入数据
    input_image = np.expand_dims(rgb_image_resized, axis=0)
    input_image = input_image.astype(np.float32) / 255.0

    # 进行预测
    prediction = model.predict(input_image)
    
    return prediction

# 使用示例:
if __name__ == "__main__":
    # 读取数据文件
    data = np.load('孙慧玲_25_30_20250630_151253.npy')

    # 分离标签和EMG数据
    labels = data[:, 0]
    emg_data = data[:, 1:17]

    # 计算组数
    num_groups = len(emg_data) // 5000

    # 重塑EMG数据
    emg_reshaped = emg_data[:num_groups * 5000].reshape(num_groups, 5000, 16)
    labels_reshaped = labels[:num_groups * 5000:5000]

    # 随机选择一组数据
    random_group_idx = random.randint(0, len(emg_reshaped) - 1)
    group_data = emg_reshaped[random_group_idx]
    current_label = labels_reshaped[random_group_idx]

    # 提取中间2000帧的数据
    window_size = 2000
    start_frame = (len(group_data) - window_size) // 2
    end_frame = start_frame + window_size
    window_data = group_data[start_frame:end_frame]
    
    # 准备数据
    all_data = np.array([window_data])
    
    # 进行预测
    prediction = process_and_predict(all_data)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    print(f"真实标签={current_label}, 预测标签={predicted_class}, 置信度={confidence:.4f}")





