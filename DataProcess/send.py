import asyncio
import websockets
import json
import numpy as np
import pandas as pd
import time
import struct
import socket
import threading
from queue import Queue
from scipy import signal
import ctypes
import os
from datetime import datetime
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import cv2  # 添加cv2导入
import tensorflow as tf
from tensorflow import keras
import random

# 全局变量
all_data = []
current_label = 0
is_recording = False
recording_start_time = 0
RECORDING_DURATION = 7
device_data_queue = Queue()
SAMPLE_INTERVAL = 0.001
raw_10_frames = []  # ⬅️ 保存前10帧的完整100字节

# 用户信息全局变量
current_user_name = ""
current_user_age = ""
current_user_status = ""

# 全局变量用于实时数据传输
connected_clients = set()
is_realtime_mode = False
emg_data_queue = Queue()

# EMG数据的缓冲区
EMG_BUFFER_SIZE = 400  # 原始缓冲区大小保持400帧
emg_data_buffer = []  # 将存储400帧EMG数据，每帧包含32通道
emg_buffer_lock = threading.Lock()  # 防止多线程同时访问缓冲区
buffer_frame_count = 0  # 计数器，记录当前缓冲区中的帧数

# 新增预测用的EMG数据缓冲区
PREDICTION_BUFFER_SIZE = 2000  # 新的预测缓冲区大小为2000帧
prediction_data_buffer = []  # 存储2000帧EMG数据用于预测
prediction_buffer_lock = threading.Lock()  # 预测缓冲区的线程锁
prediction_frame_count = 0  # 预测缓冲区的帧计数器

# 指定要保留的通道（通道索引从0开始，所以减1）
KEEP_CHANNELS = [9-1, 7-1, 10-1, 3-1, 8-1, 25-1, 14-1, 2-1, 6-1, 17-1, 1-1, 26-1, 15-1, 18-1, 11-1, 5-1]
# 保留通道对应的原始名称
CHANNEL_NAMES = [f'channel_{i+1}' for i in KEEP_CHANNELS]

# 加载模型（全局）
model_path = 'models/CNNAttention2D_dl2_si0_ex1.0_dw1_oc1.0_0.966.h5'
cnn_model = keras.models.load_model(model_path)
print("已加载CNN模型")

def apply_highpass_filter(data, cutoff_freq=20, fs=1000):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data, axis=0)

def get_volt(num):
    return ctypes.c_int32(num << 8).value >> 8

def process_emg_data(data_bytes):
    if len(data_bytes) < 100:
        print(f"数据长度不足100字节（实际为 {len(data_bytes)}），跳过处理")
        return []

    # 首先提取所有32个通道
    all_channels = []
    for i in range(32):
        start = 4 + i * 3
        byte1 = data_bytes[start]
        byte2 = data_bytes[start + 1]
        byte3 = data_bytes[start + 2]
        raw_value = (byte1 << 16) | (byte2 << 8) | byte3
        signed_value = get_volt(raw_value)
        all_channels.append(signed_value)
    
    # 只保留指定的16个通道
    kept_channels = [all_channels[i] for i in KEEP_CHANNELS]
    return kept_channels

def save_data_to_file():
    global all_data, raw_10_frames, current_user_name, current_user_age, current_user_status

    if not all_data:
        print("没有数据可保存")
        return

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建文件名
    if current_user_name and current_user_age and current_user_status:
        filename = f"{current_user_name}_{current_user_age}_{current_user_status}_{timestamp}"
    else:
        filename = f"unknown_user_{timestamp}"
    
    # 创建目录路径
    if current_user_name:
        dir_path = os.path.join("emgs", current_user_name)
    else:
        dir_path = os.path.join("emgs", "unknown")
    
    # 确保目录存在
    os.makedirs(dir_path, exist_ok=True)
    
    # 构建完整的文件路径
    excel_path = os.path.join(dir_path, f"{filename}.xlsx")
    npy_path = os.path.join(dir_path, f"{filename}.npy")

    # 保存原始数据
    np_data = np.array(all_data)
    np.save(npy_path, np_data)
    print(f"原始数据已保存为 {npy_path}")

    # # 创建列名：label + 16个通道名称
    # columns = ['label'] + [f'channel_{i+1}' for i in KEEP_CHANNELS]
    # pd.DataFrame(np_data, columns=columns).to_excel(excel_path, index=False)
    # print(f"原始数据已保存为 {excel_path}")


async def send_emg_buffer(buffer_data):
    """发送EMG缓冲区数据（仅经过去均值处理的数据）"""
    if not connected_clients:
        return
    
    if buffer_data:
        # 准备完整的EMG缓冲区数据包
        emg_data = {
            "type": "emg_buffer",
            "buffer": buffer_data,  # 发送处理后的数据
            "buffer_size": len(buffer_data),
            "timestamp": time.time(),
            "is_normalized": False,  # 标记数据未归一化
            "is_zero_centered": True,  # 标记数据已去均值，以零为中心
            "is_filtered": False  # 标记数据未进行高通滤波
        }
        
        # 发送给所有连接的客户端
        websockets_tasks = []
        for client in connected_clients:
            try:
                websockets_tasks.append(client.send(json.dumps(emg_data)))
            except Exception as e:
                print(f"发送EMG数据错误: {e}")
        
        # 异步执行所有发送任务
        if websockets_tasks:
            await asyncio.gather(*websockets_tasks, return_exceptions=True)
            print(f"已发送去均值处理后的缓冲区数据，包含{len(buffer_data)}帧")

# 保留旧的发送函数，用于兼容性
async def send_emg_data(channels_data_batch):
    if not connected_clients:
        return
    
    # channels_data_batch是一个列表，包含多个EMG数据点(每个数据点是32通道)
    if channels_data_batch:
        # 准备包含多个EMG数据点的包
        emg_data = {
            "type": "emg_data_batch",
            "data_points": channels_data_batch,  # 发送多个数据点
            "batch_size": len(channels_data_batch)
        }
        
        # 发送给所有连接的客户端
        websockets_tasks = []
        for client in connected_clients:
            try:
                websockets_tasks.append(client.send(json.dumps(emg_data)))
            except Exception as e:
                print(f"发送EMG数据错误: {e}")
        
        # 异步执行所有发送任务
        if websockets_tasks:
            await asyncio.gather(*websockets_tasks, return_exceptions=True)
            print(f"已发送{len(channels_data_batch)}个数据点的EMG数据包")

def prepare_emg_buffer():
    """准备EMG数据缓冲区，只执行去均值处理，不进行高通滤波"""
    global emg_data_buffer
    
    # 如果缓冲区为空，返回空数组
    if not emg_data_buffer:
        return []
        
    # 复制数据以避免并发问题
    with emg_buffer_lock:
        data_copy = emg_data_buffer.copy()
    
    # 使用NumPy进行高效处理
    try:
        data_np = np.array(data_copy)
        
        # 如果数据为空或只有一帧，无法执行处理
        if data_np.size == 0 or len(data_np) <= 1:
            return data_copy
            
        # 创建结果数组
        result = np.zeros_like(data_np, dtype=float)
        
        # 对每个通道分别执行处理（已经只有16个通道）
        for channel in range(data_np.shape[1]):
            # 提取该通道的所有数据
            channel_data = data_np[:, channel]
            
            # 去均值处理
            mean_value = np.mean(channel_data)
            centered_data = channel_data - mean_value
            
            # 保存去均值后的数据
            result[:, channel] = centered_data
            
        # 转换回Python列表
        return result.tolist()
        
    except Exception as e:
        print(f"数据处理时出错: {e}")
        return data_copy  # 如果出错，返回原始数据

async def process_emg_queue():
    """处理EMG数据队列并维护两个缓冲区：
    1. 400帧缓冲区用于实时显示
    2. 2000帧缓冲区用于预测
    """
    global emg_data_buffer, buffer_frame_count
    global prediction_data_buffer, prediction_frame_count
    
    while True:
        # 1. 处理队列中的数据，添加到两个缓冲区
        try:
            while not emg_data_queue.empty():
                new_data = emg_data_queue.get_nowait()
                
                # 添加到实时显示缓冲区
                with emg_buffer_lock:
                    if len(emg_data_buffer) >= EMG_BUFFER_SIZE:
                        emg_data_buffer.pop(0)
                    emg_data_buffer.append(new_data)
                    buffer_frame_count += 1
                
                # 添加到预测缓冲区
                with prediction_buffer_lock:
                    if len(prediction_data_buffer) >= PREDICTION_BUFFER_SIZE:
                        prediction_data_buffer.pop(0)
                    prediction_data_buffer.append(new_data)
                    prediction_frame_count += 1
        except Exception as e:
            pass
        
        # 2. 当实时显示缓冲区积累了400帧数据时发送
        if buffer_frame_count >= EMG_BUFFER_SIZE:
            try:
                # 获取EMG缓冲区
                buffer_data = prepare_emg_buffer()
                
                # 准备发送的数据包
                emg_package = {
                    "type": "emg_buffer",
                    "buffer": buffer_data,  # 发送处理后的数据
                    "buffer_size": len(buffer_data),
                    "timestamp": time.time(),
                    "is_normalized": False,  # 标记数据未归一化
                    "is_zero_centered": True,  # 标记数据已去均值，以零为中心
                    "is_filtered": False  # 标记数据未进行高通滤波
                }

                # 直接调用预测函数
                if len(prediction_data_buffer) >= 2000:
                    # 准备预测数据
                    prediction_data = np.array(prediction_data_buffer[-2000:])
                    prediction_data = prediction_data.reshape(1, 2000, 16)
                    
                    # 调用预测函数
                    prediction_result = process_and_predict(prediction_data)
                    predicted_class = np.argmax(prediction_result[0])
                    confidence = np.max(prediction_result[0])
                    
                    print(f"预测结果: 类别={predicted_class}, 置信度={confidence:.4f}")
                    
                    # 将预测结果添加到数据包中
                    emg_package["prediction"] = {
                        'class': int(predicted_class),
                        'confidence': float(confidence)
                    }
                
                if buffer_data:
                    # 发送数据包
                    websockets_tasks = []
                    for client in connected_clients:
                        try:
                            websockets_tasks.append(client.send(json.dumps(emg_package)))
                        except Exception as e:
                            print(f"发送EMG数据错误: {e}")
                    
                    # 异步执行所有发送任务
                    if websockets_tasks:
                        await asyncio.gather(*websockets_tasks, return_exceptions=True)
                        print(f"已发送数据包，包含{len(buffer_data)}帧EMG数据")
                    
                    # 重置计数器
                    buffer_frame_count = 0
            except Exception as e:
                print(f"处理EMG数据出错: {e}")
        


def device_data_handler():
    global is_recording, recording_start_time, current_label, all_data, raw_10_frames, is_realtime_mode

    device_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    device_server.bind(('0.0.0.0', 9000))
    device_server.listen(1)
    print("设备数据服务器已启动，监听端口: 9000")

    sample_count = 0
    task_start_time = 0
    packet_counter = 0
    process_interval = 3  # 调整为每3个包处理一次

    while True:
        try:
            device_conn, device_addr = device_server.accept()
            print(f"设备已连接: {device_addr}")

            while True:
                data = device_conn.recv(100)
                if not data or len(data) < 100:
                    continue

                # 帧头完整性校验
                if not (data[0] == 0xAA and data[1] == 0xAA):
                    continue

                # ⬅️ 保存前10帧原始完整100字节
                if len(raw_10_frames) < 10:
                    raw_10_frames.append(list(data))

                packet_counter += 1
                if packet_counter % process_interval != 0:
                    continue

                device_data_queue.put(data)

                # 实时模式下，处理并放入队列
                if is_realtime_mode:
                    channels = process_emg_data(data)
                    if len(channels) == 16:
                        # 将数据放入队列以供处理线程使用
                        emg_data_queue.put(channels)
                
                if is_recording:
                    current_time = time.time()
                    if sample_count == 0:
                        task_start_time = current_time

                    if current_time - recording_start_time <= RECORDING_DURATION:
                        channels = process_emg_data(data)
                        if len(channels) == 16 and sample_count < 5000:
                            row_data = [current_label] + channels
                            all_data.append(row_data)
                            sample_count += 1
                            if sample_count % 1000 == 0:
                                duration = current_time - task_start_time
                                current_rate = sample_count / duration if duration > 0 else 0
                                print(f"当前采样率: {current_rate:.2f} 包/秒，已采集 {sample_count} 个样本")
                        elif len(channels) != 16:
                            print(f"警告：channels 长度为 {len(channels)}，未添加到all_data")
                        else:
                            is_recording = False
                            task_end_time = current_time
                            duration = task_end_time - task_start_time
                            sampling_rate = sample_count / duration if duration > 0 else 0
                            print(f"已完成动作{current_label}的记录")
                            print(f"动作{current_label}采集到有效数据包数: {sample_count}")
                            print(f"动作{current_label}采样率: {sampling_rate:.2f} 包/秒")
                            sample_count = 0
                            task_start_time = 0
                            if current_label == 9:
                                save_data_to_file()
                    else:
                        is_recording = False
                        task_end_time = current_time
                        duration = task_end_time - task_start_time
                        sampling_rate = sample_count / duration if duration > 0 else 0
                        print(f"已完成动作{current_label}的记录")
                        print(f"动作{current_label}采集到有效数据包数: {sample_count}")
                        print(f"动作{current_label}采样率: {sampling_rate:.2f} 包/秒")
                        sample_count = 0
                        task_start_time = 0
                        if current_label == 9:
                            save_data_to_file()
        except Exception as e:
            print(f"处理设备数据时出错: {e}")
            try:
                device_conn.close()
            except:
                pass

async def handle_web_client(websocket):
    global is_recording, recording_start_time, current_label, connected_clients, is_realtime_mode
    global current_user_name, current_user_age, current_user_status

    # 添加新连接的客户端
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                cmd = json.loads(message)
                if cmd.get("command") == "start_recording":
                    current_label = cmd.get("action_label", 0)
                    # 获取用户信息
                    current_user_name = cmd.get("user_name", "")
                    current_user_age = cmd.get("user_age", "")
                    current_user_status = cmd.get("status", "")
                    
                    is_recording = True
                    recording_start_time = time.time()
                    print(f"开始记录动作{current_label}的数据，用户: {current_user_name}, 年龄: {current_user_age}, 状态: {current_user_status}")
                    await websocket.send(json.dumps({
                        "type": "recording_started",
                        "action": current_label
                    }))
                elif cmd.get("command") == "start_realtime_emg":
                    is_realtime_mode = True
                    print("开始实时EMG数据传输模式")
                    await websocket.send(json.dumps({
                        "type": "realtime_started",
                        "status": "success"
                    }))
                elif cmd.get("command") == "stop_realtime_emg":
                    is_realtime_mode = False
                    print("停止实时EMG数据传输模式")
                    await websocket.send(json.dumps({
                        "type": "realtime_stopped",
                        "status": "success"
                    }))
            except json.JSONDecodeError:
                print(f"无效的JSON消息: {message}")
    except Exception as e:
        print(f"处理Web客户端数据时出错: {e}")
    finally:
        # 客户端断开连接时移除
        connected_clients.remove(websocket)


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

    # 准备输入数据
    input_image = np.expand_dims(rgb_image_resized, axis=0)
    input_image = input_image.astype(np.float32) / 255.0

    # 进行预测
    prediction = cnn_model.predict(input_image)
    
    return prediction


async def main():
    # 启动设备数据处理线程
    device_thread = threading.Thread(target=device_data_handler, daemon=True)
    device_thread.start()
    
    # 创建EMG数据处理任务
    emg_queue_task = asyncio.create_task(process_emg_queue())

    host = "0.0.0.0"
    port = 8765

    print(f"WebSocket服务器正在启动，地址: ws://{host}:{port}")
    print(f"采样间隔设置为 {SAMPLE_INTERVAL*1000:.1f} 毫秒")
    print(f"采样率设置为 2000Hz (每3个包处理一次)")
    print(f"数据存储后将应用去均值处理")
    print(f"EMG缓冲区大小设置为 {EMG_BUFFER_SIZE} 帧，每积累 {EMG_BUFFER_SIZE} 帧发送一次数据")
    print(f"保留的EMG通道: {[ch+1 for ch in KEEP_CHANNELS]} (共{len(KEEP_CHANNELS)}个通道)")

    async with websockets.serve(handle_web_client, host, port):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("服务器已停止")
        if all_data:
            save_data_to_file()

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志

import asyncio
import websockets
import json
import numpy as np
import pandas as pd
import time
import struct
import socket
import threading
from queue import Queue
from scipy import signal
import ctypes
import os
from datetime import datetime
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import cv2  # 添加cv2导入
# 尝试导入TensorFlow和Keras，如果失败则提供替代方案
try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"成功导入TensorFlow版本: {tf.__version__}")
    PREDICTION_ENABLED = True
except ImportError as e:
    print(f"无法导入TensorFlow: {e}")
    print("预测功能将被禁用，但其他功能正常工作")
    tf = None
    keras = None
    PREDICTION_ENABLED = False

import random

# 全局变量
all_data = []
current_label = 0
is_recording = False
recording_start_time = 0
RECORDING_DURATION = 7
device_data_queue = Queue()
SAMPLE_INTERVAL = 0.001
raw_10_frames = []  # ⬅️ 保存前10帧的完整100字节

# 用户信息全局变量
current_user_name = ""
current_user_age = ""
current_user_status = ""

# 全局变量用于实时数据传输
connected_clients = set()
is_realtime_mode = False
emg_data_queue = Queue()

# EMG数据的缓冲区
EMG_BUFFER_SIZE = 400  # 原始缓冲区大小保持400帧
emg_data_buffer = []  # 将存储400帧EMG数据，每帧包含32通道
emg_buffer_lock = threading.Lock()  # 防止多线程同时访问缓冲区
buffer_frame_count = 0  # 计数器，记录当前缓冲区中的帧数

# 新增预测用的EMG数据缓冲区
PREDICTION_BUFFER_SIZE = 2000  # 新的预测缓冲区大小为2000帧
prediction_data_buffer = []  # 存储2000帧EMG数据用于预测
prediction_buffer_lock = threading.Lock()  # 预测缓冲区的线程锁
prediction_frame_count = 0  # 预测缓冲区的帧计数器

# 指定要保留的通道（通道索引从0开始，所以减1）
KEEP_CHANNELS = [9-1, 7-1, 10-1, 3-1, 8-1, 25-1, 14-1, 2-1, 6-1, 17-1, 1-1, 26-1, 15-1, 18-1, 11-1, 5-1]
# 保留通道对应的原始名称
CHANNEL_NAMES = [f'channel_{i+1}' for i in KEEP_CHANNELS]

# 加载模型（全局）
cnn_model = None
if PREDICTION_ENABLED and keras is not None:
    try:
        # 使用正斜杠路径，更兼容跨平台
        model_path = 'DataProcess/CNNAttention2D_dl2_si0_ex1.0_dw1_oc1.0_0.966.h5'
        cnn_model = keras.models.load_model(model_path)
        print("已加载CNN模型")
    except Exception as e:
        print(f"加载CNN模型失败: {e}")
        print("预测功能将被禁用")
        PREDICTION_ENABLED = False
        cnn_model = None
else:
    print("跳过模型加载，预测功能禁用")

def apply_highpass_filter(data, cutoff_freq=20, fs=1000):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data, axis=0)

def get_volt(num):
    return ctypes.c_int32(num << 8).value >> 8

def process_emg_data(data_bytes):
    if len(data_bytes) < 100:
        print(f"数据长度不足100字节（实际为 {len(data_bytes)}），跳过处理")
        return []

    # 首先提取所有32个通道
    all_channels = []
    for i in range(32):
        start = 4 + i * 3
        byte1 = data_bytes[start]
        byte2 = data_bytes[start + 1]
        byte3 = data_bytes[start + 2]
        raw_value = (byte1 << 16) | (byte2 << 8) | byte3
        signed_value = get_volt(raw_value)
        all_channels.append(signed_value)
    
    # 只保留指定的16个通道
    kept_channels = [all_channels[i] for i in KEEP_CHANNELS]
    return kept_channels

def save_data_to_file():
    global all_data, raw_10_frames, current_user_name, current_user_age, current_user_status

    if not all_data:
        print("没有数据可保存")
        return

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建文件名
    if current_user_name and current_user_age and current_user_status:
        filename = f"{current_user_name}_{current_user_age}_{current_user_status}_{timestamp}"
    else:
        filename = f"unknown_user_{timestamp}"
    
    # 创建目录路径
    if current_user_name:
        dir_path = os.path.join("emgs", current_user_name)
    else:
        dir_path = os.path.join("emgs", "unknown")
    
    # 确保目录存在
    os.makedirs(dir_path, exist_ok=True)
    
    # 构建完整的文件路径
    excel_path = os.path.join(dir_path, f"{filename}.xlsx")
    npy_path = os.path.join(dir_path, f"{filename}.npy")

    # 保存原始数据
    np_data = np.array(all_data)
    np.save(npy_path, np_data)
    print(f"原始数据已保存为 {npy_path}")

    # # 创建列名：label + 16个通道名称
    # columns = ['label'] + [f'channel_{i+1}' for i in KEEP_CHANNELS]
    # pd.DataFrame(np_data, columns=columns).to_excel(excel_path, index=False)
    # print(f"原始数据已保存为 {excel_path}")


async def send_emg_buffer(buffer_data):
    """发送EMG缓冲区数据（仅经过去均值处理的数据）"""
    if not connected_clients:
        return
    
    if buffer_data:
        # 准备完整的EMG缓冲区数据包
        emg_data = {
            "type": "emg_buffer",
            "buffer": buffer_data,  # 发送处理后的数据
            "buffer_size": len(buffer_data),
            "timestamp": time.time(),
            "is_normalized": False,  # 标记数据未归一化
            "is_zero_centered": True,  # 标记数据已去均值，以零为中心
            "is_filtered": False  # 标记数据未进行高通滤波
        }
        
        # 发送给所有连接的客户端
        websockets_tasks = []
        for client in connected_clients:
            try:
                websockets_tasks.append(client.send(json.dumps(emg_data)))
            except Exception as e:
                print(f"发送EMG数据错误: {e}")
        
        # 异步执行所有发送任务
        if websockets_tasks:
            await asyncio.gather(*websockets_tasks, return_exceptions=True)
            print(f"已发送去均值处理后的缓冲区数据，包含{len(buffer_data)}帧")

# 保留旧的发送函数，用于兼容性
async def send_emg_data(channels_data_batch):
    if not connected_clients:
        return
    
    # channels_data_batch是一个列表，包含多个EMG数据点(每个数据点是32通道)
    if channels_data_batch:
        # 准备包含多个EMG数据点的包
        emg_data = {
            "type": "emg_data_batch",
            "data_points": channels_data_batch,  # 发送多个数据点
            "batch_size": len(channels_data_batch)
        }
        
        # 发送给所有连接的客户端
        websockets_tasks = []
        for client in connected_clients:
            try:
                websockets_tasks.append(client.send(json.dumps(emg_data)))
            except Exception as e:
                print(f"发送EMG数据错误: {e}")
        
        # 异步执行所有发送任务
        if websockets_tasks:
            await asyncio.gather(*websockets_tasks, return_exceptions=True)
            print(f"已发送{len(channels_data_batch)}个数据点的EMG数据包")

def prepare_emg_buffer():
    """准备EMG数据缓冲区，只执行去均值处理，不进行高通滤波"""
    global emg_data_buffer
    
    # 如果缓冲区为空，返回空数组
    if not emg_data_buffer:
        return []
        
    # 复制数据以避免并发问题
    with emg_buffer_lock:
        data_copy = emg_data_buffer.copy()
    
    # 使用NumPy进行高效处理
    try:
        data_np = np.array(data_copy)
        
        # 如果数据为空或只有一帧，无法执行处理
        if data_np.size == 0 or len(data_np) <= 1:
            return data_copy
            
        # 创建结果数组
        result = np.zeros_like(data_np, dtype=float)
        
        # 对每个通道分别执行处理（已经只有16个通道）
        for channel in range(data_np.shape[1]):
            # 提取该通道的所有数据
            channel_data = data_np[:, channel]
            
            # 去均值处理
            mean_value = np.mean(channel_data)
            centered_data = channel_data - mean_value
            
            # 保存去均值后的数据
            result[:, channel] = centered_data
            
        # 转换回Python列表
        return result.tolist()
        
    except Exception as e:
        print(f"数据处理时出错: {e}")
        return data_copy  # 如果出错，返回原始数据

async def process_emg_queue():
    """处理EMG数据队列并维护两个缓冲区：
    1. 400帧缓冲区用于实时显示
    2. 2000帧缓冲区用于预测
    """
    global emg_data_buffer, buffer_frame_count
    global prediction_data_buffer, prediction_frame_count
    
    while True:
        # 1. 处理队列中的数据，添加到两个缓冲区
        try:
            while not emg_data_queue.empty():
                new_data = emg_data_queue.get_nowait()
                
                # 添加到实时显示缓冲区
                with emg_buffer_lock:
                    if len(emg_data_buffer) >= EMG_BUFFER_SIZE:
                        emg_data_buffer.pop(0)
                    emg_data_buffer.append(new_data)
                    buffer_frame_count += 1
                
                # 添加到预测缓冲区
                with prediction_buffer_lock:
                    if len(prediction_data_buffer) >= PREDICTION_BUFFER_SIZE:
                        prediction_data_buffer.pop(0)
                    prediction_data_buffer.append(new_data)
                    prediction_frame_count += 1
        except Exception as e:
            pass
        
        # 2. 当实时显示缓冲区积累了400帧数据时发送
        if buffer_frame_count >= EMG_BUFFER_SIZE:
            try:
                # 获取EMG缓冲区
                buffer_data = prepare_emg_buffer()
                
                # 准备发送的数据包
                emg_package = {
                    "type": "emg_buffer",
                    "buffer": buffer_data,  # 发送处理后的数据
                    "buffer_size": len(buffer_data),
                    "timestamp": time.time(),
                    "is_normalized": False,  # 标记数据未归一化
                    "is_zero_centered": True,  # 标记数据已去均值，以零为中心
                    "is_filtered": False  # 标记数据未进行高通滤波
                }

                # 直接调用预测函数（仅当预测功能启用时）
                if PREDICTION_ENABLED and cnn_model is not None and len(prediction_data_buffer) >= 2000:
                    try:
                        # 准备预测数据
                        prediction_data = np.array(prediction_data_buffer[-2000:])
                        prediction_data = prediction_data.reshape(1, 2000, 16)
                        
                        # 调用预测函数
                        prediction_result = process_and_predict(prediction_data)
                        predicted_class = np.argmax(prediction_result[0])
                        confidence = np.max(prediction_result[0])
                        
                        print(f"预测结果: 类别={predicted_class}, 置信度={confidence:.4f}")
                        
                        # 将预测结果添加到数据包中
                        emg_package["prediction"] = {
                            'class': int(predicted_class),
                            'confidence': float(confidence)
                        }
                    except Exception as pred_error:
                        print(f"预测过程中出错: {pred_error}")
                        # 不进行预测，继续发送数据包
                
                if buffer_data:
                    # 发送数据包
                    websockets_tasks = []
                    for client in connected_clients:
                        try:
                            websockets_tasks.append(client.send(json.dumps(emg_package)))
                        except Exception as e:
                            print(f"发送EMG数据错误: {e}")
                    
                    # 异步执行所有发送任务
                    if websockets_tasks:
                        await asyncio.gather(*websockets_tasks, return_exceptions=True)
                        print(f"已发送数据包，包含{len(buffer_data)}帧EMG数据")
                    
                    # 重置计数器
                    buffer_frame_count = 0
            except Exception as e:
                print(f"处理EMG数据出错: {e}")
        


def device_data_handler():
    global is_recording, recording_start_time, current_label, all_data, raw_10_frames, is_realtime_mode

    device_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    device_server.bind(('0.0.0.0', 9000))
    device_server.listen(1)
    print("设备数据服务器已启动，监听端口: 9000")

    sample_count = 0
    task_start_time = 0
    packet_counter = 0
    process_interval = 3  # 调整为每3个包处理一次

    while True:
        try:
            device_conn, device_addr = device_server.accept()
            print(f"设备已连接: {device_addr}")

            while True:
                data = device_conn.recv(100)
                if not data or len(data) < 100:
                    continue

                # 帧头完整性校验
                if not (data[0] == 0xAA and data[1] == 0xAA):
                    continue

                # ⬅️ 保存前10帧原始完整100字节
                if len(raw_10_frames) < 10:
                    raw_10_frames.append(list(data))

                packet_counter += 1
                if packet_counter % process_interval != 0:
                    continue

                device_data_queue.put(data)

                # 实时模式下，处理并放入队列
                if is_realtime_mode:
                    channels = process_emg_data(data)
                    if len(channels) == 16:
                        # 将数据放入队列以供处理线程使用
                        emg_data_queue.put(channels)
                
                if is_recording:
                    current_time = time.time()
                    if sample_count == 0:
                        task_start_time = current_time

                    if current_time - recording_start_time <= RECORDING_DURATION:
                        channels = process_emg_data(data)
                        if len(channels) == 16 and sample_count < 5000:
                            row_data = [current_label] + channels
                            all_data.append(row_data)
                            sample_count += 1
                            if sample_count % 1000 == 0:
                                duration = current_time - task_start_time
                                current_rate = sample_count / duration if duration > 0 else 0
                                print(f"当前采样率: {current_rate:.2f} 包/秒，已采集 {sample_count} 个样本")
                        elif len(channels) != 16:
                            print(f"警告：channels 长度为 {len(channels)}，未添加到all_data")
                        else:
                            is_recording = False
                            task_end_time = current_time
                            duration = task_end_time - task_start_time
                            sampling_rate = sample_count / duration if duration > 0 else 0
                            print(f"已完成动作{current_label}的记录")
                            print(f"动作{current_label}采集到有效数据包数: {sample_count}")
                            print(f"动作{current_label}采样率: {sampling_rate:.2f} 包/秒")
                            sample_count = 0
                            task_start_time = 0
                            if current_label == 9:
                                save_data_to_file()
                    else:
                        is_recording = False
                        task_end_time = current_time
                        duration = task_end_time - task_start_time
                        sampling_rate = sample_count / duration if duration > 0 else 0
                        print(f"已完成动作{current_label}的记录")
                        print(f"动作{current_label}采集到有效数据包数: {sample_count}")
                        print(f"动作{current_label}采样率: {sampling_rate:.2f} 包/秒")
                        sample_count = 0
                        task_start_time = 0
                        if current_label == 9:
                            save_data_to_file()
        except Exception as e:
            print(f"处理设备数据时出错: {e}")
            try:
                device_conn.close()
            except:
                pass

async def handle_web_client(websocket):
    global is_recording, recording_start_time, current_label, connected_clients, is_realtime_mode
    global current_user_name, current_user_age, current_user_status

    # 添加新连接的客户端
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                cmd = json.loads(message)
                if cmd.get("command") == "start_recording":
                    current_label = cmd.get("action_label", 0)
                    # 获取用户信息
                    current_user_name = cmd.get("user_name", "")
                    current_user_age = cmd.get("user_age", "")
                    current_user_status = cmd.get("status", "")
                    
                    is_recording = True
                    recording_start_time = time.time()
                    print(f"开始记录动作{current_label}的数据，用户: {current_user_name}, 年龄: {current_user_age}, 状态: {current_user_status}")
                    await websocket.send(json.dumps({
                        "type": "recording_started",
                        "action": current_label
                    }))
                elif cmd.get("command") == "start_realtime_emg":
                    is_realtime_mode = True
                    print("开始实时EMG数据传输模式")
                    await websocket.send(json.dumps({
                        "type": "realtime_started",
                        "status": "success"
                    }))
                elif cmd.get("command") == "stop_realtime_emg":
                    is_realtime_mode = False
                    print("停止实时EMG数据传输模式")
                    await websocket.send(json.dumps({
                        "type": "realtime_stopped",
                        "status": "success"
                    }))
            except json.JSONDecodeError:
                print(f"无效的JSON消息: {message}")
    except Exception as e:
        print(f"处理Web客户端数据时出错: {e}")
    finally:
        # 客户端断开连接时移除
        connected_clients.remove(websocket)


def process_and_predict(all_data):
    """
    处理EMG数据并进行预测
    
    参数:
    all_data: shape (1, 2000, 16) 的numpy数组
    
    返回:
    prediction: 模型的预测结果
    """
    # 检查模型是否可用
    if not PREDICTION_ENABLED or cnn_model is None:
        print("预测功能未启用，返回默认结果")
        return np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])  # 返回默认的10类预测结果
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

    # 准备输入数据
    input_image = np.expand_dims(rgb_image_resized, axis=0)
    input_image = input_image.astype(np.float32) / 255.0

    # 进行预测
    prediction = cnn_model.predict(input_image)
    
    return prediction


async def main():
    # 启动设备数据处理线程
    device_thread = threading.Thread(target=device_data_handler, daemon=True)
    device_thread.start()
    
    # 创建EMG数据处理任务
    emg_queue_task = asyncio.create_task(process_emg_queue())

    host = "0.0.0.0"
    port = 8765

    print(f"WebSocket服务器正在启动，地址: ws://{host}:{port}")
    print(f"采样间隔设置为 {SAMPLE_INTERVAL*1000:.1f} 毫秒")
    print(f"采样率设置为 2000Hz (每3个包处理一次)")
    print(f"数据存储后将应用去均值处理")
    print(f"EMG缓冲区大小设置为 {EMG_BUFFER_SIZE} 帧，每积累 {EMG_BUFFER_SIZE} 帧发送一次数据")
    print(f"保留的EMG通道: {[ch+1 for ch in KEEP_CHANNELS]} (共{len(KEEP_CHANNELS)}个通道)")

    async with websockets.serve(handle_web_client, host, port):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("服务器已停止")
        if all_data:
            save_data_to_file()
