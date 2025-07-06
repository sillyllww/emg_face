import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import random

# 设置GPU内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("已启用GPU内存动态增长")
    except RuntimeError as e:
        print(f"GPU设置错误: {e}")

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

class SwishLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.swish(inputs)

def evaluate_metrics(model, model_name, test_x, test_y):
    """评估模型性能"""
    x = test_x
    y = test_y

    loss, acc = model.evaluate(x, y, verbose=2)
    y_preds = np.argmax(model.predict(x), axis=1)
    f1 = f1_score(np.argmax(y, axis=1), y_preds, average='weighted')

    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

    # 创建混淆矩阵
    cm = confusion_matrix(np.argmax(y, axis=1), y_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}_{acc:.4f}.png')
    plt.show()

    return acc

def load_image_data(images_dir='images_augmented_random'):
    """读取图像数据"""
    print("开始读取图像数据...")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"找不到 {images_dir} 文件夹")
    
    images = []
    labels = []
    
    # 获取所有标签文件夹
    label_folders = [f for f in os.listdir(images_dir) if f.startswith('label_')]
    label_folders.sort()
    
    print(f"找到 {len(label_folders)} 个标签文件夹: {label_folders}")
    
    # 读取每个标签文件夹中的图像
    total_images = 0
    for label_folder in label_folders:
        label_path = os.path.join(images_dir, label_folder)
        label_num = int(label_folder.split('_')[1])
        
        image_files = [f for f in os.listdir(label_path) if f.endswith('.png')]
        print(f"标签 {label_num}: 找到 {len(image_files)} 张图片")
        
        for image_file in image_files:
            image_path = os.path.join(label_path, image_file)
            img = Image.open(image_path)
            # 确保图像是RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            if img_array.shape != (37, 37, 3):
                img_array = np.resize(img_array, (37, 37, 3))
            
            images.append(img_array)
            labels.append(label_num)
            total_images += 1
    
    # 转换为numpy数组
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels)
    
    print(f"数据读取完成:")
    print(f"总图像数: {total_images}")
    print(f"图像数组形状: {X.shape}")
    print(f"标签数组形状: {y.shape}")
    print(f"唯一标签: {np.unique(y)}")
    
    return X, y

def prepare_data(X, y, test_size=0.2, val_size=0.3):
    """准备训练、验证和测试数据"""
    num_classes = len(np.unique(y))
    y_categorical = to_categorical(y, num_classes)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, 
        test_size=test_size, 
        random_state=42, 
        stratify=y
    )
    
    # 进一步划分训练集得到验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=42,
        stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes

def get_SimpleCNN(num_classes=10, deep_layers=2, stride_increase=0, 
                  expansion=1, depthwise=1, outputchannels=1):
    """简单的CNN模型"""
    inputs = keras.Input((37, 37, 3))
    
    # Stage 1 - 使用outputchannels参数调整通道数
    base_filters_1 = int(32 * outputchannels)
    x = layers.Conv2D(base_filters_1, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Stage 2 - 使用expansion参数扩展通道数
    base_filters_2 = int(64 * outputchannels)
    expanded_filters_2 = int(base_filters_2 * expansion)
    x = layers.Conv2D(expanded_filters_2, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # 可选的depthwise卷积
    if depthwise > 0:
        x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
    
    # 使用stride_increase调整池化大小
    pool_size_2 = (2, 2 + stride_increase) if stride_increase > 0 else (2, 2)
    x = layers.MaxPooling2D(pool_size_2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Stage 3 - 使用expansion参数扩展通道数
    base_filters_3 = int(128 * outputchannels)
    expanded_filters_3 = int(base_filters_3 * expansion)
    x = layers.Conv2D(expanded_filters_3, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # 可选的depthwise卷积
    if depthwise > 0:
        x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
    
    # 使用stride_increase调整池化大小
    pool_size_3 = (2, 2 + stride_increase) if stride_increase > 0 else (2, 2)
    x = layers.MaxPooling2D(pool_size_3)(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    
    # 全连接层 - 根据deep_layers参数调整层数
    if deep_layers >= 3:
        x = layers.Dense(int(512 * outputchannels), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
    
    if deep_layers >= 2:
        x = layers.Dense(int(256 * outputchannels), activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
    
    if deep_layers >= 1:
        x = layers.Dense(int(128 * outputchannels), activation='relu')(x)
        x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

def mlp(x, hidden_units, dropout_rate):
    """多层感知机"""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def transformer_block(x, transformer_layers, projection_dim, num_heads=6):
    """Transformer块"""
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.4
        )(x1, x1)

        x2 = layers.Add()([attention_output, x])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]], dropout_rate=0.4)
        x = layers.Add()([x3, x2])

    return x

def get_CNNAttention2D(num_classes=10, deep_layers=2, stride_increase=0, 
                       expansion=1, depthwise=1, outputchannels=1):
    """CNN+Attention模型，适应16x16x3输入"""
    inputs = keras.Input((37, 37, 3))

    # Stage 1 - 使用outputchannels参数调整通道数
    base_filters_1 = int(32 * outputchannels)
    x = layers.Conv2D(base_filters_1, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Stage 2 - 使用expansion参数扩展通道数
    base_filters_2 = int(64 * outputchannels)
    expanded_filters_2 = int(base_filters_2 * expansion)
    x = layers.Conv2D(expanded_filters_2, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # 可选的depthwise卷积
    if depthwise > 0:
        x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
    
    # 使用stride_increase调整池化大小
    pool_size_2 = (2, 2 + stride_increase) if stride_increase > 0 else (2, 2)
    x = layers.MaxPooling2D(pool_size_2)(x)
    x = layers.Dropout(0.25)(x)

    # Stage 3 - 使用expansion参数扩展通道数
    base_filters_3 = int(128 * outputchannels)
    expanded_filters_3 = int(base_filters_3 * expansion)
    x = layers.Conv2D(expanded_filters_3, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # 可选的depthwise卷积
    if depthwise > 0:
        x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
    
    # 使用stride_increase调整池化大小
    pool_size_3 = (2, 2 + stride_increase) if stride_increase > 0 else (2, 2)
    x = layers.MaxPooling2D(pool_size_3)(x)
    x = layers.Dropout(0.25)(x)

    # 展平并重塑为序列以使用Transformer
    x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)
    
    # Transformer block - projection_dim根据outputchannels调整
    projection_dim = int(128 * outputchannels)
    x = transformer_block(x, transformer_layers=2, num_heads=8, projection_dim=projection_dim)
    
    # Global average pooling
    x = layers.GlobalAvgPool1D()(x)

    # 全连接层 - 根据deep_layers参数调整层数
    if deep_layers >= 3:
        x = layers.Dense(int(512 * outputchannels))(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    if deep_layers >= 2:
        x = layers.Dense(int(256 * outputchannels))(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    if deep_layers >= 1:
        x = layers.Dense(int(128 * outputchannels))(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)

def get_model_parameters():
    """获取模型参数配置"""
    print("\n=== 模型参数配置 ===")
    print("请输入模型参数 (直接按回车使用默认值):")
    
    try:
        # deep_layers参数
        deep_layers_input = input("deep_layers (全连接层数, 默认=2): ").strip()
        deep_layers = int(deep_layers_input) if deep_layers_input else 2
        
        # stride_increase参数
        stride_increase_input = input("stride_increase (池化步长增加, 默认=0): ").strip()
        stride_increase = int(stride_increase_input) if stride_increase_input else 0
        
        # expansion参数
        expansion_input = input("expansion (通道扩展因子, 默认=1): ").strip()
        expansion = float(expansion_input) if expansion_input else 1.0
        
        # depthwise参数
        depthwise_input = input("depthwise (是否使用深度卷积, 0=否 1=是, 默认=1): ").strip()
        depthwise = int(depthwise_input) if depthwise_input else 1
        
        # outputchannels参数
        outputchannels_input = input("outputchannels (输出通道倍数, 默认=1): ").strip()
        outputchannels = float(outputchannels_input) if outputchannels_input else 1.0
        
    except ValueError:
        print("输入格式错误，使用默认参数")
        deep_layers, stride_increase, expansion, depthwise, outputchannels = 2, 0, 1.0, 1, 1.0
    
    print(f"\n选择的参数:")
    print(f"  deep_layers: {deep_layers}")
    print(f"  stride_increase: {stride_increase}")
    print(f"  expansion: {expansion}")
    print(f"  depthwise: {depthwise}")
    print(f"  outputchannels: {outputchannels}")
    
    return deep_layers, stride_increase, expansion, depthwise, outputchannels

def train(model, model_name, epochs, batch_size, train_x, train_y, val_x, val_y, test_x, test_y):
    """训练模型"""
    
    # 显示模型结构
    model.summary()
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    
    # 回调函数
    checkpoint_path = f"models/{model_name}_best_weights.h5"
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5,
            patience=5, 
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 训练模型
    print(f"开始训练模型: {model_name}")
    print(f"训练数据形状: {train_x.shape}, 标签形状: {train_y.shape}")
    
    history = model.fit(
        train_x, train_y, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_data=(val_x, val_y), 
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估模型
    print("评估模型性能...")
    acc = evaluate_metrics(model, model_name, test_x, test_y)
    
    # 保存最终模型
    model_save_path = f"models/{model_name}_{acc:.3f}.h5"
    model.save(model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_name}.png')
    plt.show()
    
    return history, acc

def validate_model_on_random_samples(model, images_dir='sliced_images', num_samples=50):
    """在随机抽取的图片上验证模型识别率"""
    print(f"\n=== 随机样本验证 ===")
    print(f"从 {images_dir} 随机抽取 {num_samples} 张图片进行验证...")
    
    if not os.path.exists(images_dir):
        print(f"错误：找不到 {images_dir} 文件夹")
        return
    
    # 获取所有标签文件夹
    label_folders = [f for f in os.listdir(images_dir) if f.startswith('label_')]
    label_folders.sort()
    
    # 收集所有图片路径和标签
    all_images = []
    all_labels = []
    
    for label_folder in label_folders:
        label_path = os.path.join(images_dir, label_folder)
        label_num = int(label_folder.split('_')[1])
        
        image_files = [f for f in os.listdir(label_path) if f.endswith('.png')]
        
        for image_file in image_files:
            image_path = os.path.join(label_path, image_file)
            all_images.append(image_path)
            all_labels.append(label_num)
    
    print(f"总共找到 {len(all_images)} 张图片")
    
    # 随机抽取样本
    sample_indices = random.sample(range(len(all_images)), min(num_samples, len(all_images)))
    
    # 加载和预处理图片
    sample_images = []
    sample_labels = []
    sample_paths = []
    
    for idx in sample_indices:
        image_path = all_images[idx]
        label = all_labels[idx]
        
        # 加载图片
        img = Image.open(image_path)
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        sample_images.append(img_array)
        sample_labels.append(label)
        sample_paths.append(image_path)
    
    # 转换为numpy数组
    X_sample = np.array(sample_images)
    y_sample = np.array(sample_labels)
    
    # 模型预测
    predictions = model.predict(X_sample, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # 计算准确率
    correct_predictions = np.sum(predicted_labels == y_sample)
    accuracy = correct_predictions / len(y_sample)
    
    print(f"\n验证结果:")
    print(f"样本数量: {len(y_sample)}")
    print(f"正确预测: {correct_predictions}")
    print(f"验证准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 显示详细结果
    print(f"\n详细预测结果:")
    print("-" * 80)
    print(f"{'序号':<4} {'真实标签':<8} {'预测标签':<8} {'置信度':<8} {'结果':<6} {'图片路径'}")
    print("-" * 80)
    
    for i in range(len(y_sample)):
        true_label = y_sample[i]
        pred_label = predicted_labels[i]
        confidence = np.max(predictions[i])
        result = "✓" if true_label == pred_label else "✗"
        filename = os.path.basename(sample_paths[i])
        
        print(f"{i+1:<4} {true_label:<8} {pred_label:<8} {confidence:.4f}  {result:<6} {filename}")
    
    # 统计各类别的准确率
    print(f"\n各类别验证准确率:")
    print("-" * 40)
    for label in np.unique(y_sample):
        mask = y_sample == label
        if np.sum(mask) > 0:
            label_accuracy = np.sum(predicted_labels[mask] == label) / np.sum(mask)
            count = np.sum(mask)
            print(f"标签 {label}: {label_accuracy:.4f} ({label_accuracy*100:.2f}%) - {count} 个样本")
    
    # 可视化部分预测结果
    visualize_predictions(X_sample, y_sample, predicted_labels, predictions, sample_paths, num_display=min(37, len(y_sample)))
    
    return accuracy

def visualize_predictions(X_sample, y_true, y_pred, predictions, sample_paths, num_display=37):
    """可视化预测结果"""
    print(f"\n可视化前 {num_display} 个预测结果...")
    
    # 计算显示网格
    cols = 4
    rows = (num_display + cols - 1) // cols
    
    plt.figure(figsize=(15, 4*rows))
    
    for i in range(min(num_display, len(X_sample))):
        plt.subplot(rows, cols, i+1)
        
        # 显示图片
        img = X_sample[i]
        if img.shape[-1] == 3:  # RGB图片
            plt.imshow(img)
        else:  # 灰度图片
            plt.imshow(img.squeeze(), cmap='gray')
        
        # 设置标题
        true_label = y_true[i]
        pred_label = y_pred[i]
        confidence = np.max(predictions[i])
        
        # 根据预测结果设置颜色
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f"真实: {true_label}, 预测: {pred_label}\n置信度: {confidence:.3f}", 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 读取数据
    X, y = load_image_data('images_augmented_random')
    
    # 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = prepare_data(X, y)
    
    # 选择模型类型
    print("\n请选择模型类型:")
    print("1. SimpleCNN - 简单CNN模型")
    print("2. CNNAttention2D - CNN+Attention模型")
    
    try:
        choice = input("请输入选择 (1-2): ").strip()
        model_type = choice
    except:
        print("输入错误，使用默认SimpleCNN模型")
        model_type = '1'
    
    # 获取模型参数
    deep_layers, stride_increase, expansion, depthwise, outputchannels = get_model_parameters()
    
    # 创建模型
    if model_type == '1':
        model = get_SimpleCNN(
            num_classes=num_classes,
            deep_layers=deep_layers,
            stride_increase=stride_increase,
            expansion=expansion,
            depthwise=depthwise,
            outputchannels=outputchannels
        )
        model_name = f"SimpleCNN_dl{deep_layers}_si{stride_increase}_ex{expansion}_dw{depthwise}_oc{outputchannels}"
    elif model_type == '2':
        model = get_CNNAttention2D(
            num_classes=num_classes,
            deep_layers=deep_layers,
            stride_increase=stride_increase,
            expansion=expansion,
            depthwise=depthwise,
            outputchannels=outputchannels
        )
        model_name = f"CNNAttention2D_dl{deep_layers}_si{stride_increase}_ex{expansion}_dw{depthwise}_oc{outputchannels}"
    else:
        print("无效选择，使用默认SimpleCNN模型")
        model = get_SimpleCNN(
            num_classes=num_classes,
            deep_layers=deep_layers,
            stride_increase=stride_increase,
            expansion=expansion,
            depthwise=depthwise,
            outputchannels=outputchannels
        )
        model_name = f"SimpleCNN_dl{deep_layers}_si{stride_increase}_ex{expansion}_dw{depthwise}_oc{outputchannels}"
    
    print(f"\n创建的模型: {model_name}")
    
    # 训练参数
    print("\n=== 训练参数配置 ===")
    try:
        epochs_input = input("训练轮数 epochs (默认=50): ").strip()
        epochs = int(epochs_input) if epochs_input else 50
        
        batch_size_input = input("批次大小 batch_size (默认=32): ").strip()
        batch_size = int(batch_size_input) if batch_size_input else 32
    except:
        print("输入错误，使用默认训练参数")
        epochs = 50
        batch_size = 32
    
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    
    # 训练模型
    history, acc = train(
        model=model,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        train_x=X_train,
        train_y=y_train,
        val_x=X_val,
        val_y=y_val,
        test_x=X_test,
        test_y=y_test
    )
    
    print(f"\n训练完成！最终测试准确率: {acc:.4f}")
    print(f"模型保存为: models/{model_name}_{acc:.3f}.h5")
    
    # 随机样本验证
    print("\n" + "="*60)
    try:
        validate_input = input("是否进行随机样本验证？(y/n, 默认=y): ").strip().lower()
        if validate_input != 'n':
            # 获取验证样本数量
            try:
                num_samples_input = input("验证样本数量 (默认=50): ").strip()
                num_samples = int(num_samples_input) if num_samples_input else 50
            except ValueError:
                print("输入格式错误，使用默认值50")
                num_samples = 50
            
            # 执行验证
            validation_acc = validate_model_on_random_samples(model, 'sliced_images', num_samples)
            
            print(f"\n=== 最终结果汇总 ===")
            print(f"测试集准确率: {acc:.4f} ({acc*100:.2f}%)")
            print(f"随机验证准确率: {validation_acc:.4f} ({validation_acc*100:.2f}%)")
            
            # 比较两个准确率
            diff = abs(acc - validation_acc)
            if diff < 0.05:
                print("✓ 模型性能稳定，测试集和随机验证结果一致")
            elif diff < 0.10:
                print("⚠ 模型性能有轻微差异，建议进一步验证")
            else:
                print("⚠ 模型性能差异较大，可能存在过拟合或数据分布问题")
        else:
            print("跳过随机样本验证")
    except KeyboardInterrupt:
        print("\n用户中断验证过程")
    except Exception as e:
        print(f"验证过程出错: {e}")

if __name__ == "__main__":
    main() 