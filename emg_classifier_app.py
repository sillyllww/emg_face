import os
import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                            QComboBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize
import tensorflow as tf
from tensorflow.keras.models import load_model

# 类别标签映射
CLASS_LABELS = {
    0: "抬眉",
    1: "皱眉",
    2: "闭眼",
    3: "鼓腮",
    4: "撅嘴",
    5: "微笑",
    6: "咧嘴笑",
    7: "龇牙",
    8: "耸鼻",
    9: "静止"
}

class EMGClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("EMG 图像分类器")
        self.setMinimumSize(800, 600)
        
        # 加载模型
        self.load_model()
        
        # 设置UI
        self.init_ui()
        
    def load_model(self):
        """加载预训练的EMG CNN模型"""
        try:
            self.model = load_model('CNNAttention2D_dl2_si0_ex1.0_dw1_oc1.0_0.993.h5')
            print("模型加载成功！")
            
            # 获取模型输入形状
            self.input_shape = self.model.input_shape[1:]
            print(f"模型输入形状: {self.input_shape}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            QMessageBox.critical(self, "错误", f"无法加载模型: {e}")
            sys.exit(1)
    
    def init_ui(self):
        """初始化用户界面"""
        # 创建主布局
        main_layout = QVBoxLayout()
        
        # 创建标题标签
        title_label = QLabel("EMG 图像分类器")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # 创建图像显示区域
        self.image_label = QLabel("请选择一张图像进行分类")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; padding: 10px; background-color: #f0f0f0;")
        self.image_label.setMinimumHeight(300)
        main_layout.addWidget(self.image_label)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        
        # 创建上传按钮
        self.upload_button = QPushButton("上传图像")
        self.upload_button.setMinimumHeight(40)
        self.upload_button.clicked.connect(self.upload_image)
        button_layout.addWidget(self.upload_button)
        
        # 创建分类按钮
        self.classify_button = QPushButton("分类")
        self.classify_button.setMinimumHeight(40)
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(False)  # 初始禁用
        button_layout.addWidget(self.classify_button)
        
        main_layout.addLayout(button_layout)
        
        # 创建结果显示区域
        result_layout = QHBoxLayout()
        
        result_label = QLabel("分类结果:")
        result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        result_layout.addWidget(result_label)
        
        self.result_text = QLabel("未分类")
        self.result_text.setStyleSheet("font-size: 16px;")
        result_layout.addWidget(self.result_text)
        
        # 添加一个弹性空间
        result_layout.addStretch()
        
        main_layout.addLayout(result_layout)
        
        # 创建置信度显示区域
        confidence_layout = QVBoxLayout()
        confidence_title = QLabel("各类别置信度:")
        confidence_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        confidence_layout.addWidget(confidence_title)
        
        self.confidence_labels = []
        for i in range(10):  # 假设有10个类别
            label_text = f"类别 {i} ({CLASS_LABELS.get(i, '未知')}): 0.00%"
            label = QLabel(label_text)
            confidence_layout.addWidget(label)
            self.confidence_labels.append(label)
        
        main_layout.addLayout(confidence_layout)
        
        # 设置主窗口部件和布局
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 初始化图像变量
        self.image = None
        self.processed_image = None
        
        # 显示模型信息
        self.statusBar().showMessage(f"模型已加载 - 输入形状: {self.input_shape}")
    
    def upload_image(self):
        """上传图像文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择EMG图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)", 
            options=options
        )
        
        if file_path:
            try:
                # 读取图像
                self.image = cv2.imread(file_path)
                if self.image is None:
                    raise Exception("无法读取图像文件")
                
                # 显示图像
                self.display_image(self.image)
                
                # 预处理图像以供模型使用
                self.processed_image = self.preprocess_image(self.image)
                
                # 启用分类按钮
                self.classify_button.setEnabled(True)
                
                # 重置结果
                self.result_text.setText("未分类")
                for i, label in enumerate(self.confidence_labels):
                    label_text = f"类别 {i} ({CLASS_LABELS.get(i, '未知')}): 0.00%"
                    label.setText(label_text)
                    label.setStyleSheet("")
                
                # 显示图像信息
                self.statusBar().showMessage(f"已加载图像: {os.path.basename(file_path)} - 尺寸: {self.image.shape[1]}x{self.image.shape[0]}")
                
            except Exception as e:
                QMessageBox.warning(self, "警告", f"图像加载失败: {e}")
    
    def display_image(self, image):
        """在界面上显示图像"""
        # 转换BGR为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小以适应显示区域，保持纵横比
        h, w, c = image_rgb.shape
        display_height = self.image_label.height() - 20  # 减去一些边距
        display_width = self.image_label.width() - 20
        
        # 计算缩放比例
        scale = min(display_width / w, display_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 创建QImage和QPixmap
        image_resized = cv2.resize(image_rgb, (new_w, new_h))
        qimg = QImage(image_resized.data, new_w, new_h, new_w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # 设置图像
        self.image_label.setPixmap(pixmap)
    
    def preprocess_image(self, image):
        """预处理图像以供模型使用"""
        try:
            # 获取模型期望的输入尺寸
            target_height = self.input_shape[0]
            target_width = self.input_shape[1]
            
            # 调整大小
            resized = cv2.resize(image, (target_width, target_height))
            
            # 如果模型期望灰度图像
            if len(self.input_shape) == 2 or self.input_shape[2] == 1:
                processed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                processed = np.expand_dims(processed, axis=-1)  # 添加通道维度
            else:
                # 保持彩色图像
                processed = resized
            
            # 归一化像素值到0-1之间
            processed = processed.astype(np.float32) / 255.0
            
            # 添加批次维度
            processed = np.expand_dims(processed, axis=0)
            
            print(f"预处理后图像形状: {processed.shape}")
            return processed
            
        except Exception as e:
            print(f"图像预处理错误: {e}")
            QMessageBox.warning(self, "警告", f"图像预处理失败: {e}")
            return None
    
    def classify_image(self):
        """对上传的图像进行分类"""
        if self.processed_image is None:
            QMessageBox.warning(self, "警告", "请先上传图像")
            return
        
        try:
            # 使用模型进行预测
            predictions = self.model.predict(self.processed_image)
            
            # 获取最高置信度的类别
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            # 获取类别标签
            class_label = CLASS_LABELS.get(predicted_class, "未知")
            
            # 更新结果显示
            self.result_text.setText(f"类别 {predicted_class} ({class_label}) - 置信度: {confidence:.2f}%")
            
            # 更新所有类别的置信度
            for i, conf in enumerate(predictions[0]):
                label_text = f"类别 {i} ({CLASS_LABELS.get(i, '未知')}): {conf*100:.2f}%"
                self.confidence_labels[i].setText(label_text)
                
                # 高亮显示预测的类别
                if i == predicted_class:
                    self.confidence_labels[i].setStyleSheet("font-weight: bold; color: blue;")
                else:
                    self.confidence_labels[i].setStyleSheet("")
            
            # 更新状态栏
            self.statusBar().showMessage(f"分类完成 - 预测结果: {class_label} (置信度: {confidence:.2f}%)")
            
        except Exception as e:
            print(f"分类错误: {e}")
            QMessageBox.critical(self, "错误", f"分类过程中出错: {e}")

def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 创建并显示主窗口
    window = EMGClassifierApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 