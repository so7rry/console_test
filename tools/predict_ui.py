# -*-coding:utf-8-*-
import sys
import csv
import json
import re
import os
import argparse
import pandas as pd
import numpy as np
import requests
import serial
from os import path
from os import mkdir
from io import StringIO
from PyQt5.Qt import *
import threading
from PyQt5.QtCore import QTime, QTimer, Qt, QThread, pyqtSignal
import base64
import time
from datetime import datetime
from multiprocessing import Process, Queue
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QProgressBar, QGroupBox, QCheckBox, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QComboBox, QTimeEdit, QSpinBox, QPushButton, QWidget, QErrorMessage, QBoxLayout, QTextBrowser)
from PyQt5.QtGui import QFont, QIcon
from esp_csi_tool_gui import Ui_MainWindow
from scipy import signal
import signal as signal_key
import socket
from pandas import Index

CSI_SAMPLE_RATE = 100
g_display_raw_data = True

# Remove invalid subcarriers
CSI_VAID_SUBCARRIER_INTERVAL = 5
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_color = []
color_step = 255 // (28 // CSI_VAID_SUBCARRIER_INTERVAL + 1)

# LLTF: 52
csi_vaid_subcarrier_index += [i for i in range(0, 26, CSI_VAID_SUBCARRIER_INTERVAL)]
csi_vaid_subcarrier_color += [(i * color_step, 0, 0) for i in range(1, 26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
csi_vaid_subcarrier_index += [i for i in range(26, 52, CSI_VAID_SUBCARRIER_INTERVAL)]
csi_vaid_subcarrier_color += [(0, i * color_step, 0) for i in range(1, 26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]

DEVICE_INFO_COLUMNS_NAMES = ["type", "timestamp", "compile_time", "chip_name", "chip_revision",
                             "app_revision", "idf_revision", "total_heap", "free_heap", "router_ssid", "ip", "port"]
g_device_info_series = None

CSI_DATA_INDEX = 500
CSI_DATA_COLUMNS = len(csi_vaid_subcarrier_index)
CSI_DATA_COLUMNS_NAMES = ["type", "seq", "timestamp", "taget_seq", "taget", "mac", "rssi", "rate", "sig_mode", "mcs",
                          "cwb", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor",
                          "ampdu_cnt", "channel_primary", "channel_secondary", "local_timestamp", "ant", "sig_len",
                          "rx_state", "agc_gain", "fft_gain", "len", "first_word_invalid", "data"]
CSI_DATA_TARGETS = ["unknown", "train", "lie_down", "walk", "stand", "bend", "sit_down",
                    "fall_from_stand", "fall_from_squat", "fall_from_bed"]

g_csi_phase_array = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.int32)
g_rssi_array = np.zeros(CSI_DATA_INDEX, dtype=np.int8)
g_radio_header_pd = pd.DataFrame(np.zeros([10, len(CSI_DATA_COLUMNS_NAMES[1:-1])], dtype=np.int64),
                                 columns=pd.Index(CSI_DATA_COLUMNS_NAMES[1:-1]))


def clean_base64_string(input_str):
    """
    清理和验证Base64字符串，移除非法字符并添加适当的填充
    
    Args:
        input_str: 输入的可能是Base64的字符串
        
    Returns:
        str: 清理后的有效Base64字符串
    """
    try:
        if not input_str:
            return ""
            
        # 转换为字符串并去除空白
        str_data = str(input_str).strip()
        
        # 检查并移除非Base64字符
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        if not all(c in valid_chars for c in str_data):
            # 找到第一个无效字符的位置
            invalid_pos = next((i for i, c in enumerate(str_data) if c not in valid_chars), len(str_data))
            # 截断字符串
            str_data = str_data[:invalid_pos]
            
        # 计算并添加填充
        padding = 4 - (len(str_data) % 4) if len(str_data) % 4 != 0 else 0
        if padding and padding < 4:  # 避免全是填充的情况
            str_data += '=' * padding
            
        return str_data
    except Exception as e:
        # 静默处理异常
        return ""


def base64_decode_bin(str_data):
    try:
        if not str_data:
            return []
        
        # 清理和验证Base64字符串
        str_data = clean_base64_string(str_data)
        if not str_data:
            return []
            
        # 解码
        try:
            bin_data = base64.b64decode(str_data)
            list_data = list(bin_data)
            for i in range(len(list_data)):
                if list_data[i] > 127:
                    list_data[i] = list_data[i] - 256
            return list_data
        except Exception:
            # 静默处理异常
            return []
    except Exception:
        # 静默处理异常
        return []


def base64_encode_bin(list_data):
    try:
        # 确保输入是有效的列表
        if not isinstance(list_data, list):
            if isinstance(list_data, str):
                # 尝试解析字符串为列表
                if list_data.startswith('[') and list_data.endswith(']'):
                    try:
                        list_data = json.loads(list_data)
                    except:
                        print(f"无法解析字符串为列表: {list_data[:20]}...")
                        return ""
                else:
                    print(f"输入不是有效的列表格式: {list_data[:20]}...")
                    return ""
            else:
                print(f"输入不是列表类型: {type(list_data)}")
                return ""
        
        # 处理负数
        for i in range(len(list_data)):
            if list_data[i] < 0:
                list_data[i] = 256 + list_data[i]
        
        # 编码为Base64
        try:
            str_data = base64.b64encode(bytes(list_data)).decode('utf-8')
            return str_data
        except Exception as e:
            print(f"Base64编码异常: {e}, 数据长度: {len(list_data)}")
            return ""
    except Exception as e:
        print(f"Base64编码处理异常: {e}")
        return ""


def get_label(folder_path):
    parts = str.split(folder_path, os.path.sep)
    return parts[-1]


def evaluate_data_send(serial_queue_write, folder_path):
    label = get_label(folder_path)
    if label == "train":
        command = f"csi --train_start"
        serial_queue_write.put(command)
    tcpCliSock = socket.socket()
    device_info_series = pd.read_csv('log/device_info.csv').iloc[-1]
    print(f"connect:{device_info_series['ip']},{device_info_series['port']}")
    tcpCliSock.connect((device_info_series['ip'], device_info_series['port']))
    file_name_list = sorted(os.listdir(folder_path))
    print(file_name_list)
    for file_name in file_name_list:
        file_path = folder_path + os.path.sep + file_name
        data_pd = pd.read_csv(file_path)
        for index, data_series in data_pd.iterrows():
            csi_raw_data = json.loads(str(data_series['data']))
            data_pd.loc[index, 'data'] = base64_encode_bin(csi_raw_data)
            temp_list = base64_decode_bin(data_pd.loc[index, 'data'])
            data_str = ','.join(str(value) for value in data_pd.loc[index]) + "\n"
            data_str = data_str.encode('utf-8')
            tcpCliSock.send(data_str)
    tcpCliSock.close()
    time.sleep(1)
    if label == "train":
        command = "csi --train_stop"
        serial_queue_write.put(command)
    sys.exit(0)


class DataGraphicalWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, serial_queue_read, serial_queue_write):
        super().__init__()
        self.serial_queue_read = serial_queue_read
        self.serial_queue_write = serial_queue_write
        self.server_url = "http://8.136.10.160:12786/api/predict"  # 修改为预测接口
        self.enable_server_predict = False
        self.wifi_connected = False
        self.csi_enabled = False
        self.csi_buffer = []  # 用于缓存CSI数据
        self.buffer_size = 100  # 缓冲区大小，可以根据需要调整
        self.predict_interval = 500  # 预测间隔(ms)
        
        # 先初始化UI
        self.initUI()
        
        # 只隐藏Command和router部分，保留CSI图像显示
        try:
            # 只隐藏这些特定组件
            components_to_hide = [
                'groupBox_16',  # Command组
                'label',        # custom标签
                'lineEdit_command',  # 命令输入框
                'comboBox_command',  # 命令下拉框
                'pushButton_command',  # 命令按钮
                'groupBox_router',  # router组
                'label_delay_2',  # ssid标签
                'label_duration_2',  # password标签
                'lineEdit_router_ssid_old',  # 旧的SSID输入框
                'lineEdit_router_password_old',  # 旧的密码输入框
                'pushButton_router_connect_old'  # 旧的连接按钮
            ]
            
            for component_name in components_to_hide:
                if hasattr(self, component_name):
                    component = getattr(self, component_name)
                    if component and hasattr(component, 'hide'):
                        component.hide()
                        print(f"已隐藏组件: {component_name}")
        except Exception as e:
            print(f"隐藏组件时出错: {e}")
        
        QTimer.singleShot(3000, lambda: self.textBrowser_log.append(
            f"<font color='yellow'>系统初始化中，将自动设置CSI并连接WiFi</font>"))
        
        # 启动预测定时器
        self.predict_timer = QTimer()
        self.predict_timer.timeout.connect(self.predict_action)
        self.predict_timer.setInterval(self.predict_interval)

    def initUI(self):
        self.setupUi(self)
        self.setWindowTitle("ESP CSI Tool - Real-time Action Detection")
        
        # 保留这些重要组件，不要删除
        important_components = [
            'textBrowser_log',
            'graphicsView_subcarrier',
            'graphicsView_rssi',
            'groupBox_subcarrier',
            'groupBox_rssi',
            'groupBox_13',  # 日志组
            'centralwidget',
            'splitter_raw_data',
            'splitter_display',
            'layoutWidget'
        ]
        
        # 隐藏和删除不需要的组件
        components_to_remove = [
            'groupBox_radar_model',
            'QWidget_evaluate_info',
            'groupBox_20',
            'groupBox_eigenvalues',
            'groupBox_eigenvalues_table',
            'groupBox_statistics',
            'groupBox_predict',
            'checkBox_raw_data',
            'checkBox_radar_model',
            'checkBox_wave_filtering',
            'groupBox_radioHeader'  # 添加info区域到隐藏列表
        ]
        
        # 从删除列表中移除重要组件
        components_to_remove = [c for c in components_to_remove if c not in important_components]
        
        # 仅隐藏不需要的组件，不删除它们
        for component_name in components_to_remove:
            if hasattr(self, component_name):
                component = getattr(self, component_name)
                if component and isinstance(component, QWidget):
                    # 安全地获取父组件
                    parent = component.parent()
                    if parent and isinstance(parent, QWidget) and hasattr(parent, 'layout'):
                        parent_layout = parent.layout()
                        if parent_layout:
                            try:
                                parent_layout.removeWidget(component)
                            except Exception as e:
                                print(f"无法从布局中移除组件 {component_name}: {e}")
                    # 只隐藏组件，不删除它
                    component.hide()
                    # 不再调用deleteLater()
                    # 不再设置为None
        
        # 处理布局组件
        layout_components = [
            'horizontalLayout_4',
            'horizontalLayout_11',
            'horizontalLayout_router'
        ]
        
        # 隐藏布局中的所有组件
        for layout_name in layout_components:
            if hasattr(self, layout_name):
                layout = getattr(self, layout_name)
                if layout and hasattr(layout, 'count'):
                    try:
                        # 隐藏布局中的所有组件，但不删除它们
                        for i in range(layout.count()):
                            item = layout.itemAt(i)
                            if item and item.widget():
                                widget = item.widget()
                                if widget and widget.objectName() not in important_components:
                                    widget.hide()
                    except Exception as e:
                        print(f"处理布局 {layout_name} 时出错: {e}")
        
        # 获取主布局
        if not hasattr(self, 'main_layout'):
            self.main_layout = self.centralwidget.layout()
            if not self.main_layout:
                self.main_layout = QVBoxLayout(self.centralwidget)
        
        # 配置新的组件
        self.setup_wifi_connection()
        self.setup_server_settings()
        self.create_detection_panel()
        
        # 配置图表
        self.setup_graphs()
        
        # 设置黑色背景
        if hasattr(self, 'textBrowser_log') and self.textBrowser_log:
            self.textBrowser_log.setStyleSheet("background:black")
        
        # 启动初始化定时器
        self.timer_boot_command = QTimer()
        self.timer_boot_command.timeout.connect(self.command_boot)
        self.timer_boot_command.setInterval(3000)
        self.timer_boot_command.start()

    def create_detection_panel(self):
        """创建实时检测面板"""
        self.detection_group = QGroupBox("实时动作检测")
        self.detection_group.setFont(QFont("Arial", 10))
        detection_layout = QVBoxLayout(self.detection_group)
        
        # 添加控制按钮和预测结果显示
        control_layout = QHBoxLayout()
        
        # 开始/停止检测按钮
        self.toggleDetectionButton = QPushButton("开始检测")
        self.toggleDetectionButton.clicked.connect(self.toggle_detection)
        self.toggleDetectionButton.setMinimumWidth(120)
        control_layout.addWidget(self.toggleDetectionButton)
        
        # 添加预测结果显示
        self.predictionLabel = QLabel("当前动作: -")
        self.predictionLabel.setFont(QFont("Arial", 12, QFont.Bold))
        # 使用数值4代替Qt.AlignHCenter
        self.predictionLabel.setStyleSheet("color: cyan; text-align: center;")
        control_layout.addWidget(self.predictionLabel, 1)
        
        detection_layout.addLayout(control_layout)
        
        # 添加预测间隔设置
        interval_layout = QHBoxLayout()
        interval_label = QLabel("预测间隔(ms):")
        self.predict_interval_spinbox = QSpinBox()
        self.predict_interval_spinbox.setRange(100, 2000)
        self.predict_interval_spinbox.setValue(self.predict_interval)
        self.predict_interval_spinbox.setSingleStep(100)
        self.predict_interval_spinbox.valueChanged.connect(self.update_predict_interval)
        
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.predict_interval_spinbox)
        interval_layout.addStretch(1)
        
        detection_layout.addLayout(interval_layout)
        
        # 将检测面板添加到主布局
        self.main_layout.addWidget(self.detection_group)

    def setup_graphs(self):
        """配置图表显示"""
        # 确保图表组件存在
        if not hasattr(self, 'graphicsView_subcarrier') or not self.graphicsView_subcarrier:
            print("警告: graphicsView_subcarrier组件不存在")
            return
            
        if not hasattr(self, 'graphicsView_rssi') or not self.graphicsView_rssi:
            print("警告: graphicsView_rssi组件不存在")
            return
            
        try:
            # 配置子载波图
            self.graphicsView_subcarrier.setYRange(-20, 20, padding=0)
            self.graphicsView_subcarrier.addLegend()
            self.graphicsView_subcarrier.setMouseEnabled(x=True, y=False)
            self.graphicsView_subcarrier.enableAutoRange(axis='y', enable=False)
            self.graphicsView_subcarrier.setMenuEnabled(False)
            
            # 配置RSSI图
            self.graphicsView_rssi.setYRange(-100, 0, padding=0)
            self.graphicsView_rssi.setMouseEnabled(x=True, y=False)
            self.graphicsView_rssi.enableAutoRange(axis='y', enable=False)
            self.graphicsView_rssi.setMenuEnabled(False)
            
            # 创建曲线
            self.curve_subcarrier = []
            for i in range(CSI_DATA_COLUMNS):
                curve = self.graphicsView_subcarrier.plot(g_csi_phase_array[:, i], name=str(i),
                                                        pen=csi_vaid_subcarrier_color[i])
                self.curve_subcarrier.append(curve)
            self.curve_rssi = self.graphicsView_rssi.plot(g_rssi_array, name='rssi', pen=(255, 255, 255))
            
            print("CSI图表配置成功")
        except Exception as e:
            print(f"配置图表时出错: {e}")

    def setup_server_settings(self):
        """配置服务器设置相关的UI和功能"""
        # 创建服务器设置组
        server_group = QGroupBox("服务器设置")
        server_group.setFont(QFont("Arial", 10))
        server_layout = QVBoxLayout(server_group)
        
        # 服务器地址设置
        server_url_layout = QHBoxLayout()
        server_label = QLabel("服务器地址:")
        self.lineEdit_server_url = QLineEdit()
        self.lineEdit_server_url.setText(self.server_url)
        self.lineEdit_server_url.setPlaceholderText("输入预测服务器地址")
        server_url_layout.addWidget(server_label)
        server_url_layout.addWidget(self.lineEdit_server_url)
        
        # 启用预测复选框
        self.checkBox_server_predict = QCheckBox("启用实时预测")
        self.checkBox_server_predict.stateChanged.connect(self.toggle_server_predict)
        server_url_layout.addWidget(self.checkBox_server_predict)
        
        # 添加到服务器设置组
        server_layout.addLayout(server_url_layout)
        
        # 添加到主布局
        if self.main_layout.count() > 1:
            self.main_layout.insertWidget(1, server_group)
        else:
            self.main_layout.addWidget(server_group)

    def setup_wifi_connection(self):
        """配置WiFi连接相关的UI和功能"""
        # 创建WiFi设置组
        wifi_group = QGroupBox("WiFi设置")
        wifi_group.setFont(QFont("Arial", 10))
        wifi_layout = QVBoxLayout(wifi_group)

        # WiFi连接控件
        connection_layout = QHBoxLayout()
        
        # SSID
        ssid_label = QLabel("SSID:")
        self.lineEdit_router_ssid = QLineEdit()
        self.lineEdit_router_ssid.setPlaceholderText("输入WiFi名称")
        
        # 密码
        password_label = QLabel("密码:")
        self.lineEdit_router_password = QLineEdit()
        self.lineEdit_router_password.setPlaceholderText("输入WiFi密码")
        self.lineEdit_router_password.setEchoMode(QLineEdit.Password)
        
        # 连接按钮
        self.pushButton_router_connect = QPushButton("连接")
        self.pushButton_router_connect.released.connect(self.command_router_connect)
        self.pushButton_router_connect.setMinimumWidth(100)
        
        # 添加到连接布局
        connection_layout.addWidget(ssid_label)
        connection_layout.addWidget(self.lineEdit_router_ssid)
        connection_layout.addWidget(password_label)
        connection_layout.addWidget(self.lineEdit_router_password)
        connection_layout.addWidget(self.pushButton_router_connect)
        
        # 添加到WiFi组
        wifi_layout.addLayout(connection_layout)
        
        # 从配置文件加载WiFi设置
        try:
            with open("./config/gui_config.json") as file:
                gui_config = json.load(file)
                if len(gui_config.get('router_ssid', '')) > 0:
                    self.lineEdit_router_ssid.setText(gui_config['router_ssid'])
                if len(gui_config.get('router_password', '')) >= 8:
                    self.lineEdit_router_password.setText(gui_config['router_password'])
        except Exception as e:
            print(f"加载WiFi配置失败: {e}")
            
        # 添加到主布局的顶部
        if self.main_layout.count() > 0:
            self.main_layout.insertWidget(0, wifi_group)
        else:
            self.main_layout.addWidget(wifi_group)

    def toggle_detection(self):
        if self.toggleDetectionButton.text() == "开始检测":
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        if not self.wifi_connected:
            QMessageBox.warning(self, "错误", "请先连接WiFi")
            return
            
        if not self.enable_server_predict:
            QMessageBox.warning(self, "错误", "请先启用服务器预测")
            return
            
        self.toggleDetectionButton.setText("停止检测")
        self.toggleDetectionButton.setStyleSheet("color: red")
        self.predict_timer.start()
        self.textBrowser_log.append("<font color='green'>开始实时动作检测</font>")

    def stop_detection(self):
        self.toggleDetectionButton.setText("开始检测")
        self.toggleDetectionButton.setStyleSheet("color: black")
        self.predict_timer.stop()
        self.csi_buffer.clear()
        self.predictionLabel.setText("当前动作: -")
        self.textBrowser_log.append("<font color='yellow'>停止实时动作检测</font>")

    def predict_action(self):
        """发送CSI数据到服务器进行实时动作识别"""
        if not self.csi_buffer:
            return
            
        try:
            # 准备数据
            data = {
                'csi_data': self.csi_buffer,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            }
            
            # 发送预测请求
            self.textBrowser_log.append(f"<font color='cyan'>发送 {len(self.csi_buffer)} 条CSI数据进行预测...</font>")
            response = requests.post(
                self.server_url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=2  # 适当增加超时时间
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'prediction' in result:
                    action = result['prediction']
                    confidence = result.get('confidence', 0)
                    
                    # 更新UI显示
                    self.predictionLabel.setText(f"当前动作: {action}")
                    
                    # 根据置信度设置不同颜色
                    if confidence > 0.8:
                        self.predictionLabel.setStyleSheet("color: lime; text-align: center;")
                    elif confidence > 0.5:
                        self.predictionLabel.setStyleSheet("color: yellow; text-align: center;")
                    else:
                        self.predictionLabel.setStyleSheet("color: orange; text-align: center;")
                        
                    self.textBrowser_log.append(
                        f"<font color='cyan'>检测到动作: {action}, 置信度: {confidence:.2f}</font>")
            else:
                self.textBrowser_log.append(
                    f"<font color='yellow'>预测请求失败: HTTP {response.status_code}</font>")
                
        except requests.exceptions.Timeout:
            self.textBrowser_log.append(f"<font color='yellow'>预测请求超时</font>")
        except requests.exceptions.ConnectionError:
            self.textBrowser_log.append(f"<font color='yellow'>无法连接到预测服务器</font>")
        except Exception as e:
            self.textBrowser_log.append(f"<font color='yellow'>预测过程发生错误: {str(e)}</font>")
        
        # 清空缓冲区
        self.csi_buffer.clear()

    def toggle_server_predict(self, state):
        if state == Qt.CheckState.Checked:
            server_url = self.lineEdit_server_url.text().strip()
            if not server_url:
                QMessageBox.warning(self, "服务器地址错误", "服务器地址不能为空")
                self.checkBox_server_predict.setChecked(False)
                return
                
            # 验证URL格式
            if not server_url.startswith(('http://', 'https://')):
                server_url = 'http://' + server_url
                
            # 更新服务器地址
            self.server_url = server_url
            self.enable_server_predict = True
            
            # 测试服务器连接
            try:
                response = requests.get(server_url.replace('/api/predict', '/api/test_connection'), timeout=5)
                if response.status_code == 200:
                    self.textBrowser_log.append(f"<font color='green'>预测服务器连接成功</font>")
                else:
                    self.textBrowser_log.append(
                        f"<font color='yellow'>警告: 服务器返回异常状态码: {response.status_code}</font>")
            except Exception as e:
                self.textBrowser_log.append(f"<font color='yellow'>警告: 服务器连接测试失败: {str(e)}</font>")
                
        else:
            self.enable_server_predict = False
            self.stop_detection()  # 停止检测
            self.textBrowser_log.append("<font color='yellow'>已禁用实时预测</font>")

    def process_csi_data(self, data_series):
        """处理接收到的CSI数据"""
        if not self.enable_server_predict or not self.predict_timer.isActive():
            return
            
        try:
            # 提取CSI数据的关键特征
            csi_data = {
                'timestamp': data_series.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]),
                'rssi': int(data_series.get('rssi', -100)),
                'mac': data_series.get('mac', ''),
                'data': data_series.get('data', [])
            }
            
            # 将数据添加到缓冲区
            self.csi_buffer.append(csi_data)
            
            # 如果缓冲区达到指定大小，触发预测
            if len(self.csi_buffer) >= self.buffer_size:
                self.predict_action()
                
        except Exception as e:
            print(f"处理CSI数据异常: {e}")

    def show_router_auto_connect(self):
        with open("./config/gui_config.json", "r") as file:
            gui_config = json.load(file)
            gui_config['router_auto_connect'] = self.checkBox_router_auto_connect.isChecked()
        with open("./config/gui_config.json", "w") as file:
            json.dump(gui_config, file)

    def command_boot(self):
        """初始化设备，设置CSI格式并启用CSI功能"""
        print("正在初始化...")
        self.textBrowser_log.append(f"<font color='cyan'>正在初始化...</font>")
        command = f"radar --csi_output_type LLFT --csi_output_format base64"
        self.serial_queue_write.put(command)
        self.textBrowser_log.append(f"<font color='green'>CSI 输出格式已设置</font>")
        QTimer.singleShot(500, lambda: self.enable_csi())
        # 自动连接WiFi，且按钮状态同步
        QTimer.singleShot(1500, lambda: self.command_router_connect() if not self.wifi_connected and self.pushButton_router_connect.text() == "连接" else None)
        self.timer_boot_command.stop()

    def enable_csi(self):
        """启用 CSI 功能"""
        command = f"radar --csi_en 1"
        self.serial_queue_write.put(command)
        self.textBrowser_log.append(f"<font color='green'>CSI 功能已启用</font>")
        self.csi_enabled = True
        
        # 启动实时数据显示定时器
        self.timer_curve_subcarrier = QTimer()
        self.timer_curve_subcarrier.timeout.connect(self.show_curve_subcarrier)
        self.timer_curve_subcarrier.setInterval(300)
        self.timer_curve_subcarrier.start()

    def command_router_connect(self):
        """连接或断开WiFi"""
        if self.pushButton_router_connect.text() == "连接":
            self.pushButton_router_connect.setText("断开")
            self.pushButton_router_connect.setStyleSheet("color: red")

            ssid = self.lineEdit_router_ssid.text()
            password = self.lineEdit_router_password.text()

            if len(ssid) == 0:
                self.textBrowser_log.append(f"<font color='red'>错误: SSID 不能为空</font>")
                self.pushButton_router_connect.setText("连接")
                self.pushButton_router_connect.setStyleSheet("color: black")
                return

            command = f"wifi_config --ssid \"{ssid}\""
            if len(password) >= 8:
                command += f" --password {password}"

            self.textBrowser_log.append(f"<font color='cyan'>正在连接 WiFi: {ssid}</font>")
            self.serial_queue_write.put(command)

            # 设置WiFi连接标志
            self.wifi_connected = True
        else:
            self.pushButton_router_connect.setText("连接")
            self.pushButton_router_connect.setStyleSheet("color: black")

            command = "ping --abort"
            self.serial_queue_write.put(command)
            command = "wifi_config --disconnect"
            self.serial_queue_write.put(command)
            self.textBrowser_log.append(f"<font color='yellow'>正在断开 WiFi 连接...</font>")

            # 重置WiFi连接标志
            self.wifi_connected = False

        # 保存配置
        try:
            with open("./config/gui_config.json", "r") as file:
                gui_config = json.load(file)
                gui_config['router_ssid'] = self.lineEdit_router_ssid.text()
                gui_config['router_password'] = self.lineEdit_router_password.text()
            with open("./config/gui_config.json", "w") as file:
                json.dump(gui_config, file)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def command_custom(self):
        command = self.lineEdit_command.text()
        self.serial_queue_write.put(command)

    def command_collect_target_start(self):
        """开始采集目标数据"""
        if self.is_collecting:
            return

        self.is_collecting = True

        # 发送用户名称到串口处理进程（如果没有在开始前发送）
        user_name = self.lineEdit_user.text().strip()
        if user_name:
            self.serial_queue_write.put(f"set_user:{user_name}")

        # 获取当前动作和新序号
        target = self.comboBox_collect_target.currentText()
        sequence = self.get_next_sequence(user_name, target)
        
        # 生成任务ID，格式：动作_序号，例如：stand_02
        task_id = f"{target}_{sequence:02d}"
        
        # 发送任务开始命令
        self.serial_queue_write.put(f"start_task:{task_id}")

        # 发送采集命令
        duration = self.spinBox_collect_duration.value()
        command = (f"radar --collect_number 1" +
                   f" --collect_tagets {target}" +
                   f" --collect_duration {duration}")
        self.serial_queue_write.put(command)

        # 更新UI显示
        self.textBrowser_log.append(
            f"<font color='cyan'>采集开始: {target} - 序号: {sequence:02d}，持续时间: {duration}ms</font>")
        self.textBrowser_log.append(
            f"<font color='cyan'>文件将保存为: {target}_{user_name}_{sequence:02d}.csv</font>")

        if self.enable_server_save:
            self.textBrowser_log.append(f"<font color='cyan'>数据将同时保存到服务器: {self.server_url}</font>")

        # 启动持续时间定时器
        self.collection_duration_timer.start(duration + 200)

        # 更新进度条
        self.update_progress(0, 100)

        # 启动进度更新定时器
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(lambda: self.update_progress(
            duration + 200 - self.collection_duration_timer.remainingTime(),
            duration + 200
        ))
        self.progress_timer.start(100)

        # 采集完成后自动恢复按钮
        QTimer.singleShot(duration + 300, self.finish_collection)

    def command_collect_target_stop(self):
        """停止采集目标数据"""
        if not self.is_collecting:
            return

        # 发送停止采集命令
        command = "radar --collect_number 0 --collect_tagets unknown"
        self.serial_queue_write.put(command)

        # 发送任务结束命令
        self.serial_queue_write.put("end_task:")

        # 停止当前采集
        self.is_collecting = False

        # 停止定时器
        if self.collection_duration_timer.isActive():
            self.collection_duration_timer.stop()

        # 停止进度条更新
        if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
            self.progress_timer.stop()

        # 设置进度条为100%
        self.progressBar.setValue(100)

        # 记录日志
        self.textBrowser_log.append(f"<font color='yellow'>采集暂停</font>")

    def on_delay_timeout(self):
        """延时结束，开始采集"""
        # 重置样式
        self.timeEdit_collect_delay.setStyleSheet("color: black")

        # 开始第一次采集
        self.command_collect_target_start()

    def on_duration_timeout(self):
        """单次采集持续时间结束"""
        # 停止当前采集
        self.command_collect_target_stop()

        # 添加短暂延迟，确保停止命令已被处理
        QTimer.singleShot(100, self.process_next_collection)

    def process_next_collection(self):
        """处理下一次采集"""
        # 检查是否需要继续采集
        if self.current_collect_count < self.target_collect_count:
            # 继续下一次采集
            self.command_collect_target_start()
        else:
            # 采集完成，恢复初始状态
            self.finish_collection()

    def finish_collection(self):
        """结束整个采集过程，恢复状态"""
        self.is_collecting = False
        self.timeEdit_collect_delay.setTime(self.label_delay)
        self.pushButton_collect_start.setStyleSheet("color: black")
        self.timeEdit_collect_delay.setStyleSheet("color: black")
        self.pushButton_collect_start.setText("start")

        if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
            self.progress_timer.stop()
        if self.timer_collect_delay.isActive():
            self.timer_collect_delay.stop()
        if self.collection_delay_timer.isActive():
            self.collection_delay_timer.stop()
        if self.collection_duration_timer.isActive():
            self.collection_duration_timer.stop()

        user_name = self.lineEdit_user.text().strip()
        target = self.comboBox_collect_target.currentText()
        self.textBrowser_log.append(
            f"<font color='green'>采集完成：用户 {user_name} 的 {target} 动作</font>")
        folder_path = f"data/{target}"
        self.textBrowser_log.append(f"<font color='green'>数据已保存到目录: {os.path.abspath(folder_path)}</font>")
        if self.enable_server_save:
            self.textBrowser_log.append(f"<font color='green'>数据已同步到服务器: {self.server_url}</font>")

    def timeEdit_collect_delay_show(self):
        time_temp = self.timeEdit_collect_delay.time()
        second = time_temp.hour() * 3600 + time_temp.minute() * 60 + time_temp.second()
        if second > 0:
            self.timeEdit_collect_delay.setTime(time_temp.addSecs(-1))
            self.timer_collect_delay.start()
            if second <= 5:
                self.textBrowser_log.append(f"<font color='cyan'>Starting in {second} seconds...</font>")
        else:
            self.timer_collect_delay.stop()
            self.on_delay_timeout()

    def pushButton_collect_show(self):
        """采集按钮点击处理"""
        if self.pushButton_collect_start.text() == "start":
            # 检查参数是否有效
            if self.comboBox_collect_target.currentIndex() == 0:
                err = QErrorMessage(self)
                err.setWindowTitle('Label parameter error')
                err.showMessage("Please check whether 'target' is set")
                err.show()
                return

            user_name = self.lineEdit_user.text().strip()
            if not user_name:
                err = QErrorMessage(self)
                err.setWindowTitle('User name error')
                err.showMessage("Please enter user name")
                err.show()
                return

            self.user_name = user_name
            self.serial_queue_write.put(f"set_user:{user_name}")
            self.label_delay = self.timeEdit_collect_delay.time()
            self.pushButton_collect_start.setText("stop")
            self.pushButton_collect_start.setStyleSheet("color: red")
            self.timeEdit_collect_delay.setStyleSheet("color: red")
            self.command_collect_target_start()
        else:
            self.finish_collection()

    def pushButton_collect_clean_show(self):
        folder_path = 'data'
        message = QMessageBox.warning(
            self, 'Warning', "will delete all files in 'data'", QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        if message == QMessageBox.Cancel:
            return
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def show_textBrowser_log(self, str):
        self.textBrowser_log.append(str)

    def show_device_info(self, device_info_series):
        global g_device_info_series
        g_device_info_series = device_info_series
        if device_info_series['type'] == 'DEVICE_INFO':
            self.textBrowser_log.append(
                f"<font color='green'>设备连接成功: {device_info_series['app_revision']}</font>")

    def closeEvent(self, event):
        self.serial_queue_write.put("exit")
        time.sleep(0.5)
        event.accept()
        try:
            os._exit(0)
        except Exception as e:
            print(f"GUI closeEvent: {e}")

    def comboBox_command_show(self):
        command = self.comboBox_command.currentText()
        self.lineEdit_command.setText(command)

    def toggle_server_save(self, state):
        if state == Qt.CheckState.Checked:
            server_url = self.lineEdit_server_url.text().strip()
            if not server_url:
                QMessageBox.warning(self, "服务器地址错误", "服务器地址不能为空，请输入有效的服务器地址")
                self.checkBox_server_save.setChecked(False)
                return

            # 验证URL格式
            if not server_url.startswith(('http://', 'https://')):
                server_url = 'http://' + server_url

            # 规范化URL
            server_url = server_url.rstrip('/')
            if not server_url.endswith('/api/csi_data'):
                server_url = server_url + '/api/csi_data'

            # 更新界面显示和实例变量
            self.lineEdit_server_url.setText(server_url)
            self.server_url = server_url
            self.enable_server_save = True

            # 发送命令
            self.serial_queue_write.put("enable_server_save")
            self.serial_queue_write.put(f"set_server_url:{server_url}")

            # 测试服务器连接
            self.textBrowser_log.append(f"<font color='cyan'>正在测试服务器连接: {server_url}</font>")
            try:
                response = requests.get(server_url.replace('/api/csi_data', '/api/test_connection'), timeout=5)
                if response.status_code == 200:
                    self.textBrowser_log.append(f"<font color='green'>服务器连接成功: {server_url}</font>")
                else:
                    self.textBrowser_log.append(
                        f"<font color='yellow'>警告: 服务器返回异常状态码: {response.status_code}</font>")
            except Exception as e:
                self.textBrowser_log.append(f"<font color='yellow'>警告: 服务器连接测试失败: {str(e)}</font>")

            self.textBrowser_log.append(f"<font color='green'>已启用服务器保存，服务器地址: {server_url}</font>")
        else:
            self.serial_queue_write.put("disable_server_save")
            self.enable_server_save = False
            self.textBrowser_log.append(f"<font color='yellow'>已禁用服务器保存</font>")

    def update_progress(self, current, total):
        """更新进度条显示"""
        progress = int((current / total) * 100)
        self.progressBar.setValue(progress)
        if progress % 10 == 0:  # 每10%更新一次日志，避免日志过多
            self.textBrowser_log.append(f"<font color='cyan'>采集进度: {progress}%</font>")

    def handle_wifi_status(self, data):
        """处理WiFi状态信息，在连接成功后更新状态"""
        # 检查是否包含WiFi连接成功的信息
        if (
                'WiFi已连接' in data or 'connected' in data.lower() or 'wifi:state:' in data.lower() and 'run' in data.lower()):
            # 只显示连接成功的提示，不再重复启用CSI功能
            if not self.wifi_connected:
                self.textBrowser_log.append(f"<font color='green'>WiFi连接成功！</font>")
                self.wifi_connected = True

        # 检查是否包含IP地址获取成功的信息
        if 'ip' in data.lower() and ('got' in data.lower() or '获取' in data or 'assigned' in data.lower()):
            # 尝试从日志中提取IP地址
            ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', data)
            if ip_match:
                ip_address = ip_match.group(0)
                self.textBrowser_log.append(f"<font color='green'>网络连接成功！IP地址: {ip_address}</font>")
            else:
                self.textBrowser_log.append(f"<font color='green'>网络连接成功！</font>")

        # 特别检测CSI数据接收，作为WiFi连接成功的额外指标
        elif "CSI_DATA" in data:
            # 如果接收到CSI数据，说明WiFi已经连接成功
            static_counter = getattr(self, 'csi_data_counter', 0)
            if static_counter == 0:
                # 只在第一次检测到CSI数据时发送连接成功消息
                self.textBrowser_log.append(f"<font color='green'>检测到CSI数据，WiFi连接正常</font>")
                self.wifi_connected = True

            # 更新计数器
            setattr(self, 'csi_data_counter', static_counter + 1)

    def get_next_sequence(self, user_name, action):
        """获取下一个序号"""
        folder = f"data/{action}"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            return 1
            
        files = os.listdir(folder)
        pattern = re.compile(rf"^{re.escape(action)}_{re.escape(user_name)}_(\d+)\.csv$")
        max_seq = 0
        for fname in files:
            m = pattern.match(fname)
            if m:
                try:
                    seq = int(m.group(1))
                    if seq > max_seq:
                        max_seq = seq
                except:
                    pass
        return max_seq + 1

    def slot_close(self):
        self.close()

    def show_curve_subcarrier(self):
        """显示CSI数据波形"""
        try:
            # 确保图表组件存在
            if not hasattr(self, 'graphicsView_subcarrier') or not self.graphicsView_subcarrier:
                print("警告: graphicsView_subcarrier组件不存在，无法更新CSI波形")
                return
                
            if not hasattr(self, 'curve_subcarrier') or not self.curve_subcarrier:
                print("警告: curve_subcarrier未初始化，无法更新CSI波形")
                return
                
            # 计算滤波参数
            wn = 20 / (CSI_SAMPLE_RATE / 2)
            butter_result = signal.butter(8, wn, 'lowpass')
            if butter_result is not None:
                b, a = butter_result[0], butter_result[1]
            else:
                b, a = [1], [1]

            # 应用中值滤波
            self.median_filtering(g_csi_phase_array)
            csi_filtfilt_data = signal.filtfilt(b, a, g_csi_phase_array.T).T

            # 更新显示范围
            data_min = np.nanmin(csi_filtfilt_data)
            data_max = np.nanmax(csi_filtfilt_data)
            if np.isnan(data_min):
                data_min = -20
            if np.isnan(data_max):
                data_max = 20

            # 设置显示范围
            self.graphicsView_subcarrier.setYRange(data_min - 2, data_max + 2, padding=0)

            # 更新曲线数据
            if len(self.curve_subcarrier) != csi_filtfilt_data.shape[1]:
                self.graphicsView_subcarrier.clear()
                self.graphicsView_subcarrier.addLegend()
                self.curve_subcarrier = []
                for i in range(min(CSI_DATA_COLUMNS, csi_filtfilt_data.shape[1])):
                    if i < len(csi_vaid_subcarrier_color):
                        curve = self.graphicsView_subcarrier.plot(csi_filtfilt_data[:, i], name=str(i),
                                                                  pen=csi_vaid_subcarrier_color[i])
                    else:
                        curve = self.graphicsView_subcarrier.plot(csi_filtfilt_data[:, i], name=str(i),
                                                                  pen=(255, 255, 255))
                    self.curve_subcarrier.append(curve)
            else:
                for i in range(min(len(self.curve_subcarrier), csi_filtfilt_data.shape[1])):
                    self.curve_subcarrier[i].setData(csi_filtfilt_data[:, i])

            # 更新RSSI数据
            if hasattr(self, 'curve_rssi') and self.curve_rssi:
                csi_filtfilt_rssi = signal.filtfilt(b, a, g_rssi_array).astype(np.int32)
                self.curve_rssi.setData(csi_filtfilt_rssi)
        except Exception as e:
            print(f"显示CSI数据异常: {e}")

    def median_filtering(self, waveform):
        """对CSI数据进行中值滤波"""
        tmp = waveform.copy()
        for i in range(1, waveform.shape[0] - 1):
            outliers_count = 0
            for j in range(waveform.shape[1]):
                if ((waveform[i - 1, j] - waveform[i, j] > 2 and waveform[i + 1, j] - waveform[i, j] > 2)
                        or (waveform[i - 1, j] - waveform[i, j] < -2 and waveform[i + 1, j] - waveform[i, j] < -2)):
                    outliers_count += 1
                    continue
            if outliers_count > 16:
                for x in range(1, waveform.shape[1] - 1):
                    tmp[i, x] = (waveform[i - 1, x] + waveform[i + 1, x]) / 2
        waveform[:] = tmp

    def update_predict_interval(self, value):
        """更新预测间隔"""
        self.predict_interval = value
        if self.predict_timer.isActive():
            self.predict_timer.setInterval(value)
            self.textBrowser_log.append(f"<font color='cyan'>预测间隔已更新为 {value}ms</font>")


def quit(signum, frame):
    print("Exit the system")
    sys.exit()


class DataHandleThread(QThread):
    signal_device_info = pyqtSignal(pd.Series)
    signal_log_msg = pyqtSignal(str)
    signal_exit = pyqtSignal()
    signal_wifi_status = pyqtSignal(str)
    signal_csi_data = pyqtSignal(pd.Series)  # 添加CSI数据信号

    def __init__(self, queue_read):
        super(DataHandleThread, self).__init__()
        self.queue_read = queue_read

    def run(self):
        while True:
            if not self.queue_read.empty():
                data_series = self.queue_read.get()

                if data_series['type'] == 'CSI_DATA':
                    self.handle_csi_data(data_series)
                    # 发送CSI数据到UI线程
                    self.signal_csi_data.emit(data_series)
                elif data_series['type'] == 'DEVICE_INFO':
                    self.signal_device_info.emit(data_series)
                    self.signal_log_msg.emit(
                        f"<font color='green'>接收到设备信息: {data_series['app_revision']}</font>")
                elif data_series['type'] == 'LOG_DATA':
                    # 获取日志数据
                    log_data = data_series['data']

                    # 特别处理WiFi和IP相关的日志
                    if ('wifi' in log_data.lower() or 'ip' in log_data.lower() or 'connected' in log_data.lower()
                            or 'disconnected' in log_data.lower() or '连接' in log_data or '断开' in log_data):
                        # 检测WiFi连接相关状态
                        wifi_connected = False

                        # 检查各种可能的WiFi连接成功标志
                        if 'wifi:state:' in log_data.lower() and 'run' in log_data.lower():
                            wifi_connected = True
                            self.signal_log_msg.emit(f"<font color='green'>WiFi连接成功! {log_data}</font>")
                        elif 'sta:' in log_data.lower() and (
                                'connected' in log_data.lower() or 'got ip' in log_data.lower()):
                            wifi_connected = True
                            self.signal_log_msg.emit(f"<font color='green'>WiFi连接成功: {log_data}</font>")
                        elif 'ip' in log_data.lower() and ('got' in log_data.lower() or 'assigned' in log_data.lower()):
                            wifi_connected = True
                            self.signal_log_msg.emit(f"<font color='green'>网络连接成功: {log_data}</font>")
                        elif 'wifi' in log_data.lower() and 'connected' in log_data.lower():
                            wifi_connected = True
                            self.signal_log_msg.emit(f"<font color='green'>WiFi连接成功: {log_data}</font>")
                        elif 'disconnected' in log_data.lower() or '断开' in log_data:
                            self.signal_log_msg.emit(f"<font color='yellow'>WiFi已断开: {log_data}</font>")
                        else:
                            self.signal_log_msg.emit(f"<font color='cyan'>WiFi状态: {log_data}</font>")

                        # 发送WiFi状态到处理函数
                        if wifi_connected:
                            self.signal_wifi_status.emit("WiFi已连接 " + log_data)
                        else:
                            self.signal_wifi_status.emit(log_data)
                    else:
                        # 显示其他日志信息
                        if data_series['tag'] == 'E':
                            self.signal_log_msg.emit(f"<font color='red'>{log_data}</font>")
                        elif data_series['tag'] == 'W':
                            self.signal_log_msg.emit(f"<font color='yellow'>{log_data}</font>")
                        else:
                            self.signal_log_msg.emit(f"<font color='white'>{log_data}</font>")
                elif data_series['type'] == 'FAIL_EVENT':
                    self.signal_log_msg.emit(
                        f"<font color='red'>{data_series['data']}</font>")
                    self.signal_exit.emit()
                    break

    def handle_csi_data(self, data):
        """处理CSI数据"""
        try:
            g_csi_phase_array[:-1] = g_csi_phase_array[1:]
            g_rssi_array[:-1] = g_rssi_array[1:]
            g_radio_header_pd.iloc[1:] = g_radio_header_pd.iloc[:-1]

            csi_raw_data = data['data']
            if not isinstance(csi_raw_data, list):
                try:
                    if isinstance(csi_raw_data, str):
                        if csi_raw_data.startswith('[') and csi_raw_data.endswith(']'):
                            # 尝试解析JSON字符串
                            try:
                                csi_raw_data = json.loads(csi_raw_data)
                            except json.JSONDecodeError:
                                # 静默处理JSON解析错误
                                csi_raw_data = base64_decode_bin(csi_raw_data)
                        else:
                            # 尝试Base64解码
                            csi_raw_data = base64_decode_bin(csi_raw_data)
                    else:
                        # 尝试转换为列表
                        try:
                            csi_raw_data = list(csi_raw_data)
                        except Exception:
                            csi_raw_data = []
                except Exception:
                    # 静默处理异常
                    csi_raw_data = []

            # 确保数据长度正确
            max_index = max([index * 2 for index in csi_vaid_subcarrier_index]) if csi_vaid_subcarrier_index else 104
            if not csi_raw_data:
                csi_raw_data = [0] * max_index
            elif len(csi_raw_data) < max_index:
                csi_raw_data.extend([0] * (max_index - len(csi_raw_data)))
            elif len(csi_raw_data) > max_index:
                csi_raw_data = csi_raw_data[:max_index]

            # 处理CSI数据
            for i in range(min(CSI_DATA_COLUMNS, len(csi_vaid_subcarrier_index))):
                try:
                    if i >= len(csi_vaid_subcarrier_index):
                        continue
                    index = csi_vaid_subcarrier_index[i]
                    if index * 2 >= len(csi_raw_data) or index * 2 - 1 < 0:
                        continue
                    real_part = csi_raw_data[index * 2]
                    imag_part = csi_raw_data[index * 2 - 1]
                    data_complex = complex(real_part, imag_part)
                    if i < g_csi_phase_array.shape[1]:
                        g_csi_phase_array[-1][i] = np.abs(data_complex)
                except Exception:
                    # 静默处理异常
                    if i < g_csi_phase_array.shape[1]:
                        g_csi_phase_array[-1][i] = 0

            # 处理RSSI数据
            try:
                g_rssi_array[-1] = int(data['rssi'])
            except (ValueError, TypeError):
                # 静默处理异常
                g_rssi_array[-1] = -100

            # 处理无线电头信息
            radio_header_data = {}
            for col in g_radio_header_pd.columns:
                try:
                    if col in data:
                        radio_header_data[col] = int(data[col])
                    else:
                        radio_header_data[col] = 0
                except (ValueError, TypeError):
                    radio_header_data[col] = 0
            g_radio_header_pd.loc[0] = pd.Series(radio_header_data)

        except Exception:
            # 静默处理异常
            # 尝试恢复到安全状态
            g_csi_phase_array[-1] = np.zeros(CSI_DATA_COLUMNS)
            g_rssi_array[-1] = -100


def parse_task_id(task_id):
    """
    从任务ID中解析出动作名称和序号
    例如: "walk_01" -> ("walk", "01")
         "lie_down_03" -> ("lie_down", "03")

    Args:
        task_id: 任务ID字符串

    Returns:
        tuple: (动作名称, 序号)
    """
    try:
        task_parts = task_id.split('_')
        if len(task_parts) < 2:
            # 格式不正确，返回默认值
            return task_id, "01"

        # 最后一部分应该是序号
        sequence = task_parts[-1]

        # 检查最后一部分是否是数字格式
        if sequence.isdigit() or (len(sequence) == 2 and sequence.isdigit()):
            # 合并前面所有部分作为动作名称
            action = '_'.join(task_parts[:-1])
            return action, sequence
        else:
            # 如果最后一部分不是数字，则整个字符串是动作名称，序号默认为01
            return task_id, "01"
    except Exception as e:
        print(f"解析任务ID异常: {e}")
        return task_id, "01"


def save_and_send_task_data(task_data_buffer, task_id, user_name, server_url, enable_server_save):
    if not task_data_buffer:
        print("task_data_buffer 为空，未保存或发送数据")
        return
    try:
        # 使用通用函数解析任务ID
        action, sequence = parse_task_id(task_id)

        folder = f"data/{action}"
        if not path.exists(folder):
            mkdir(folder)

        # 生成文件名
        filename = f"{action}_{user_name}_{sequence}.csv"
        filepath = os.path.join(folder, filename)

        print(f"保存数据到文件: {filepath}, 动作: {action}, 序号: {sequence}")

        # 保存到本地文件
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(task_data_buffer[0].keys())
            for row in task_data_buffer:
                writer.writerow(row.values())
        print(f"本地保存文件: {filename}, 数据量: {len(task_data_buffer)} 条")

        if enable_server_save:
            print(f"准备发送批量数据: {len(task_data_buffer)} 条")
            print(f"服务器地址: {server_url}")

            # 为每条数据添加元数据
            for data in task_data_buffer:
                data['user_name'] = user_name
                data['action'] = action
                data['sequence'] = sequence
                data['file_name'] = filename

            success = send_data_to_server(task_data_buffer, server_url)
            if success:
                print(f"服务器数据发送成功: {filename}")
            else:
                print(f"警告: 服务器数据发送失败: {filename}")
        else:
            print("服务器保存功能未启用，跳过发送")
    except Exception as e:
        print(f"保存任务数据时发生错误: {type(e).__name__}: {str(e)}")


def send_data_to_server(data, server_url):
    try:
        # 验证并规范化服务器URL
        server_url = server_url.strip()
        if not server_url:
            print("错误: 服务器URL为空")
            return False

        # 确保URL有正确的协议前缀
        if not server_url.startswith(('http://', 'https://')):
            server_url = 'http://' + server_url

        # 规范化URL路径
        server_url = server_url.rstrip('/')
        if not server_url.endswith('/api/csi_data'):
            server_url = server_url + '/api/csi_data'

        # 验证URL格式
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', server_url):
            print(f"错误: 无效的服务器URL格式: {server_url}")
            return False

        # 限制数据大小，防止内存问题
        max_batch_size = 1000
        if len(data) > max_batch_size:
            print(f"警告: 数据量过大 ({len(data)}条)，将分批发送 (每批{max_batch_size}条)")
            batches = [data[i:i + max_batch_size] for i in range(0, len(data), max_batch_size)]
        else:
            batches = [data]

        # 创建会话并配置
        session = requests.Session()
        session.trust_env = False

        # 配置重试参数
        max_retries = 3
        retry_delay = 2  # 秒
        success_count = 0

        # 分批处理数据
        for batch_idx, batch in enumerate(batches):
            batch_success = False

            for attempt in range(max_retries):
                try:
                    # 设置超时和请求头
                    headers = {'Content-Type': 'application/json'}
                    response = session.post(
                        server_url,
                        json=batch,
                        headers=headers,
                        timeout=15  # 增加超时时间
                    )

                    print(f"服务器响应状态码: {response.status_code}")

                    if response.status_code == 200:
                        batch_success = True
                        success_count += 1
                        break
                    else:
                        print(f"批次 {batch_idx + 1} 发送失败: HTTP {response.status_code}")
                        print(f"响应内容: {response.text[:200]}...")

                        # 特殊状态码处理
                        if response.status_code == 400:
                            print("服务器报告请求格式错误，检查数据格式")
                        elif response.status_code == 401 or response.status_code == 403:
                            print("服务器认证失败或拒绝访问")
                        elif response.status_code >= 500:
                            print("服务器内部错误，稍后重试")

                        if attempt == max_retries - 1:  # 最后一次尝试
                            print(f"批次 {batch_idx + 1} 已达到最大重试次数，放弃此批次")
                        else:
                            retry_time = retry_delay * (attempt + 1)
                            print(f"将在 {retry_time} 秒后重试...")
                            time.sleep(retry_time)

                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    print(f"网络错误: {type(e).__name__}: {str(e)}")
                    if attempt == max_retries - 1:
                        print(f"批次 {batch_idx + 1} 已达到最大重试次数，放弃此批次")
                    else:
                        retry_time = retry_delay * (attempt + 1)
                        print(f"将在 {retry_time} 秒后重试...")
                        time.sleep(retry_time)

                except Exception as e:
                    print(f"发送请求时发生未预期错误: {type(e).__name__}: {str(e)}")
                    if attempt == max_retries - 1:
                        print(f"批次 {batch_idx + 1} 已达到最大重试次数，放弃此批次")
                    else:
                        retry_time = retry_delay * (attempt + 1)
                        print(f"将在 {retry_time} 秒后重试...")
                        time.sleep(retry_time)

        # 返回总体成功状态
        if success_count == len(batches):
            return True
        elif success_count > 0:
            print(f"部分成功: {success_count}/{len(batches)} 个批次发送成功")
            return True  # 至少有一个批次成功就返回True
        else:
            print("所有数据批次发送失败")
            return False

    except Exception as e:
        print(f"发送数据到服务器时发生错误: {type(e).__name__}: {str(e)}")
        return False


def serial_handle(queue_read, queue_write, port):
    ser = None
    try:
        # 增加超时时间，提高稳定性
        ser = serial.Serial(port=port, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=0.5)
        print(f"串口成功打开: {port}, 波特率: 115200, 超时: 0.5秒")
    except Exception as e:
        print(f"串口打开失败: {e}")
        data_series = pd.Series(index=['type', 'data'], data=['FAIL_EVENT', f"无法打开串口: {e}"])
        queue_read.put(data_series)
        sys.exit()
        return
    print("打开串口: ", port)
    print("CSI数据过滤已禁用，将保存所有数据包")
    ser.reset_input_buffer()
    folder_list = ['log', 'data']
    for folder in folder_list:
        if not path.exists(folder):
            try:
                mkdir(folder)
            except Exception as e:
                print(f"创建文件夹失败: {folder}, 错误: {e}")
                data_series = pd.Series(index=['type', 'data'],
                                        data=['FAIL_EVENT', f"创建文件夹失败: {folder}, 错误: {e}"])
                queue_read.put(data_series)
                if ser:
                    ser.close()
                sys.exit()
                return
    data_valid_list = pd.DataFrame(
        columns=pd.Index(['type', 'columns_names', 'file_name', 'file_fd', 'file_writer']),
        data=[["CSI_DATA", CSI_DATA_COLUMNS_NAMES, "log/csi_data.csv", None, None],
              ["DEVICE_INFO", DEVICE_INFO_COLUMNS_NAMES, "log/device_info.csv", None, None]]
    )
    log_data_writer = None
    try:
        for data_valid in data_valid_list.iloc:
            data_valid['file_fd'] = open(data_valid['file_name'], 'w')
            data_valid['file_writer'] = csv.writer(data_valid['file_fd'])
            data_valid['file_writer'].writerow(data_valid['columns_names'])
        log_data_writer = open("log/log_data.txt", 'w+')
    except Exception as e:
        print(f"打开文件失败: {e}")
        data_series = pd.Series(index=['type', 'data'], data=['FAIL_EVENT', f"打开文件失败: {e}"])
        queue_read.put(data_series)
        if ser:
            ser.close()
        sys.exit()
        return

    # 使用混合方法：保留缓冲区但使用更简单的处理逻辑
    data_buffer = ""
    packet_lock = threading.Lock()  # 保留锁以确保线程安全
    taget_last = 'unknown'
    taget_seq_last = 0
    csi_target_data_file_fd = None
    taget_data_writer = None
    last_queue_full_warning_time = 0
    last_buffer_warning_time = 0
    current_user_name = "unknown"
    enable_server_save = False
    server_url = "http://8.136.10.160:12786/api/csi_data"
    created_files = {}
    task_data_buffer = []
    current_task_id = None
    current_sequence = 0

    # 缓冲区大小限制，使用适中的值
    MAX_BUFFER_SIZE = 5000  # 增大缓冲区大小
    TRUNCATE_SIZE = 2500  # 增大截断保留大小

    try:
        # 不发送重启命令，避免干扰数据接收
        print(f"当前服务器状态：URL={server_url}, 保存={enable_server_save}")
    except Exception as e:
        print(f"初始化异常: {e}")
        data_series = pd.Series(index=['type', 'data'], data=['FAIL_EVENT', f"初始化异常: {e}"])
        queue_read.put(data_series)
        if ser:
            ser.close()
        sys.exit()
        return

    while True:
        try:
            if not queue_write.empty():
                command = queue_write.get()
                if command == "exit":
                    for data_valid in data_valid_list.iloc:
                        if data_valid['file_fd']:
                            data_valid['file_fd'].close()
                    if log_data_writer:
                        log_data_writer.close()
                    if csi_target_data_file_fd:
                        csi_target_data_file_fd.close()
                    if ser:
                        ser.close()
                    sys.exit()
                    break
                if command.startswith("set_user:"):
                    current_user_name = command.split(":")[1]
                    print(f"设置当前用户为: {current_user_name}")
                    continue
                if command == "enable_server_save":
                    enable_server_save = True
                    print("已启用服务器保存")
                    continue
                if command == "disable_server_save":
                    enable_server_save = False
                    print("已禁用服务器保存")
                    continue
                if command.startswith("set_server_url:"):
                    new_url = command.split(":", 1)[1]
                    if new_url and new_url != "http":
                        server_url = new_url
                        print(f"设置服务器地址为: {server_url}")
                    continue
                if command.startswith("start_task:"):
                    current_task_id = command.split(":")[1]
                    action, sequence_str = parse_task_id(current_task_id)
                    try:
                        current_sequence = int(sequence_str)
                    except ValueError:
                        current_sequence = 1
                    task_data_buffer.clear()
                    print(f"开始新任务: {current_task_id}, 动作: {action}, 序号: {current_sequence}")
                    continue
                if command == "end_task" or command.startswith("end_task:"):
                    if task_data_buffer and current_task_id:
                        print(
                            f"处理end_task命令，准备保存和发送数据：任务ID={current_task_id}，数据量={len(task_data_buffer)}，服务器保存={enable_server_save}")
                        save_and_send_task_data(task_data_buffer, current_task_id, current_user_name, server_url,
                                                enable_server_save)
                    task_data_buffer.clear()
                    current_task_id = None
                    continue
                command = command + "\r\n"
                ser.write(command.encode('utf-8'))
                print(f"{datetime.now()}, 串口写入: {command}")
                continue

            # 读取串口数据
            strings = str(ser.readline())
            if not strings:
                continue
        except Exception as e:
            print(f"串口操作异常: {e}")
            data_series = pd.Series(index=['type', 'data'], data=['FAIL_EVENT', f"串口操作异常: {e}"])
            queue_read.put(data_series)
            for data_valid in data_valid_list.iloc:
                if data_valid['file_fd']:
                    data_valid['file_fd'].close()
            if log_data_writer:
                log_data_writer.close()
            if csi_target_data_file_fd:
                csi_target_data_file_fd.close()
            if ser:
                ser.close()
            sys.exit()
            break

        # 处理串口数据
        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        if not strings:
            continue

        # 混合处理方法：保留缓冲区但使用更简单的处理逻辑
        try:
            # 添加到缓冲区前检查大小
            if len(data_buffer) + len(strings) > MAX_BUFFER_SIZE:
                # 静默截断缓冲区，不输出警告
                data_buffer = data_buffer[-TRUNCATE_SIZE:]  # 只保留较少的字节

            # 添加到缓冲区
            data_buffer += strings

            # 处理数据包
            for data_valid in data_valid_list.iloc:
                index = data_buffer.find(data_valid['type'])
                if index >= 0:
                    # 找到下一个数据包开始位置
                    packet_start = index
                    next_packet = data_buffer.find(data_valid['type'], packet_start + 1)

                    # 如果没有找到下一个数据包，继续等待更多数据
                    if next_packet == -1:
                        continue

                    # 提取当前数据包
                    packet = data_buffer[packet_start:next_packet]
                    # 更新缓冲区，移除已处理的数据
                    data_buffer = data_buffer[next_packet:]

                    # 解析CSV数据
                    try:
                        csv_reader = csv.reader(StringIO(packet))
                        data = next(csv_reader)

                        # 检查数据是否完整
                        if len(data) == len(data_valid['columns_names']):
                            data_series = pd.Series(data, index=data_valid['columns_names'])

                            # 处理时间戳
                            try:
                                datetime.strptime(str(data_series['timestamp']), '%Y-%m-%d %H:%M:%S.%f')
                            except Exception as e:
                                data_series['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                            # 处理CSI数据
                            if data_series['type'] == 'CSI_DATA':
                                with packet_lock:
                                    try:
                                        # 检查并清理数据字段
                                        if 'data' in data_series:
                                            # 确保数据是字符串类型
                                            data_str = str(data_series['data'])
                                            # 检查数据是否包含非法字符
                                            if ':' in data_str or ',' in data_str:
                                                # 找到第一个非法字符的位置
                                                invalid_pos = min(
                                                    data_str.find(':') if data_str.find(':') != -1 else len(data_str),
                                                    data_str.find(',') if data_str.find(',') != -1 else len(data_str)
                                                )
                                                data_series['data'] = data_str[:invalid_pos]
                                        
                                        # 解码Base64数据
                                        try:
                                            csi_raw_data = base64_decode_bin(data_series['data'])
                                        except Exception:
                                            # 备用解码方法
                                            try:
                                                str_data = str(data_series['data']).strip()
                                                # 确保长度是4的倍数
                                                padding = 4 - (len(str_data) % 4) if len(str_data) % 4 != 0 else 0
                                                if padding:
                                                    str_data += '=' * padding
                                                csi_raw_data = list(base64.b64decode(str_data))
                                                # 处理大于127的值
                                                for i in range(len(csi_raw_data)):
                                                    if csi_raw_data[i] > 127:
                                                        csi_raw_data[i] = csi_raw_data[i] - 256
                                            except Exception:
                                                csi_raw_data = []

                                        # 检查数据长度
                                        if len(csi_raw_data) != int(data_series['len']):
                                            if len(csi_raw_data) < int(data_series['len']):
                                                csi_raw_data.extend([0] * (int(data_series['len']) - len(csi_raw_data)))
                                            elif len(csi_raw_data) > int(data_series['len']):
                                                csi_raw_data = csi_raw_data[:int(data_series['len'])]
                                    except Exception:
                                        csi_raw_data = [0] * 128
                                        data_series['len'] = len(csi_raw_data)

                                    # 更新数据系列
                                    data_series['data'] = csi_raw_data

                                    # 发送到队列
                                    if not queue_read.full():
                                        queue_read.put(data_series)
                                    # 队列满时静默丢弃数据，不输出警告

                                    # 检测是否是第一次接收到CSI数据
                                    static_counter = getattr(serial_handle, 'csi_data_counter', 0)
                                    if static_counter == 0:
                                        # 只在第一次检测到CSI数据时发送连接成功消息
                                        success_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                                   data=['LOG_DATA', 'I', datetime.now().strftime(
                                                                       '%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                                         "WiFi连接成功！开始接收CSI数据"])
                                        if not queue_read.full():
                                            queue_read.put(success_series)

                                    # 更新计数器
                                    setattr(serial_handle, 'csi_data_counter', static_counter + 1)

                                    # 处理任务数据
                                    if data_series['taget'] != 'unknown' and current_task_id:
                                        # 添加数据到缓冲区
                                        task_data_buffer.append(data_series.astype(str).to_dict())

                                        # 使用解析的任务ID
                                        action, sequence_str = parse_task_id(current_task_id)

                                        # 生成文件名
                                        sequence = f"{current_sequence:02d}"
                                        current_file_key = f"{action}_{current_user_name}_{sequence}"

                                        # 创建文件（如果需要）
                                        if current_file_key not in created_files:
                                            try:
                                                folder = f"data/{action}"
                                                if not path.exists(folder):
                                                    mkdir(folder)
                                                csi_target_data_file_name = f"{folder}/{current_file_key}.csv"
                                                csi_target_data_file_fd = open(csi_target_data_file_name, 'w+')
                                                taget_data_writer = csv.writer(csi_target_data_file_fd)
                                                taget_data_writer.writerow(data_series.index)
                                                created_files[current_file_key] = csi_target_data_file_fd
                                                print(f"创建新文件: {csi_target_data_file_name}")
                                            except Exception as e:
                                                print(f"创建文件失败: {e}")

                                        # 写入数据
                                        try:
                                            if current_file_key in created_files and created_files[current_file_key]:
                                                taget_data_writer = csv.writer(created_files[current_file_key])
                                                taget_data_writer.writerow(data_series.astype(str))
                                                created_files[current_file_key].flush()
                                        except Exception as e:
                                            print(f"写入文件失败: {e}, 文件: {current_file_key}")

                                    # 更新上一次目标信息
                                    taget_last = data_series['taget']
                                    taget_seq_last = data_series['taget_seq']
                            else:
                                # 非CSI数据直接发送到队列
                                queue_read.put(data_series)

                            # 写入文件
                            data_valid['file_writer'].writerow(data_series.astype(str))
                            data_valid['file_fd'].flush()
                            break
                    except Exception as e:
                        print(f"CSV解析异常: {e}")
                        continue
            else:
                # 处理日志数据
                if data_buffer.find("CSI_DATA") == -1 and data_buffer.find("DEVICE_INFO") == -1:
                    strings = re.sub(r'\\x1b.*?m', '', strings)
                    log_data_writer.writelines(strings + "\n")
                    log_data_writer.flush()  # 立即刷新日志

                    log = re.match(r'.*([DIWE]) \((\d+)\) (.*)', strings, re.I)
                    if log:
                        data_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                data=['LOG_DATA', log.group(1), log.group(2), log.group(3)])
                        if not queue_read.full():
                            queue_read.put(data_series)
                    # 特别处理WiFi和IP相关的日志，即使它们不符合标准日志格式
                    elif ('wifi' in strings.lower() or 'ip' in strings.lower() or 'connected' in strings.lower()
                          or 'disconnected' in strings.lower() or '连接' in strings or '断开' in strings):
                        # 检测WiFi连接相关状态
                        wifi_connected = False

                        # 检查各种可能的WiFi连接成功标志
                        if 'wifi:state:' in strings.lower() and 'run' in strings.lower():
                            wifi_connected = True
                        elif 'sta:' in strings.lower() and (
                                'connected' in strings.lower() or 'got ip' in strings.lower()):
                            wifi_connected = True
                        elif 'ip' in strings.lower() and ('got' in strings.lower() or 'assigned' in strings.lower()):
                            wifi_connected = True
                        elif 'wifi' in strings.lower() and 'connected' in strings.lower():
                            wifi_connected = True

                        # 根据连接状态创建数据系列
                        if wifi_connected:
                            # 这是WiFi连接成功的标志
                            data_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                    data=['LOG_DATA', 'I',
                                                          datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                          "WiFi连接成功! " + strings])

                            # 强制发送一个明确的连接成功消息，确保UI能够看到
                            success_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                       data=['LOG_DATA', 'I',
                                                             datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                             "WiFi已连接"])
                            if not queue_read.full():
                                queue_read.put(success_series)

                            # 如果检测到run状态，额外发送一个更明确的成功消息
                            if 'wifi:state:' in strings.lower() and 'run' in strings.lower():
                                extra_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                         data=['LOG_DATA', 'I',
                                                               datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                               "WiFi连接成功！ESP32已进入运行状态"])
                                if not queue_read.full():
                                    queue_read.put(extra_series)
                        else:
                            data_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                    data=['LOG_DATA', 'I',
                                                          datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                          strings])

                        if not queue_read.full():
                            queue_read.put(data_series)
        except Exception as e:
            print(f"处理数据异常: {e}")
            continue


if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print(" Python version should >= 3.6")
        exit()
    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device")
    parser.add_argument('--user_id', dest='user_id', type=str, default="01",
                        help="User ID for data collection (default=01)")
    parser.add_argument('--desc', dest='description', type=str, default="",
                        help="Optional description to add to data files")
    args = parser.parse_args()
    serial_port = args.port
    user_id = args.user_id
    description = args.description
    # 调整队列大小为中等值，避免太小或太大
    serial_queue_read = Queue(maxsize=128)
    serial_queue_write = Queue(maxsize=64)
    signal_key.signal(signal_key.SIGINT, quit)
    signal_key.signal(signal_key.SIGTERM, quit)
    serial_handle_process = Process(target=serial_handle, args=(serial_queue_read, serial_queue_write, serial_port))
    serial_handle_process.start()
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('../../../docs/_static/icon.png'))
    window = DataGraphicalWindow(serial_queue_read, serial_queue_write)
    data_handle_thread = DataHandleThread(serial_queue_read)
    data_handle_thread.signal_device_info.connect(window.show_device_info)
    data_handle_thread.signal_log_msg.connect(window.show_textBrowser_log)
    data_handle_thread.signal_exit.connect(window.slot_close)
    data_handle_thread.signal_wifi_status.connect(window.handle_wifi_status)
    data_handle_thread.signal_csi_data.connect(window.process_csi_data)  # 连接CSI数据信号
    data_handle_thread.start()
    window.show()
    exit_code = app.exec()
    serial_handle_process.join()
    sys.exit(exit_code)