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
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QProgressBar, QGroupBox, QCheckBox, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QComboBox, QTimeEdit, QSpinBox, QPushButton, QWidget, QErrorMessage, QBoxLayout, QTextBrowser, QGridLayout, QTabWidget)
from PyQt5.QtGui import QFont, QIcon
from esp_csi_tool_gui import Ui_MainWindow
from scipy import signal
import signal as signal_key
import socket
from pandas import Index
import pyqtgraph as pg

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
# 修改为多设备支持的数据结构
g_device_info_series = {}  # 使用字典存储不同设备的信息

CSI_DATA_INDEX = 500
CSI_DATA_COLUMNS = len(csi_vaid_subcarrier_index)
CSI_DATA_COLUMNS_NAMES = ["type", "seq", "timestamp", "taget_seq", "taget", "mac", "rssi", "rate", "sig_mode", "mcs",
                          "cwb", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor",
                          "ampdu_cnt", "channel_primary", "channel_secondary", "local_timestamp", "ant", "sig_len",
                          "rx_state", "agc_gain", "fft_gain", "len", "first_word_invalid", "data"]
CSI_DATA_TARGETS = ["unknown", "train", "lie_down", "walk", "stand", "bend", "sit_down",
                    "fall_from_stand", "fall_from_squat", "fall_from_bed"]

# 修改为多设备支持的数据结构
g_csi_phase_array = {}  # 使用字典存储不同设备的CSI数据
g_rssi_array = {}  # 使用字典存储不同设备的RSSI数据
g_radio_header_pd = {}  # 使用字典存储不同设备的无线电头信息

# 为默认设备初始化数据结构
g_csi_phase_array['esp32_1'] = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.int32)
g_rssi_array['esp32_1'] = np.zeros(CSI_DATA_INDEX, dtype=np.int8)
g_radio_header_pd['esp32_1'] = pd.DataFrame(np.zeros([10, len(CSI_DATA_COLUMNS_NAMES[1:-1])], dtype=np.int64),
                                 columns=pd.Index(CSI_DATA_COLUMNS_NAMES[1:-1]))

# 新增函数：为新设备初始化数据结构
def init_device_data(device_id):
    """为新设备初始化数据结构"""
    if device_id not in g_csi_phase_array:
        g_csi_phase_array[device_id] = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.int32)
    if device_id not in g_rssi_array:
        g_rssi_array[device_id] = np.zeros(CSI_DATA_INDEX, dtype=np.int8)
    if device_id not in g_radio_header_pd:
        g_radio_header_pd[device_id] = pd.DataFrame(np.zeros([10, len(CSI_DATA_COLUMNS_NAMES[1:-1])], dtype=np.int64),
                                       columns=pd.Index(CSI_DATA_COLUMNS_NAMES[1:-1]))
    return g_radio_header_pd[device_id]


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
    def __init__(self, serial_queues_read, serial_queues_write):
        super().__init__()
        # 支持多个设备的串口队列
        if isinstance(serial_queues_read, dict) and isinstance(serial_queues_write, dict):
            # 多设备模式
            self.serial_queues_read = serial_queues_read
            self.serial_queues_write = serial_queues_write
            self.multi_device_mode = True
        else:
            # 单设备模式（向后兼容）
            self.serial_queues_read = {'esp32_1': serial_queues_read}
            self.serial_queues_write = {'esp32_1': serial_queues_write}
            self.multi_device_mode = False
        
        self.active_device_id = 'esp32_1'  # 默认活跃设备
        self.server_url = "http://8.136.10.160:12786/api/csi_data"
        self.enable_server_predict = False
        self.wifi_connected = {}  # 每个设备的WiFi连接状态
        for device_id in self.serial_queues_read.keys():
            self.wifi_connected[device_id] = False
            
        self.csi_enabled = {}  # 每个设备的CSI启用状态
        for device_id in self.serial_queues_read.keys():
            self.csi_enabled[device_id] = False
            
        self.csi_buffer = {}  # 每个设备的CSI数据缓冲
        for device_id in self.serial_queues_read.keys():
            self.csi_buffer[device_id] = []
            
        self.buffer_size = 100  # 缓冲区大小，可以根据需要调整
        self.predict_interval = 500  # 预测间隔(ms)
        
        # 创建设备曲线数据结构（必须在setupUi之前）
        self.curve_subcarrier = {}
        self.curve_rssi = {}
        
        # 添加设备数据字典
        self.device_task_buffers = {}
        for device_id in self.serial_queues_read.keys():
            self.device_task_buffers[device_id] = []
            
        # 当前任务ID
        self.current_task_id = None
        
        # 用户名和任务设置（确保在setupUi前初始化）
        self.current_user_name = "user01"  # 默认用户名
        
        # 先初始化UI
        self.setupUi()
        
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

        # 启动实时数据显示定时器
        self.timer_curve_subcarrier = QTimer()
        self.timer_curve_subcarrier.timeout.connect(self.show_curve_subcarrier)
        self.timer_curve_subcarrier.setInterval(300)
        self.timer_curve_subcarrier.start()
        
        # 为每个设备创建定时器
        self.timer_boot_commands = {}
        for i, device_id in enumerate(self.serial_queues_read.keys()):
            self.timer_boot_commands[device_id] = QTimer()
            self.timer_boot_commands[device_id].timeout.connect(
                lambda dev_id=device_id: self.command_boot(dev_id))
            # 错开启动时间，避免同时发送命令
            startup_delay = 2000 + i * 2000
            self.timer_boot_commands[device_id].setInterval(startup_delay)
            self.timer_boot_commands[device_id].setSingleShot(True)  # 只触发一次
            self.timer_boot_commands[device_id].start()
            
        # 添加设备切换下拉框
        self.setup_device_switcher()
        
    def setupUi(self):
        """初始化UI界面"""
        # 设置UI组件
        super().setupUi(self)  # 调用父类的setupUi方法而不是自己的
        self.setWindowTitle("ESP CSI Tool - 多设备数据融合")
        
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

    def setup_device_switcher(self):
        """添加设备切换下拉框"""
        if not self.multi_device_mode or len(self.serial_queues_read.keys()) <= 1:
            return
            
        # 创建设备切换面板
        device_group = QGroupBox("设备切换")
        device_group.setFont(QFont("Arial", 10))
        device_layout = QHBoxLayout(device_group)
        
        # 添加下拉框
        self.device_selector = QComboBox()
        for device_id in self.serial_queues_read.keys():
            self.device_selector.addItem(f"设备 {device_id}")
        self.device_selector.currentIndexChanged.connect(self.switch_active_device)
        
        # 添加合并模式复选框
        self.merge_view_checkbox = QCheckBox("合并视图")
        self.merge_view_checkbox.stateChanged.connect(self.toggle_merge_view)
        
        # 添加设备状态标签
        self.device_status = {}
        for device_id in self.serial_queues_read.keys():
            self.device_status[device_id] = QLabel(f"{device_id}: 未连接")
            self.device_status[device_id].setStyleSheet("color: gray")
        
        # 添加到布局
        device_layout.addWidget(QLabel("选择设备:"))
        device_layout.addWidget(self.device_selector)
        device_layout.addWidget(self.merge_view_checkbox)
        
        # 添加状态标签
        status_layout = QHBoxLayout()
        for device_id in self.serial_queues_read.keys():
            status_layout.addWidget(self.device_status[device_id])
        
        # 创建垂直布局容器
        device_container = QVBoxLayout()
        device_container.addLayout(device_layout)
        device_container.addLayout(status_layout)
        
        # 设置设备组的布局
        device_group.setLayout(device_container)
        
        # 添加到主布局的顶部
        if self.main_layout and self.main_layout.count() > 0:
            self.main_layout.insertWidget(0, device_group)
        
    def switch_active_device(self, index):
        """切换活跃设备"""
        if index >= 0 and index < len(self.serial_queues_read):
            device_id = list(self.serial_queues_read.keys())[index]
            self.active_device_id = device_id
            self.textBrowser_log.append(f"<font color='cyan'>切换到设备: {device_id}</font>")
            
            # 更新图表显示
            self.update_curve_display()
    
    def toggle_merge_view(self, state):
        """切换是否显示合并视图"""
        self.update_curve_display()
        
        if state == 2:  # Qt.Checked
            self.textBrowser_log.append("<font color='cyan'>已启用合并视图，同时显示所有设备数据</font>")
        else:
            self.textBrowser_log.append(f"<font color='cyan'>已禁用合并视图，仅显示当前设备: {self.active_device_id}</font>")
    
    def update_curve_display(self):
        """根据当前模式更新图表显示"""
        # 实现在show_curve_subcarrier方法中
        pass
    
    def command_boot(self, device_id=None):
        """初始化设备，设置CSI格式并启用CSI功能"""
        if device_id is None:
            device_id = self.active_device_id
            
        # 简化日志输出
        self.textBrowser_log.append(f"<font color='cyan'>初始化设备 {device_id}</font>")
        
        # 确保此设备的数据结构已初始化
        if device_id not in g_csi_phase_array:
            init_device_data(device_id)
        
        # 设置CSI输出格式
        command = f"radar --csi_output_type LLFT --csi_output_format base64"
        
        if device_id in self.serial_queues_write:
            # 直接访问Queue对象，而不是通过字典访问put方法
            queue = self.serial_queues_write[device_id]
            if hasattr(queue, 'put'):
                try:
                    queue.put(command)
                    # 减少日志输出
                    
                    # 添加额外命令以确保CSI正确初始化
                    # 重置CSI (先禁用再启用)
                    QTimer.singleShot(200, lambda dev_id=device_id: self.reset_csi(dev_id))
                    
                    # 启用CSI功能
                    QTimer.singleShot(1000, lambda dev_id=device_id: self.enable_csi(dev_id))
                    
                    # 自动连接WiFi，且按钮状态同步
                    QTimer.singleShot(2000, lambda dev_id=device_id: self.command_router_connect(dev_id) 
                        if not self.wifi_connected.get(dev_id, False) and self.pushButton_router_connect.text() == "连接" 
                        else None)
                except Exception as e:
                    self.textBrowser_log.append(f"<font color='red'>设备 {device_id} 初始化失败: {str(e)}</font>")
                    
        # 停止此设备的启动定时器（已经不需要了）
        if device_id in self.timer_boot_commands and self.timer_boot_commands[device_id].isActive():
            self.timer_boot_commands[device_id].stop()
            
    def reset_csi(self, device_id=None):
        """重置CSI功能（先禁用）"""
        if device_id is None:
            device_id = self.active_device_id
            
        command = f"radar --csi_en 0"
        
        if device_id in self.serial_queues_write:
            # 直接访问Queue对象
            queue = self.serial_queues_write[device_id]
            if hasattr(queue, 'put'):
                try:
                    queue.put(command)
                    # 减少日志输出
                    self.csi_enabled[device_id] = False
                except Exception as e:
                    self.textBrowser_log.append(f"<font color='red'>设备 {device_id} 重置失败: {str(e)}</font>")

    def enable_csi(self, device_id=None):
        """启用 CSI 功能"""
        if device_id is None:
            device_id = self.active_device_id
            
        command = f"radar --csi_en 1"
        
        if device_id in self.serial_queues_write:
            # 直接访问Queue对象
            queue = self.serial_queues_write[device_id]
            if hasattr(queue, 'put'):
                try:
                    queue.put(command)
                    self.textBrowser_log.append(f"<font color='green'>设备 {device_id} CSI已启用</font>")
                    self.csi_enabled[device_id] = True
                    
                    # 添加调试命令，查看设备CSI状态
                    QTimer.singleShot(500, lambda dev_id=device_id: self.check_csi_status(dev_id))
                except Exception as e:
                    self.textBrowser_log.append(f"<font color='red'>设备 {device_id} 启用CSI失败: {str(e)}</font>")
                    
    def check_csi_status(self, device_id=None):
        """检查CSI状态"""
        if device_id is None:
            device_id = self.active_device_id
            
        command = f"radar --get_config"
        
        if device_id in self.serial_queues_write:
            # 直接访问Queue对象
            queue = self.serial_queues_write[device_id]
            if hasattr(queue, 'put'):
                try:
                    queue.put(command)
                    # 减少日志输出
                except Exception as e:
                    self.textBrowser_log.append(f"<font color='red'>检查设备 {device_id} 状态失败: {str(e)}</font>")

    def command_router_connect(self, device_id=None):
        """连接或断开WiFi"""
        if device_id is None:
            device_id = self.active_device_id
            
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

            self.textBrowser_log.append(f"<font color='cyan'>正在连接设备 {device_id} 的WiFi: {ssid}</font>")
            
            if device_id in self.serial_queues_write:
                # 直接访问Queue对象
                queue = self.serial_queues_write[device_id]
                if hasattr(queue, 'put'):
                    queue.put(command)

            # 设置WiFi连接标志
            self.wifi_connected[device_id] = True
            
            # 更新状态标签
            if hasattr(self, 'device_status') and device_id in self.device_status:
                self.device_status[device_id].setText(f"{device_id}: 正在连接")
                self.device_status[device_id].setStyleSheet("color: orange")
        else:
            self.pushButton_router_connect.setText("连接")
            self.pushButton_router_connect.setStyleSheet("color: black")

            if device_id in self.serial_queues_write:
                # 直接访问Queue对象
                queue = self.serial_queues_write[device_id]
                if hasattr(queue, 'put'):
                    command = "ping --abort"
                    queue.put(command)
                    command = "wifi_config --disconnect"
                    queue.put(command)
                
            self.textBrowser_log.append(f"<font color='yellow'>正在断开设备 {device_id} 的WiFi连接...</font>")

            # 重置WiFi连接标志
            self.wifi_connected[device_id] = False
            
            # 更新状态标签
            if hasattr(self, 'device_status') and device_id in self.device_status:
                self.device_status[device_id].setText(f"{device_id}: 已断开")
                self.device_status[device_id].setStyleSheet("color: red")

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
        
        if self.active_device_id in self.serial_queues_write:
            # 直接访问Queue对象
            queue = self.serial_queues_write[self.active_device_id]
            if hasattr(queue, 'put'):
                queue.put(command)
        else:
            self.textBrowser_log.append(f"<font color='red'>错误: 设备 {self.active_device_id} 未连接</font>")

    def process_csi_data(self, data_series):
        """处理接收到的CSI数据"""
        # 获取设备ID，默认使用活跃设备
        device_id = data_series.get('device_id', self.active_device_id)
        
        # --- 实时预测逻辑 ---
        # 仅当启用服务器预测时才执行
        if self.enable_server_predict and self.predict_timer.isActive():
            # 确保设备ID存在于缓冲区中
            if device_id not in self.csi_buffer:
                self.csi_buffer[device_id] = []
            
            try:
                # 提取CSI数据的关键特征
                csi_data = {
                    'timestamp': data_series.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]),
                    'rssi': int(data_series.get('rssi', -100)),
                    'mac': data_series.get('mac', ''),
                    'data': data_series.get('data', []),
                    'device_id': device_id  # 添加设备ID
                }
                
                # 将数据添加到对应设备的缓冲区
                self.csi_buffer[device_id].append(csi_data)
                
                # 如果缓冲区达到指定大小，触发预测
                if len(self.csi_buffer[device_id]) >= self.buffer_size:
                    self.predict_action(device_id)
                    
            except Exception as e:
                print(f"处理设备 {device_id} 的CSI数据异常: {e}")

        # --- 任务数据采集逻辑 ---
        # 仅当任务正在进行时才执行
        if self.current_task_id and data_series.get('taget', 'unknown') != 'unknown':
            if device_id in self.device_task_buffers:
                data_dict = data_series.astype(str).to_dict()
                # 确保设备ID存在
                data_dict['device_id'] = device_id
                self.device_task_buffers[device_id].append(data_dict)

    def predict_action(self, device_id=None):
        """发送CSI数据到服务器进行实时动作识别"""
        # 如果指定了设备ID，就使用该设备的数据，否则使用活跃设备的数据
        if device_id is None:
            device_id = self.active_device_id
        
        # 检查设备的缓冲区是否有数据
        if device_id not in self.csi_buffer or not self.csi_buffer[device_id]:
            return
            
        try:
            # 准备数据
            data = {
                'csi_data': self.csi_buffer[device_id],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'device_id': device_id
            }
            
            # 发送预测请求
            self.textBrowser_log.append(f"<font color='cyan'>发送设备 {device_id} 的 {len(self.csi_buffer[device_id])} 条CSI数据进行预测...</font>")
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
                    self.predictionLabel.setText(f"设备 {device_id} 当前动作: {action}")
                    
                    # 根据置信度设置不同颜色
                    if confidence > 0.8:
                        self.predictionLabel.setStyleSheet("color: lime; text-align: center;")
                    elif confidence > 0.5:
                        self.predictionLabel.setStyleSheet("color: yellow; text-align: center;")
                    else:
                        self.predictionLabel.setStyleSheet("color: orange; text-align: center;")
                        
                    self.textBrowser_log.append(
                        f"<font color='cyan'>设备 {device_id} 检测到动作: {action}, 置信度: {confidence:.2f}</font>")
            else:
                self.textBrowser_log.append(
                    f"<font color='yellow'>设备 {device_id} 预测请求失败: HTTP {response.status_code}</font>")
                
        except requests.exceptions.Timeout:
            self.textBrowser_log.append(f"<font color='yellow'>设备 {device_id} 预测请求超时</font>")
        except requests.exceptions.ConnectionError:
            self.textBrowser_log.append(f"<font color='yellow'>设备 {device_id} 无法连接到预测服务器</font>")
        except Exception as e:
            self.textBrowser_log.append(f"<font color='yellow'>设备 {device_id} 预测过程发生错误: {str(e)}</font>")
        
        # 清空缓冲区
        self.csi_buffer[device_id].clear()

    def handle_wifi_status(self, data, device_id=None):
        """处理WiFi状态信息，在连接成功后更新状态"""
        if device_id is None:
            device_id = self.active_device_id
            
        # 检查是否包含WiFi连接成功的信息
        if (
                'WiFi已连接' in data or 'connected' in data.lower() or 'wifi:state:' in data.lower() and 'run' in data.lower()):
            # 只显示连接成功的提示，不再重复启用CSI功能
            if not self.wifi_connected.get(device_id, False):
                self.textBrowser_log.append(f"<font color='green'>设备 {device_id} WiFi连接成功！</font>")
                self.wifi_connected[device_id] = True
                
                # 更新状态标签
                if hasattr(self, 'device_status') and device_id in self.device_status:
                    self.device_status[device_id].setText(f"{device_id}: 已连接")
                    self.device_status[device_id].setStyleSheet("color: lime")

        # 检查是否包含IP地址获取成功的信息
        if 'ip' in data.lower() and ('got' in data.lower() or '获取' in data or 'assigned' in data.lower()):
            # 尝试从日志中提取IP地址
            ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', data)
            if ip_match:
                ip_address = ip_match.group(0)
                self.textBrowser_log.append(f"<font color='green'>设备 {device_id} 网络连接成功！IP地址: {ip_address}</font>")
            else:
                self.textBrowser_log.append(f"<font color='green'>设备 {device_id} 网络连接成功！</font>")

        # 特别检测CSI数据接收，作为WiFi连接成功的额外指标
        elif "CSI_DATA" in data:
            # 如果接收到CSI数据，说明WiFi已经连接成功
            static_counter = getattr(self, f'csi_data_counter_{device_id}', 0)
            if static_counter == 0:
                # 只在第一次检测到CSI数据时发送连接成功消息
                self.textBrowser_log.append(f"<font color='green'>检测到设备 {device_id} 的CSI数据，WiFi连接正常</font>")
                self.wifi_connected[device_id] = True
                
                # 更新状态标签
                if hasattr(self, 'device_status') and device_id in self.device_status:
                    self.device_status[device_id].setText(f"{device_id}: 数据接收中")
                    self.device_status[device_id].setStyleSheet("color: cyan")

            # 更新计数器
            setattr(self, f'csi_data_counter_{device_id}', static_counter + 1)

    def closeEvent(self, event):
        # 向所有设备发送退出命令
        for device_id, queue in self.serial_queues_write.items():
            try:
                if hasattr(queue, 'put'):
                    queue.put("exit")
            except Exception as e:
                print(f"关闭设备 {device_id} 时出错: {e}")
                
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
        if state == 2:  # Qt.Checked
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

            # 发送命令到当前活跃设备
            if self.active_device_id in self.serial_queues_write:
                queue = self.serial_queues_write[self.active_device_id]
                if hasattr(queue, 'put'):
                    queue.put("enable_server_save")
                    queue.put(f"set_server_url:{server_url}")

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
            # 发送命令到当前活跃设备
            if self.active_device_id in self.serial_queues_write:
                queue = self.serial_queues_write[self.active_device_id]
                if hasattr(queue, 'put'):
                    queue.put("disable_server_save")
                    
            self.enable_server_save = False
            self.textBrowser_log.append(f"<font color='yellow'>已禁用服务器保存</font>")

    def update_progress(self, current, total):
        """更新进度条显示"""
        progress = int((current / total) * 100)
        self.progressBar.setValue(progress)
        if progress % 10 == 0:  # 每10%更新一次日志，避免日志过多
            self.textBrowser_log.append(f"<font color='cyan'>采集进度: {progress}%</font>")

    def show_router_auto_connect(self):
        with open("./config/gui_config.json", "r") as file:
            gui_config = json.load(file)
            gui_config['router_auto_connect'] = self.checkBox_router_auto_connect.isChecked()
        with open("./config/gui_config.json", "w") as file:
            json.dump(gui_config, file)

    def show_textBrowser_log(self, str):
        self.textBrowser_log.append(str)

    def show_device_info(self, device_info_series, device_id=None):
        """显示设备信息"""
        if device_id is None:
            device_id = self.active_device_id
            
        # 保存设备信息
        if device_id not in g_device_info_series:
            g_device_info_series[device_id] = device_info_series
        else:
            g_device_info_series[device_id] = device_info_series
            
        if device_info_series['type'] == 'DEVICE_INFO':
            self.textBrowser_log.append(
                f"<font color='green'>设备 {device_id} 连接成功: {device_info_series['app_revision']}</font>")
            
            # 更新设备状态标签
            if hasattr(self, 'device_status') and device_id in self.device_status:
                self.device_status[device_id].setText(f"{device_id}: 已连接")
                self.device_status[device_id].setStyleSheet("color: green")

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
            # 创建多设备标签页界面
            if self.multi_device_mode and len(self.serial_queues_read.keys()) > 1:
                from PyQt5.QtWidgets import QTabWidget
                
                # 创建标签页组件
                self.devices_tab = QTabWidget()
                self.devices_tab.setFont(QFont("Arial", 10))
                # 设置标签页最小高度，确保内容完整显示
                self.devices_tab.setMinimumHeight(500)
                
                # 创建合并视图标签页，使用原始图表
                all_devices_widget = QWidget()
                all_layout = QVBoxLayout(all_devices_widget)
                # 设置合适的间距
                all_layout.setContentsMargins(5, 5, 5, 5)
                all_layout.setSpacing(5)
                
                # 将原始图表添加到合并视图标签页
                if hasattr(self, 'groupBox_subcarrier') and self.groupBox_subcarrier:
                    all_layout.addWidget(self.groupBox_subcarrier)
                if hasattr(self, 'groupBox_rssi') and self.groupBox_rssi:
                    all_layout.addWidget(self.groupBox_rssi)
                
                # 添加合并视图标签页
                self.devices_tab.addTab(all_devices_widget, "所有设备")
                
                # 为每个设备创建独立标签页
                self.device_views = {}
                for device_id in self.serial_queues_read.keys():
                    # 创建设备页
                    device_widget = QWidget()
                    device_layout = QVBoxLayout(device_widget)
                    # 设置合适的间距
                    device_layout.setContentsMargins(5, 5, 5, 5)
                    device_layout.setSpacing(10)
                    
                    # 复制原始图表的容器
                    device_subcarrier_group = QGroupBox(f"{device_id} 子载波")
                    device_subcarrier_group.setFont(QFont("Arial", 10, QFont.Bold))
                    device_subcarrier_layout = QVBoxLayout(device_subcarrier_group)
                    device_subcarrier_layout.setContentsMargins(5, 15, 5, 5)
                    device_subcarrier_layout.setSpacing(0)
                    
                    # 创建子载波图表
                    device_graphicsView_subcarrier = self.graphicsView_subcarrier.__class__()
                    device_graphicsView_subcarrier.setYRange(-20, 20)
                    device_graphicsView_subcarrier.addLegend()
                    device_graphicsView_subcarrier.setMouseEnabled(x=True, y=False)
                    device_graphicsView_subcarrier.enableAutoRange(axis='y', enable=False)
                    device_graphicsView_subcarrier.setMenuEnabled(False)
                    # 设置更大的图表尺寸，确保完整显示
                    device_graphicsView_subcarrier.setMinimumHeight(250)
                    device_graphicsView_subcarrier.setMinimumWidth(800)
                    device_subcarrier_layout.addWidget(device_graphicsView_subcarrier)
                    
                    # 创建RSSI图表
                    device_rssi_group = QGroupBox(f"{device_id} RSSI")
                    device_rssi_group.setFont(QFont("Arial", 10, QFont.Bold))
                    device_rssi_layout = QVBoxLayout(device_rssi_group)
                    device_rssi_layout.setContentsMargins(5, 15, 5, 5)
                    device_rssi_layout.setSpacing(0)
                    device_graphicsView_rssi = self.graphicsView_rssi.__class__()
                    device_graphicsView_rssi.setYRange(-100, 0)
                    device_graphicsView_rssi.setMouseEnabled(x=True, y=False)
                    device_graphicsView_rssi.enableAutoRange(axis='y', enable=False)
                    device_graphicsView_rssi.setMenuEnabled(False)
                    # 设置更大的图表尺寸，确保完整显示
                    device_graphicsView_rssi.setMinimumHeight(150)
                    device_graphicsView_rssi.setMinimumWidth(800)
                    device_rssi_layout.addWidget(device_graphicsView_rssi)
                    
                    # 添加组件到设备标签页
                    device_layout.addWidget(device_subcarrier_group)
                    device_layout.addWidget(device_rssi_group)
                    
                    # 保存设备视图引用
                    self.device_views[device_id] = {
                        'widget': device_widget,
                        'subcarrier': device_graphicsView_subcarrier,
                        'rssi': device_graphicsView_rssi
                    }
                    
                    # 添加设备标签页
                    self.devices_tab.addTab(device_widget, f"设备 {device_id}")
                
                # 添加标签页切换事件
                self.devices_tab.currentChanged.connect(self.on_tab_changed)
                
                # 将标签页组件添加到主布局
                main_layout_index = 3  # 假设前三个位置被其他组件占用
                if self.main_layout.count() > main_layout_index:
                    self.main_layout.insertWidget(main_layout_index, self.devices_tab)
                else:
                    self.main_layout.addWidget(self.devices_tab)
            
            # 配置子载波图
            self.graphicsView_subcarrier.setYRange(-20, 20)
            self.graphicsView_subcarrier.addLegend()
            self.graphicsView_subcarrier.setMouseEnabled(x=True, y=False)
            self.graphicsView_subcarrier.enableAutoRange(axis='y', enable=False)
            self.graphicsView_subcarrier.setMenuEnabled(False)
            # 设置更大的图表尺寸，确保完整显示
            self.graphicsView_subcarrier.setMinimumHeight(250)
            self.graphicsView_subcarrier.setMinimumWidth(800)
            
            # 配置RSSI图
            self.graphicsView_rssi.setYRange(-100, 0)
            self.graphicsView_rssi.setMouseEnabled(x=True, y=False)
            self.graphicsView_rssi.enableAutoRange(axis='y', enable=False)
            self.graphicsView_rssi.setMenuEnabled(False)
            # 设置更大的图表尺寸，确保完整显示
            self.graphicsView_rssi.setMinimumHeight(150)
            self.graphicsView_rssi.setMinimumWidth(800)
            
            # 为每个设备创建曲线
            for device_id in self.serial_queues_read.keys():
                # 初始化数据结构（如果尚未初始化）
                if device_id not in g_csi_phase_array:
                    init_device_data(device_id)
                
                # 主视图中的子载波曲线
                self.curve_subcarrier[device_id] = []
                for i in range(CSI_DATA_COLUMNS):
                    # 为不同设备设置不同颜色系列
                    if device_id == 'esp32_1':
                        # 第一个设备使用红绿色系
                        color = csi_vaid_subcarrier_color[i]
                    else:
                        # 其他设备使用蓝黄色系
                        r, g, b = csi_vaid_subcarrier_color[i]
                        color = (b, g, 255-r)  # 交换红蓝通道
                    
                    # 创建曲线并添加设备标识
                    curve = self.graphicsView_subcarrier.plot(
                        g_csi_phase_array[device_id][:, i], 
                        name=f"{device_id}-{i}",
                        pen=color
                    )
                    self.curve_subcarrier[device_id].append(curve)
                
                # 主视图中的RSSI曲线
                if device_id == 'esp32_1':
                    self.curve_rssi[device_id] = self.graphicsView_rssi.plot(
                        g_rssi_array[device_id], 
                        name=f'rssi-{device_id}', 
                        pen=(255, 255, 255)
                    )
                else:
                    # 第二个设备使用黄色
                    self.curve_rssi[device_id] = self.graphicsView_rssi.plot(
                        g_rssi_array[device_id], 
                        name=f'rssi-{device_id}', 
                        pen=(255, 255, 0)
                    )
                
                # 如果是多设备模式，还要为每个设备的独立视图创建曲线
                if self.multi_device_mode and hasattr(self, 'device_views') and device_id in self.device_views:
                    device_view = self.device_views[device_id]
                    
                    # 为设备独立视图创建子载波曲线
                    device_curve_subcarrier = []
                    for i in range(CSI_DATA_COLUMNS):
                        # 在单设备视图中显示更多数据点
                        color = csi_vaid_subcarrier_color[i]
                        curve = device_view['subcarrier'].plot(
                            g_csi_phase_array[device_id][:, i], 
                            name=f"SC-{i}",
                            pen=color
                        )
                        device_curve_subcarrier.append(curve)
                    
                    # 为设备独立视图创建RSSI曲线
                    device_curve_rssi = device_view['rssi'].plot(
                        g_rssi_array[device_id], 
                        name='RSSI', 
                        pen=(255, 255, 255)
                    )
                    
                    # 保存曲线引用
                    device_view['curve_subcarrier'] = device_curve_subcarrier
                    device_view['curve_rssi'] = device_curve_rssi
                    
        except Exception as e:
            print(f"配置图表时出错: {e}")
            
    def on_tab_changed(self, index):
        """处理标签页切换事件"""
        if hasattr(self, 'devices_tab'):
            tab_title = self.devices_tab.tabText(index)
            
            if tab_title == "所有设备":
                # 切换到合并视图模式
                if hasattr(self, 'merge_view_checkbox'):
                    self.merge_view_checkbox.setChecked(True)
            else:
                # 切换到单设备模式
                device_id = tab_title.replace("设备 ", "")
                self.active_device_id = device_id
                
                # 更新设备选择器
                if hasattr(self, 'device_selector'):
                    device_ids = list(self.serial_queues_read.keys())
                    if device_id in device_ids:
                        self.device_selector.setCurrentIndex(device_ids.index(device_id))
                
                # 禁用合并视图
                if hasattr(self, 'merge_view_checkbox'):
                    self.merge_view_checkbox.setChecked(False)

    def show_curve_subcarrier(self):
        """显示CSI数据波形"""
        try:
            # 确保图表组件存在
            if not hasattr(self, 'graphicsView_subcarrier') or not self.graphicsView_subcarrier:
                return
            
            # 检查当前显示模式（单设备或合并模式）
            show_all_devices = hasattr(self, 'merge_view_checkbox') and self.merge_view_checkbox and self.merge_view_checkbox.isChecked()
            
            # 计算滤波参数
            wn = 20 / (CSI_SAMPLE_RATE / 2)
            butter_result = signal.butter(8, wn, 'lowpass')
            if butter_result is not None:
                b, a = butter_result[0], butter_result[1]
            else:
                b, a = [1], [1]
            
            # 准备处理数据的设备列表
            devices_to_process = list(self.serial_queues_read.keys())
            
            # 处理每个设备的数据
            for device_id in devices_to_process:
                # 跳过不存在的设备数据
                if device_id not in g_csi_phase_array:
                    continue
                
                # 应用中值滤波
                self.median_filtering(g_csi_phase_array[device_id])
                csi_filtfilt_data = signal.filtfilt(b, a, g_csi_phase_array[device_id].T).T
                
                # 更新单个设备的独立视图（如果存在）
                if self.multi_device_mode and hasattr(self, 'device_views') and device_id in self.device_views:
                    device_view = self.device_views[device_id]
                    
                    if 'curve_subcarrier' in device_view and 'curve_rssi' in device_view:
                        # 更新子载波曲线
                        for i, curve in enumerate(device_view['curve_subcarrier']):
                            if i < csi_filtfilt_data.shape[1]:
                                curve.setData(csi_filtfilt_data[:, i])
                        
                        # 获取当前设备数据的显示范围
                        device_min = np.nanmin(csi_filtfilt_data)
                        device_max = np.nanmax(csi_filtfilt_data)
                        
                        # 防止数据异常导致显示范围不正确
                        if np.isnan(device_min) or device_min == float('inf'):
                            device_min = -20
                        if np.isnan(device_max) or device_max == float('-inf'):
                            device_max = 20
                            
                        # 设置单设备图表的显示范围，添加余量使图像完整显示
                        device_view['subcarrier'].setYRange(device_min - 5, device_max + 5)
                        
                        # 更新RSSI曲线
                        csi_filtfilt_rssi = signal.filtfilt(b, a, g_rssi_array[device_id]).astype(np.int32)
                        device_view['curve_rssi'].setData(csi_filtfilt_rssi)
                        
                        # 设置更合适的RSSI显示范围
                        rssi_min = np.nanmin(csi_filtfilt_rssi)
                        rssi_max = np.nanmax(csi_filtfilt_rssi)
                        if not np.isnan(rssi_min) and not np.isnan(rssi_max):
                            device_view['rssi'].setYRange(min(rssi_min - 5, -100), max(rssi_max + 5, -20))
                
                # 主视图显示范围初始化
                data_min = float('inf')
                data_max = float('-inf')
                
                # 更新显示范围
                curr_min = np.nanmin(csi_filtfilt_data)
                curr_max = np.nanmax(csi_filtfilt_data)
                
                if not np.isnan(curr_min) and curr_min < data_min:
                    data_min = curr_min
                if not np.isnan(curr_max) and curr_max > data_max:
                    data_max = curr_max
                
                # 更新主视图的曲线数据
                if device_id in self.curve_subcarrier and device_id in g_csi_phase_array:
                    device_curves = self.curve_subcarrier[device_id]
                    
                    # 如果曲线数量与数据不匹配，需要重新创建
                    if len(device_curves) != csi_filtfilt_data.shape[1]:
                        # 重新创建此设备的曲线
                        for curve in device_curves:
                            self.graphicsView_subcarrier.removeItem(curve)
                        
                        device_curves = []
                        for i in range(csi_filtfilt_data.shape[1]):
                            # 选择设备特定的颜色
                            if device_id == 'esp32_1':
                                color = csi_vaid_subcarrier_color[i] if i < len(csi_vaid_subcarrier_color) else (255, 255, 255)
                            else:
                                r, g, b = csi_vaid_subcarrier_color[i] if i < len(csi_vaid_subcarrier_color) else (255, 255, 255)
                                color = (b, g, 255-r)  # 交换红蓝通道
                                
                            curve = self.graphicsView_subcarrier.plot(
                                csi_filtfilt_data[:, i], 
                                name=f"{device_id}-{i}",
                                pen=color
                            )
                            device_curves.append(curve)
                            
                        self.curve_subcarrier[device_id] = device_curves
                    else:
                        # 更新现有曲线
                        for i in range(min(len(device_curves), csi_filtfilt_data.shape[1])):
                            # 如果是单设备模式或曲线是活跃设备的，则更新；否则隐藏
                            if show_all_devices or device_id == self.active_device_id:
                                device_curves[i].setData(csi_filtfilt_data[:, i])
                                device_curves[i].setPen(device_curves[i].opts['pen'])  # 确保曲线可见
                            else:
                                # 在单设备模式下隐藏非活跃设备的曲线
                                device_curves[i].setData([0])  # 使用空数据隐藏曲线
                
                # 更新RSSI数据
                if device_id in self.curve_rssi and device_id in g_rssi_array:
                    csi_filtfilt_rssi = signal.filtfilt(b, a, g_rssi_array[device_id]).astype(np.int32)
                    
                    # 如果是合并模式或是当前活跃设备，则更新曲线
                    if show_all_devices or device_id == self.active_device_id:
                        self.curve_rssi[device_id].setData(csi_filtfilt_rssi)
                        self.curve_rssi[device_id].setPen(self.curve_rssi[device_id].opts['pen'])  # 确保曲线可见
                    else:
                        # 在单设备模式下隐藏非活跃设备的曲线
                        self.curve_rssi[device_id].setData([0])
                
                # 防止数据异常导致显示范围不正确
                if np.isnan(data_min) or data_min == float('inf'):
                    data_min = -20
                if np.isnan(data_max) or data_max == float('-inf'):
                    data_max = 20
                
                # 设置显示范围（添加小的边距）
                if hasattr(self, 'graphicsView_subcarrier') and self.graphicsView_subcarrier:
                    self.graphicsView_subcarrier.setYRange(data_min - 2, data_max + 2)
            
        except Exception:
            # 静默处理异常
            pass

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
        self.pushButton_router_connect.released.connect(lambda: self.command_router_connect())
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
        # 使用csi_data接口作为默认服务器地址，方便数据采集
        data_url = "http://8.136.10.160:12786/api/csi_data"
        self.lineEdit_server_url.setText(data_url)
        self.lineEdit_server_url.setPlaceholderText("输入数据服务器地址")
        server_url_layout.addWidget(server_label)
        server_url_layout.addWidget(self.lineEdit_server_url)
        
        # 启用预测复选框
        self.checkBox_server_predict = QCheckBox("启用实时预测")
        self.checkBox_server_predict.stateChanged.connect(self.toggle_server_predict)
        server_url_layout.addWidget(self.checkBox_server_predict)
        
        # 添加到服务器设置组
        server_layout.addLayout(server_url_layout)
        
        # 添加启用服务器保存选项
        server_save_layout = QHBoxLayout()
        self.checkBox_server_save = QCheckBox("保存数据到服务器")
        self.checkBox_server_save.stateChanged.connect(self.toggle_server_save)
        server_save_layout.addWidget(self.checkBox_server_save)
        server_layout.addLayout(server_save_layout)
        
        # 添加到主布局
        if self.main_layout.count() > 1:
            self.main_layout.insertWidget(1, server_group)
        else:
            self.main_layout.addWidget(server_group)
            
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
        
        # 添加任务采集控制面板
        task_group = QGroupBox("数据采集任务")
        task_group.setFont(QFont("Arial", 10))
        task_layout = QVBoxLayout(task_group)
        
        # 任务ID输入
        task_id_layout = QHBoxLayout()
        task_id_label = QLabel("任务ID:")
        self.task_id_input = QLineEdit()
        self.task_id_input.setPlaceholderText("动作名称_序号 (例如: walk_01)")
        
        task_id_layout.addWidget(task_id_label)
        task_id_layout.addWidget(self.task_id_input)
        
        # 用户ID输入
        user_id_layout = QHBoxLayout()
        user_id_label = QLabel("用户ID:")
        self.user_id_input = QLineEdit()
        self.user_id_input.setText(self.current_user_name)
        self.user_id_input.setPlaceholderText("输入用户ID")
        self.user_id_input.textChanged.connect(self.update_user_id)
        
        user_id_layout.addWidget(user_id_label)
        user_id_layout.addWidget(self.user_id_input)
        
        # 在任务ID输入上方添加动作类型选择
        action_type_layout = QHBoxLayout()
        action_type_label = QLabel("动作类型:")
        self.action_type_combo = QComboBox()
        self.action_type_combo.addItems([
            "walk", "sit", "stand", "lie_down", "bend", "fall_from_stand", "fall_from_squat", "fall_from_bed"
        ])
        action_type_layout.addWidget(action_type_label)
        action_type_layout.addWidget(self.action_type_combo)
        task_layout.addLayout(action_type_layout)
        
        # 添加到任务布局
        task_layout.addLayout(task_id_layout)
        task_layout.addLayout(user_id_layout)
        
        # 添加任务控制按钮
        task_buttons_layout = QHBoxLayout()
        
        # 开始任务按钮
        self.start_task_button = QPushButton("开始采集")
        self.start_task_button.clicked.connect(self.start_task)
        self.start_task_button.setStyleSheet("background-color: green; color: white;")
        
        # 结束任务按钮
        self.end_task_button = QPushButton("结束采集")
        self.end_task_button.clicked.connect(self.end_task)
        self.end_task_button.setStyleSheet("background-color: red; color: white;")
        self.end_task_button.setEnabled(False)  # 初始时禁用
        
        task_buttons_layout.addWidget(self.start_task_button)
        task_buttons_layout.addWidget(self.end_task_button)
        
        task_layout.addLayout(task_buttons_layout)
        
        # 将任务面板添加到主布局
        self.main_layout.addWidget(task_group)
        
        # 将检测面板添加到主布局
        self.main_layout.addWidget(self.detection_group)
        
    def update_user_id(self, text):
        """更新用户ID"""
        self.current_user_name = text
        
    def start_task(self):
        """点击开始任务按钮的处理函数"""
        task_id = self.task_id_input.text().strip()
        
        if not task_id:
            QMessageBox.warning(self, "错误", "请输入任务ID")
            return
            
        # 禁用开始按钮，启用结束按钮
        self.start_task_button.setEnabled(False)
        self.end_task_button.setEnabled(True)
        
        # 按钮视觉反馈
        self.start_task_button.setText("采集中...")
        self.start_task_button.setStyleSheet("background-color: lime; color: black;")
        self.end_task_button.setText("结束采集")
        self.end_task_button.setStyleSheet("background-color: red; color: white;")
        
        # 日志反馈
        self.textBrowser_log.append(f"<font color='lime'>【提示】已开始采集任务: {task_id}</font>")
        
        # 开始任务
        self.start_task_collection(task_id)
        
    def end_task(self):
        """点击结束任务按钮的处理函数"""
        # 启用开始按钮，禁用结束按钮
        self.start_task_button.setEnabled(True)
        self.end_task_button.setEnabled(False)
        
        # 按钮视觉反馈
        self.start_task_button.setText("开始采集")
        self.start_task_button.setStyleSheet("background-color: green; color: white;")
        self.end_task_button.setText("已结束")
        self.end_task_button.setStyleSheet("background-color: gray; color: white;")
        
        # 日志反馈
        self.textBrowser_log.append(f"<font color='orange'>【提示】已结束采集任务</font>")
        
        # 结束任务
        self.end_task_collection()

    def toggle_detection(self):
        """切换检测开始/停止状态"""
        if self.toggleDetectionButton.text() == "开始检测":
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        """开始动作检测"""
        # 检查是否有连接的设备
        connected_devices = [device_id for device_id, status in self.wifi_connected.items() if status]
        if not connected_devices:
            QMessageBox.warning(self, "错误", "请先连接至少一个设备的WiFi")
            return
            
        if not self.enable_server_predict:
            QMessageBox.warning(self, "错误", "请先启用服务器预测")
            return
            
        self.toggleDetectionButton.setText("停止检测")
        self.toggleDetectionButton.setStyleSheet("color: red")
        self.predict_timer.start()
        self.textBrowser_log.append("<font color='green'>开始实时动作检测</font>")

    def stop_detection(self):
        """停止动作检测"""
        self.toggleDetectionButton.setText("开始检测")
        self.toggleDetectionButton.setStyleSheet("color: black")
        self.predict_timer.stop()
        
        # 清空所有设备的缓冲区
        for device_id in self.csi_buffer:
            self.csi_buffer[device_id].clear()
            
        self.predictionLabel.setText("当前动作: -")
        self.textBrowser_log.append("<font color='yellow'>停止实时动作检测</font>")
        
    def toggle_server_predict(self, state):
        """切换是否启用服务器预测功能"""
        if state == 2:  # Qt.Checked
            server_url = self.lineEdit_server_url.text().strip()
            if not server_url:
                QMessageBox.warning(self, "服务器地址错误", "服务器地址不能为空")
                self.checkBox_server_predict.setChecked(False)
                return
                
            # 验证URL格式
            if not server_url.startswith(('http://', 'https://')):
                server_url = 'http://' + server_url
                
            # 规范化URL，确保使用predict接口
            base_url = server_url.rstrip('/').replace('/api/csi_data', '')
            predict_url = base_url + '/api/predict'
                
            # 更新服务器地址
            self.lineEdit_server_url.setText(predict_url)
            self.server_url = predict_url
            self.enable_server_predict = True
            
            # 测试服务器连接
            try:
                test_url = base_url + '/api/test_connection'
                response = requests.get(test_url, timeout=5)
                if response.status_code == 200:
                    self.textBrowser_log.append(f"<font color='green'>预测服务器连接成功: {predict_url}</font>")
                else:
                    self.textBrowser_log.append(
                        f"<font color='yellow'>警告: 服务器返回异常状态码: {response.status_code}</font>")
            except Exception as e:
                self.textBrowser_log.append(f"<font color='yellow'>警告: 服务器连接测试失败: {str(e)}</font>")
                
        else:
            # 禁用预测功能时恢复csi_data接口
            base_url = self.server_url.rstrip('/').replace('/api/predict', '')
            data_url = base_url + '/api/csi_data'
            self.lineEdit_server_url.setText(data_url)
            self.server_url = data_url
            
            self.enable_server_predict = False
            self.stop_detection()  # 停止检测
            self.textBrowser_log.append("<font color='yellow'>已禁用实时预测</font>")

    def handle_csi_data(self, data, device_id=None):
        """处理CSI数据"""
        if device_id is None:
            device_id = self.active_device_id
            
        try:
            # 确保设备数据结构已初始化
            if device_id not in g_csi_phase_array:
                init_device_data(device_id)
                
            g_csi_phase_array[device_id][:-1] = g_csi_phase_array[device_id][1:]
            g_rssi_array[device_id][:-1] = g_rssi_array[device_id][1:]
            g_radio_header_pd[device_id].iloc[1:] = g_radio_header_pd[device_id].iloc[:-1]

            # 添加设备ID到数据中
            data['device_id'] = device_id
            
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
                    if i < g_csi_phase_array[device_id].shape[1]:
                        g_csi_phase_array[device_id][-1][i] = np.abs(data_complex)
                except Exception:
                    # 静默处理异常
                    if i < g_csi_phase_array[device_id].shape[1]:
                        g_csi_phase_array[device_id][-1][i] = 0

            # 处理RSSI数据
            try:
                g_rssi_array[device_id][-1] = int(data['rssi'])
            except (ValueError, TypeError):
                # 静默处理异常
                g_rssi_array[device_id][-1] = -100

            # 处理无线电头信息
            radio_header_data = {}
            for col in g_radio_header_pd[device_id].columns:
                try:
                    if col in data:
                        radio_header_data[col] = int(data[col])
                    else:
                        radio_header_data[col] = 0
                except (ValueError, TypeError):
                    radio_header_data[col] = 0
            g_radio_header_pd[device_id].loc[0] = pd.Series(radio_header_data)

        except Exception as e:
            # 添加错误日志但不中断流程
            print(f"处理设备 {device_id} 的CSI数据异常: {e}")
            # 尝试恢复到安全状态
            try:
                g_csi_phase_array[device_id][-1] = np.zeros(CSI_DATA_COLUMNS)
                g_rssi_array[device_id][-1] = -100
            except:
                pass

    def start_task_collection(self, task_id):
        """开始采集任务数据"""
        self.current_task_id = task_id
        # 清空所有设备的任务缓冲区
        for device_id in self.device_task_buffers:
            self.device_task_buffers[device_id] = []
        # 获取当前选中的动作类型
        taget_action = self.action_type_combo.currentText()
        # 向所有设备发送开始任务命令和动作类型
        for device_id, queue in self.serial_queues_write.items():
            if hasattr(queue, 'put'):
                # 设置用户名
                user_command = f"set_user:{self.current_user_name}"
                queue.put(user_command)
                # 设置动作类型
                queue.put(f"set_taget:{taget_action}")
                # 开始任务
                command = f"start_task:{task_id}"
                queue.put(command)
                self.textBrowser_log.append(f"<font color='cyan'>设备 {device_id} 开始任务: {task_id}，动作类型: {taget_action}</font>")
    
    def end_task_collection(self):
        """结束采集任务数据并合并"""
        if not self.current_task_id:
            self.textBrowser_log.append("<font color='yellow'>没有正在进行的任务，无法结束</font>")
            return
            
        # 向所有设备发送结束任务命令
        for device_id, queue in self.serial_queues_write.items():
            if hasattr(queue, 'put'):
                queue.put("end_task")
                self.textBrowser_log.append(f"<font color='cyan'>设备 {device_id} 结束任务: {self.current_task_id}</font>")
        
        # 如果是双设备模式，等待数据处理完成后再尝试合并数据
        if self.multi_device_mode and len(self.serial_queues_read.keys()) > 1:
            self.textBrowser_log.append("<font color='cyan'>双设备模式：将在3秒后尝试合并数据...</font>")
            # 等待3秒，确保数据都已保存
            QTimer.singleShot(3000, lambda: self.merge_device_data())
        else:
            # 单设备模式，直接清空任务ID
            self.current_task_id = None

    def merge_device_data(self):
        """合并两个设备的数据"""
        if not self.current_task_id:
            return
        # 从两个设备的处理线程获取数据
        device1_data = self.device_task_buffers.get('esp32_1', [])
        device2_data = self.device_task_buffers.get('esp32_2', [])
        if not device1_data or not device2_data:
            self.textBrowser_log.append("<font color='yellow'>至少有一个设备的数据为空，无法合并</font>")
            return
        # 合并数据
        merged_file = merge_task_data(device1_data, device2_data, self.current_task_id, self.current_user_name)
        if merged_file:
            self.textBrowser_log.append(f"<font color='green'>已合并设备数据并保存到: {merged_file}</font>")
            # 新增：自动上传融合后的csv内容到服务器
            try:
                import csv
                merged_data = []
                with open(merged_file, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        merged_data.append(row)
                # 上传到服务器（直接调用本地函数，不再import）
                success = send_data_to_server(merged_data, self.server_url)
                if success:
                    self.textBrowser_log.append(f"<font color='green'>融合数据已上传服务器: {merged_file}</font>")
                else:
                    self.textBrowser_log.append(f"<font color='red'>融合数据上传服务器失败: {merged_file}</font>")
            except Exception as e:
                self.textBrowser_log.append(f"<font color='red'>融合数据上传异常: {e}</font>")
        # 清空任务ID和缓冲区
        self.current_task_id = None
        for device_id in self.device_task_buffers:
            self.device_task_buffers[device_id] = []


def quit(signum, frame):
    print("Exit the system")
    sys.exit()


class DataHandleThread(QThread):
    signal_device_info = pyqtSignal(pd.Series, str)  # 添加设备ID参数
    signal_log_msg = pyqtSignal(str)
    signal_exit = pyqtSignal()
    signal_wifi_status = pyqtSignal(str, str)  # 添加设备ID参数
    signal_csi_data = pyqtSignal(pd.Series, str)  # 添加设备ID参数

    def __init__(self, queue_read, device_id='esp32_1'):
        super(DataHandleThread, self).__init__()
        self.queue_read = queue_read
        self.device_id = device_id  # 存储设备ID

    def run(self):
        # 只保留简短的启动信息
        self.signal_log_msg.emit(f"<font color='cyan'>设备 {self.device_id} 准备就绪</font>")
        last_data_time = time.time()
        data_counter = 0
        warning_sent = False
        
        while True:
            try:
                if not self.queue_read.empty():
                    data_series = self.queue_read.get()
                    
                    # 重置无数据警告标志
                    warning_sent = False
                    
                    try:
                        if data_series['type'] == 'CSI_DATA':
                            # 更新最后数据接收时间
                            last_data_time = time.time()
                            data_counter += 1
                            
                            # 减少日志频率，从100改为1000
                            if data_counter % 1000 == 0:
                                self.signal_log_msg.emit(
                                    f"<font color='cyan'>设备 {self.device_id}: {data_counter} 数据包</font>")
                            
                            self.handle_csi_data(data_series)
                            # 发送CSI数据到UI线程，包括设备ID
                            self.signal_csi_data.emit(data_series, self.device_id)
                        elif data_series['type'] == 'DEVICE_INFO':
                            # 发送设备信息，包括设备ID
                            self.signal_device_info.emit(data_series, self.device_id)
                            self.signal_log_msg.emit(
                                f"<font color='green'>设备 {self.device_id} 已连接: {data_series['app_revision']}</font>")
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
                                    self.signal_log_msg.emit(f"<font color='green'>设备 {self.device_id} WiFi已连接</font>")
                                elif 'sta:' in log_data.lower() and (
                                        'connected' in log_data.lower() or 'got ip' in log_data.lower()):
                                    wifi_connected = True
                                    # 简化日志
                                    if 'got ip' in log_data.lower():
                                        ip_match = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', log_data)
                                        if ip_match:
                                            ip_address = ip_match.group(0)
                                            self.signal_log_msg.emit(f"<font color='green'>设备 {self.device_id} IP: {ip_address}</font>")
                                elif 'ip' in log_data.lower() and ('got' in log_data.lower() or 'assigned' in log_data.lower()):
                                    wifi_connected = True
                                    # 简化日志，不重复显示
                                    pass
                                elif 'wifi' in log_data.lower() and 'connected' in log_data.lower():
                                    wifi_connected = True
                                    # 简化日志，不重复显示
                                    pass
                                elif 'disconnected' in log_data.lower() or '断开' in log_data:
                                    self.signal_log_msg.emit(f"<font color='yellow'>设备 {self.device_id} WiFi已断开</font>")
                                else:
                                    # 仅显示重要的WiFi状态，其他省略
                                    pass

                                # 发送WiFi状态到处理函数，包括设备ID
                                if wifi_connected:
                                    self.signal_wifi_status.emit("WiFi已连接", self.device_id)
                                else:
                                    self.signal_wifi_status.emit(log_data, self.device_id)
                            # 检测CSI相关信息，但减少日志输出
                            elif ('csi' in log_data.lower() or 'radar' in log_data.lower()) and 'error' in log_data.lower():
                                # 只显示错误信息
                                self.signal_log_msg.emit(f"<font color='red'>设备 {self.device_id} CSI错误: {log_data}</font>")
                            else:
                                # 仅显示错误和警告级别的日志，减少其他日志
                                if data_series['tag'] == 'E':
                                    prefix = f"设备 {self.device_id}: "
                                    self.signal_log_msg.emit(f"<font color='red'>{prefix}{log_data}</font>")
                                elif data_series['tag'] == 'W':
                                    prefix = f"设备 {self.device_id}: "
                                    self.signal_log_msg.emit(f"<font color='yellow'>{prefix}{log_data}</font>")
                        elif data_series['type'] == 'FAIL_EVENT':
                            self.signal_log_msg.emit(
                                f"<font color='red'>设备 {self.device_id} 错误: {data_series['data']}</font>")
                            self.signal_exit.emit()
                            break
                    except Exception as e:
                        self.signal_log_msg.emit(
                            f"<font color='red'>设备 {self.device_id} 数据处理异常: {str(e)}</font>")
                else:
                    # 检查是否长时间没有接收到数据
                    current_time = time.time()
                    if current_time - last_data_time > 30 and not warning_sent and data_counter > 0:  # 增加超时时间
                        self.signal_log_msg.emit(
                            f"<font color='yellow'>设备 {self.device_id} {int(current_time - last_data_time)} 秒未接收数据</font>")
                        warning_sent = True
                    
                    # 避免CPU占用过高
                    time.sleep(0.01)
            except Exception as e:
                self.signal_log_msg.emit(
                    f"<font color='red'>设备 {self.device_id} 异常: {str(e)}</font>")
                time.sleep(1)  # 发生异常时暂停一下
                
    def handle_csi_data(self, data):
        """处理CSI数据"""
        try:
            # 确保设备数据结构已初始化
            if self.device_id not in g_csi_phase_array:
                init_device_data(self.device_id)
                
            g_csi_phase_array[self.device_id][:-1] = g_csi_phase_array[self.device_id][1:]
            g_rssi_array[self.device_id][:-1] = g_rssi_array[self.device_id][1:]
            g_radio_header_pd[self.device_id].iloc[1:] = g_radio_header_pd[self.device_id].iloc[:-1]

            # 添加设备ID到数据中
            data['device_id'] = self.device_id
            
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
                    if i < g_csi_phase_array[self.device_id].shape[1]:
                        g_csi_phase_array[self.device_id][-1][i] = np.abs(data_complex)
                except Exception:
                    # 静默处理异常
                    if i < g_csi_phase_array[self.device_id].shape[1]:
                        g_csi_phase_array[self.device_id][-1][i] = 0

            # 处理RSSI数据
            try:
                g_rssi_array[self.device_id][-1] = int(data['rssi'])
            except (ValueError, TypeError):
                # 静默处理异常
                g_rssi_array[self.device_id][-1] = -100

            # 处理无线电头信息
            radio_header_data = {}
            for col in g_radio_header_pd[self.device_id].columns:
                try:
                    if col in data:
                        radio_header_data[col] = int(data[col])
                    else:
                        radio_header_data[col] = 0
                except (ValueError, TypeError):
                    radio_header_data[col] = 0
            g_radio_header_pd[self.device_id].loc[0] = pd.Series(radio_header_data)

        except Exception as e:
            # 添加错误日志但不中断流程
            print(f"处理设备 {self.device_id} 的CSI数据异常: {e}")
            # 尝试恢复到安全状态
            try:
                g_csi_phase_array[self.device_id][-1] = np.zeros(CSI_DATA_COLUMNS)
                g_rssi_array[self.device_id][-1] = -100
            except:
                pass


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


def save_and_send_task_data(task_data_buffer, task_id, user_name, server_url, enable_server_save, device_id=None):
    """保存和发送任务数据，支持多设备区分"""
    if not task_data_buffer:
        print("task_data_buffer 为空，未保存或发送数据")
        return
    try:
        # 使用通用函数解析任务ID
        action, sequence = parse_task_id(task_id)

        folder = f"data/{action}"
        if not path.exists(folder):
            mkdir(folder)

        # 生成文件名，包含设备ID以避免覆盖
        device_suffix = f"_{device_id}" if device_id else ""
        filename = f"{action}_{user_name}_{sequence}{device_suffix}.csv"
        filepath = os.path.join(folder, filename)

        print(f"保存数据到文件: {filepath}, 动作: {action}, 序号: {sequence}, 设备: {device_id or 'unknown'}")

        # 保存到本地文件（始终保存）
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(task_data_buffer[0].keys())
            for row in task_data_buffer:
                writer.writerow(row.values())
        print(f"本地保存文件: {filename}, 数据量: {len(task_data_buffer)} 条")

        # 无论enable_server_save是否为True，只要有数据都尝试上传
        print(f"准备发送批量数据: {len(task_data_buffer)} 条")
        print(f"服务器地址: {server_url}")
        # 为每条数据添加元数据
        for data in task_data_buffer:
            data['user_name'] = user_name
            data['action'] = action
            data['sequence'] = sequence
            data['file_name'] = filename
            # 添加设备ID区分数据来源
            if device_id:
                data['device_id'] = device_id
        success = send_data_to_server(task_data_buffer, server_url)
        if success:
            print(f"服务器数据发送成功: {filename}")
        else:
            print(f"警告: 服务器数据发送失败: {filename}")
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


def serial_handle(queue_read, queue_write, port, device_id=None):
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
    print(f"打开串口: {port}, 设备ID: {device_id or '未指定'}")
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
    # 为每个设备创建独立的文件
    device_suffix = f"_{device_id}" if device_id else ""
    data_valid_list = pd.DataFrame(
        columns=pd.Index(['type', 'columns_names', 'file_name', 'file_fd', 'file_writer']),
        data=[["CSI_DATA", CSI_DATA_COLUMNS_NAMES, f"log/csi_data{device_suffix}.csv", None, None],
              ["DEVICE_INFO", DEVICE_INFO_COLUMNS_NAMES, f"log/device_info{device_suffix}.csv", None, None]]
    )
    log_data_writer = None
    try:
        for data_valid in data_valid_list.iloc:
            data_valid['file_fd'] = open(data_valid['file_name'], 'w')
            data_valid['file_writer'] = csv.writer(data_valid['file_fd'])
            data_valid['file_writer'].writerow(data_valid['columns_names'])
        log_data_writer = open(f"log/log_data{device_suffix}.txt", 'w+')
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
    current_user_name = "user01"  # 默认用户名，可通过命令更新
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
        print(f"当前服务器状态：URL={server_url}, 保存={enable_server_save}, 设备ID={device_id}")
    except Exception as e:
        print(f"初始化异常: {e}")
        data_series = pd.Series(index=['type', 'data'], data=['FAIL_EVENT', f"初始化异常: {e}"])
        queue_read.put(data_series)
        if ser:
            ser.close()
        sys.exit()
        return

    # 添加设备ID计数器标记
    csi_data_counter_name = f'csi_data_counter_{device_id}' if device_id else 'csi_data_counter'
    setattr(serial_handle, csi_data_counter_name, 0)

    current_taget = "unknown"  # 新增：保存当前动作类型

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
                if command.startswith("set_taget:"):
                    current_taget = command.split(":", 1)[1]
                    print(f"设置当前taget为: {current_taget}")
                    continue
                if command.startswith("start_task:"):
                    current_task_id = command.split(":")[1]
                    action, sequence_str = parse_task_id(current_task_id)
                    try:
                        current_sequence = int(sequence_str)
                    except ValueError:
                        current_sequence = 1
                    task_data_buffer.clear()
                    print(f"开始新任务: {current_task_id}, 动作: {action}, 序号: {current_sequence}, 设备ID: {device_id}")
                    continue
                if command == "end_task" or command.startswith("end_task:"):
                    if task_data_buffer and current_task_id:
                        print(
                            f"处理end_task命令，准备保存和发送数据：任务ID={current_task_id}，数据量={len(task_data_buffer)}，服务器保存={enable_server_save}, 设备ID={device_id or '未指定'}")
                        save_and_send_task_data(task_data_buffer, current_task_id, current_user_name, server_url,
                                                enable_server_save, device_id)
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

                                    # 如果设置了设备ID，添加到数据中
                                    if device_id:
                                        data_series['device_id'] = device_id

                                    # 先同步taget字段
                                    data_series['taget'] = current_taget

                                    # 发送到队列
                                    if not queue_read.full():
                                        queue_read.put(data_series)
                                    # 队列满时静默丢弃数据，不输出警告

                                    # 检测是否是第一次接收到CSI数据
                                    static_counter = getattr(serial_handle, csi_data_counter_name, 0)
                                    if static_counter == 0:
                                        # 只在第一次检测到CSI数据时发送连接成功消息
                                        success_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                                   data=['LOG_DATA', 'I', datetime.now().strftime(
                                                                       '%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                                         f"WiFi连接成功！设备 {device_id} 开始接收CSI数据"])
                                        if not queue_read.full():
                                            queue_read.put(success_series)

                                    # 更新计数器
                                    setattr(serial_handle, csi_data_counter_name, static_counter + 1)

                                    # 处理任务数据
                                    if current_task_id:
                                        # 添加设备ID到数据中
                                        data_dict = data_series.astype(str).to_dict()
                                        if device_id:
                                            data_dict['device_id'] = device_id
                                        # 添加数据到缓冲区
                                        task_data_buffer.append(data_dict)

                                        # 使用解析的任务ID
                                        action, sequence_str = parse_task_id(current_task_id)

                                        # 生成文件名，包含设备ID以避免覆盖
                                        sequence = f"{current_sequence:02d}"
                                        device_suffix = f"_{device_id}" if device_id else ""
                                        current_file_key = f"{action}_{current_user_name}_{sequence}{device_suffix}"

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
                                if device_id:
                                    data_series['device_id'] = device_id
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
                        # 添加设备ID
                        if device_id:
                            data_series['device_id'] = device_id
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
                                                          f"WiFi连接成功! {strings}"])
                            # 添加设备ID
                            if device_id:
                                data_series['device_id'] = device_id

                            # 强制发送一个明确的连接成功消息，确保UI能够看到
                            success_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                       data=['LOG_DATA', 'I',
                                                             datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                             "WiFi已连接"])
                            # 添加设备ID
                            if device_id:
                                success_series['device_id'] = device_id
                            if not queue_read.full():
                                queue_read.put(success_series)

                            # 如果检测到run状态，额外发送一个更明确的成功消息
                            if 'wifi:state:' in strings.lower() and 'run' in strings.lower():
                                extra_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                         data=['LOG_DATA', 'I',
                                                               datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                               f"WiFi连接成功！设备 {device_id} 已进入运行状态"])
                                if device_id:
                                    extra_series['device_id'] = device_id
                                if not queue_read.full():
                                    queue_read.put(extra_series)
                        else:
                            data_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                    data=['LOG_DATA', 'I',
                                                          datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                          strings])
                            # 添加设备ID
                            if device_id:
                                data_series['device_id'] = device_id

                        if not queue_read.full():
                            queue_read.put(data_series)
                            
                    # 添加对CSI相关日志的特殊处理
                    elif 'csi' in strings.lower() or 'radar' in strings.lower():
                        csi_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                               data=['LOG_DATA', 'I',
                                                     datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                     f"CSI状态: {strings}"])
                        # 添加设备ID
                        if device_id:
                            csi_series['device_id'] = device_id
                        if not queue_read.full():
                            queue_read.put(csi_series)
        except Exception as e:
            print(f"处理数据异常: {e}")
            continue


def merge_task_data(task_data1, task_data2, task_id, user_name, output_folder="data/merged"):
    """合并两个设备的任务数据
    
    Args:
        task_data1: 第一个设备的数据
        task_data2: 第二个设备的数据
        task_id: 任务ID
        user_name: 用户名
        output_folder: 输出文件夹，默认为data/merged
        
    Returns:
        合并后的文件名
    """
    if not task_data1 or not task_data2:
        print("至少有一个设备的数据为空，无法合并")
        return None
        
    try:
        # 创建输出文件夹
        if not path.exists(output_folder):
            mkdir(output_folder)
            
        # 解析任务ID
        action, sequence = parse_task_id(task_id)
        
        # 为两个设备的数据添加标识
        for data in task_data1:
            data['device_id'] = 'esp32_1'
        
        for data in task_data2:
            data['device_id'] = 'esp32_2'
            
        # 合并数据，并按时间戳排序
        merged_data = task_data1 + task_data2
        
        # 尝试按时间戳排序
        try:
            merged_data.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S.%f'))
        except Exception:
            # 如果排序失败，保持原来的顺序
            print("按时间戳排序失败，保持原来的顺序")
            
        # 生成合并文件名
        merged_filename = f"{action}_{user_name}_{sequence}_merged.csv"
        merged_filepath = os.path.join(output_folder, merged_filename)
        
        # 保存到文件
        with open(merged_filepath, 'w', newline='') as f:
            if merged_data:
                writer = csv.writer(f)
                writer.writerow(merged_data[0].keys())
                for row in merged_data:
                    writer.writerow(row.values())
                    
        print(f"合并数据已保存到: {merged_filepath}, 总数据量: {len(merged_data)} 条")
        return merged_filepath
    except Exception as e:
        print(f"合并数据时发生错误: {type(e).__name__}: {str(e)}")
        return None


if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print(" Python version should >= 3.6")
        exit()
    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of first ESP32 device")
    parser.add_argument('-p2', '--port2', dest='port2', action='store', required=False,
                        help="Serial port number of second ESP32 device (optional)")
    parser.add_argument('--user_id', dest='user_id', type=str, default="01",
                        help="User ID for data collection (default=01)")
    parser.add_argument('--desc', dest='description', type=str, default="",
                        help="Optional description to add to data files")
    args = parser.parse_args()
    serial_port = args.port
    serial_port2 = args.port2  # 可能为None
    user_id = args.user_id
    description = args.description
    
    # 单设备或双设备模式设置
    if serial_port2:
        # 双设备模式：创建两个设备的队列
        serial_queues_read = {
            'esp32_1': Queue(maxsize=128),
            'esp32_2': Queue(maxsize=128)
        }
        serial_queues_write = {
            'esp32_1': Queue(maxsize=64),
            'esp32_2': Queue(maxsize=64)
        }
        
        # 初始化每个设备的数据结构
        init_device_data('esp32_1')
        init_device_data('esp32_2')
        
        # 启动两个串口处理进程
        serial_handle_process1 = Process(target=serial_handle, 
                                       args=(serial_queues_read['esp32_1'], 
                                             serial_queues_write['esp32_1'], 
                                             serial_port,
                                             'esp32_1'))  # 传递设备ID
        serial_handle_process2 = Process(target=serial_handle, 
                                       args=(serial_queues_read['esp32_2'], 
                                             serial_queues_write['esp32_2'], 
                                             serial_port2,
                                             'esp32_2'))  # 传递设备ID
        serial_handle_process1.start()
        serial_handle_process2.start()
        
        # 输出模式信息
        print(f"双设备模式: 设备1 = {serial_port}, 设备2 = {serial_port2}")
    else:
        # 单设备模式：使用原来的方式
        serial_queues_read = {
            'esp32_1': Queue(maxsize=128)
        }
        serial_queues_write = {
            'esp32_1': Queue(maxsize=64)
        }
        
        # 初始化设备数据结构
        init_device_data('esp32_1')
        
        # 启动串口处理进程
        serial_handle_process = Process(target=serial_handle, 
                                      args=(serial_queues_read['esp32_1'], 
                                            serial_queues_write['esp32_1'], 
                                            serial_port,
                                            'esp32_1'))  # 传递设备ID
        serial_handle_process.start()
        
        # 输出模式信息
        print(f"单设备模式: 设备 = {serial_port}")
    
    # 设置信号处理
    signal_key.signal(signal_key.SIGINT, quit)
    signal_key.signal(signal_key.SIGTERM, quit)
    
    # 创建应用和主窗口
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('../../../docs/_static/icon.png'))
    window = DataGraphicalWindow(serial_queues_read, serial_queues_write)
    
    # 为每个设备创建并启动数据处理线程
    data_handle_threads = {}
    for device_id, queue in serial_queues_read.items():
        data_handle_thread = DataHandleThread(queue, device_id)
        # 连接信号
        data_handle_thread.signal_device_info.connect(
            lambda device_info, d_id=device_id: window.show_device_info(device_info, d_id))
        data_handle_thread.signal_log_msg.connect(window.show_textBrowser_log)
        data_handle_thread.signal_exit.connect(window.slot_close)
        data_handle_thread.signal_wifi_status.connect(
            lambda status, d_id=device_id: window.handle_wifi_status(status, d_id))
        data_handle_thread.signal_csi_data.connect(window.process_csi_data)
        data_handle_thread.start()
        data_handle_threads[device_id] = data_handle_thread
    
    # 等待短暂时间让串口处理进程初始化
    time.sleep(2)
    
    # 主动为每个设备发送CSI初始化命令，确保双设备都能正确配置
    print("初始化设备CSI功能...")
    for device_id, queue in serial_queues_write.items():
        if hasattr(queue, 'put'):
            # 设置CSI输出格式
            queue.put(f"radar --csi_output_type LLFT --csi_output_format base64")
            time.sleep(0.5)
            # 确保CSI功能被禁用（重置状态）
            queue.put(f"radar --csi_en 0")
            time.sleep(0.5)
            # 启用CSI功能
            queue.put(f"radar --csi_en 1") 
            time.sleep(0.5)
            # 检查CSI状态
            queue.put(f"radar --get_config")
            print(f"已初始化设备: {device_id}")

    # 显示窗口并启动事件循环
    window.show()
    window.textBrowser_log.append("<font color='green'>系统准备就绪</font>")
    exit_code = app.exec()
    
    # 关闭处理进程
    if serial_port2:
        serial_handle_process1.join()
        serial_handle_process2.join()
    else:
        serial_handle_process.join()
    
    sys.exit(exit_code)