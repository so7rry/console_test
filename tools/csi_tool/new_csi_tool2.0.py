# -*-coding:utf-8-*-
import sys
import csv
import json
import re
import os
import argparse
import pandas as pd
import numpy as np
import requests  # 添加requests库用于HTTP请求

import serial
from os import path
from os import mkdir
from io import StringIO
from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import pyqtgraph as pq
import threading

from PyQt5.QtCore import QDate, QDate, QTime, QDateTime
import base64
import time
from datetime import datetime
from multiprocessing import Process, Queue
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
# from PyQt5.QtChart import QChart, QLineSeries,QValueAxis
from esp_csi_tool_gui import Ui_MainWindow
from scipy import signal
import signal as signal_key
import socket

CSI_SAMPLE_RATE = 100

# 定义全局变量
g_display_raw_data = True

# Remove invalid subcarriers
# secondary channel : below, HT, 40 MHz, non STBC, v, HT-LFT: 0~63, -64~-1, 384
CSI_VAID_SUBCARRIER_INTERVAL = 5
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_color = []
color_step = 255 // (28 // CSI_VAID_SUBCARRIER_INTERVAL + 1)

# LLTF: 52
# 26  red
csi_vaid_subcarrier_index += [i for i in range(
    0, 26, CSI_VAID_SUBCARRIER_INTERVAL)]
csi_vaid_subcarrier_color += [(i * color_step, 0, 0)
                              for i in range(1, 26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
# 26  green
csi_vaid_subcarrier_index += [i for i in range(
    26, 52, CSI_VAID_SUBCARRIER_INTERVAL)]
csi_vaid_subcarrier_color += [(0, i * color_step, 0)
                              for i in range(1, 26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]

DEVICE_INFO_COLUMNS_NAMES = ["type", "timestamp", "compile_time", "chip_name", "chip_revision",
                             "app_revision", "idf_revision", "total_heap", "free_heap", "router_ssid", "ip", "port"]
g_device_info_series = None

CSI_DATA_INDEX = 500  # buffer size
CSI_DATA_COLUMNS = len(csi_vaid_subcarrier_index)
CSI_DATA_COLUMNS_NAMES = ["type", "seq", "timestamp", "taget_seq", "taget", "mac", "rssi", "rate", "sig_mode", "mcs",
                          "cwb", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi", "noise_floor",
                          "ampdu_cnt", "channel_primary", "channel_secondary", "local_timestamp", "ant", "sig_len",
                          "rx_state", "agc_gain", "fft_gain", "len", "first_word_invalid", "data"]
CSI_DATA_TARGETS = ["unknown", "train",
                    "lie_down", "walk", "stand", "bend", "sit_down",
                    "fall_from_stand", "fall_from_squat", "fall_from_bed"]

g_csi_phase_array = np.zeros(
    [CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.int32)
g_rssi_array = np.zeros(CSI_DATA_INDEX, dtype=np.int8)
g_radio_header_pd = pd.DataFrame(np.zeros([10, len(
    CSI_DATA_COLUMNS_NAMES[1:-1])], dtype=np.int32), columns=CSI_DATA_COLUMNS_NAMES[1:-1])


def base64_decode_bin(str_data):
    try:
        # 检查输入数据是否为None或空
        if not str_data:
            # print("Base64数据为空")
            return []

        # 处理长度不是4的倍数的Base64字符串（静默处理）
        padding_needed = len(str_data) % 4
        if padding_needed > 0:
            str_data += '=' * (4 - padding_needed)
            # 移除输出，避免影响观看
            # print(f"修正后的Base64字符串长度: {len(str_data)}")

        bin_data = base64.b64decode(str_data)
    except Exception as e:
        print(f"Base64解码异常: {e}")  # 简化输出，不再显示数据内容
        return []

    try:
        # 将二进制数据转换为列表
        list_data = list(bin_data)

        # 转换为有符号数
        for i in range(len(list_data)):
            if list_data[i] > 127:
                list_data[i] = list_data[i] - 256

        return list_data
    except Exception as e:
        print(f"数据处理异常: {e}")
        return []


def base64_encode_bin(list_data):
    for i in range(len(list_data)):
        if list_data[i] < 0:
            list_data[i] = 256 + list_data[i]
    # print(list_data)

    str_data = "test"
    try:
        str_data = base64.b64encode(bytes(list_data)).decode('utf-8')
    except Exception as e:
        print(f"Exception: {e}, data: {list_data}")
    return str_data


def get_label(folder_path):
    parts = str.split(folder_path, os.path.sep)
    return parts[-1]


def evaluate_data_send(serial_queue_write, folder_path):
    label = get_label(folder_path)
    if label == "train":
        command = f"csi --train_start"
        serial_queue_write.put(command)

    tcpCliSock = socket.socket()
    device_info_series = pd.read_csv('../log/device_info.csv').iloc[-1]

    print(f"connect:{device_info_series['ip']},{device_info_series['port']}")
    tcpCliSock.connect((device_info_series['ip'], device_info_series['port']))

    file_name_list = sorted(os.listdir(folder_path))
    print(file_name_list)
    for file_name in file_name_list:
        file_path = folder_path + os.path.sep + file_name
        data_pd = pd.read_csv(file_path)
        for index, data_series in enumerate(data_pd.iloc):
            csi_raw_data = json.loads(data_series['data'])
            data_pd.loc[index, 'data'] = base64_encode_bin(csi_raw_data)
            temp_list = base64_decode_bin(data_pd.loc[index, 'data'])
            data_str = ','.join(str(value)
                                for value in data_pd.loc[index]) + "\n"
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
        self.user_name = "unknown"  # 默认用户名称

        # 服务器配置
        self.server_url = "http://8.136.10.160:18306/api/csi_data"  # 默认服务器地址
        self.enable_server_save = False  # 是否启用服务器保存

        self.initUI()

    def initUI(self):
        self.setupUi(self)
        global g_display_raw_data

        # 设置窗口标题
        self.setWindowTitle("ESP CSI Tool")

        # 如果存在Raw data标题，删除它
        for child in self.findChildren(QGroupBox):
            if child.title() == "Raw data":
                child.setTitle("")  # 清空标题

        # 创建采集窗口
        self.create_collect_window()

        # 将采集窗口添加到主窗口
        if hasattr(self, 'verticalLayout_17'):
            self.verticalLayout_17.addWidget(self.collect_group)

        # 添加服务器保存控制
        self.checkBox_server_save = QCheckBox("保存到服务器")
        self.checkBox_server_save.setChecked(False)
        self.checkBox_server_save.stateChanged.connect(self.toggle_server_save)
        if hasattr(self, 'verticalLayout_17'):
            self.verticalLayout_17.addWidget(self.checkBox_server_save)

        # 添加服务器地址输入框
        server_layout = QHBoxLayout()
        server_label = QLabel("服务器地址:")
        self.lineEdit_server_url = QLineEdit()
        self.lineEdit_server_url.setText("http://8.136.10.160:12786/api/csi_data")  # 确保包含完整的URL
        self.lineEdit_server_url.setPlaceholderText("输入服务器地址 (例如: http://8.136.10.160:12786/api/csi_data)")
        server_layout.addWidget(server_label)
        server_layout.addWidget(self.lineEdit_server_url)
        if hasattr(self, 'verticalLayout_17'):
            self.verticalLayout_17.addLayout(server_layout)

        # 添加tableView_values属性
        self.tableView_values = {
            'sig_mode': ['Non HT(11bg)', 'HT(11n)', 'VHT(11ac)', 'HE(11ax)'],
            'mcs': ['RATE_6M', 'RATE_9M', 'RATE_12M', 'RATE_18M', 'RATE_24M', 'RATE_36M', 'RATE_48M', 'RATE_54M',
                    'MCS0', 'MCS1', 'MCS2', 'MCS3', 'MCS4', 'MCS5', 'MCS6', 'MCS7',
                    'MCS8', 'MCS9', 'MCS10', 'MCS11', 'MCS12', 'MCS13', 'MCS14', 'MCS15',
                    'MCS16', 'MCS17', 'MCS18', 'MCS19', 'MCS20', 'MCS21', 'MCS22', 'MCS23'],
            'cwb': ['20MHz', '40MHz'],
            'aggregation': ['MPDU', 'AMPDU'],
            'stbc': ['Non STBC', 'STBC'],
            'fec_coding': ['BCC', 'LDPC'],
            'sgi': ['Long GI', 'Short GI'],
            'smoothing': ['No', 'Yes'],
            'not_sounding': ['Yes', 'No'],
            'channel_secondary': ['None', 'Above', 'Below']
        }

        with open("../config/gui_config.json") as file:
            gui_config = json.load(file)
            if len(gui_config['router_ssid']) > 0:
                self.lineEdit_router_ssid.setText(gui_config['router_ssid'])
            if len(gui_config['router_password']) >= 8:
                self.lineEdit_router_password.setText(gui_config['router_password'])

            g_display_raw_data = gui_config.get('display_raw_data', True)
            if g_display_raw_data:
                self.checkBox_raw_data.setChecked(True)
            else:
                self.checkBox_raw_data.setChecked(False)

            if gui_config.get('router_auto_connect', False):
                self.checkBox_router_auto_connect.setChecked(True)
            else:
                self.checkBox_router_auto_connect.setChecked(False)

        # 隐藏不需要的组件
        if hasattr(self, 'groupBox_radar_model'):
            self.groupBox_radar_model.hide()
        if hasattr(self, 'QWidget_evaluate_info'):
            self.QWidget_evaluate_info.hide()
        if hasattr(self, 'groupBox_20'):  # Evaluate组
            self.groupBox_20.hide()
        if hasattr(self, 'groupBox_eigenvalues'):  # 特征值显示
            self.groupBox_eigenvalues.hide()
        if hasattr(self, 'groupBox_eigenvalues_table'):  # 特征值表格
            self.groupBox_eigenvalues_table.hide()
        if hasattr(self, 'groupBox_statistics'):  # 统计相关
            self.groupBox_statistics.hide()
        if hasattr(self, 'groupBox_predict'):  # 预测相关
            self.groupBox_predict.hide()

        # 删除顶部的复选框
        if hasattr(self, 'checkBox_raw_data'):
            if self.checkBox_raw_data.parent():
                parent_layout = self.checkBox_raw_data.parent().layout()
                if parent_layout:
                    parent_layout.removeWidget(self.checkBox_raw_data)
                self.checkBox_raw_data.hide()

        if hasattr(self, 'checkBox_radar_model'):
            if self.checkBox_radar_model.parent():
                parent_layout = self.checkBox_radar_model.parent().layout()
                if parent_layout:
                    parent_layout.removeWidget(self.checkBox_radar_model)
                self.checkBox_radar_model.hide()

        # 删除"display:"标签
        for widget in self.findChildren(QLabel):
            if hasattr(widget, 'text') and widget.text() == "display:":
                if widget.parent():
                    parent_layout = widget.parent().layout()
                    if parent_layout:
                        parent_layout.removeWidget(widget)
                widget.hide()

        # 完全删除info界面
        for child in self.findChildren(QGroupBox):
            if child.title().lower() == "info":
                if child.parent():
                    parent_widget = child.parent()
                    if hasattr(parent_widget, 'layout'):
                        layout = parent_widget.layout()
                        if layout:
                            layout.removeWidget(child)
                child.setParent(None)
                child.deleteLater()

        # 创建采集窗口相关变量
        self.label_number = 0  # 存储数值，不是标签对象
        self.label_delay = QTime(0, 0, 3)  # 默认3秒

        # 采集控制变量
        self.is_collecting = False
        self.target_collect_count = 0
        self.current_collect_count = 0

        # 删除现有定时器并创建新的定时器
        if hasattr(self, 'timer_collect_duration'):
            del self.timer_collect_duration
        if hasattr(self, 'collection_timer'):
            del self.collection_timer

        # 创建新的定时器
        self.collection_delay_timer = QTimer()
        self.collection_delay_timer.setSingleShot(True)
        self.collection_delay_timer.timeout.connect(self.on_delay_timeout)

        self.collection_duration_timer = QTimer()
        self.collection_duration_timer.setSingleShot(True)
        self.collection_duration_timer.timeout.connect(self.on_duration_timeout)

        # Initialize CSI and RSSI plots
        self.curve_subcarrier_range = np.array([10, 20])
        self.graphicsView_subcarrier.setYRange(
            self.curve_subcarrier_range[0], self.curve_subcarrier_range[1], padding=0)
        self.graphicsView_subcarrier.addLegend()

        # 禁用鼠标滚轮引起的自动缩放
        self.graphicsView_subcarrier.setMouseEnabled(x=True, y=False)
        self.graphicsView_subcarrier.enableAutoRange(axis='y', enable=False)
        # 设置视图交互模式，禁用某些缩放行为
        self.graphicsView_subcarrier.setMenuEnabled(False)

        # 配置RSSI图表
        self.graphicsView_rssi.setYRange(-100, 0, padding=0)
        self.graphicsView_rssi.setMouseEnabled(x=True, y=False)
        self.graphicsView_rssi.enableAutoRange(axis='y', enable=False)
        self.graphicsView_rssi.setMenuEnabled(False)

        self.curve_subcarrier = []
        self.serial_queue_write = serial_queue_write

        for i in range(CSI_DATA_COLUMNS):
            curve = self.graphicsView_subcarrier.plot(
                g_csi_phase_array[:, i], name=str(i), pen=csi_vaid_subcarrier_color[i])
            self.curve_subcarrier.append(curve)

        self.curve_rssi = self.graphicsView_rssi.plot(
            g_rssi_array, name='rssi', pen=(255, 255, 255))

        self.wave_filtering_flag = self.checkBox_wave_filtering.isCheckable()
        self.checkBox_wave_filtering.released.connect(
            self.show_curve_subcarrier_filter)
        self.checkBox_router_auto_connect.released.connect(
            self.show_router_auto_connect)

        # Connect signals
        self.pushButton_router_connect.released.connect(
            self.command_router_connect)
        self.pushButton_command.released.connect(self.command_custom)
        self.comboBox_command.activated.connect(self.comboBox_command_show)

        self.checkBox_raw_data.released.connect(self.checkBox_raw_data_show)

        # Initialize timers
        self.timer_boot_command = QTimer()
        self.timer_boot_command.timeout.connect(self.command_boot)
        self.timer_boot_command.setInterval(3000)
        self.timer_boot_command.start()

        self.timer_curve_subcarrier = QTimer()
        self.timer_curve_subcarrier.timeout.connect(self.show_curve_subcarrier)
        self.timer_curve_subcarrier.setInterval(300)

        self.timer_collect_duration = QTimer()
        self.timer_collect_delay = QTimer()
        self.timer_collect_delay.setInterval(1000)
        self.timer_collect_delay.timeout.connect(self.timeEdit_collect_delay_show)

        # Initialize UI state
        self.checkBox_raw_data_show()
        self.textBrowser_log.setStyleSheet("background:black")

    def create_collect_window(self):
        """创建采集窗口"""
        # 创建采集窗口
        self.collect_group = QGroupBox(self.centralwidget)
        self.collect_group.setTitle("Collect")
        self.collect_group.setFont(QFont("Arial", 10))

        # 创建垂直布局
        collect_layout = QVBoxLayout(self.collect_group)
        collect_layout.setContentsMargins(5, 5, 5, 5)

        # 创建水平布局
        horizontal_layout = QHBoxLayout()

        # 添加用户名称标签和输入框
        label_user = QLabel("user")
        horizontal_layout.addWidget(label_user)

        self.lineEdit_user = QLineEdit()
        self.lineEdit_user.setPlaceholderText("输入用户名称")
        self.lineEdit_user.setFixedWidth(100)
        horizontal_layout.addWidget(self.lineEdit_user)

        # 添加标签和下拉框
        label_target = QLabel("target")
        horizontal_layout.addWidget(label_target)

        self.comboBox_collect_target = QComboBox()
        for target in CSI_DATA_TARGETS:
            self.comboBox_collect_target.addItem(target)
        horizontal_layout.addWidget(self.comboBox_collect_target)

        # 添加延时标签和时间编辑器
        label_delay = QLabel("delay")
        horizontal_layout.addWidget(label_delay)

        self.timeEdit_collect_delay = QTimeEdit()
        self.timeEdit_collect_delay.setDisplayFormat("HH:mm:ss")
        self.timeEdit_collect_delay.setTime(QTime(0, 0, 3))  # 默认3秒
        horizontal_layout.addWidget(self.timeEdit_collect_delay)

        # 添加持续时间标签和微调框
        label_duration = QLabel("duration(ms)")
        horizontal_layout.addWidget(label_duration)

        self.spinBox_collect_duration = QSpinBox()
        self.spinBox_collect_duration.setMinimum(100)
        self.spinBox_collect_duration.setMaximum(10000)
        self.spinBox_collect_duration.setSingleStep(100)
        self.spinBox_collect_duration.setValue(500)
        horizontal_layout.addWidget(self.spinBox_collect_duration)

        # 添加次数标签和微调框
        number_label = QLabel("number")
        horizontal_layout.addWidget(number_label)

        self.spinBox_collect_number = QSpinBox()
        self.spinBox_collect_number.setMinimum(1)
        self.spinBox_collect_number.setMaximum(100)
        self.spinBox_collect_number.setValue(1)
        horizontal_layout.addWidget(self.spinBox_collect_number)

        # 添加清除和开始按钮
        self.pushButton_collect_clean = QPushButton("clean")
        horizontal_layout.addWidget(self.pushButton_collect_clean)

        self.pushButton_collect_start = QPushButton("start")
        horizontal_layout.addWidget(self.pushButton_collect_start)

        # 将水平布局添加到垂直布局
        collect_layout.addLayout(horizontal_layout)

        # 将采集窗口添加到UI中
        # 先尝试找到log区域
        log_container = None
        for widget in self.findChildren(QWidget):
            if hasattr(widget, 'objectName') and 'log' in widget.objectName().lower():
                log_container = widget
                break

        # 如果找到了log区域，则把采集窗口放在它下面
        if log_container and log_container.parent() and hasattr(log_container.parent(), 'layout'):
            parent_layout = log_container.parent().layout()
            index = -1
            for i in range(parent_layout.count()):
                if parent_layout.itemAt(i).widget() == log_container:
                    index = i
                    break

            if index >= 0:
                parent_layout.insertWidget(index + 1, self.collect_group)
            else:
                # 如果找不到log区域的位置，则添加到主布局
                if hasattr(self, 'verticalLayout_17'):
                    self.verticalLayout_17.addWidget(self.collect_group)
        else:
            # 如果找不到log区域，则添加到主布局
            if hasattr(self, 'verticalLayout_17'):
                self.verticalLayout_17.addWidget(self.collect_group)

        # 连接信号槽
        self.pushButton_collect_start.released.connect(self.pushButton_collect_show)
        self.pushButton_collect_clean.released.connect(self.pushButton_collect_clean_show)

    def checkBox_raw_data_show(self):
        global g_display_raw_data
        g_display_raw_data = self.checkBox_raw_data.isChecked()

        if self.checkBox_raw_data.isChecked():
            self.groupBox_raw_data.show()
            self.timer_curve_subcarrier.start()
        else:
            self.groupBox_raw_data.hide()
            self.timer_curve_subcarrier.stop()

        with open("../config/gui_config.json", "r") as file:
            gui_config = json.load(file)
            gui_config['display_raw_data'] = self.checkBox_raw_data.isChecked()
        with open("../config/gui_config.json", "w") as file:
            json.dump(gui_config, file)

    def show_router_auto_connect(self):
        with open("../config/gui_config.json", "r") as file:
            gui_config = json.load(file)
            gui_config['router_auto_connect'] = self.checkBox_router_auto_connect.isChecked()
        with open("../config/gui_config.json", "w") as file:
            json.dump(gui_config, file)

    def median_filtering(self, waveform):
        tmp = waveform
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
        waveform = tmp

    def show_curve_subcarrier(self):
        try:
            wn = 20 / (CSI_SAMPLE_RATE / 2)
            b, a = signal.butter(8, wn, 'lowpass')

            if self.wave_filtering_flag:
                self.median_filtering(g_csi_phase_array)
                csi_filtfilt_data = signal.filtfilt(b, a, g_csi_phase_array.T).T
            else:
                csi_filtfilt_data = g_csi_phase_array

            # 计算数据范围，确保图表不会频繁自动调整
            data_min = np.nanmin(csi_filtfilt_data)
            data_max = np.nanmax(csi_filtfilt_data)

            # 防止NaN值导致的问题
            if np.isnan(data_min):
                data_min = 0
            if np.isnan(data_max):
                data_max = 10

            # 仅当数据超出当前范围较多时才调整范围
            need_update_range = False
            if data_min < self.curve_subcarrier_range[0]:
                self.curve_subcarrier_range[0] = data_min - 2
                need_update_range = True
            if data_max > self.curve_subcarrier_range[1]:
                self.curve_subcarrier_range[1] = data_max + 2
                need_update_range = True

            if need_update_range:
                self.graphicsView_subcarrier.setYRange(
                    self.curve_subcarrier_range[0], self.curve_subcarrier_range[1], padding=0)

            # 确保曲线数量与数据列数一致
            if len(self.curve_subcarrier) != csi_filtfilt_data.shape[1]:
                # 如果曲线数量不一致，重新创建曲线
                self.graphicsView_subcarrier.clear()
                self.graphicsView_subcarrier.addLegend()
                self.curve_subcarrier = []

                for i in range(min(CSI_DATA_COLUMNS, csi_filtfilt_data.shape[1])):
                    if i < len(csi_vaid_subcarrier_color):
                        curve = self.graphicsView_subcarrier.plot(
                            csi_filtfilt_data[:, i], name=str(i), pen=csi_vaid_subcarrier_color[i])
                    else:
                        # 默认颜色
                        curve = self.graphicsView_subcarrier.plot(
                            csi_filtfilt_data[:, i], name=str(i), pen=(255, 255, 255))
                    self.curve_subcarrier.append(curve)
            else:
                # 更新现有曲线数据
                for i in range(min(len(self.curve_subcarrier), csi_filtfilt_data.shape[1])):
                    self.curve_subcarrier[i].setData(csi_filtfilt_data[:, i])

            if self.wave_filtering_flag:
                csi_filtfilt_rssi = signal.filtfilt(
                    b, a, g_rssi_array).astype(np.int32)
            else:
                csi_filtfilt_rssi = g_rssi_array
            self.curve_rssi.setData(csi_filtfilt_rssi)
        except Exception as e:
            print(f"显示CSI数据异常: {e}")

    def show_curve_subcarrier_filter(self):
        self.wave_filtering_flag = self.checkBox_wave_filtering.isChecked()

    def command_boot(self):
        # 使用原始版本中的正确命令格式
        command = f"radar --csi_output_type LLFT --csi_output_format base64"
        self.serial_queue_write.put(command)

        if self.checkBox_router_auto_connect.isChecked() and len(self.lineEdit_router_ssid.text()) > 0:
            self.command_router_connect()

        self.timer_boot_command.stop()

    def command_router_connect(self):
        if self.pushButton_router_connect.text() == "connect":
            self.pushButton_router_connect.setText("disconnect")
            self.pushButton_router_connect.setStyleSheet("color: red")

            command = "wifi_config --ssid " + ("\"%s\"" % self.lineEdit_router_ssid.text())
            if len(self.lineEdit_router_password.text()) >= 8:
                command += " --password " + self.lineEdit_router_password.text()
            self.serial_queue_write.put(command)
        else:
            self.pushButton_router_connect.setText("connect")
            self.pushButton_router_connect.setStyleSheet("color: black")
            command = "ping --abort"
            self.serial_queue_write.put(command)
            command = "wifi_config --disconnect"
            self.serial_queue_write.put(command)

        with open("../config/gui_config.json", "r") as file:
            gui_config = json.load(file)
            gui_config['router_ssid'] = self.lineEdit_router_ssid.text()
            gui_config['router_password'] = self.lineEdit_router_password.text()
        with open("../config/gui_config.json", "w") as file:
            json.dump(gui_config, file)

    def command_custom(self):
        command = self.lineEdit_command.text()
        self.serial_queue_write.put(command)

    def command_collect_target_start(self):
        """开始采集目标数据"""
        if self.is_collecting:
            return

        self.is_collecting = True
        self.current_collect_count += 1

        # 发送用户名称到串口处理进程（如果没有在开始前发送）
        user_name = self.lineEdit_user.text().strip()
        if user_name:
            self.serial_queue_write.put(f"set_user:{user_name}")

        # 发送采集命令
        target = self.comboBox_collect_target.currentText()
        duration = self.spinBox_collect_duration.value()
        command = (f"radar --collect_number 1" +
                   f" --collect_tagets {target}" +
                   f" --collect_duration {duration}")
        self.serial_queue_write.put(command)

        # 更新UI显示
        self.textBrowser_log.append(
            f"<font color='cyan'>采集开始: {target} - 第 {self.current_collect_count}/{self.target_collect_count} 次，持续时间: {duration}ms</font>")

        # 启动持续时间定时器
        self.collection_duration_timer.start(duration + 200)  # 添加200ms缓冲时间，确保采集命令有足够时间完成

    def command_collect_target_stop(self):
        """停止采集目标数据"""
        if not self.is_collecting:
            return

        command = "radar --collect_number 0 --collect_tagets unknown"
        self.serial_queue_write.put(command)

        # 停止当前采集
        self.is_collecting = False

        # 停止定时器
        if self.collection_duration_timer.isActive():
            self.collection_duration_timer.stop()

        # 记录日志
        self.textBrowser_log.append(f"<font color='yellow'>采集暂停</font>")

    def on_delay_timeout(self):
        """延时结束，开始采集"""
        # 保存目标采集次数
        self.target_collect_count = self.spinBox_collect_number.value()
        self.current_collect_count = 0

        # 更新UI
        self.spinBox_collect_number.setStyleSheet("color: red")
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
        # 确保当前采集已停止
        if self.is_collecting:
            self.command_collect_target_stop()

        # 恢复UI状态
        self.spinBox_collect_number.setValue(self.label_number)
        self.timeEdit_collect_delay.setTime(self.label_delay)
        self.pushButton_collect_start.setStyleSheet("color: black")
        self.spinBox_collect_number.setStyleSheet("color: black")
        self.timeEdit_collect_delay.setStyleSheet("color: black")
        self.pushButton_collect_start.setText("start")

        # 停止所有定时器
        if self.timer_collect_delay.isActive():
            self.timer_collect_delay.stop()
        if self.collection_delay_timer.isActive():
            self.collection_delay_timer.stop()
        if self.collection_duration_timer.isActive():
            self.collection_duration_timer.stop()

        # 显示采集完成消息
        self.textBrowser_log.append(f"<font color='green'>采集完成：共 {self.current_collect_count} 次</font>")

    def timeEdit_collect_delay_show(self):
        """延时倒计时逻辑"""
        time_temp = self.timeEdit_collect_delay.time()
        second = time_temp.hour() * 3600 + time_temp.minute() * 60 + time_temp.second()

        if second > 0:
            # 显示当前倒计时
            self.timeEdit_collect_delay.setTime(time_temp.addSecs(-1))
            self.timer_collect_delay.start()

            # 在倒计时过程中显示倒计时信息
            if second <= 5:  # 最后5秒显示倒计时
                self.textBrowser_log.append(f"<font color='cyan'>Starting in {second} seconds...</font>")
        else:
            # 倒计时结束，停止延时定时器
            self.timer_collect_delay.stop()

            # 调用延时结束处理
            self.on_delay_timeout()

    def pushButton_collect_show(self):
        """采集按钮点击处理"""
        if self.pushButton_collect_start.text() == "start":
            # 检查参数是否有效
            if self.comboBox_collect_target.currentIndex() == 0 or self.spinBox_collect_number.value() == 0:
                err = QErrorMessage(self)
                err.setWindowTitle('Label parameter error')
                err.showMessage(
                    "Please check whether 'target' or 'number' is set")
                err.show()
                return

            # 检查用户名称是否已输入
            user_name = self.lineEdit_user.text().strip()
            if not user_name:
                err = QErrorMessage(self)
                err.setWindowTitle('User name error')
                err.showMessage("Please enter user name")
                err.show()
                return

            # 设置当前用户名称并发送到串口处理进程
            self.user_name = user_name
            self.serial_queue_write.put(f"set_user:{user_name}")

            # 保存当前设置
            self.label_number = self.spinBox_collect_number.value()
            self.label_delay = self.timeEdit_collect_delay.time()

            # 更新UI
            self.pushButton_collect_start.setText("stop")
            self.pushButton_collect_start.setStyleSheet("color: red")
            self.timeEdit_collect_delay.setStyleSheet("color: red")

            # 显示采集准备信息
            target = self.comboBox_collect_target.currentText()
            number = self.spinBox_collect_number.value()
            delay = self.timeEdit_collect_delay.time().toString("HH:mm:ss")
            duration = self.spinBox_collect_duration.value()
            self.textBrowser_log.append(
                f"<font color='cyan'>Preparing to collect: {target}, Count: {number}, Delay: {delay}, Duration: {duration}ms</font>")
            self.textBrowser_log.append(
                f"<font color='cyan'>文件将保存为: {target}_user{user_name}_[序列号].csv</font>")

            # 启动延时计时器
            self.timer_collect_delay.start()
        else:
            # 强制结束采集
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
        # self.textBrowser_log.moveCursor(self.textBrowser_log.textCursor().End)

    def show_device_info(self, device_info_series):
        """记录设备信息但不显示"""
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
        """处理命令下拉框选择事件"""
        command = self.comboBox_command.currentText()
        self.lineEdit_command.setText(command)

    def toggle_server_save(self, state):
        """切换服务器保存状态"""
        if state == Qt.Checked:
            # 发送启用服务器保存的命令
            self.serial_queue_write.put("enable_server_save")
            # 发送服务器地址
            server_url = self.lineEdit_server_url.text().strip()
            if server_url:
                # 清理和规范化URL
                server_url = server_url.strip()

                # 如果URL不包含协议，添加http://
                if not server_url.startswith(('http://', 'https://')):
                    server_url = 'http://' + server_url

                # 移除URL末尾的斜杠
                server_url = server_url.rstrip('/')

                # 如果URL不包含/api/csi_data，添加它
                if not server_url.endswith('/api/csi_data'):
                    server_url = server_url + '/api/csi_data'

                print(f"设置服务器地址为: {server_url}")
                # 使用特殊分隔符来避免URL中的冒号导致的问题
                self.serial_queue_write.put(f"set_server_url:{server_url}")
        else:
            # 发送禁用服务器保存的命令
            self.serial_queue_write.put("disable_server_save")


def quit(signum, frame):
    print("Exit the system")
    sys.exit()


class DataHandleThread(QThread):
    """数据处理线程类"""
    signal_device_info = pyqtSignal(pd.Series)
    signal_log_msg = pyqtSignal(str)
    signal_exit = pyqtSignal()

    def __init__(self, queue_read):
        super(DataHandleThread, self).__init__()
        self.queue_read = queue_read

    def run(self):
        while True:
            if not self.queue_read.empty():
                data_series = self.queue_read.get()

                if data_series['type'] == 'CSI_DATA':
                    self.handle_csi_data(data_series)
                elif data_series['type'] == 'DEVICE_INFO':
                    self.signal_device_info.emit(data_series)
                    self.signal_log_msg.emit(
                        f"<font color='green'>接收到设备信息: {data_series['app_revision']}</font>")
                elif data_series['type'] == 'LOG_DATA':
                    if data_series['tag'] == 'E':
                        self.signal_log_msg.emit(
                            f"<font color='red'>{data_series['data']}</font>")
                    elif data_series['tag'] == 'W':
                        self.signal_log_msg.emit(
                            f"<font color='yellow'>{data_series['data']}</font>")
                    else:
                        self.signal_log_msg.emit(
                            f"<font color='white'>{data_series['data']}</font>")
                elif data_series['type'] == 'FAIL_EVENT':
                    self.signal_log_msg.emit(
                        f"<font color='red'>{data_series['data']}</font>")
                    self.signal_exit.emit()
                    break

    def handle_csi_data(self, data):
        """处理CSI数据"""
        try:
            # 旋转数据
            g_csi_phase_array[:-1] = g_csi_phase_array[1:]
            g_rssi_array[:-1] = g_rssi_array[1:]
            g_radio_header_pd.iloc[1:] = g_radio_header_pd.iloc[:-1]

            # 解析CSI数据
            csi_raw_data = data['data']

            # 确保csi_raw_data是列表
            if not isinstance(csi_raw_data, list):
                # 静默转换，不输出日志
                try:
                    if isinstance(csi_raw_data, str):
                        # 尝试作为字符串解析
                        if csi_raw_data.startswith('[') and csi_raw_data.endswith(']'):
                            # 可能是JSON格式的列表
                            csi_raw_data = json.loads(csi_raw_data)
                        else:
                            # 尝试作为逗号分隔的字符串解析
                            csi_raw_data = [int(x) for x in csi_raw_data.split(',') if x.strip()]
                    else:
                        # 尝试转换为列表
                        csi_raw_data = list(csi_raw_data)
                except:
                    # 转换失败，使用空列表
                    csi_raw_data = []

            # 检查数据长度是否符合预期
            max_index = 104  # 使用固定值避免可能的异常
            try:
                max_index = max([index * 2 for index in csi_vaid_subcarrier_index])
            except:
                pass  # 使用默认值

            # 静默补全数据，不输出日志
            if not csi_raw_data:
                csi_raw_data = [0] * max_index
            elif len(csi_raw_data) < max_index:
                csi_raw_data.extend([0] * (max_index - len(csi_raw_data)))
            elif len(csi_raw_data) > max_index:
                csi_raw_data = csi_raw_data[:max_index]

            # 安全处理子载波数据
            for i in range(min(CSI_DATA_COLUMNS, len(csi_vaid_subcarrier_index))):
                try:
                    # 确保索引在范围内
                    if i >= len(csi_vaid_subcarrier_index):
                        continue

                    index = csi_vaid_subcarrier_index[i]
                    if index * 2 >= len(csi_raw_data) or index * 2 - 1 < 0:
                        continue

                    # 计算复数值
                    real_part = csi_raw_data[index * 2]
                    imag_part = csi_raw_data[index * 2 - 1]
                    data_complex = complex(real_part, imag_part)

                    # 安全地设置数组值，避免索引越界
                    if i < g_csi_phase_array.shape[1]:
                        g_csi_phase_array[-1][i] = np.abs(data_complex)
                except:
                    # 出错时设置为0，避免显示异常，静默处理
                    if i < g_csi_phase_array.shape[1]:
                        g_csi_phase_array[-1][i] = 0

            # 更新RSSI和无线电头信息
            try:
                g_rssi_array[-1] = int(data['rssi'])
            except (ValueError, TypeError):
                g_rssi_array[-1] = -100  # 默认值，静默处理

            # 将字符串数据转换为整数类型
            radio_header_data = {}
            for col in g_radio_header_pd.columns:
                try:
                    # 尝试转换为整数
                    if col in data:
                        radio_header_data[col] = int(data[col])
                    else:
                        radio_header_data[col] = 0
                except (ValueError, TypeError):
                    # 无法转换为整数，使用0作为默认值
                    radio_header_data[col] = 0

            # 将转换后的数据应用到DataFrame
            g_radio_header_pd.loc[0] = pd.Series(radio_header_data)

            # 记录动作采集数据
            if data['taget'] != 'unknown':
                self.signal_log_msg.emit(
                    f"<font color='cyan'>采集到动作数据: {data['taget']} - 序列 {data['taget_seq']}</font>")
        except Exception as e:
            # 只有关键错误才输出
            if "complex" in str(e) or "index" in str(e) or "shape" in str(e):
                self.signal_log_msg.emit(f"<font color='red'>CSI数据处理异常: {e}</font>")


def serial_handle(queue_read, queue_write, port):
    """处理串口通信的函数"""
    ser = None
    try:
        ser = serial.Serial(port=port, baudrate=115200,
                            bytesize=8, parity='N', stopbits=1, timeout=0.1)
    except Exception as e:
        print(f"串口打开失败: {e}")
        data_series = pd.Series(index=['type', 'data'],
                                data=['FAIL_EVENT', f"无法打开串口: {e}"])
        queue_read.put(data_series)
        sys.exit()
        return

    print("打开串口: ", port)
    print("CSI数据过滤已禁用，将保存所有数据包")

    # 等待串口初始化
    ser.flushInput()

    # 创建必要的文件夹
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

    # 初始化数据验证列表
    data_valid_list = pd.DataFrame(columns=['type', 'columns_names', 'file_name', 'file_fd', 'file_writer'],
                                   data=[["CSI_DATA", CSI_DATA_COLUMNS_NAMES, "log/csi_data.csv", None, None],
                                         ["DEVICE_INFO", DEVICE_INFO_COLUMNS_NAMES, "log/device_info.csv", None, None]])

    # 打开数据文件
    log_data_writer = None
    try:
        for data_valid in data_valid_list.iloc:
            data_valid['file_fd'] = open(data_valid['file_name'], 'w')
            data_valid['file_writer'] = csv.writer(data_valid['file_fd'])
            data_valid['file_writer'].writerow(data_valid['columns_names'])

        log_data_writer = open("../log/log_data.txt", 'w+')
    except Exception as e:
        print(f"打开文件失败: {e}")
        data_series = pd.Series(index=['type', 'data'],
                                data=['FAIL_EVENT', f"打开文件失败: {e}"])
        queue_read.put(data_series)
        if ser:
            ser.close()
        sys.exit()
        return

    # 数据包缓存和同步相关变量
    data_buffer = ""  # 用于缓存不完整的数据包
    last_complete_packet = None  # 用于存储最后一个完整的数据包
    packet_lock = threading.Lock()  # 用于同步数据包处理

    taget_last = 'unknown'
    taget_seq_last = 0
    csi_target_data_file_fd = None
    taget_data_writer = None

    # 数据处理控制参数
    last_queue_full_warning_time = 0  # 上次队列满警告的时间
    current_user_name = "unknown"  # 当前用户名称

    # 服务器配置
    enable_server_save = False  # 是否启用服务器保存
    server_url = "http://127.0.0.1:5000/api/csi_data"  # 默认服务器地址

    # 保存已创建的文件记录，格式为 {action}_{user}:{count}
    created_files = {}
    current_file_key = None  # 当前正在写入的文件标识
    current_file_count = 0  # 当前文件的计数

    # 发送重启命令
    try:
        ser.write("restart\r\n".encode('utf-8'))
        time.sleep(0.01)
    except Exception as e:
        print(f"发送重启命令失败: {e}")
        data_series = pd.Series(index=['type', 'data'],
                                data=['FAIL_EVENT', f"发送重启命令失败: {e}"])
        queue_read.put(data_series)
        if ser:
            ser.close()
        sys.exit()
        return

    while True:
        try:
            # 处理写队列中的命令
            if not queue_write.empty():
                command = queue_write.get()

                if command == "exit":
                    # 关闭所有资源
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

                # 检查是否是设置用户名称的命令
                if command.startswith("set_user:"):
                    current_user_name = command.split(":")[1]
                    print(f"设置当前用户为: {current_user_name}")
                    continue

                # 检查是否是启用服务器保存的命令
                if command == "enable_server_save":
                    enable_server_save = True
                    print("已启用服务器保存")
                    continue

                # 检查是否是禁用服务器保存的命令
                if command == "disable_server_save":
                    enable_server_save = False
                    print("已禁用服务器保存")
                    continue

                # 检查是否是设置服务器地址的命令
                if command.startswith("set_server_url:"):
                    new_url = command.split(":", 1)[1]  # 使用split(":", 1)只分割第一个冒号
                    if new_url and new_url != "http":  # 确保新URL不为空且不是"http"
                        server_url = new_url
                        print(f"设置服务器地址为: {server_url}")
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
            data_series = pd.Series(index=['type', 'data'],
                                    data=['FAIL_EVENT', f"串口操作异常: {e}"])
            queue_read.put(data_series)
            # 关闭所有资源
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

        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        if not strings:
            continue

        # 处理数据
        try:
            # 将新数据添加到缓冲区
            data_buffer += strings

            # 检查是否有完整的数据包
            for data_valid in data_valid_list.iloc:
                index = data_buffer.find(data_valid['type'])
                if index >= 0:
                    # 找到数据包开始位置
                    packet_start = index
                    # 尝试找到数据包结束位置（下一个数据包开始或缓冲区结束）
                    next_packet = data_buffer.find(data_valid['type'], packet_start + 1)
                    if next_packet == -1:
                        # 如果没有找到下一个数据包，说明当前数据包可能不完整
                        continue

                    # 提取完整的数据包
                    packet = data_buffer[packet_start:next_packet]
                    data_buffer = data_buffer[next_packet:]  # 更新缓冲区

                    # 处理数据包
                    csv_reader = csv.reader(StringIO(packet))
                    data = next(csv_reader)

                    if len(data) == len(data_valid['columns_names']):
                        data_series = pd.Series(
                            data, index=data_valid['columns_names'])

                        try:
                            datetime.strptime(
                                data_series['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                        except Exception as e:
                            data_series['timestamp'] = datetime.now().strftime(
                                '%Y-%m-%d %H:%M:%S.%f')[:-3]

                        if data_series['type'] == 'CSI_DATA':
                            with packet_lock:  # 使用锁确保数据包处理的同步性
                                try:
                                    csi_raw_data = base64_decode_bin(
                                        data_series['data'])
                                    if len(csi_raw_data) != int(data_series['len']):
                                        # 静默处理数据长度不匹配
                                        # print(f"CSI数据长度不匹配, 期望: {data_series['len']}, 实际: {len(csi_raw_data)}")
                                        # 尝试补齐或截断数据
                                        if len(csi_raw_data) < int(data_series['len']):
                                            # 数据不足，补零
                                            csi_raw_data.extend([0] * (int(data_series['len']) - len(csi_raw_data)))
                                            # print(f"数据不足，已补齐: {len(csi_raw_data)}")
                                        elif len(csi_raw_data) > int(data_series['len']):
                                            # 数据过长，截断
                                            csi_raw_data = csi_raw_data[:int(data_series['len'])]
                                            # print(f"数据过长，已截断: {len(csi_raw_data)}")
                                except Exception as e:
                                    # 解码错误，创建一个空数组，静默处理
                                    # print(f"base64解码错误: {e}")
                                    csi_raw_data = [0] * 128  # 假设最大数据长度为128
                                    data_series['len'] = len(csi_raw_data)
                                    # print(f"已创建空数据数组: {len(csi_raw_data)}")

                                # 更新数据
                                data_series['data'] = csi_raw_data

                                # 将解析后的原始数据写入日志
                                if not queue_read.full():
                                    queue_read.put(data_series)
                                else:
                                    # 记录队列已满
                                    curr_time = time.time()
                                    if curr_time - last_queue_full_warning_time > 5:  # 每5秒最多警告一次
                                        print("队列已满，丢弃CSI数据")
                                        last_queue_full_warning_time = curr_time

                                if data_series['taget'] != 'unknown':
                                    try:
                                        # 生成当前数据文件的标识
                                        file_key = f"{data_series['taget']}_{current_user_name}"

                                        # 检查是否需要创建新文件
                                        create_new_file = False

                                        if csi_target_data_file_fd is None:
                                            # 当前没有打开的文件
                                            create_new_file = True
                                        elif file_key != current_file_key:
                                            # 动作或用户变化
                                            create_new_file = True
                                        elif data_series['taget_seq'] == '1' and data_series['taget'] != taget_last:
                                            # 当序列号为1且动作改变时创建新文件
                                            create_new_file = True

                                        if create_new_file:
                                            # 关闭之前的文件
                                            if csi_target_data_file_fd:
                                                csi_target_data_file_fd.close()

                                            # 更新当前文件标识
                                            current_file_key = file_key

                                            # 如果是新的动作或用户，重置计数
                                            if file_key not in created_files:
                                                created_files[file_key] = 0

                                            # 增加计数
                                            created_files[file_key] += 1
                                            current_file_count = created_files[file_key]

                                            # 创建新的文件
                                            folder = f"data/{data_series['taget']}"
                                            if not path.exists(folder):
                                                mkdir(folder)

                                            # 生成新的文件名
                                            seq_formatted = f"{current_file_count:02d}"
                                            csi_target_data_file_name = f"{folder}/{data_series['taget']}_{current_user_name}_{seq_formatted}.csv"
                                            print(f"创建新的数据文件: {csi_target_data_file_name}")

                                            csi_target_data_file_fd = open(
                                                csi_target_data_file_name, 'w+')
                                            taget_data_writer = csv.writer(
                                                csi_target_data_file_fd)
                                            taget_data_writer.writerow(data_series.index)

                                        # 确保有打开的文件进行写入
                                        if csi_target_data_file_fd and taget_data_writer:
                                            # 将数据转换为CSV格式
                                            csv_data = {}
                                            for col in data_series.index:
                                                csv_data[col] = str(data_series[col])

                                            taget_data_writer.writerow(
                                                data_series.astype(str))
                                            # 立即刷新文件，确保数据写入磁盘
                                            csi_target_data_file_fd.flush()

                                            # 如果启用了服务器保存，发送数据到服务器
                                            if enable_server_save:
                                                # 准备要发送的数据
                                                server_data = {
                                                    'user_name': current_user_name,
                                                    'action': data_series['taget'],
                                                    'sequence': current_file_count,  # 使用当前文件计数作为序列号
                                                    'timestamp': data_series['timestamp'],
                                                    'csi_data': data_series['data'],
                                                    'rssi': data_series['rssi'],
                                                    'mac': data_series['mac'],
                                                    'csv_data': csv_data
                                                }
                                                send_data_to_server(server_data, server_url)

                                        # 更新最后的目标和序列号
                                        taget_last = data_series['taget']
                                        taget_seq_last = data_series['taget_seq']
                                    except Exception as e:
                                        # 只记录关键错误
                                        if "permission" in str(e) or "file" in str(e) or "directory" in str(e):
                                            print(f"写入目标数据文件失败: {e}")

                                # 总是更新最后的目标和序列号
                                taget_last = data_series['taget']
                                taget_seq_last = data_series['taget_seq']
                        else:
                            queue_read.put(data_series)

                        data_valid['file_writer'].writerow(data_series.astype(str))
                        data_valid['file_fd'].flush()
                        break
                else:
                    # 处理日志数据
                    strings = re.sub(r'\\x1b.*?m', '', strings)
                    log_data_writer.writelines(strings + "\n")
                    log_data_writer.flush()  # 立即刷新日志

                    log = re.match(r'.*([DIWE]) \((\d+)\) (.*)', strings, re.I)
                    if log:
                        data_series = pd.Series(index=['type', 'tag', 'timestamp', 'data'],
                                                data=['LOG_DATA', log.group(1), log.group(2), log.group(3)])
                        if not queue_read.full():
                            queue_read.put(data_series)
        except Exception as e:
            print(f"处理数据异常: {e}")
            continue


def send_data_to_server(data, server_url):
    """发送数据到后端服务器"""
    try:
        # 清理和规范化URL
        server_url = server_url.strip()

        # 如果URL不包含协议，添加http://
        if not server_url.startswith(('http://', 'https://')):
            server_url = 'http://' + server_url

        # 移除URL末尾的斜杠
        server_url = server_url.rstrip('/')

        # 如果URL不包含/api/csi_data，添加它
        if not server_url.endswith('/api/csi_data'):
            server_url = server_url + '/api/csi_data'

        # 处理序列号
        try:
            sequence = int(data.get('sequence', 0))
        except (ValueError, TypeError):
            sequence = 0

        # 确保CSI数据是可以序列化为JSON的格式
        csi_data = data.get('csi_data', [])
        if isinstance(csi_data, (list, tuple)):
            # 如果是列表，确保所有元素都是整数
            csi_data_list = [int(x) for x in csi_data]
        else:
            # 如果不是列表或元组，尝试转换为字符串
            try:
                csi_data_list = [int(x) for x in str(csi_data).split(',')]
            except:
                # 如果转换失败，使用空列表
                print("CSI数据格式无效，使用空列表")
                csi_data_list = []

        # 准备要发送的数据
        server_data = {
            'user_name': data.get('user_name', 'unknown'),
            'action': data.get('action', 'unknown'),
            'sequence': sequence,
            'timestamp': data.get('timestamp', ''),
            'csi_data': csi_data_list,
            'rssi': data.get('rssi', 0),
            'mac': data.get('mac', ''),
            'file_format': 'csv',  # 指定文件格式为CSV
            'file_name': f"{data.get('action', 'unknown')}_{data.get('user_name', 'unknown')}_{sequence:02d}.csv",
            # 与本地文件命名保持一致
            'csv_data': data.get('csv_data', {})  # 添加CSV数据
        }

        print(f"正在发送数据到服务器: {server_url}")
        print(f"文件名: {server_data['file_name']}")

        # 设置请求会话，禁用代理
        session = requests.Session()
        session.trust_env = False  # 禁用环境变量中的代理设置

        # 添加超时和重试机制
        for attempt in range(3):  # 最多重试3次
            try:
                response = session.post(server_url, json=server_data, timeout=5)
                if response.status_code == 200:
                    print(f"数据成功发送到服务器: {response.json()}")
                    return True
                else:
                    print(f"发送数据失败: {response.status_code} - {response.text}")
                    # 如果不是超时或网络错误，无需重试
                    if response.status_code != 408 and response.status_code < 500:
                        return False
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                print(f"发送数据到服务器超时或连接错误，尝试重试 ({attempt + 1}/3): {e}")
                time.sleep(1)  # 等待1秒后重试
                continue

        print("已达到最大重试次数，放弃发送")
        return False
    except Exception as e:
        print(f"发送数据到服务器时发生错误: {e}")
        return False


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
    user_id = args.user_id  # 获取用户ID
    description = args.description  # 获取描述信息

    # 增加队列容量，避免队列满的问题
    serial_queue_read = Queue(maxsize=512)
    serial_queue_write = Queue(maxsize=128)

    signal_key.signal(signal_key.SIGINT, quit)
    signal_key.signal(signal_key.SIGTERM, quit)

    serial_handle_process = Process(target=serial_handle, args=(
        serial_queue_read, serial_queue_write, serial_port))
    serial_handle_process.start()

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('../../../docs/_static/icon.png'))

    window = DataGraphicalWindow(serial_queue_read, serial_queue_write)
    data_handle_thread = DataHandleThread(serial_queue_read)
    data_handle_thread.signal_device_info.connect(window.show_device_info)
    data_handle_thread.signal_log_msg.connect(window.show_textBrowser_log)
    data_handle_thread.signal_exit.connect(window.close)
    data_handle_thread.start()

    window.show()
    sys.exit(app.exec())
    serial_handle_process.join()