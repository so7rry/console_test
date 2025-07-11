# -*-coding:utf-8-*-
import sys
import csv
import json
import re
import os
import argparse
import pandas as pd
import numpy as np

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
                              for i in range(1,  26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]
# 26  green
csi_vaid_subcarrier_index += [i for i in range(
    26, 52, CSI_VAID_SUBCARRIER_INTERVAL)]
csi_vaid_subcarrier_color += [(0, i * color_step, 0)
                              for i in range(1,  26 // CSI_VAID_SUBCARRIER_INTERVAL + 2)]

DEVICE_INFO_COLUMNS_NAMES = ["type", "timestamp", "compile_time", "chip_name", "chip_revision",
                             "app_revision", "idf_revision", "total_heap", "free_heap", "router_ssid", "ip", "port"]
g_device_info_series = None

CSI_DATA_INDEX = 500  # buffer size
CSI_DATA_COLUMNS = len(csi_vaid_subcarrier_index)
CSI_DATA_COLUMNS_NAMES = ["type", "seq", "timestamp", "taget_seq", "taget", "mac", "rssi", "rate", "sig_mode", "mcs",
                          "cwb", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding","sgi", "noise_floor", 
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
        bin_data = base64.b64decode(str_data)
    except Exception as e:
        print(f"Exception: {e}, data: {str_data}")

    list_data = list(bin_data)

    for i in range(len(list_data)):
        if list_data[i] > 127:
            list_data[i] = list_data[i] - 256

    return list_data


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
        command = f"radar --train_start"
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
            # print(f"temp_list: {temp_list}")
            data_str = ','.join(str(value)
                                for value in data_pd.loc[index]) + "\n"
            data_str = data_str.encode('utf-8')
            # print(f"data_str: {data_str}")
            tcpCliSock.send(data_str)

    tcpCliSock.close()
    time.sleep(1)

    if label == "train":
        command = "radar --train_stop"
        serial_queue_write.put(command)

    sys.exit(0)


class DataGraphicalWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, serial_queue_read, serial_queue_write, csi_packet_threshold=5):
        super().__init__()
        self.serial_queue_read = serial_queue_read
        self.serial_queue_write = serial_queue_write
        self.csi_packet_threshold = csi_packet_threshold
        self.user_name = "unknown"  # 默认用户名称
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
        wn = 20 / (CSI_SAMPLE_RATE / 2)
        b, a = signal.butter(8, wn, 'lowpass')

        if self.wave_filtering_flag:
            self.median_filtering(g_csi_phase_array)
            csi_filtfilt_data = signal.filtfilt(b, a, g_csi_phase_array.T).T
        else:
            csi_filtfilt_data = g_csi_phase_array

        # 计算数据范围，确保图表不会频繁自动调整
        data_min = csi_filtfilt_data.min()
        data_max = csi_filtfilt_data.max()

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

        for i in range(CSI_DATA_COLUMNS):
            self.curve_subcarrier[i].setData(csi_filtfilt_data[:, i])

        if self.wave_filtering_flag:
            csi_filtfilt_rssi = signal.filtfilt(
                b, a, g_rssi_array).astype(np.int32)
        else:
            csi_filtfilt_rssi = g_rssi_array
        self.curve_rssi.setData(csi_filtfilt_rssi)

    def show_curve_subcarrier_filter(self):
        self.wave_filtering_flag = self.checkBox_wave_filtering.isChecked()

    def command_boot(self):
        command = f"radar --csi_output_type LLFT --csi_output_format base64"
        self.serial_queue_write.put(command)

        if self.checkBox_router_auto_connect.isChecked() and len(self.lineEdit_router_ssid.text()) > 0:
            self.command_router_connect()

        self.timer_boot_command.stop()

    def command_predict_config(self):
        command = (f"radar --predict_someone_sensitivity {self.doubleSpinBox_predict_someone_sensitivity.value()}" +
                   f" --predict_move_sensitivity {self.doubleSpinBox_predict_move_sensitivity.value()}")
        self.serial_queue_write.put(command)
        command = (f"radar --predict_buff_size {self.spinBox_predict_buffer_size.text()}" +
                   f" --predict_outliers_number {self.spinBox_predict_outliers_number.text()}")
        self.serial_queue_write.put(command)

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
        self.textBrowser_log.append(f"<font color='cyan'>采集开始: {target} - 第 {self.current_collect_count}/{self.target_collect_count} 次，持续时间: {duration}ms</font>")
        
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
            self.textBrowser_log.append(f"<font color='cyan'>Preparing to collect: {target}, Count: {number}, Delay: {delay}, Duration: {duration}ms</font>")
            self.textBrowser_log.append(f"<font color='cyan'>文件将保存为: {target}_user{user_name}_[序列号].csv</font>")
            
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
            self.textBrowser_log.append(f"<font color='green'>设备连接成功: {device_info_series['app_revision']}</font>")

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
        # 旋转数据
        g_csi_phase_array[:-1] = g_csi_phase_array[1:]
        g_rssi_array[:-1] = g_rssi_array[1:]
        g_radio_header_pd.iloc[1:] = g_radio_header_pd.iloc[:-1]

        # 解析CSI数据
        csi_raw_data = data['data']
        for i in range(CSI_DATA_COLUMNS):
            data_complex = complex(csi_raw_data[csi_vaid_subcarrier_index[i] * 2],
                              csi_raw_data[csi_vaid_subcarrier_index[i] * 2 - 1])
            g_csi_phase_array[-1][i] = np.abs(data_complex)

        # 更新RSSI和无线电头信息
        g_rssi_array[-1] = int(data['rssi'])
        
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

def serial_handle(queue_read, queue_write, port, csi_filter=5):
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
    print(f"CSI数据过滤率: 每{csi_filter}个数据包处理1个")

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

    taget_last = 'unknown'
    taget_seq_last = 0
    csi_target_data_file_fd = None
    taget_data_writer = None
    
    # 数据处理控制参数
    csi_packet_counter = 0
    csi_packet_threshold = csi_filter  # 使用传入的过滤率参数
    last_queue_full_warning_time = 0  # 上次队列满警告的时间
    current_user_name = "unknown"  # 当前用户名称
    
    # 保存已创建的文件记录，格式为 {action}_{user}:{count}
    created_files = {}

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
            for data_valid in data_valid_list.iloc:
                index = strings.find(data_valid['type'])
                if index >= 0:
                    strings = strings[index:]
                    csv_reader = csv.reader(StringIO(strings))
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
                            try:
                                csi_raw_data = base64_decode_bin(
                                    data_series['data'])
                                if len(csi_raw_data) != int(data_series['len']):
                                    print(
                                        f"CSI数据长度不匹配, 期望: {data_series['len']}, 实际: {len(csi_raw_data)}")
                                    break
                            except Exception as e:
                                print(
                                    f"base64解码错误: {e}, 数据: {data_series['data']}")
                                break

                            data_series['data'] = csi_raw_data

                            if data_series['taget'] != 'unknown':
                                try:
                                    if data_series['taget'] != taget_last or data_series['taget_seq'] != taget_seq_last:
                                        # 先关闭之前的文件
                                        if csi_target_data_file_fd:
                                            csi_target_data_file_fd.close()
                                            
                                        folder = f"data/{data_series['taget']}"
                                        if not path.exists(folder):
                                            mkdir(folder)

                                        # 使用动作名称和用户名生成文件标识
                                        action_name = data_series['taget']
                                        file_key = f"{action_name}_{current_user_name}"
                                        
                                        # 检查这个组合是否已经存在，如果存在则递增计数
                                        if file_key in created_files:
                                            created_files[file_key] += 1
                                        else:
                                            created_files[file_key] = 1
                                            
                                        # 生成两位数的序号
                                        seq_formatted = f"{created_files[file_key]:02d}"
                                        
                                        csi_target_data_file_name = f"{folder}/{action_name}_{current_user_name}_{seq_formatted}.csv"
                                        print(f"创建数据文件: {csi_target_data_file_name}")
                                        csi_target_data_file_fd = open(
                                            csi_target_data_file_name, 'w+')
                                        taget_data_writer = csv.writer(
                                            csi_target_data_file_fd)
                                        taget_data_writer.writerow(data_series.index)

                                    taget_data_writer.writerow(
                                        data_series.astype(str))
                                    # 立即刷新文件，确保数据写入磁盘
                                    csi_target_data_file_fd.flush()
                                except Exception as e:
                                    print(f"写入目标数据文件失败: {e}")

                            taget_last = data_series['taget']
                            taget_seq_last = data_series['taget_seq']

                            if not queue_read.full():
                                queue_read.put(data_series)
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


if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print(" Python version should >= 3.6")
        exit()

    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device")
    parser.add_argument('--csi_filter', dest='csi_filter', type=int, default=5,
                        help="Filter rate for CSI data (process 1 out of N packets, default=5)")
    parser.add_argument('--user_id', dest='user_id', type=str, default="01",
                        help="User ID for data collection (default=01)")
    parser.add_argument('--desc', dest='description', type=str, default="",
                        help="Optional description to add to data files")

    args = parser.parse_args()
    serial_port = args.port
    csi_packet_threshold = args.csi_filter  # 获取用户设置的CSI数据包过滤率
    user_id = args.user_id  # 获取用户ID
    description = args.description  # 获取描述信息

    serial_queue_read = Queue(maxsize=128)
    serial_queue_write = Queue(maxsize=64)

    signal_key.signal(signal_key.SIGINT, quit)
    signal_key.signal(signal_key.SIGTERM, quit)

    serial_handle_process = Process(target=serial_handle, args=(
        serial_queue_read, serial_queue_write, serial_port, csi_packet_threshold))
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
