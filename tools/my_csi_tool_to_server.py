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
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QProgressBar, QGroupBox, QCheckBox, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QComboBox, QTimeEdit, QSpinBox, QPushButton, QWidget, QErrorMessage, QBoxLayout)
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


def base64_decode_bin(str_data):
    try:
        if not str_data:
            return []
        # 移除额外的填充处理，直接解码
        bin_data = base64.b64decode(str_data)
        list_data = list(bin_data)
        for i in range(len(list_data)):
            if list_data[i] > 127:
                list_data[i] = list_data[i] - 256
        return list_data
    except Exception as e:
        print(f"Base64解码异常: {e}")
        return []


def base64_encode_bin(list_data):
    for i in range(len(list_data)):
        if list_data[i] < 0:
            list_data[i] = 256 + list_data[i]
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
        self.user_name = "unknown"
        self.server_url = "http://8.136.10.160:12786/api/csi_data"
        self.enable_server_save = False
        self.current_collect_count = 0
        self.target_collect_count = 0
        self.collect_counters = {}  # {user_name: {action: count}}
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setFormat("%p%")
        self.csi_enabled = False
        self.wifi_connected = False
        self.initUI()
        QTimer.singleShot(3000, lambda: self.textBrowser_log.append(
            f"<font color='yellow'>系统初始化中，将自动设置CSI并连接WiFi</font>"))

    def initUI(self):
        self.setupUi(self)
        global g_display_raw_data
        self.setWindowTitle("ESP CSI Tool")
        for child in self.findChildren(QGroupBox):
            if child.title() == "Raw data":
                child.setTitle("")
        self.create_collect_window()
        if hasattr(self, 'verticalLayout_17'):
            self.verticalLayout_17.addWidget(self.collect_group)
        self.checkBox_server_save = QCheckBox("保存到服务器")
        self.checkBox_server_save.setChecked(True)
        self.checkBox_server_save.stateChanged.connect(self.toggle_server_save)
        if hasattr(self, 'verticalLayout_17'):
            self.verticalLayout_17.addWidget(self.checkBox_server_save)
        server_layout = QHBoxLayout()
        server_label = QLabel("服务器地址:")
        self.lineEdit_server_url = QLineEdit()
        self.lineEdit_server_url.setText(self.server_url)
        self.lineEdit_server_url.setPlaceholderText("输入服务器地址 (例如: http://8.136.10.160:12786/api/csi_data)")
        server_layout.addWidget(server_label)
        server_layout.addWidget(self.lineEdit_server_url)
        if hasattr(self, 'verticalLayout_17'):
            self.verticalLayout_17.addLayout(server_layout)
        if hasattr(self, 'verticalLayout_17'):
            self.verticalLayout_17.addWidget(self.progressBar)
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
        with open("./config/gui_config.json") as file:
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
        if hasattr(self, 'groupBox_radar_model'):
            self.groupBox_radar_model.hide()
        if hasattr(self, 'QWidget_evaluate_info'):
            self.QWidget_evaluate_info.hide()
        if hasattr(self, 'groupBox_20'):
            self.groupBox_20.hide()
        if hasattr(self, 'groupBox_eigenvalues'):
            self.groupBox_eigenvalues.hide()
        if hasattr(self, 'groupBox_eigenvalues_table'):
            self.groupBox_eigenvalues_table.hide()
        if hasattr(self, 'groupBox_statistics'):
            self.groupBox_statistics.hide()
        if hasattr(self, 'groupBox_predict'):
            self.groupBox_predict.hide()
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
        for widget in self.findChildren(QLabel):
            if hasattr(widget, 'text') and widget.text() == "display:":
                parent = widget.parent()
                if parent and isinstance(parent, QWidget):
                    parent_layout = parent.layout()
                    if parent_layout:
                        parent_layout.removeWidget(widget)
                widget.hide()
        for child in self.findChildren(QGroupBox):
            if child.title().lower() == "info":
                if child.parent():
                    parent_widget = child.parent()
                    if isinstance(parent_widget, QWidget):
                        layout = parent_widget.layout()
                        if layout:
                            layout.removeWidget(child)
                child.setParent(None)
                child.deleteLater()
        self.label_number = 0
        self.label_delay = QTime(0, 0, 3)
        self.is_collecting = False
        self.collection_delay_timer = QTimer()
        self.collection_delay_timer.setSingleShot(True)
        self.collection_delay_timer.timeout.connect(self.on_delay_timeout)
        self.collection_duration_timer = QTimer()
        self.collection_duration_timer.setSingleShot(True)
        self.collection_duration_timer.timeout.connect(self.on_duration_timeout)
        self.curve_subcarrier_range = np.array([10, 20])
        self.graphicsView_subcarrier.setYRange(self.curve_subcarrier_range[0], self.curve_subcarrier_range[1],
                                               padding=0)
        self.graphicsView_subcarrier.addLegend()
        self.graphicsView_subcarrier.setMouseEnabled(x=True, y=False)
        self.graphicsView_subcarrier.enableAutoRange(axis='y', enable=False)
        self.graphicsView_subcarrier.setMenuEnabled(False)
        self.graphicsView_rssi.setYRange(-100, 0, padding=0)
        self.graphicsView_rssi.setMouseEnabled(x=True, y=False)
        self.graphicsView_rssi.enableAutoRange(axis='y', enable=False)
        self.graphicsView_rssi.setMenuEnabled(False)
        self.curve_subcarrier = []
        self.serial_queue_write = serial_queue_write
        for i in range(CSI_DATA_COLUMNS):
            curve = self.graphicsView_subcarrier.plot(g_csi_phase_array[:, i], name=str(i),
                                                      pen=csi_vaid_subcarrier_color[i])
            self.curve_subcarrier.append(curve)
        self.curve_rssi = self.graphicsView_rssi.plot(g_rssi_array, name='rssi', pen=(255, 255, 255))
        self.wave_filtering_flag = self.checkBox_wave_filtering.isCheckable()
        self.checkBox_wave_filtering.released.connect(self.show_curve_subcarrier_filter)
        self.pushButton_router_connect.released.connect(self.command_router_connect)
        self.pushButton_command.released.connect(self.command_custom)
        self.comboBox_command.activated.connect(self.comboBox_command_show)
        self.checkBox_raw_data.released.connect(self.checkBox_raw_data_show)
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
        self.checkBox_raw_data_show()
        self.textBrowser_log.setStyleSheet("background:black")

        # 程序启动时自动启用服务器保存
        QTimer.singleShot(1000, lambda: self.toggle_server_save(Qt.CheckState.Checked))

    def create_collect_window(self):
        self.collect_group = QGroupBox(self.centralwidget)
        self.collect_group.setTitle("Collect")
        self.collect_group.setFont(QFont("Arial", 10))
        collect_layout = QVBoxLayout(self.collect_group)
        collect_layout.setContentsMargins(5, 5, 5, 5)
        horizontal_layout = QHBoxLayout()
        label_user = QLabel("user")
        horizontal_layout.addWidget(label_user)
        self.lineEdit_user = QLineEdit()
        self.lineEdit_user.setPlaceholderText("输入用户名称")
        self.lineEdit_user.setFixedWidth(100)
        horizontal_layout.addWidget(self.lineEdit_user)
        label_target = QLabel("target")
        horizontal_layout.addWidget(label_target)
        self.comboBox_collect_target = QComboBox()
        for target in CSI_DATA_TARGETS:
            self.comboBox_collect_target.addItem(target)
        horizontal_layout.addWidget(self.comboBox_collect_target)
        label_delay = QLabel("delay")
        horizontal_layout.addWidget(label_delay)
        self.timeEdit_collect_delay = QTimeEdit()
        self.timeEdit_collect_delay.setDisplayFormat("HH:mm:ss")
        self.timeEdit_collect_delay.setTime(QTime(0, 0, 3))
        horizontal_layout.addWidget(self.timeEdit_collect_delay)
        label_duration = QLabel("duration(ms)")
        horizontal_layout.addWidget(label_duration)
        self.spinBox_collect_duration = QSpinBox()
        self.spinBox_collect_duration.setMinimum(100)
        self.spinBox_collect_duration.setMaximum(10000)
        self.spinBox_collect_duration.setSingleStep(100)
        self.spinBox_collect_duration.setValue(500)
        horizontal_layout.addWidget(self.spinBox_collect_duration)
        # 移除number相关控件
        # 自动采集按钮
        self.pushButton_collect_clean = QPushButton("clean")
        horizontal_layout.addWidget(self.pushButton_collect_clean)
        self.pushButton_collect_start = QPushButton("start")
        horizontal_layout.addWidget(self.pushButton_collect_start)
        collect_layout.addLayout(horizontal_layout)
        log_container = None
        for widget in self.findChildren(QWidget):
            if hasattr(widget, 'objectName') and 'log' in widget.objectName().lower():
                log_container = widget
                break
        parent = log_container.parent() if log_container and log_container.parent() else None
        parent_layout = parent.layout() if isinstance(parent, QWidget) and parent.layout() is not None else None
        if parent_layout is not None:
            index = -1
            for i in range(parent_layout.count()):
                item = parent_layout.itemAt(i)
                if item is not None and hasattr(item, 'widget') and item.widget() == log_container:
                    index = i
                    break
            if index >= 0 and isinstance(parent_layout, QBoxLayout):
                try:
                    parent_layout.insertWidget(index + 1, self.collect_group)
                except Exception:
                    if hasattr(self, 'verticalLayout_17'):
                        self.verticalLayout_17.addWidget(self.collect_group)
            else:
                if hasattr(self, 'verticalLayout_17'):
                    self.verticalLayout_17.addWidget(self.collect_group)
        else:
            if hasattr(self, 'verticalLayout_17'):
                self.verticalLayout_17.addWidget(self.collect_group)
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
        with open("./config/gui_config.json", "r") as file:
            gui_config = json.load(file)
            gui_config['display_raw_data'] = self.checkBox_raw_data.isChecked()
        with open("./config/gui_config.json", "w") as file:
            json.dump(gui_config, file)

    def show_router_auto_connect(self):
        with open("./config/gui_config.json", "r") as file:
            gui_config = json.load(file)
            gui_config['router_auto_connect'] = self.checkBox_router_auto_connect.isChecked()
        with open("./config/gui_config.json", "w") as file:
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
            butter_result = signal.butter(8, wn, 'lowpass')
            if butter_result is not None:
                b, a = butter_result[0], butter_result[1]
            else:
                b, a = [1], [1]
            if self.wave_filtering_flag:
                self.median_filtering(g_csi_phase_array)
                csi_filtfilt_data = signal.filtfilt(b, a, g_csi_phase_array.T).T
            else:
                csi_filtfilt_data = g_csi_phase_array
            data_min = np.nanmin(csi_filtfilt_data)
            data_max = np.nanmax(csi_filtfilt_data)
            if np.isnan(data_min):
                data_min = 0
            if np.isnan(data_max):
                data_max = 10
            need_update_range = False
            if data_min < self.curve_subcarrier_range[0]:
                self.curve_subcarrier_range[0] = data_min - 2
                need_update_range = True
            if data_max > self.curve_subcarrier_range[1]:
                self.curve_subcarrier_range[1] = data_max + 2
                need_update_range = True
            if need_update_range:
                self.graphicsView_subcarrier.setYRange(self.curve_subcarrier_range[0], self.curve_subcarrier_range[1],
                                                       padding=0)
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
            if self.wave_filtering_flag:
                csi_filtfilt_rssi = signal.filtfilt(b, a, g_rssi_array).astype(np.int32)
            else:
                csi_filtfilt_rssi = g_rssi_array
            self.curve_rssi.setData(csi_filtfilt_rssi)
        except Exception as e:
            print(f"显示CSI数据异常: {e}")

    def show_curve_subcarrier_filter(self):
        self.wave_filtering_flag = self.checkBox_wave_filtering.isChecked()

    def command_boot(self):
        """初始化设备，设置CSI格式并启用CSI功能"""
        print("正在初始化...")
        self.textBrowser_log.append(f"<font color='cyan'>正在初始化...</font>")
        command = f"radar --csi_output_type LLFT --csi_output_format base64"
        self.serial_queue_write.put(command)
        self.textBrowser_log.append(f"<font color='green'>CSI 输出格式已设置</font>")
        QTimer.singleShot(500, lambda: self.enable_csi())
        # 自动连接WiFi，且按钮状态同步
        QTimer.singleShot(1500, lambda: self.command_router_connect() if not self.wifi_connected else None)
        self.timer_boot_command.stop()

    def enable_csi(self):
        """启用 CSI 功能"""
        command = f"radar --csi_en 1"
        self.serial_queue_write.put(command)
        self.textBrowser_log.append(f"<font color='green'>CSI 功能已启用</font>")
        self.csi_enabled = True

    def command_router_connect(self):
        """连接或断开WiFi"""
        if self.pushButton_router_connect.text() == "connect":
            self.pushButton_router_connect.setText("disconnect")
            self.pushButton_router_connect.setStyleSheet("color: red")

            ssid = self.lineEdit_router_ssid.text()
            password = self.lineEdit_router_password.text()

            if len(ssid) == 0:
                self.textBrowser_log.append(f"<font color='red'>错误: SSID 不能为空</font>")
                self.pushButton_router_connect.setText("connect")
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
            self.pushButton_router_connect.setText("connect")
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
        self.serial_queue_write.put("end_task")

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
        self.textBrowser_log.append(f"<font color='yellow'>采集已停止</font>")

    def on_delay_timeout(self):
        """延时结束，开始采集"""
        # 重置样式
        self.timeEdit_collect_delay.setStyleSheet("color: black")
        self.timeEdit_collect_delay.setTime(self.label_delay)  # 恢复原始设置的延时值
        
        # 停止延迟计时器
        if self.collection_delay_timer.isActive():
            self.collection_delay_timer.stop()
        if self.timer_collect_delay.isActive():
            self.timer_collect_delay.stop()
        
        # 开始采集
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

        # 停止所有计时器
        if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
            self.progress_timer.stop()
        if self.timer_collect_delay.isActive():
            self.timer_collect_delay.stop()
        if self.collection_delay_timer.isActive():
            self.collection_delay_timer.stop()
        if self.collection_duration_timer.isActive():
            self.collection_duration_timer.stop()
        
        # 中止任何正在进行的采集
        command = "radar --collect_number 0 --collect_tagets unknown"
        self.serial_queue_write.put(command)
        
        # 发送任务结束命令
        self.serial_queue_write.put("end_task")

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
            if second <= 5 or second % 5 == 0:  # 每5秒显示一次，最后5秒每秒显示
                self.textBrowser_log.append(f"<font color='cyan'>将在 {second} 秒后开始采集...</font>")
        else:
            self.timer_collect_delay.stop()
            self.timeEdit_collect_delay.setStyleSheet("color: black")  # 重置样式
            self.textBrowser_log.append(f"<font color='green'>延时结束，开始采集...</font>")
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
            
            # 获取延迟时间（秒）
            delay_seconds = self.label_delay.hour() * 3600 + self.label_delay.minute() * 60 + self.label_delay.second()
            
            if delay_seconds > 0:
                # 如果有延迟，启动延迟计时器并开始倒计时
                self.collection_delay_timer.start(delay_seconds * 1000)  # 毫秒为单位
                self.timer_collect_delay.start()
                self.textBrowser_log.append(f"<font color='cyan'>将在 {delay_seconds} 秒后开始采集...</font>")
            else:
                # 如果没有延迟，直接开始采集
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


def quit(signum, frame):
    print("Exit the system")
    sys.exit()


class DataHandleThread(QThread):
    signal_device_info = pyqtSignal(pd.Series)
    signal_log_msg = pyqtSignal(str)
    signal_exit = pyqtSignal()
    signal_wifi_status = pyqtSignal(str)

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
        try:
            g_csi_phase_array[:-1] = g_csi_phase_array[1:]
            g_rssi_array[:-1] = g_rssi_array[1:]
            g_radio_header_pd.iloc[1:] = g_radio_header_pd.iloc[:-1]

            csi_raw_data = data['data']
            if not isinstance(csi_raw_data, list):
                try:
                    if isinstance(csi_raw_data, str):
                        if csi_raw_data.startswith('[') and csi_raw_data.endswith(']'):
                            csi_raw_data = json.loads(csi_raw_data)
                        else:
                            csi_raw_data = [int(x) for x in csi_raw_data.split(',') if x.strip()]
                    else:
                        csi_raw_data = list(csi_raw_data)
                except:
                    csi_raw_data = []

            # 确保数据长度正确
            max_index = max([index * 2 for index in csi_vaid_subcarrier_index]) if csi_vaid_subcarrier_index else 104
            if not csi_raw_data:
                csi_raw_data = [0] * max_index
            elif len(csi_raw_data) < max_index:
                csi_raw_data.extend([0] * (max_index - len(csi_raw_data)))
            elif len(csi_raw_data) > max_index:
                csi_raw_data = csi_raw_data[:max_index]

            # 直接处理CSI数据
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
                except:
                    if i < g_csi_phase_array.shape[1]:
                        g_csi_phase_array[-1][i] = 0

            # 处理RSSI数据
            try:
                g_rssi_array[-1] = int(data['rssi'])
            except (ValueError, TypeError):
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

            # 显示目标数据信息
            if data['taget'] != 'unknown':
                self.signal_log_msg.emit(
                    f"<font color='cyan'>采集到动作数据: {data['taget']} - 序列 {data['taget_seq']}</font>")

        except Exception as e:
            # 只记录关键错误
            if "complex" in str(e) or "index" in str(e) or "shape" in str(e):
                self.signal_log_msg.emit(f"<font color='red'>CSI数据处理异常: {e}</font>")


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
                                        # 解码Base64数据
                                        csi_raw_data = base64_decode_bin(data_series['data'])

                                        # 检查数据长度
                                        if len(csi_raw_data) != int(data_series['len']):
                                            if len(csi_raw_data) < int(data_series['len']):
                                                csi_raw_data.extend([0] * (int(data_series['len']) - len(csi_raw_data)))
                                            elif len(csi_raw_data) > int(data_series['len']):
                                                csi_raw_data = csi_raw_data[:int(data_series['len'])]
                                    except Exception as e:
                                        csi_raw_data = [0] * 128
                                        data_series['len'] = len(csi_raw_data)
                                        print(f"CSI数据解码异常: {e}")

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
    # 连接WiFi状态信号
    data_handle_thread.signal_wifi_status.connect(window.handle_wifi_status)
    data_handle_thread.start()
    window.show()
    exit_code = app.exec()
    serial_handle_process.join()
    sys.exit(exit_code)