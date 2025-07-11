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
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import pyqtgraph as pq
import threading
from PyQt5.QtCore import QDate, QTime, QDateTime
import base64
import time
from datetime import datetime
from multiprocessing import Process, Queue
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from esp_csi_tool_gui import Ui_MainWindow
from scipy import signal
import signal as signal_key
import socket

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
g_radio_header_pd = pd.DataFrame(np.zeros([10, len(CSI_DATA_COLUMNS_NAMES[1:-1])], dtype=np.int32), columns=CSI_DATA_COLUMNS_NAMES[1:-1])

def base64_decode_bin(str_data):
    try:
        if not str_data:
            return []
        padding_needed = len(str_data) % 4
        if padding_needed > 0:
            str_data += '=' * (4 - padding_needed)
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
        # 修改为嵌套字典,外层是用户名,内层是动作类型
        self.collect_counters = {}  # {user_name: {action: count}}
        self.initUI()

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
                if widget.parent():
                    parent_layout = widget.parent().layout()
                    if parent_layout:
                        parent_layout.removeWidget(widget)
                widget.hide()
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
        self.graphicsView_subcarrier.setYRange(self.curve_subcarrier_range[0], self.curve_subcarrier_range[1], padding=0)
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
            curve = self.graphicsView_subcarrier.plot(g_csi_phase_array[:, i], name=str(i), pen=csi_vaid_subcarrier_color[i])
            self.curve_subcarrier.append(curve)
        self.curve_rssi = self.graphicsView_rssi.plot(g_rssi_array, name='rssi', pen=(255, 255, 255))
        self.wave_filtering_flag = self.checkBox_wave_filtering.isCheckable()
        self.checkBox_wave_filtering.released.connect(self.show_curve_subcarrier_filter)
        self.checkBox_router_auto_connect.released.connect(self.show_router_auto_connect)
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
        QTimer.singleShot(1000, lambda: self.toggle_server_save(Qt.Checked))

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
                self.graphicsView_subcarrier.setYRange(self.curve_subcarrier_range[0], self.curve_subcarrier_range[1], padding=0)
            if len(self.curve_subcarrier) != csi_filtfilt_data.shape[1]:
                self.graphicsView_subcarrier.clear()
                self.graphicsView_subcarrier.addLegend()
                self.curve_subcarrier = []
                for i in range(min(CSI_DATA_COLUMNS, csi_filtfilt_data.shape[1])):
                    if i < len(csi_vaid_subcarrier_color):
                        curve = self.graphicsView_subcarrier.plot(csi_filtfilt_data[:, i], name=str(i), pen=csi_vaid_subcarrier_color[i])
                    else:
                        curve = self.graphicsView_subcarrier.plot(csi_filtfilt_data[:, i], name=str(i), pen=(255, 255, 255))
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
        if self.is_collecting:
            return
        self.is_collecting = True
        
        # 获取当前用户和动作类型
        user_name = self.lineEdit_user.text().strip()
        target = self.comboBox_collect_target.currentText()
        
        # 初始化用户和动作的计数器
        if user_name not in self.collect_counters:
            self.collect_counters[user_name] = {}
        if target not in self.collect_counters[user_name]:
            self.collect_counters[user_name][target] = 0
            
        # 增加当前用户和动作的计数
        self.collect_counters[user_name][target] += 1
        
        # 生成包含序号的任务ID
        task_id = f"{target}_{self.collect_counters[user_name][target]:02d}"
        print(f"开始新的采集任务: {task_id}, 用户: {user_name}, 当前计数: {self.collect_counters[user_name][target]}")
        
        self.serial_queue_write.put(f"start_task:{task_id}")
        if user_name:
            self.serial_queue_write.put(f"set_user:{user_name}")
        duration = self.spinBox_collect_duration.value()
        command = (f"radar --collect_number 1" +
                   f" --collect_tagets {target}" +
                   f" --collect_duration {duration}")
        self.serial_queue_write.put(command)
        self.textBrowser_log.append(
            f"<font color='cyan'>采集开始: {target} - 用户: {user_name} - 第 {self.collect_counters[user_name][target]} 次，持续时间: {duration}ms</font>")
        self.collection_duration_timer.start(duration + 200)

    def command_collect_target_stop(self):
        if not self.is_collecting:
            return
        self.serial_queue_write.put("radar --collect_number 0 --collect_tagets unknown")
        self.serial_queue_write.put("end_task:")
        self.is_collecting = False
        if self.collection_duration_timer.isActive():
            self.collection_duration_timer.stop()
        self.textBrowser_log.append(f"<font color='yellow'>采集暂停</font>")

    def on_delay_timeout(self):
        self.timeEdit_collect_delay.setStyleSheet("color: black")
        self.command_collect_target_start()

    def on_duration_timeout(self):
        self.command_collect_target_stop()
        QTimer.singleShot(100, self.process_next_collection)

    def process_next_collection(self):
        if self.current_collect_count < self.target_collect_count:
            print(f"处理下一次采集: 当前计数 {self.current_collect_count}, 目标次数 {self.target_collect_count}")  # 修改日志
            self.command_collect_target_start()
        else:
            self.finish_collection()

    def finish_collection(self):
        if self.is_collecting:
            self.command_collect_target_stop()
        self.timeEdit_collect_delay.setTime(self.label_delay)
        self.pushButton_collect_start.setStyleSheet("color: black")
        self.timeEdit_collect_delay.setStyleSheet("color: black")
        self.pushButton_collect_start.setText("start")
        if self.timer_collect_delay.isActive():
            self.timer_collect_delay.stop()
        if self.collection_delay_timer.isActive():
            self.collection_delay_timer.stop()
        if self.collection_duration_timer.isActive():
            self.collection_duration_timer.stop()
        
        # 获取当前用户和动作类型
        user_name = self.lineEdit_user.text().strip()
        target = self.comboBox_collect_target.currentText()
        
        if user_name in self.collect_counters and target in self.collect_counters[user_name]:
            print(f"采集完成: 用户 {user_name} 的 {target} 动作共 {self.collect_counters[user_name][target]} 次")
            self.textBrowser_log.append(f"<font color='green'>采集完成：用户 {user_name} 的 {target} 动作共 {self.collect_counters[user_name][target]} 次</font>")
        
        # 重置计数器，为下一次采集做准备
        self.current_collect_count = 0
        self.target_collect_count = 0

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
        if self.pushButton_collect_start.text() == "start":
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
                
            # 如果用户名改变,重置所有动作类型的计数器
            if user_name != self.user_name:
                self.collect_counters[user_name] = {}
                print(f"用户改变: {self.user_name} -> {user_name}, 重置所有计数器")
                
            self.user_name = user_name
            self.serial_queue_write.put(f"set_user:{user_name}")
            self.label_delay = self.timeEdit_collect_delay.time()
            self.pushButton_collect_start.setText("stop")
            self.pushButton_collect_start.setStyleSheet("color: red")
            self.timeEdit_collect_delay.setStyleSheet("color: red")
            target = self.comboBox_collect_target.currentText()
            delay = self.timeEdit_collect_delay.time().toString("HH:mm:ss")
            duration = self.spinBox_collect_duration.value()
            self.textBrowser_log.append(
                f"<font color='cyan'>准备采集: {target}, 延迟: {delay}, 持续时间: {duration}ms</font>")
            self.textBrowser_log.append(
                f"<font color='cyan'>文件将保存为: {target}_{user_name}_[序列号].csv</font>")
            self.timer_collect_delay.start()
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
        if state == Qt.Checked:
            self.serial_queue_write.put("enable_server_save")
            server_url = self.lineEdit_server_url.text().strip()
            if server_url:
                server_url = server_url.strip()
                if not server_url.startswith(('http://', 'https://')):
                    server_url = 'http://' + server_url
                server_url = server_url.rstrip('/')
                if not server_url.endswith('/api/csi_data'):
                    server_url = server_url + '/api/csi_data'
                print(f"设置服务器地址为: {server_url}")
                self.serial_queue_write.put(f"set_server_url:{server_url}")
                self.textBrowser_log.append(f"<font color='green'>已启用服务器保存，服务器地址: {server_url}</font>")
            else:
                self.textBrowser_log.append(f"<font color='yellow'>警告：服务器地址为空，请设置有效的服务器地址</font>")
        else:
            self.serial_queue_write.put("disable_server_save")
            self.textBrowser_log.append(f"<font color='yellow'>已禁用服务器保存</font>")

def quit(signum, frame):
    print("Exit the system")
    sys.exit()

class DataHandleThread(QThread):
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
            max_index = 104
            try:
                max_index = max([index * 2 for index in csi_vaid_subcarrier_index])
            except:
                pass
            if not csi_raw_data:
                csi_raw_data = [0] * max_index
            elif len(csi_raw_data) < max_index:
                csi_raw_data.extend([0] * (max_index - len(csi_raw_data)))
            elif len(csi_raw_data) > max_index:
                csi_raw_data = csi_raw_data[:max_index]
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
            try:
                g_rssi_array[-1] = int(data['rssi'])
            except (ValueError, TypeError):
                g_rssi_array[-1] = -100
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
            if data['taget'] != 'unknown':
                self.signal_log_msg.emit(
                    f"<font color='cyan'>采集到动作数据: {data['taget']} - 序列 {data['taget_seq']}</font>")
        except Exception as e:
            if "complex" in str(e) or "index" in str(e) or "shape" in str(e):
                self.signal_log_msg.emit(f"<font color='red'>CSI数据处理异常: {e}</font>")

def save_and_send_task_data(task_data_buffer, task_id, user_name, server_url, enable_server_save):
    if not task_data_buffer:
        print("task_data_buffer 为空，未保存或发送数据")
        return
    try:
        # 从task_id中提取动作和序号
        task_parts = task_id.split('_')
        # 处理带下划线的动作名称
        if len(task_parts) > 2:  # 如果动作名称包含下划线
            action = '_'.join(task_parts[:-1])  # 合并动作名称部分
            sequence = task_parts[-1]  # 最后一部分是序号
        else:
            action = task_parts[0]
            sequence = task_parts[1] if len(task_parts) > 1 else "01"
        
    folder = f"data/{action}"
    if not path.exists(folder):
        mkdir(folder)
        
        # 生成文件名
        filename = f"{action}_{user_name}_{sequence}.csv"
        filepath = os.path.join(folder, filename)
        
        print(f"保存数据到文件: {filepath}, 序号: {sequence}")
        
        # 保存到本地文件
        with open(filepath, 'w+', newline='') as f:
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
        server_url = server_url.strip()
        if not server_url.startswith(('http://', 'https://')):
            server_url = 'http://' + server_url
        server_url = server_url.rstrip('/')
        if not server_url.endswith('/api/csi_data'):
            server_url = server_url + '/api/csi_data'
        
        print(f"正在发送数据到服务器: {server_url}, 数据量: {len(data)} 条")
        
        # 在请求前打印一些调试信息
        print(f"数据类型: {type(data)}, 第一条数据键: {list(data[0].keys()) if data and isinstance(data[0], dict) else '未知'}")
        
        session = requests.Session()
        session.trust_env = False
        for attempt in range(3):
            try:
                print(f"尝试发送请求到服务器({attempt + 1}/3)...")
                response = session.post(server_url, json=data, timeout=10)  # 增加超时时间
                print(f"服务器响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"数据成功发送到服务器: {response.text}")
                    return True
                else:
                    print(f"发送数据失败: {response.status_code} - {response.text}")
                    # 尝试下一次重试而不是立即返回
                    if attempt == 2:  # 最后一次尝试
                    return False
                    time.sleep(2)  # 在重试前等待更长时间
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                print(f"发送数据到服务器超时或连接错误，尝试重试 ({attempt + 1}/3): {e}")
                time.sleep(2)  # 增加重试间隔
                continue
            except Exception as e:
                print(f"发送请求时发生未预期错误: {type(e).__name__}: {str(e)}")
                # 尝试以字符串形式查看数据的前100个字符
                try:
                    print(f"请求数据预览: {str(data)[:100]}...")
                except:
                    print("无法打印数据预览")
                if attempt == 2:  # 最后一次尝试
                return False
                time.sleep(2)
        print("已达到最大重试次数，放弃发送")
        return False
    except Exception as e:
        print(f"发送数据到服务器时发生错误: {type(e).__name__}: {str(e)}")
        return False

def serial_handle(queue_read, queue_write, port):
    ser = None
    try:
        ser = serial.Serial(port=port, baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=0.1)
    except Exception as e:
        print(f"串口打开失败: {e}")
        data_series = pd.Series(index=['type', 'data'], data=['FAIL_EVENT', f"无法打开串口: {e}"])
        queue_read.put(data_series)
        sys.exit()
        return
    print("打开串口: ", port)
    print("CSI数据过滤已禁用，将保存所有数据包")
    ser.flushInput()
    folder_list = ['log', 'data']
    for folder in folder_list:
        if not path.exists(folder):
            try:
                mkdir(folder)
            except Exception as e:
                print(f"创建文件夹失败: {folder}, 错误: {e}")
                data_series = pd.Series(index=['type', 'data'], data=['FAIL_EVENT', f"创建文件夹失败: {folder}, 错误: {e}"])
                queue_read.put(data_series)
                if ser:
                    ser.close()
                sys.exit()
                return
    data_valid_list = pd.DataFrame(columns=['type', 'columns_names', 'file_name', 'file_fd', 'file_writer'],
                                   data=[["CSI_DATA", CSI_DATA_COLUMNS_NAMES, "log/csi_data.csv", None, None],
                                         ["DEVICE_INFO", DEVICE_INFO_COLUMNS_NAMES, "log/device_info.csv", None, None]])
    log_data_writer = None
    try:
        for data_valid in data_valid_list.iloc:
            data_valid['file_fd'] = open(data_valid['file_name'], 'w')
            data_valid['file_writer'] = csv.writer(data_valid['file_fd'])
            data_valid['file_writer'].writerow(data_valid['columns_names'])
        log_data_writer = open("../log/log_data.txt", 'w+')
    except Exception as e:
        print(f"打开文件失败: {e}")
        data_series = pd.Series(index=['type', 'data'], data=['FAIL_EVENT', f"打开文件失败: {e}"])
        queue_read.put(data_series)
        if ser:
            ser.close()
        sys.exit()
        return
    data_buffer = ""
    last_complete_packet = None
    packet_lock = threading.Lock()
    taget_last = 'unknown'
    taget_seq_last = 0
    csi_target_data_file_fd = None
    taget_data_writer = None
    last_queue_full_warning_time = 0
    current_user_name = "unknown"
    enable_server_save = False
    server_url = "http://8.136.10.160:12786/api/csi_data"
    created_files = {}
    task_data_buffer = []
    current_task_id = None
    current_sequence = 0  # 添加当前序号计数器
    try:
        ser.write("restart\r\n".encode('utf-8'))
        time.sleep(0.01)
        print(f"当前服务器状态：URL={server_url}, 保存={enable_server_save}")
    except Exception as e:
        print(f"发送重启命令失败: {e}")
        data_series = pd.Series(index=['type', 'data'], data=['FAIL_EVENT', f"发送重启命令失败: {e}"])
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
                    task_parts = current_task_id.split('_')
                    if len(task_parts) > 1:
                        current_sequence = int(task_parts[-1])  # 更新当前序号
                    task_data_buffer.clear()
                    print(f"开始新任务: {current_task_id}, 序号: {current_sequence}")
                    continue
                if command == "end_task" or command.startswith("end_task:"):
                    if task_data_buffer and current_task_id:
                        print(f"处理end_task命令，准备保存和发送数据：任务ID={current_task_id}，数据量={len(task_data_buffer)}，服务器保存={enable_server_save}")
                        save_and_send_task_data(task_data_buffer, current_task_id, current_user_name, server_url, enable_server_save)
                    task_data_buffer.clear()
                    current_task_id = None
                    continue
                command = command + "\r\n"
                ser.write(command.encode('utf-8'))
                print(f"{datetime.now()}, 串口写入: {command}")
                continue
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
        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        if not strings:
            continue
        try:
            data_buffer += strings
            for data_valid in data_valid_list.iloc:
                index = data_buffer.find(data_valid['type'])
                if index >= 0:
                    packet_start = index
                    next_packet = data_buffer.find(data_valid['type'], packet_start + 1)
                    if next_packet == -1:
                        continue
                    packet = data_buffer[packet_start:next_packet]
                    data_buffer = data_buffer[next_packet:]
                    csv_reader = csv.reader(StringIO(packet))
                    data = next(csv_reader)
                    if len(data) == len(data_valid['columns_names']):
                        data_series = pd.Series(data, index=data_valid['columns_names'])
                        try:
                            datetime.strptime(data_series['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                        except Exception as e:
                            data_series['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        if data_series['type'] == 'CSI_DATA':
                            with packet_lock:
                                try:
                                    csi_raw_data = base64_decode_bin(data_series['data'])
                                    if len(csi_raw_data) != int(data_series['len']):
                                        if len(csi_raw_data) < int(data_series['len']):
                                            csi_raw_data.extend([0] * (int(data_series['len']) - len(csi_raw_data)))
                                        elif len(csi_raw_data) > int(data_series['len']):
                                            csi_raw_data = csi_raw_data[:int(data_series['len'])]
                                except Exception as e:
                                    csi_raw_data = [0] * 128
                                    data_series['len'] = len(csi_raw_data)
                                data_series['data'] = csi_raw_data
                                if not queue_read.full():
                                    queue_read.put(data_series)
                                else:
                                    curr_time = time.time()
                                    if curr_time - last_queue_full_warning_time > 5:
                                        print("队列已满，丢弃CSI数据")
                                        last_queue_full_warning_time = curr_time
                                if data_series['taget'] != 'unknown' and current_task_id:
                                    task_data_buffer.append(data_series.astype(str).to_dict())
                                    # 从current_task_id中提取动作和序号
                                    task_parts = current_task_id.split('_')
                                    # 处理带下划线的动作名称
                                    if len(task_parts) > 2:  # 如果动作名称包含下划线
                                        action = '_'.join(task_parts[:-1])  # 合并动作名称部分
                                    else:
                                        action = task_parts[0]
                                    
                                    # 使用当前序号生成文件名
                                    sequence = f"{current_sequence:02d}"
                                    file_key = f"{action}_{current_user_name}_{sequence}"
                                    
                                    # 如果这个文件还没有被创建
                                    if file_key not in created_files:
                                        folder = f"data/{action}"
                                        if not path.exists(folder):
                                            mkdir(folder)
                                        csi_target_data_file_name = f"{folder}/{file_key}.csv"
                                        csi_target_data_file_fd = open(csi_target_data_file_name, 'w+')
                                        taget_data_writer = csv.writer(csi_target_data_file_fd)
                                        taget_data_writer.writerow(data_series.index)
                                        created_files[file_key] = csi_target_data_file_fd
                                        print(f"创建新文件: {csi_target_data_file_name}")
                                    
                                    # 写入数据到对应的文件
                                    if file_key in created_files:
                                        taget_data_writer = csv.writer(created_files[file_key])
                                        taget_data_writer.writerow(data_series.astype(str))
                                        created_files[file_key].flush()
                                    
                                taget_last = data_series['taget']
                                taget_seq_last = data_series['taget_seq']
                        else:
                            queue_read.put(data_series)
                        data_valid['file_writer'].writerow(data_series.astype(str))
                        data_valid['file_fd'].flush()
                        break
                else:
                    strings = re.sub(r'\\x1b.*?m', '', strings)
                    log_data_writer.writelines(strings + "\n")
                    log_data_writer.flush()
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
    parser.add_argument('--user_id', dest='user_id', type=str, default="01",
                        help="User ID for data collection (default=01)")
    parser.add_argument('--desc', dest='description', type=str, default="",
                        help="Optional description to add to data files")
    args = parser.parse_args()
    serial_port = args.port
    user_id = args.user_id
    description = args.description
    serial_queue_read = Queue(maxsize=512)
    serial_queue_write = Queue(maxsize=128)
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
    data_handle_thread.signal_exit.connect(window.close)
    data_handle_thread.start()
    window.show()
    sys.exit(app.exec())
    serial_handle_process.join()