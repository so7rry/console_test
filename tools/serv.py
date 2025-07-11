#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSI数据接收服务器示例代码
此代码用于接收ESP-CSI工具发送的批量数据并保存为CSV文件
同时提供实时动作识别API，使用预训练模型进行预测
"""

# 配置GPU
import os
import tensorflow as tf

# GPU内存设置
try:
    # 允许GPU内存动态增长而不是一次性分配全部
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"已启用GPU内存动态增长: {len(gpus)}个GPU可用")
    else:
        print("没有检测到GPU，将使用CPU")
except Exception as e:
    print(f"GPU配置错误: {e}")

from flask import Flask, request, jsonify
import csv
import logging
import numpy as np  # 添加numpy导入，确保np可用
import time
import json
from datetime import datetime
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 保存数据的目录
DATA_DIR = "server_data"
FILES_DIR = os.path.join(DATA_DIR, "files")
MODEL_DIR = os.path.join(DATA_DIR, "models")

# 预训练模型路径
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'action_recognition_model.pkl')
# 可以在这里指定其他模型路径
CUSTOM_MODEL_PATH = os.environ.get('CSI_MODEL_PATH', DEFAULT_MODEL_PATH)

# 全局变量用于存储加载的模型
loaded_model = None

# 添加全局变量，用于跟踪动作变化
last_predictions = deque(maxlen=10)  # 保存最近10次预测结果
last_csi_features = deque(maxlen=3)  # 保存最近3次CSI特征

# 确保目录存在
def ensure_directories():
    try:
        os.makedirs(FILES_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info(f"数据保存目录: {os.path.abspath(FILES_DIR)}")
        logger.info(f"模型目录: {os.path.abspath(MODEL_DIR)}")
    except Exception as e:
        logger.error(f"创建目录失败: {e}")
        raise

# 初始化目录
ensure_directories()

# 加载预训练模型
def load_model(model_path=None):
    global loaded_model
    
    if model_path is None:
        model_path = CUSTOM_MODEL_PATH
    
    try:
        import pickle
        
        # 检查是否有同名的SavedModel目录
        # 例如: 如果model_path是"xxx/model.pkl"，则检查"xxx/model"是否是SavedModel目录
        saved_model_dir = None
        if model_path.endswith('.pkl'):
            saved_model_dir = model_path[:-4]  # 移除.pkl后缀
        
        # 首先尝试使用TensorFlow的方式加载SavedModel
        if saved_model_dir and os.path.isdir(saved_model_dir) and 'saved_model.pb' in os.listdir(saved_model_dir):
            try:
                logger.info(f"检测到SavedModel目录: {saved_model_dir}")
                loaded_model = tf.saved_model.load(saved_model_dir)
                logger.info(f"模型加载成功(SavedModel): {saved_model_dir}")
                
                # 检查模型信息
                model_info = {}
                if hasattr(loaded_model, 'signatures'):
                    model_info['signatures'] = list(loaded_model.signatures.keys())
                    logger.info(f"模型签名: {model_info['signatures']}")
                    
                    # 检查签名的输入和输出
                    for sig_name in model_info['signatures']:
                        sig = loaded_model.signatures[sig_name]
                        try:
                            if hasattr(sig, 'structured_input_signature'):
                                input_spec = sig.structured_input_signature
                                if len(input_spec) > 1 and input_spec[1]:
                                    logger.info(f"签名 {sig_name} 的输入键: {list(input_spec[1].keys())}")
                            
                            if hasattr(sig, 'structured_outputs'):
                                output_spec = sig.structured_outputs
                                logger.info(f"签名 {sig_name} 的输出键: {list(output_spec.keys())}")
                        except Exception as e:
                            logger.warning(f"检查签名 {sig_name} 失败: {e}")
                
                # 尝试获取模型的输入/输出信息
                if hasattr(loaded_model, '__call__'):
                    logger.info("模型有__call__方法，可以直接调用")
                else:
                    logger.warning("模型没有__call__方法，需要使用signatures调用")
                
                return True
            except Exception as e:
                logger.warning(f"使用tf.saved_model.load加载失败: {e}")
                # 继续尝试pickle方式加载
        
        # 检查原始路径是否是SavedModel目录
        if os.path.isdir(model_path) and 'saved_model.pb' in os.listdir(model_path):
            try:
                logger.info(f"检测到路径是SavedModel目录: {model_path}")
                loaded_model = tf.saved_model.load(model_path)
                logger.info(f"模型加载成功(SavedModel): {model_path}")
                
                # 检查模型信息
                model_info = {}
                if hasattr(loaded_model, 'signatures'):
                    model_info['signatures'] = list(loaded_model.signatures.keys())
                    logger.info(f"模型签名: {model_info['signatures']}")
                    
                    # 检查签名的输入和输出
                    for sig_name in model_info['signatures']:
                        sig = loaded_model.signatures[sig_name]
                        try:
                            if hasattr(sig, 'structured_input_signature'):
                                input_spec = sig.structured_input_signature
                                if len(input_spec) > 1 and input_spec[1]:
                                    logger.info(f"签名 {sig_name} 的输入键: {list(input_spec[1].keys())}")
                            
                            if hasattr(sig, 'structured_outputs'):
                                output_spec = sig.structured_outputs
                                logger.info(f"签名 {sig_name} 的输出键: {list(output_spec.keys())}")
                        except Exception as e:
                            logger.warning(f"检查签名 {sig_name} 失败: {e}")
                
                return True
            except Exception as e:
                logger.warning(f"使用tf.saved_model.load加载失败: {e}")
                # 继续尝试pickle方式加载
        
        # 使用pickle加载模型
        if os.path.exists(model_path) and os.path.isfile(model_path):
            try:
                with open(model_path, 'rb') as f:
                    loaded_model = pickle.load(f)
                logger.info(f"模型加载成功(pickle): {model_path}")
                
                # 确保TensorFlow模型使用正确的设备配置
                if hasattr(loaded_model, 'compile'):
                    try:
                        # 重新编译模型以确保使用正确的配置
                        loaded_model.compile(
                            loss='categorical_crossentropy',
                            optimizer='adam', 
                            metrics=['accuracy']
                        )
                        logger.info("已重新编译TensorFlow模型")
                    except Exception as comp_e:
                        logger.warning(f"重新编译模型失败: {comp_e}")
                
                # 检查模型类型和输出类别
                if hasattr(loaded_model, 'classes_'):
                    logger.info(f"模型类别: {loaded_model.classes_}")
                    logger.info(f"类别数量: {len(loaded_model.classes_)}")
                elif isinstance(loaded_model, tf.keras.Model):
                    # 检查Keras模型的输出层
                    try:
                        output_shape = loaded_model.output_shape
                        logger.info(f"模型输出形状: {output_shape}")
                        if isinstance(output_shape, tuple) and len(output_shape) > 1:
                            num_classes = output_shape[1]
                            logger.info(f"输出类别数量: {num_classes}")
                    except Exception as e:
                        logger.warning(f"获取模型输出形状失败: {e}")
                
                return True
            except Exception as e:
                logger.error(f"使用pickle加载模型失败: {e}")
                return False
        else:
            logger.warning(f"模型文件不存在: {model_path}")
            return False
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return False

# 尝试加载模型
load_model()

# 动作识别函数 - 使用预训练模型或模拟预测
def predict_action(csi_data):
    """
    基于CSI数据预测动作
    
    Args:
        csi_data: 包含CSI数据的列表
    
    Returns:
        dict: 包含预测结果和置信度的字典
    """
    try:
        global loaded_model
        
        # 如果模型未加载，尝试重新加载
        if loaded_model is None:
            load_model()
        
        # 如果模型加载成功，使用模型预测
        if loaded_model is not None:
            try:
                # 从CSI数据中提取特征
                features = extract_features_from_csi(csi_data)
                
                # 记录模型输入信息以便调试
                logger.info(f"模型输入形状: {features.shape}, 非零元素比例: {np.count_nonzero(features)/features.size:.2%}")
                
                # 确保输入数据类型正确
                features = features.astype(np.float32)
                
                # 使用模型预测 - 区分不同类型的模型
                if isinstance(loaded_model, tf.keras.Model):
                    # Keras/TensorFlow 模型
                    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                        try:
                            predictions = loaded_model.predict(features, verbose=0)
                            prediction_idx = np.argmax(predictions[0])
                            confidence = float(predictions[0][prediction_idx])
                        except Exception as gpu_err:
                            logger.warning(f"GPU预测失败，尝试CPU: {gpu_err}")
                            # 如果GPU失败，尝试CPU
                            with tf.device('/CPU:0'):
                                predictions = loaded_model.predict(features, verbose=0)
                                prediction_idx = np.argmax(predictions[0])
                                confidence = float(predictions[0][prediction_idx])
                elif hasattr(loaded_model, 'predict'):
                    # Scikit-learn 模型
                    prediction_idx = loaded_model.predict([features.reshape(-1)])[0]
                    
                    # 获取预测概率（如果模型支持）
                    if hasattr(loaded_model, 'predict_proba'):
                        probabilities = loaded_model.predict_proba([features.reshape(-1)])[0]
                        confidence = float(probabilities[prediction_idx])
                    else:
                        confidence = 0.7  # 默认置信度
                else:
                    # 处理使用tf.saved_model.load()加载的SavedModel
                    try:
                        # 对于SavedModel，尝试使用不同的调用方式
                        if hasattr(loaded_model, '__call__') or hasattr(loaded_model, 'signatures'):
                            # 如果模型是可调用的或有signatures
                            try:
                                # 检查模型的输入要求
                                # 先转换特征数据为张量
                                tensor_features = tf.convert_to_tensor(features, dtype=tf.float32)
                                
                                # 尝试直接调用模型
                                if hasattr(loaded_model, '__call__'):
                                    logger.info("使用__call__方法调用SavedModel")
                                    
                                    # 尝试不同的输入格式
                                    try:
                                        # 直接使用特征张量
                                        predictions = loaded_model(tensor_features)
                                        logger.info("SavedModel直接调用成功")
                                    except Exception as e:
                                        logger.warning(f"直接调用失败，尝试使用字典输入: {e}")
                                        # 尝试使用字典作为输入
                                        predictions = loaded_model({'input_1': tensor_features})
                                
                                # 尝试使用默认signature
                                elif 'serving_default' in loaded_model.signatures:
                                    logger.info("使用serving_default signature调用SavedModel")
                                    
                                    # 获取signature的输入细节
                                    serving_fn = loaded_model.signatures['serving_default']
                                    
                                    # 检查signature的输入键
                                    try:
                                        input_keys = list(serving_fn.structured_input_signature[1].keys())
                                        logger.info(f"Signature输入键: {input_keys}")
                                        # 使用第一个输入键
                                        input_dict = {input_keys[0]: tensor_features}
                                        predictions = serving_fn(**input_dict)
                                    except (IndexError, AttributeError, TypeError) as e:
                                        logger.warning(f"获取signature输入键失败: {e}，尝试直接调用")
                                        # 如果无法获取输入键，直接尝试调用
                                        predictions = serving_fn(inputs=tensor_features)
                                
                                else:
                                    # 尝试第一个可用的signature
                                    signature_key = list(loaded_model.signatures.keys())[0]
                                    logger.info(f"使用signature {signature_key}调用SavedModel")
                                    serving_fn = loaded_model.signatures[signature_key]
                                    
                                    # 检查signature的输入键
                                    try:
                                        input_keys = list(serving_fn.structured_input_signature[1].keys())
                                        logger.info(f"Signature输入键: {input_keys}")
                                        # 使用第一个输入键
                                        input_dict = {input_keys[0]: tensor_features}
                                        predictions = serving_fn(**input_dict)
                                    except (IndexError, AttributeError, TypeError) as e:
                                        logger.warning(f"获取signature输入键失败: {e}，尝试直接调用")
                                        # 如果无法获取输入键，直接尝试调用
                                        predictions = serving_fn(inputs=tensor_features)
                                
                                # 处理预测结果
                                if isinstance(predictions, dict):
                                    # 如果返回值是字典（常见于SignatureDef结果）
                                    output_key = list(predictions.keys())[0]  # 获取第一个输出键
                                    prediction_tensor = predictions[output_key]
                                    
                                    # 转换为numpy数组并获取最大值索引
                                    prediction_values = prediction_tensor.numpy()
                                    logger.info(f"预测输出形状: {prediction_values.shape}, 值范围: [{prediction_values.min():.4f}, {prediction_values.max():.4f}]")
                                    
                                    if len(prediction_values.shape) > 1:
                                        prediction_idx = np.argmax(prediction_values[0])
                                        confidence = float(prediction_values[0][prediction_idx])
                                        logger.info(f"预测值: {prediction_values[0]}")
                                    else:
                                        prediction_idx = np.argmax(prediction_values)
                                        confidence = float(prediction_values[prediction_idx])
                                        logger.info(f"预测值: {prediction_values}")
                                else:
                                    # 如果返回值是张量
                                    prediction_values = predictions.numpy()
                                    logger.info(f"预测输出形状: {prediction_values.shape}, 值范围: [{prediction_values.min():.4f}, {prediction_values.max():.4f}]")
                                    
                                    if len(prediction_values.shape) > 1:
                                        prediction_idx = np.argmax(prediction_values[0])
                                        confidence = float(prediction_values[0][prediction_idx])
                                        logger.info(f"预测值: {prediction_values[0]}")
                                    else:
                                        prediction_idx = np.argmax(prediction_values)
                                        confidence = float(prediction_values[prediction_idx])
                                        logger.info(f"预测值: {prediction_values}")
                                
                                logger.info(f"SavedModel预测成功，预测索引: {prediction_idx}, 置信度: {confidence:.2f}")
                            except Exception as call_err:
                                logger.error(f"调用SavedModel失败: {call_err}")
                                import traceback
                                logger.error(traceback.format_exc())
                                raise
                        else:
                            logger.error("SavedModel不支持调用或没有signatures")
                            raise ValueError("SavedModel不支持调用或没有signatures")
                    except Exception as saved_model_err:
                        logger.error(f"SavedModel预测失败: {saved_model_err}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise
                
                # 获取动作标签 - 使用更健壮的方式处理
                try:
                    # 尝试从模型中获取类别名称
                    if hasattr(loaded_model, 'classes_'):
                        actions = loaded_model.classes_
                    else:
                        # 使用默认标签列表
                        actions = ["站立", "行走", "坐下", "躺下", "弯腰", "跌倒"]
                        
                    if isinstance(prediction_idx, np.ndarray) and prediction_idx.size == 1:
                        prediction_idx = prediction_idx.item()
                    
                    action = actions[prediction_idx] if prediction_idx < len(actions) else "未知"
                except Exception as label_err:
                    logger.error(f"获取标签出错: {label_err}")
                    action = f"类别{prediction_idx}"
                
                logger.info(f"模型预测结果: {action}, 置信度: {confidence:.2f}")
                return {
                    "prediction": action,
                    "confidence": confidence,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    "method": "model"
                }
            except Exception as e:
                logger.error(f"模型预测失败，使用模拟预测: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # 如果模型预测失败，回退到模拟预测
        
        # 使用模拟预测
        logger.info("使用模拟预测")
        # 从CSI数据中提取特征
        data_length = len(csi_data)
        rssi_values = [entry.get('rssi', -100) for entry in csi_data]
        avg_rssi = sum(rssi_values) / len(rssi_values) if rssi_values else -100
        
        # 简单的规则模拟不同动作
        actions = ["站立", "行走", "坐下", "躺下", "弯腰", "跌倒"]
        
        # 基于数据量和RSSI模拟不同动作
        if data_length < 10:
            action = "未知"
            confidence = 0.2
        else:
            # 使用时间戳作为随机种子，使结果有一定变化但相近时间内保持一致
            timestamp = int(time.time()) // 5  # 每5秒变化一次
            np.random.seed(timestamp)
            
            # 随机选择一个动作，但让结果有一定的连续性
            action_idx = np.random.choice(len(actions), p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1])
            action = actions[action_idx]
            
            # 生成一个合理的置信度
            base_confidence = 0.7
            noise = np.random.uniform(-0.2, 0.2)
            confidence = min(max(base_confidence + noise, 0.4), 0.95)
        
        return {
            "prediction": action,
            "confidence": confidence,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "method": "simulation"
        }
    except Exception as e:
        logger.error(f"预测过程发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "prediction": "错误",
            "confidence": 0.0,
            "error": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "method": "error"
        }

def extract_features_from_csi(csi_data):
    """
    从CSI数据中提取特征，适配DS_GFT模型的输入要求
    CSI数据格式: 每个子载波有一对值[虚部,实部]，共52个子载波，即104个值
    
    Args:
        csi_data: 包含CSI数据的列表
    
    Returns:
        numpy.ndarray: 适合模型输入的特征数组，形状为 (1, 300, 52)
    """
    try:
        # 创建一个零矩阵作为输入，形状为 (1, 300, 52)
        features = np.zeros((1, 300, 52))
        
        # 首先尝试理解数据结构
        if len(csi_data) == 0:
            logger.warning("CSI数据为空")
            return features
        
        # 获取第一条记录的结构
        sample = csi_data[0]
        logger.info(f"处理CSI数据，可用字段: {list(sample.keys())}")
        
        # 如果每条记录有完整的CSI数据数组
        if 'data' in sample and isinstance(sample['data'], list):
            logger.info("使用CSI data数组填充特征矩阵")
            
            # 收集所有数据点以便进行归一化
            all_magnitudes = []
            
            # 第一遍：收集所有幅值，用于归一化
            for entry in csi_data:
                if 'data' in entry and entry['data']:
                    raw_data = entry['data']
                    
                    # 确保数据长度是偶数 (每对是[虚部,实部])
                    if len(raw_data) % 2 != 0:
                        raw_data = raw_data[:-1]
                    
                    for j in range(0, min(104, len(raw_data)), 2):
                        if j+1 < len(raw_data):
                            imag = raw_data[j]      # 虚部
                            real = raw_data[j+1]    # 实部
                            # 计算复数的模(幅值): √(虚部²+实部²)
                            magnitude = np.sqrt(imag**2 + real**2)
                            all_magnitudes.append(magnitude)
            
            # 计算幅值统计信息，用于归一化
            if all_magnitudes:
                mag_min = np.min(all_magnitudes)
                mag_max = np.max(all_magnitudes)
                mag_range = mag_max - mag_min if mag_max > mag_min else 1
                mag_mean = np.mean(all_magnitudes)
                mag_std = np.std(all_magnitudes) if np.std(all_magnitudes) > 0 else 1
                logger.info(f"CSI幅值范围: min={mag_min:.2f}, max={mag_max:.2f}, range={mag_range:.2f}, mean={mag_mean:.2f}, std={mag_std:.2f}")
            else:
                mag_min = 0
                mag_range = 1
                mag_mean = 0
                mag_std = 1
            
            # 第二遍：填充特征矩阵，并应用归一化
            for i, entry in enumerate(csi_data):
                if i >= 300:  # 最多使用300个时间步
                    break
                    
                if 'data' in entry and entry['data']:
                    raw_data = entry['data']
                    
                    # 确保数据长度是偶数 (每对是[虚部,实部])
                    if len(raw_data) % 2 != 0:
                        logger.warning(f"CSI数据长度不是偶数: {len(raw_data)}")
                        raw_data = raw_data[:-1]  # 去掉最后一个元素使长度为偶数
                    
                    # 提取幅值并归一化
                    subcarrier_features = []
                    for j in range(0, min(104, len(raw_data)), 2):
                        if j+1 < len(raw_data):
                            imag = raw_data[j]      # 虚部
                            real = raw_data[j+1]    # 实部
                            # 计算复数的模(幅值): √(虚部²+实部²)
                            magnitude = np.sqrt(imag**2 + real**2)
                            
                            # 尝试使用Z-score归一化代替Min-Max归一化
                            # 这可以更好地处理异常值和不同量级的数据
                            normalized_magnitude = (magnitude - mag_mean) / mag_std
                            
                            # 限制到合理范围
                            normalized_magnitude = np.clip(normalized_magnitude, -3, 3)
                            
                            # 重新缩放到[0,1]范围
                            normalized_magnitude = (normalized_magnitude + 3) / 6
                            
                            subcarrier_features.append(normalized_magnitude)
                    
                    # 确保特征数量正确
                    if len(subcarrier_features) > 52:
                        subcarrier_features = subcarrier_features[:52]
                    
                    # 填充到特征矩阵
                    features[0, i, :len(subcarrier_features)] = subcarrier_features
            
        # 验证特征是否包含非零值
        if np.count_nonzero(features) == 0:
            logger.warning("警告：提取的特征全为零，可能表示数据格式无法正确解析")
        
        # 应用一些额外的处理来增强特征
        # 1. 确保数据范围适合神经网络 (通常[-1,1]或[0,1]是好的范围)
        features = np.clip(features, 0, 1)
        
        # 2. 如果特征太稀疏，可以应用平滑处理
        if np.count_nonzero(features)/features.size < 0.1:  # 如果非零元素少于10%
            # 应用简单的时间维度平滑
            for i in range(1, features.shape[1]-1):
                # 使用简单的3点移动平均
                features[0, i] = (features[0, i-1] + features[0, i] + features[0, i+1]) / 3
        
        logger.info(f"特征提取成功：形状={features.shape}，非零元素数量={np.count_nonzero(features)}，非零元素比例={np.count_nonzero(features)/features.size:.2%}")
        return features
        
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 返回一个正确形状的默认特征矩阵
        return np.zeros((1, 300, 52))

# 服务器连接测试
@app.route('/api/test_connection', methods=['POST', 'GET'])
def test_connection():
    try:
        if request.method == 'POST':
            data = request.json
            logger.info(f"收到连接测试: {data}")
        else:  # GET request
            logger.info("收到GET连接测试")
        
        # 返回模型加载状态
        model_status = "loaded" if loaded_model is not None else "not_loaded"
        model_path = CUSTOM_MODEL_PATH if loaded_model is not None else "none"
        
        return jsonify({
            "status": "success", 
            "message": "服务器连接正常",
            "model_status": model_status,
            "model_path": model_path
        })
    except Exception as e:
        logger.error(f"连接测试错误: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# 接收批量CSI数据并保存为CSV
@app.route('/api/csi_data', methods=['POST'])
def receive_csi_data():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            logger.error("无效的批量数据")
            return jsonify({"status": "error", "message": "无效的批量数据"}), 400
        
        logger.info(f"收到批量CSI数据: {len(data)}条")
        
        # 提取元数据
        first_record = data[0]
        user_name = first_record.get('user_name', 'unknown')
        action = first_record.get('action', first_record.get('taget', 'unknown'))
        sequence = first_record.get('sequence', '01')
        
        # 创建动作对应的目录
        action_dir = os.path.join(FILES_DIR, action)
        os.makedirs(action_dir, exist_ok=True)
        
        # 生成文件名
        filename = f"{action}_{user_name}_{sequence}.csv"
        filepath = os.path.join(action_dir, filename)
        
        # 总是创建新文件，不追加
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入列名
            writer.writerow(data[0].keys())
            # 写入数据行
            for row in data:
                writer.writerow(row.values())
        
        logger.info(f"成功保存{len(data)}条数据到新文件: {os.path.join(action, filename)}")
        return jsonify({
            "status": "success",
            "message": f"成功保存{len(data)}条数据到新文件: {os.path.join(action, filename)}",
            "file": os.path.join(action, filename)
        })
    except Exception as e:
        logger.error(f"保存数据失败: {e}")
        return jsonify({"status": "error", "message": f"保存数据失败: {e}"}), 500

# 接收CSV文件
@app.route('/api/upload_file', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logger.error("没有文件")
            return jsonify({"status": "error", "message": "没有文件"}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("没有选择文件")
            return jsonify({"status": "error", "message": "没有选择文件"}), 400
            
        # 从文件名中提取动作类型
        filename = file.filename
        # 处理带下划线的动作名称
        parts = filename.split('_')
        if len(parts) > 2:  # 如果动作名称包含下划线
            action = '_'.join(parts[:-2])  # 合并动作名称部分（排除用户名和序号）
        else:
            action = parts[0]
        
        # 创建动作对应的目录
        action_dir = os.path.join(FILES_DIR, action)
        os.makedirs(action_dir, exist_ok=True)
        
        # 保存文件到对应目录
        filepath = os.path.join(action_dir, filename)
        file.save(filepath)
        logger.info(f"文件保存成功: {filepath}")
        
        return jsonify({"status": "success", "message": "文件上传成功", "file": os.path.join(action, filename)})
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return jsonify({"status": "error", "message": f"文件上传失败: {e}"}), 500

# 实时动作预测API
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'csi_data' not in data:
            logger.error("无效的预测请求数据")
            return jsonify({"status": "error", "message": "无效的预测请求数据"}), 400
        
        csi_data = data['csi_data']
        logger.info(f"收到实时预测请求: {len(csi_data)}条CSI数据")
        
        # 记录CSI数据格式以便调试
        if len(csi_data) > 0:
            sample_entry = csi_data[0]
            logger.info(f"CSI数据样例: {sample_entry}")
            logger.info(f"CSI数据键: {list(sample_entry.keys())}")
            if 'data' in sample_entry and sample_entry['data']:
                if isinstance(sample_entry['data'], list):
                    logger.info(f"CSI data字段长度: {len(sample_entry['data'])}")
                    logger.info(f"CSI data字段前几个值: {sample_entry['data'][:5] if len(sample_entry['data']) > 5 else sample_entry['data']}")
                else:
                    logger.info(f"CSI data字段类型: {type(sample_entry['data'])}")
        
        # 计算当前CSI数据的统计特征，用于比较不同时刻的CSI数据差异
        csi_stats = analyze_csi_data(csi_data)
        
        # 调用预测函数
        result = predict_action(csi_data)
        
        # 保存当前预测结果用于检测动作变化
        last_predictions.append((result["prediction"], result["confidence"]))
        
        # 检测预测结果是否一直不变
        if len(last_predictions) >= 5:  # 至少有5个预测结果时才进行检查
            predictions_set = set(pred[0] for pred in last_predictions)
            if len(predictions_set) == 1:  # 如果最近的预测都是同一个动作
                logger.warning(f"注意：最近{len(last_predictions)}次预测结果都是 {list(predictions_set)[0]}，可能存在模型过度拟合或特征提取问题")
                
                # 计算CSI数据的变化程度
                if hasattr(csi_stats, 'get') and len(last_csi_features) > 0:
                    avg_change = sum(csi_stats.get('diff_from_last', 0.0) for _ in range(min(len(last_csi_features), 3))) / min(len(last_csi_features), 3)
                    logger.info(f"CSI数据平均变化程度: {avg_change:.4f}")
                    
                    # 如果CSI数据变化明显但预测不变，可能是模型问题
                    if avg_change > 0.2:  # 如果变化超过20%
                        logger.warning("CSI数据变化显著但预测结果不变，可能是模型对输入不敏感或过度拟合")
        
        # 返回预测结果
        response = {
            "status": "success",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "timestamp": result["timestamp"],
            "method": result.get("method", "unknown")
        }
        
        logger.info(f"预测结果: {result['prediction']}, 置信度: {result['confidence']:.2f}, 方法: {result.get('method', 'unknown')}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"预测请求处理失败: {e}")
        return jsonify({
            "status": "error", 
            "message": f"预测请求处理失败: {e}"
        }), 500

# 添加CSI数据分析函数，计算连续CSI数据间的差异
def analyze_csi_data(csi_data):
    """
    分析CSI数据，计算统计特征和与上次数据的差异
    """
    try:
        # 如果数据为空，返回空结果
        if not csi_data:
            return {"status": "empty"}
            
        # 提取原始CSI数据值
        raw_values = []
        for entry in csi_data:
            if 'data' in entry and entry['data']:
                raw_values.extend(entry['data'])
        
        # 如果没有提取到值，返回空结果
        if not raw_values:
            return {"status": "no_values"}
        
        # 计算基本统计特征
        stats = {
            "mean": float(np.mean(raw_values)),
            "std": float(np.std(raw_values)),
            "min": float(np.min(raw_values)),
            "max": float(np.max(raw_values)),
            "median": float(np.median(raw_values))
        }
        
        # 从CSI数据计算特征指纹（简化版本）
        fingerprint = []
        for entry in csi_data:
            if 'data' in entry and entry['data'] and len(entry['data']) >= 10:
                # 使用前10个值作为指纹特征的一部分
                fingerprint.extend(entry['data'][:10])
        
        # 如果指纹为空，使用原始数据
        if not fingerprint:
            fingerprint = raw_values[:min(100, len(raw_values))]
        
        # 计算与上次CSI数据的差异
        diff_from_last = 0.0
        if last_csi_features:
            last_fingerprint = last_csi_features[-1].get('fingerprint', [])
            
            # 计算两个指纹之间的差异（欧几里得距离）
            if last_fingerprint and fingerprint:
                # 确保两个指纹长度相同
                min_length = min(len(fingerprint), len(last_fingerprint))
                fp1 = np.array(fingerprint[:min_length])
                fp2 = np.array(last_fingerprint[:min_length])
                
                # 计算归一化欧几里得距离
                if min_length > 0:
                    diff = np.sqrt(np.sum((fp1 - fp2)**2)) / min_length
                    diff_from_last = float(diff)
                    logger.info(f"当前CSI数据与上次的差异: {diff_from_last:.4f}")
        
        # 将当前CSI数据的特征保存到历史记录
        current_features = {
            "fingerprint": fingerprint,
            "stats": stats,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        }
        last_csi_features.append(current_features)
        
        # 添加差异值到结果
        stats["diff_from_last"] = diff_from_last
        stats["status"] = "success"
        
        return stats
        
    except Exception as e:
        logger.error(f"CSI数据分析失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}

# 重新加载模型API
@app.route('/api/reload_model', methods=['POST', 'GET'])
def reload_model():
    try:
        if request.method == 'POST' and request.json and 'model_path' in request.json:
            model_path = request.json['model_path']
        else:
            model_path = CUSTOM_MODEL_PATH
            
        success = load_model(model_path)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"模型重新加载成功: {model_path}"
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"模型加载失败: {model_path}"
            }), 400
    except Exception as e:
        logger.error(f"重新加载模型失败: {e}")
        return jsonify({
            "status": "error", 
            "message": f"重新加载模型失败: {e}"
        }), 500

# 添加一个API端点来获取模型输出的概率分布
@app.route('/api/debug_prediction', methods=['POST'])
def debug_prediction():
    try:
        data = request.json
        if not data or 'csi_data' not in data:
            logger.error("无效的预测请求数据")
            return jsonify({"status": "error", "message": "无效的预测请求数据"}), 400
        
        csi_data = data['csi_data']
        logger.info(f"收到调试预测请求: {len(csi_data)}条CSI数据")
        
        # 从CSI数据中提取特征
        features = extract_features_from_csi(csi_data)
        tensor_features = tf.convert_to_tensor(features, dtype=tf.float32)
        
        # 获取模型信息
        model_info = {"model_type": str(type(loaded_model))}
        
        # 检测模型类型并获取预测
        if loaded_model is None:
            return jsonify({
                "status": "error",
                "message": "模型未加载"
            }), 400
            
        # 尝试获取原始预测值
        raw_predictions = None
        try:
            # 根据模型类型选择调用方式
            if isinstance(loaded_model, tf.keras.Model):
                raw_predictions = loaded_model.predict(features, verbose=0)[0]
                model_info["call_method"] = "keras_predict"
            elif hasattr(loaded_model, '__call__'):
                prediction_output = loaded_model(tensor_features)
                if isinstance(prediction_output, dict):
                    output_key = list(prediction_output.keys())[0]
                    raw_predictions = prediction_output[output_key].numpy()[0]
                else:
                    raw_predictions = prediction_output.numpy()[0]
                model_info["call_method"] = "direct_call"
            elif hasattr(loaded_model, 'signatures') and 'serving_default' in loaded_model.signatures:
                serving_fn = loaded_model.signatures['serving_default']
                try:
                    input_keys = list(serving_fn.structured_input_signature[1].keys())
                    if input_keys:
                        input_dict = {input_keys[0]: tensor_features}
                        prediction_output = serving_fn(**input_dict)
                    else:
                        prediction_output = serving_fn(inputs=tensor_features)
                        
                    if isinstance(prediction_output, dict):
                        output_key = list(prediction_output.keys())[0]
                        raw_predictions = prediction_output[output_key].numpy()[0]
                    else:
                        raw_predictions = prediction_output.numpy()[0]
                except Exception as e:
                    logger.error(f"获取预测失败: {e}")
                    raw_predictions = None
                model_info["call_method"] = "serving_default"
            else:
                model_info["call_method"] = "unknown"
        except Exception as e:
            logger.error(f"获取原始预测值失败: {e}")
            raw_predictions = None
            
        # 准备响应
        response = {
            "status": "success",
            "model_info": model_info,
            "features_info": {
                "shape": list(features.shape),
                "non_zero_ratio": float(np.count_nonzero(features) / features.size),
                "min": float(np.min(features)),
                "max": float(np.max(features)),
                "mean": float(np.mean(features)),
                "std": float(np.std(features))
            }
        }
        
        # 如果成功获取原始预测值，添加到响应中
        if raw_predictions is not None:
            # 转换为Python列表以便JSON序列化
            raw_list = [float(x) for x in raw_predictions]
            response["raw_predictions"] = raw_list
            
            # 获取类别标签
            try:
                if hasattr(loaded_model, 'classes_'):
                    classes = loaded_model.classes_
                else:
                    classes = ["站立", "行走", "坐下", "躺下", "弯腰", "跌倒", "其他"]
                
                # 将预测与类别标签匹配
                predictions_with_labels = []
                for i, prob in enumerate(raw_list):
                    if i < len(classes):
                        predictions_with_labels.append({
                            "label": classes[i],
                            "probability": prob
                        })
                    else:
                        predictions_with_labels.append({
                            "label": f"类别{i}",
                            "probability": prob
                        })
                
                response["predictions_with_labels"] = predictions_with_labels
            except Exception as e:
                logger.error(f"获取类别标签失败: {e}")
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"调试预测失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": f"调试预测失败: {e}"
        }), 500

# 主函数
if __name__ == '__main__':
    logger.info("启动CSI数据接收服务器...")
    logger.info(f"数据保存目录: {os.path.abspath(FILES_DIR)}")
    logger.info(f"使用模型路径: {os.path.abspath(CUSTOM_MODEL_PATH)}")
    logger.info(f"模型加载状态: {'成功' if loaded_model is not None else '失败'}")
    logger.info("API端点:")
    logger.info("  - /api/test_connection  - 测试服务器连接")
    logger.info("  - /api/csi_data         - 接收批量CSI数据")
    logger.info("  - /api/upload_file      - 上传CSV文件")
    logger.info("  - /api/predict          - 实时动作预测")
    logger.info("  - /api/reload_model     - 重新加载模型")
    logger.info("  - /api/debug_prediction - 调试模型预测")
    
    app.run(host='0.0.0.0', port=7799, debug=True)