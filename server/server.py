from flask import Flask, request, jsonify
import os
import csv
from datetime import datetime

app = Flask(__name__)

# 创建服务器数据存储目录
SERVER_DATA_DIR = "server_data"
if not os.path.exists(SERVER_DATA_DIR):
    os.makedirs(SERVER_DATA_DIR)

@app.route('/api/csi_data', methods=['POST'])
def save_csi_data():
    try:
        data = request.json
        
        # 获取数据信息
        action = data.get('action', 'unknown')
        user_name = data.get('user_name', 'unknown')
        sequence = data.get('sequence', 0)
        file_name = data.get('file_name', f"{action}_{user_name}_{sequence:02d}.csv")
        
        # 创建动作目录
        action_dir = os.path.join(SERVER_DATA_DIR, action)
        os.makedirs(action_dir, exist_ok=True)
        
        # 构建完整的文件路径
        file_path = os.path.join(action_dir, file_name)
        
        # 检查文件是否存在
        file_exists = os.path.exists(file_path)
        
        # 将数据写入CSV文件
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data['csv_data'].keys())
            
            # 如果是新文件，写入表头
            if not file_exists:
                writer.writeheader()
            
            # 写入数据行
            writer.writerow(data['csv_data'])
        
        print(f"数据已保存到: {file_path}")
        return jsonify({
            "status": "success",
            "message": "数据保存成功",
            "file": file_path
        })
        
    except Exception as e:
        print(f"保存数据时发生错误: {e}")
        return jsonify({
            "status": "error",
            "message": f"保存数据失败: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 