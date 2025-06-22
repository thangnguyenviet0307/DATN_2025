import paramiko
from scp import SCPClient
import os

# Thông tin Raspberry Pi
pi_host = "192.168.0.104"
pi_user = "pi"
pi_password = "11112222"

# Đường dẫn thư mục gốc trên máy tính
output_dir = "D:/DATN/MediaPipe_VideoData/hand_gesture_dataset/output_data"

# Định nghĩa các file nguồn trên máy tính
local_json_file = os.path.join(output_dir, "file_json", "label_mapping.json")
local_tflite_file = os.path.join(output_dir, "models", "models_tflite", "hand_gesture_model_new.tflite")
local_scaler_file = os.path.join(output_dir, "models", "models_scaler", "scaler_new.npy")
local_final_code_file = "D:/DATN/MediaPipe_VideoData/src/test_code_pi.py"  # File mới

# Định nghĩa các đường dẫn đích trên Raspberry Pi
remote_json_file_path = "/home/pi/hand_gesture/models/json/label_mapping.json"
remote_tflite_file_path = "/home/pi/hand_gesture/models/model/hand_gesture_model_new.tflite"
remote_scaler_file_path = "/home/pi/hand_gesture/models/scaler/scaler_new.npy"
remote_final_code_file_path = "/home/pi/hand_gesture/src/test_code_pi.py"  # Đường dẫn đích cho file mới

def check_file_access(file_path):
    """
    Kiểm tra quyền truy cập file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Không có quyền đọc: {file_path}")
        return True
    except Exception as e:
        print(f"Lỗi kiểm tra quyền truy cập {file_path}: {str(e)}")
        return False

def scp_file_to_pi():
    try:
        # Kiểm tra quyền truy cập các file
        files_to_check = [
            ("JSON file", local_json_file),
            ("TFLite file", local_tflite_file),
            ("Scaler file", local_scaler_file),
            ("Final code file", local_final_code_file),  # Thêm file mới vào danh sách kiểm tra
        ]
        for name, path in files_to_check:
            if not check_file_access(path):
                return

        # Tạo SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Kết nối đến Raspberry Pi
        print(f"Đang kết nối đến {pi_host}...")
        ssh.connect(pi_host, username=pi_user, password=pi_password)

        # Tạo SCP client
        scp = SCPClient(ssh.get_transport())

        # Copy file JSON
        print(f"Copy {local_json_file} đến {remote_json_file_path}...")
        scp.put(local_json_file, remote_json_file_path)

        # Copy file TFLite
        print(f"Copy {local_tflite_file} đến {remote_tflite_file_path}...")
        scp.put(local_tflite_file, remote_tflite_file_path)

        # Copy file Scaler (.npy)
        print(f"Copy {local_scaler_file} đến {remote_scaler_file_path}...")
        scp.put(local_scaler_file, remote_scaler_file_path)

        # Copy file final_code_pi.py
        print(f"Copy {local_final_code_file} đến {remote_final_code_file_path}...")
        scp.put(local_final_code_file, remote_final_code_file_path)

        print("Copy file thành công!")

        # Đóng kết nối
        scp.close()
        ssh.close()

    except Exception as e:
        print(f"Lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    scp_file_to_pi()