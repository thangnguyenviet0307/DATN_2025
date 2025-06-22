import cv2
import mediapipe as mp
import numpy as np
import os
import re
import json
from tabulate import tabulate

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Đường dẫn gốc
output_dir = r"D:\DATN\MediaPipe_VideoData\hand_gesture_dataset"
video_dir = os.path.join(output_dir, "video_data")
output_data_dir = os.path.join(output_dir, "output_data")
file_json_dir = os.path.join(output_data_dir, "file_json")
file_npy_dir = os.path.join(output_data_dir, "file_npy")
stats_dir = os.path.join(output_dir, "data")  # Thư mục mới để lưu thông tin thống kê

# Tạo các thư mục nếu chưa tồn tại
for directory in [output_data_dir, file_json_dir, file_npy_dir, stats_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Load label mapping
label_mapping_path = os.path.join(file_json_dir, 'label_mapping.json')
if not os.path.exists(label_mapping_path):
    print(f"Error: Label mapping file not found at {label_mapping_path}")
    exit()
with open(label_mapping_path, 'r') as f:
    label_mapping = json.load(f)
gestures = [label_mapping[str(i)] for i in range(len(label_mapping))]
print(f"Loaded gestures: {gestures}")

# Danh sách để lưu thông tin thống kê
stats_table = []

# Hàm trích xuất landmarks từ một khung hình
def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks), hand_landmarks
    return None, None

# Hàm vẽ text với nền
def put_text_with_background(frame, text, position, font_scale, color, thickness, bg_color=(0, 0, 0)):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    top_left = (position[0], position[1] - text_height - baseline)
    bottom_right = (position[0] + text_width, position[1] + baseline)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, -1)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Hàm xử lý video và lưu thành .npy
def video_to_npy(video_path, gesture_label, gesture_name):
    cap = cv2.VideoCapture(video_path)
    data = []
    labels = []
    frame_count = 0
    detected_count = 0

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False, [], [], 0, 0

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
        print(f"Warning: Invalid FPS for {video_path}. Using default FPS: 30")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        landmarks, hand_landmarks = extract_landmarks(frame)

        if hand_landmarks is not None:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            data.append(landmarks)
            labels.append(gesture_label)
            detected_count += 1

        font_scale = 0.7
        color = (0, 255, 0)
        thickness = 2
        bg_color = (0, 0, 0)

        put_text_with_background(frame, f"Recording: {gesture_name}", (10, 30), font_scale, color, thickness, bg_color)
        put_text_with_background(frame, f"Frame: {frame_count}", (10, 60), font_scale, color, thickness, bg_color)
        put_text_with_background(frame, f"Hands Detected: {detected_count}", (10, 90), font_scale, color, thickness, bg_color)

        cv2.imshow(f'Processing Video: {os.path.basename(video_path)} - Press Q to Skip', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames for {video_path}, detected hands in {detected_count} frames")

    if len(data) == 0:
        print(f"Warning: No hand detected in {video_path}")
        return False, [], [], frame_count, detected_count

    return True, data, labels, frame_count, detected_count

# Xử lý tất cả video
successful_videos = 0
all_gesture_data = {}
all_gesture_labels = {}

for i, gesture in enumerate(gestures):
    gesture_dir = os.path.join(video_dir, gesture)
    if not os.path.exists(gesture_dir):
        print(f"Directory not found: {gesture_dir}")
        continue

    video_files = [f for f in os.listdir(gesture_dir) if re.match(rf"^{gesture}_\d+\.mp4$", f)]
    if not video_files:
        print(f"No videos found for gesture: {gesture}")
        continue

    gesture_data = []
    gesture_labels = []
    total_frames = 0
    total_detected = 0

    for video_file in video_files:
        video_path = os.path.join(gesture_dir, video_file)
        print(f"Processing video: {video_path}")
        success, data, labels, frame_count, detected_count = video_to_npy(video_path, i, gesture)
        if success:
            successful_videos += 1
            gesture_data.extend(data)
            gesture_labels.extend(labels)
        total_frames += frame_count
        total_detected += detected_count

    if gesture_data:
        gesture_output_dir = os.path.join(file_npy_dir, gesture)
        if not os.path.exists(gesture_output_dir):
            os.makedirs(gesture_output_dir)
        np.save(os.path.join(gesture_output_dir, f"{gesture}_data.npy"), np.array(gesture_data))
        np.save(os.path.join(gesture_output_dir, f"{gesture}_labels.npy"), np.array(gesture_labels))
        print(f"Saved data to {os.path.join(gesture_output_dir, f'{gesture}_data.npy')}")
        all_gesture_data[gesture] = np.array(gesture_data)
        all_gesture_labels[gesture] = np.array(gesture_labels)
        stats_table.append([gesture, total_frames, total_detected, i])
    else:
        print(f"Warning: No valid data processed for gesture: {gesture}")
        stats_table.append([gesture, total_frames, total_detected, i])

if successful_videos == 0:
    print("Error: No videos were processed successfully.")
    exit()

# In bảng thống kê
print("\nSummary of Processed Frames and Labels:")
table_str = tabulate(stats_table, headers=["Gesture", "Total Frames", "Detected Frames", "Label"], tablefmt="grid")
print(table_str)

# Lưu bảng thống kê vào file trong thư mục data
stats_file_path = os.path.join(stats_dir, "summary_stats.txt")
with open(stats_file_path, 'w', encoding='utf-8') as f:
    f.write("Summary of Processed Frames and Labels:\n")
    f.write(table_str)
print(f"Saved summary statistics to {stats_file_path}")

# Gộp tất cả dữ liệu
all_data = []
all_labels = []

for gesture in gestures:
    if gesture in all_gesture_data and all_gesture_data[gesture].size > 0:
        all_data.append(all_gesture_data[gesture])
        all_labels.append(all_gesture_labels[gesture])
    else:
        print(f"Warning: No data to combine for {gesture}")

if not all_data:
    print("Error: No data to combine.")
    exit()

X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)

# Lưu dữ liệu gộp
np.save(os.path.join(file_npy_dir, 'all_gestures_data.npy'), X)
np.save(os.path.join(file_npy_dir, 'all_gestures_labels.npy'), y)
print(f"Saved combined data with shape {X.shape} to {os.path.join(file_npy_dir, 'all_gestures_data.npy')}")
print(f"Saved combined labels with shape {y.shape} to {os.path.join(file_npy_dir, 'all_gestures_labels.npy')}")

# Giải phóng tài nguyên
hands.close()