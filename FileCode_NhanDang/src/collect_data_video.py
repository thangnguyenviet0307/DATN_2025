import cv2
import os
import time
import re
import json

# Đường dẫn gốc lưu video
output_dir = r"D:\DATN\MediaPipe_VideoData\hand_gesture_dataset"
video_dir = os.path.join(output_dir, "video_data")
file_json_dir = os.path.join(output_dir, "output_data", "file_json")

# Tạo thư mục gốc và thư mục JSON nếu chưa tồn tại
for directory in [video_dir, file_json_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Thời gian quay mỗi video (giây)
video_duration = 10

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Please check connection and permissions.")
    exit()

# Lấy kích thước khung hình và FPS từ webcam
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 30
    print("Warning: Invalid FPS from webcam. Using default FPS: 30")

# Hàm tìm chỉ số lớn nhất của video hiện có
def get_max_video_index(gesture_dir, gesture):
    max_index = 0
    pattern = re.compile(rf"^{gesture}_(\d+)\.mp4$")
    for filename in os.listdir(gesture_dir):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    return max_index

# Hàm quay video cho một cử chỉ
def record_gesture(gesture, gesture_dir, index):
    video_path = os.path.join(gesture_dir, f"{gesture}_{index}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Could not create video writer for {video_path}")
        return

    print(f"\nRecording video for gesture: {gesture} (Video {index})")
    print(f"Video will be saved to: {video_path}")
    print("Press 'q' to stop recording early.")

    start_time = time.time()
    end_time = start_time + video_duration

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        remaining_time = int(end_time - time.time())
        cv2.putText(frame, f"Recording: {gesture} (Video {index})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time remaining: {remaining_time}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Recording Gesture - Press Q to Stop', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped early by user.")
            break

    out.release()
    print(f"Finished recording video for gesture: {gesture} (Video {index})")

# Tải ánh xạ nhãn hiện có (nếu tồn tại) và kiểm tra thư mục video_data
label_mapping_path = os.path.join(file_json_dir, 'label_mapping.json')
label_mapping = {}
existing_gestures = []

# Tải ánh xạ nhãn từ file JSON nếu tồn tại
if os.path.exists(label_mapping_path):
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    print(f"Loaded existing label mapping: {label_mapping}")
    existing_gestures = list(label_mapping.values())

# Kiểm tra các thư mục trong video_data để thêm các cử chỉ chưa có trong label_mapping
existing_dirs = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
next_label = len(label_mapping)
for gesture in existing_dirs:
    if gesture not in existing_gestures:
        label_mapping[str(next_label)] = gesture
        existing_gestures.append(gesture)
        next_label += 1
print(f"Updated label mapping after checking video_data: {label_mapping}")

# Nhập số lượng cử chỉ và video
while True:
    try:
        num_gestures = int(input("Enter the number of gestures to record (e.g., 3): ").strip())
        if num_gestures <= 0:
            print("Error: Number of gestures must be a positive integer.")
            continue
        break
    except ValueError:
        print("Error: Please enter a valid integer.")

while True:
    try:
        num_videos_per_gesture = int(input("Enter the number of videos per gesture (e.g., 5): ").strip())
        if num_videos_per_gesture <= 0:
            print("Error: Number of videos must be a positive integer.")
            continue
        break
    except ValueError:
        print("Error: Please enter a valid integer.")

# Nhập tên cử chỉ và tạo thư mục nếu cần
gestures = []
for i in range(num_gestures):
    while True:
        gesture = input(f"Enter the name of gesture {i + 1}/{num_gestures}: ").strip()
        if not gesture:
            print("Error: Gesture name cannot be empty.")
            continue
        gesture_dir = os.path.join(video_dir, gesture)
        if not os.path.exists(gesture_dir):
            os.makedirs(gesture_dir)
            print(f"Created directory for gesture: {gesture_dir}")
            label_mapping[str(next_label)] = gesture
            existing_gestures.append(gesture)
            next_label += 1
        gestures.append((gesture, gesture_dir))
        break

# Lưu ánh xạ nhãn cập nhật vào file JSON
with open(label_mapping_path, 'w') as f:
    json.dump(label_mapping, f, indent=4)
print(f"Saved updated label mapping to {label_mapping_path}")

# Quay video cho từng cử chỉ
for gesture, gesture_dir in gestures:
    max_index = get_max_video_index(gesture_dir, gesture)
    print(f"Existing videos for {gesture}: {max_index}")

    for i in range(num_videos_per_gesture):
        index = max_index + i + 1
        print(f"\nPrepare to record gesture: {gesture} (Video {index})")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break
            cv2.putText(frame, f"Prepare to record: {gesture} (Video {index})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'S' to start recording", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'Q' to skip this video", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Prepare to Record - Press S to Start', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                break
            if key == ord('q') or key == ord('Q'):
                print(f"Skipped recording for gesture: {gesture} (Video {index})")
                break

        if key == ord('q') or key == ord('Q'):
            continue

        record_gesture(gesture, gesture_dir, index)

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
print("\nFinished recording all gestures.")