import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import json
import time
import csv
import datetime
from collections import Counter

# Đường dẫn gốc
output_dir = r"D:\DATN\MediaPipe_VideoData\hand_gesture_dataset"
output_data_dir = os.path.join(output_dir, "output_data")
file_json_dir = os.path.join(output_data_dir, "file_json")
models_tflite_dir = os.path.join(output_data_dir, "models", "models_tflite")
models_scaler_dir = os.path.join(output_data_dir, "models", "models_scaler")
video_output_dir = os.path.join(output_data_dir, "video_output")

# Tạo thư mục video output nếu chưa tồn tại
if not os.path.exists(video_output_dir):
    os.makedirs(video_output_dir)
    print(f"Created directory: {video_output_dir}")

# Load label mapping
label_mapping_path = os.path.join(file_json_dir, 'label_mapping.json')
if not os.path.exists(label_mapping_path):
    print(f"Label mapping file not found at {label_mapping_path}")
    exit()
with open(label_mapping_path, 'r') as f:
    label_mapping = json.load(f)
gestures = [label_mapping[str(i)] for i in range(len(label_mapping))]
print(f"Loaded gestures: {gestures}")

# Load scaler
scaler_path = os.path.join(models_scaler_dir, 'scaler_new.npy')
if not os.path.exists(scaler_path):
    print(f"Scaler file not found at {scaler_path}")
    exit()
scaler = np.load(scaler_path, allow_pickle=True).item()
print("Scaler loaded successfully")

# Load TFLite model
model_path = os.path.join(models_tflite_dir, 'hand_gesture_model_new.tflite')
if not os.path.exists(model_path):
    print(f"TFLite model not found at {model_path}")
    exit()
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded successfully")

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()
print("Webcam initialized successfully")

# Lấy thông số khung hình
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0:
    fps = 30
    print("Warning: Invalid FPS from webcam. Using default FPS: 30")

# Khởi tạo video writer
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_video_path = os.path.join(video_output_dir, f"gesture_recognition_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
print(f"Recording output video to: {output_video_path}")

# Khởi tạo file CSV
csv_path = os.path.join(video_output_dir, f"gesture_log_{timestamp}.csv")
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Gesture', 'Confidence', 'FPS', 'Processing Time (ms)'])
print(f"Recording predictions to: {csv_path}")

# Khởi tạo bộ đếm và thống kê
gesture_counter = Counter()
gesture_times = {gesture: [] for gesture in gestures}
fps_values = []
MAX_GESTURE_COUNT = 100

# Chuẩn bị sequence
sequence_length = 5
sequence = []
prev_time = time.time()

def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks), frame
    return None, frame

def predict_sequence(sequence):
    if len(sequence) != sequence_length:
        return None, 0.0

    sequence_array = np.array(sequence)
    sequence_scaled = scaler.transform(sequence_array)
    sequence_scaled = sequence_scaled.reshape(1, sequence_length, 63)

    interpreter.set_tensor(input_details[0]['index'], sequence_scaled.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    gesture_idx = np.argmax(output_data[0])
    confidence = output_data[0][gesture_idx]
    return gesture_idx, confidence

# Main loop
print("Starting gesture recognition...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame from webcam")
        break

    current_time = time.time()
    fps_value = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    fps_values.append(fps_value)
    prev_time = current_time

    landmarks, frame = extract_landmarks(frame)

    if landmarks is not None:
        sequence.append(landmarks)
        if len(sequence) > sequence_length:
            sequence.pop(0)

        if len(sequence) == sequence_length:
            start_time = time.time()
            gesture_idx, confidence = predict_sequence(sequence)
            processing_time = (time.time() - start_time) * 1000

            if gesture_idx is not None and confidence > 0.7:
                gesture_name = gestures[gesture_idx]
                cv2.putText(frame, f"Gesture: {gesture_name} ({confidence:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if gesture_counter[gesture_name] < MAX_GESTURE_COUNT:
                    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_writer.writerow([current_timestamp, gesture_name, f"{confidence:.2f}",
                                         f"{fps_value:.2f}", f"{processing_time:.2f}"])
                    gesture_counter[gesture_name] += 1
                    gesture_times[gesture_name].append(processing_time)

    cv2.putText(frame, f"FPS: {fps_value:.2f}",
                (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    out.write(frame)
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User stopped the program")
        break

# Tính thống kê
avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
gesture_avg_times = {gesture: (sum(times) / len(times) if times else 0.0)
                     for gesture, times in gesture_times.items()}

# Lưu bảng thống kê
table_path = os.path.join(video_output_dir, f"gesture_stats_{timestamp}.txt")
with open(table_path, 'w') as f:
    f.write("Gesture Recognition Statistics (Max 100 per Gesture)\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'Gesture':<20} {'Count':>10} {'Avg Time (ms)':>15}\n")
    f.write("-" * 50 + "\n")
    for gesture in gestures:
        count = min(gesture_counter.get(gesture, 0), MAX_GESTURE_COUNT)
        avg_time = gesture_avg_times.get(gesture, 0.0)
        f.write(f"{gesture:<20} {count:>10} {avg_time:>15.2f}\n")
    f.write("=" * 50 + "\n")
    f.write(f"Average FPS: {avg_fps:.2f}\n")
print(f"Statistics saved to: {table_path}")

# In bảng thống kê
print("\nGesture Recognition Statistics (Max 100 per Gesture)")
print("=" * 50)
print(f"{'Gesture':<20} {'Count':>10} {'Avg Time (ms)':>15}")
print("-" * 50)
for gesture in gestures:
    count = min(gesture_counter.get(gesture, 0), MAX_GESTURE_COUNT)
    avg_time = gesture_avg_times.get(gesture, 0.0)
    print(f"{gesture:<20} {count:>10} {avg_time:>15.2f}")
print("=" * 50)
print(f"Average FPS: {avg_fps:.2f}")

# Tạo biểu đồ bằng Chart.js
counts = [min(gesture_counter.get(gesture, 0), MAX_GESTURE_COUNT) for gesture in gestures]
avg_times = [gesture_avg_times.get(gesture, 0.0) for gesture in gestures]

chart_recognitions = f"""
<canvas id="recognitionChart"></canvas>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const ctx1 = document.getElementById('recognitionChart').getContext('2d');
new Chart(ctx1, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(gestures)},
    datasets: [{{
      label: 'Number of Recognitions',
      data: {json.dumps(counts)},
      backgroundColor: 'rgba(54, 162, 235, 0.6)',
      borderColor: 'rgba(54, 162, 235, 1)',
      borderWidth: 1
    }}]
  }},
  options: {{
    plugins: {{
      title: {{
        display: true,
        text: 'Number of Successful Recognitions per Gesture (Max 100)'
      }}
    }},
    scales: {{
      y: {{
        beginAtZero: true,
        title: {{
          display: true,
          text: 'Count'
        }}
      }},
      x: {{
        title: {{
          display: true,
          text: 'Gesture'
        }}
      }}
    }}
  }}
}});
</script>
"""
chart_times = f"""
<canvas id="timeChart"></canvas>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const ctx2 = document.getElementById('timeChart').getContext('2d');
new Chart(ctx2, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(gestures)},
    datasets: [{{
      label: 'Average Recognition Time (ms)',
      data: {json.dumps(avg_times)},
      backgroundColor: 'rgba(75, 192, 192, 0.6)',
      borderColor: 'rgba(75, 192, 192, 1)',
      borderWidth: 1
    }}]
  }},
  options: {{
    plugins: {{
      title: {{
        display: true,
        text: 'Average Recognition Time per Gesture'
      }}
    }},
    scales: {{
      y: {{
        beginAtZero: true,
        title: {{
          display: true,
          text: 'Time (ms)'
        }}
      }},
      x: {{
        title: {{
          display: true,
          text: 'Gesture'
        }}
      }}
    }}
  }}
}});
</script>
"""
chart_fps = f"""
<canvas id="fpsChart"></canvas>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const ctx3 = document.getElementById('fpsChart').getContext('2d');
new Chart(ctx3, {{
  type: 'bar',
  data: {{
    labels: ['Average FPS', 'Webcam FPS'],
    datasets: [{{
      label: 'FPS',
      data: [{avg_fps}, {fps}],
      backgroundColor: ['rgba(255, 99, 132, 0.6)', 'rgba(153, 102, 255, 0.6)'],
      borderColor: ['rgba(255, 99, 132, 1)', 'rgba(153, 102, 255, 1)'],
      borderWidth: 1
    }}]
  }},
  options: {{
    plugins: {{
      title: {{
        display: true,
        text: 'FPS Comparison'
      }}
    }},
    scales: {{
      y: {{
        beginAtZero: true,
        title: {{
          display: true,
          text: 'FPS'
        }}
      }}
    }}
  }}
}});
</script>
"""

# Lưu biểu đồ Chart.js vào file HTML
chart_path = os.path.join(video_output_dir, f"gesture_charts_{timestamp}.html")
with open(chart_path, 'w') as f:
    f.write("<html><body>")
    f.write(chart_recognitions)
    f.write(chart_times)
    f.write(chart_fps)
    f.write("</body></html>")
print(f"Charts saved to: {chart_path}")

# Giải phóng tài nguyên
csv_file.close()
cap.release()
out.release()
cv2.destroyAllWindows()
hands.close()
print(f"Output video saved to: {output_video_path}")
print(f"Predictions saved to: {csv_path}")
print("Program ended successfully")