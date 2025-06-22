import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import mediapipe as mp
import os
import json
import time
from collections import Counter
import warnings
from mqtt_connection import MQTTConnection
from mqtt_config import MQTT_TOPIC_CONTROL
from picamera2 import Picamera2
import datetime

# Initialize MQTT Connection
mqtt_conn = MQTTConnection()
if not mqtt_conn.connect():
    print("Failed to connect to MQTT broker, continuing without MQTT...")

# Root directory path
output_dir = r"/home/pi/hand_gesture"

# Define subdirectories
file_json_dir = os.path.join(output_dir, "models", "json")
models_tflite_dir = os.path.join(output_dir, "models", "model")
models_scaler_dir = os.path.join(output_dir, "models", "scaler")

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
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded successfully")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pi Camera
frame_width = 640
frame_height = 480
picam2 = Picamera2()
try:
    config = picam2.create_video_configuration(main={"size": (frame_width, frame_height), "format": "RGB888"})
    picam2.configure(config)
    picam2.set_controls({"FrameRate": 15, "ExposureTime": 0, "AwbEnable": True})  # Auto exposure and white balance
    picam2.start()
except Exception as e:
    print(f"Error: Could not initialize Pi Camera: {e}")
    exit()

print("Pi Camera initialized successfully with auto exposure and white balance enabled")

# Prepare sequence to store frames
sequence_length = 5
sequence = []

# Variables for FPS and prediction
prev_time = time.time()
frame_count = 0
last_gesture = "Unknown"
last_confidence = 0.0
last_mqtt_time = 0
MQTT_DELAY = 3

# Initialize counters and processing times
gesture_counter = Counter()
gesture_times = {gesture: [] for gesture in gestures}
fps_values = []
MAX_GESTURE_COUNT = 20

# Target FPS
target_fps = 15
frame_delay = 1.0 / target_fps

def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            x_coords = []
            y_coords = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
                x_coords.append(lm.x)
                y_coords.append(lm.y)

            # Calculate bounding box
            x_min = max(0, min(x_coords) * frame_width - 20)
            x_max = min(frame_width, max(x_coords) * frame_width + 20)
            y_min = max(0, min(y_coords) * frame_height - 20)
            y_max = min(frame_height, max(y_coords) * frame_height + 20)

            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            return np.array(landmarks), frame

    return None, frame

def predict_sequence(sequence):
    global last_mqtt_time
    if len(sequence) != sequence_length:
        return None, 0.0

    sequence_array = np.array(sequence)
    sequence_scaled = scaler.transform(sequence_array).reshape(1, sequence_length, 63)
    interpreter.set_tensor(input_details[0]['index'], sequence_scaled.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    gesture_idx = np.argmax(output_data[0])
    confidence = output_data[0][gesture_idx]

    current_time = time.time()
    if confidence > 0.7 and (current_time - last_mqtt_time) >= MQTT_DELAY:
        gesture = gestures[gesture_idx]
        if mqtt_conn.publish(MQTT_TOPIC_CONTROL, gesture):
            last_mqtt_time = current_time
            print(f"Published gesture command: {gesture} (after {MQTT_DELAY}s delay)")

    return gesture_idx, confidence

def get_rpi_temp():
    try:
        temp = os.popen("vcgencmd measure_temp").readline()
        return temp.replace("temp=", "").replace("'C\n", "")
    except Exception:
        return "N/A"

# Main loop
print("Starting gesture recognition...")
try:
    while True:
        loop_start_time = time.time()
        frame = picam2.capture_array()
        if frame is None or frame.size == 0:
            print("Error: Failed to capture valid frame from Pi Camera. Retrying...")
            time.sleep(0.1)
            continue

        # Calculate FPS
        current_time = time.time()
        fps_value = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        fps_values.append(fps_value)
        prev_time = current_time

        landmarks, frame = extract_landmarks(frame)
        if landmarks is not None:
            sequence.append(landmarks)
            if len(sequence) > sequence_length:
                sequence.pop(0)

            frame_count += 1
            if frame_count % 15 == 0 and len(sequence) == sequence_length:
                start_time = time.time()
                gesture_idx, confidence = predict_sequence(sequence)
                processing_time = (time.time() - start_time) * 1000

                if gesture_idx is not None and confidence > 0.7:
                    gesture_name = gestures[gesture_idx]
                    last_gesture = gesture_name
                    last_confidence = confidence
                    if gesture_counter[gesture_name] < MAX_GESTURE_COUNT:
                        gesture_counter[gesture_name] += 1
                        gesture_times[gesture_name].append(processing_time)

        # Display information on frame
        cv2.putText(frame, f"Gesture: {last_gesture} ({last_confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps_value:.2f}", (frame_width - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"Temp: {get_rpi_temp()}", (frame_width - 100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        y_position = frame_height - 20 * (len(gestures) + 1)
        cv2.putText(frame, "Gesture Counts:", (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        for gesture in gestures:
            y_position += 20
            cv2.putText(frame, f"{gesture}: {gesture_counter[gesture]}", (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User stopped the program")
            break

        # Control frame rate
        elapsed_time = time.time() - loop_start_time
        time.sleep(max(0, frame_delay - elapsed_time))

except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    # Calculate averages
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
    gesture_avg_times = {g: (sum(t) / len(t) if t else 0.0) for g, t in gesture_times.items()}
    print(f"Average FPS: {avg_fps:.2f}")
    print("Average processing times:", gesture_avg_times)

    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
    hands.close()
    mqtt_conn.disconnect()
    print("Program ended successfully")