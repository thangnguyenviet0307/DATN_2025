import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
from datetime import datetime

# Thiết lập logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_log_dir = os.path.join("D:\\DATN\\MediaPipe_VideoData\\hand_gesture_dataset", "output_data", "file_datatxt")
for subdir in [base_log_dir]:
    if not os.path.exists(subdir):
        os.makedirs(subdir)
        logging.info(f"Created directory: {subdir}")

# Configure logging for general training log
general_log_filename = os.path.join(base_log_dir, f"hand_gesture_training_{timestamp}.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(general_log_filename),
        logging.StreamHandler()
    ]
)

# Configure logging for input info file
input_log_filename = os.path.join(base_log_dir, f"input_info_{timestamp}.txt")
input_logger = logging.getLogger('input_logger')
input_logger.setLevel(logging.INFO)
input_fh = logging.FileHandler(input_log_filename)
input_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
input_logger.addHandler(input_fh)

# Configure logging for LSTM layer info file
lstm_log_filename = os.path.join(base_log_dir, f"lstm_layer_info_{timestamp}.txt")
lstm_logger = logging.getLogger('lstm_logger')
lstm_logger.setLevel(logging.INFO)
lstm_fh = logging.FileHandler(lstm_log_filename)
lstm_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
lstm_logger.addHandler(lstm_fh)

# Đường dẫn gốc
output_dir = r"D:\DATN\MediaPipe_VideoData\hand_gesture_dataset"
output_data_dir = os.path.join(output_dir, "output_data")
file_json_dir = os.path.join(output_data_dir, "file_json")
file_npy_dir = os.path.join(output_data_dir, "file_npy")
models_dir = os.path.join(output_data_dir, "models")
models_h5_dir = os.path.join(models_dir, "models_h5")
models_tflite_dir = os.path.join(models_dir, "models_tflite")
models_scaler_dir = os.path.join(models_dir, "models_scaler")
output_images_dir = os.path.join(output_data_dir, "image_results")

# Tạo các thư mục nếu chưa tồn tại
for directory in [file_json_dir, file_npy_dir, models_h5_dir, models_tflite_dir, models_scaler_dir, output_images_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and display label mapping
label_mapping_path = os.path.join(file_json_dir, 'label_mapping.json')
if not os.path.exists(label_mapping_path):
    logging.error(f"Error: Label mapping file not found at {label_mapping_path}")
    exit()
with open(label_mapping_path, 'r') as f:
    label_mapping = json.load(f)
logging.info(f"Content of label_mapping.json: {json.dumps(label_mapping, indent=4)}")
gestures = [label_mapping[str(i)] for i in range(len(label_mapping))]
num_classes = len(gestures)
logging.info(f"Gestures: {gestures}")

# Load and display labels from all_gestures_labels.npy
labels_path = os.path.join(file_npy_dir, 'all_gestures_labels.npy')
if not os.path.exists(labels_path):
    logging.error(f"Error: Labels file not found at {labels_path}")
    exit()
y = np.load(labels_path)
unique_labels = np.unique(y)
logging.info(f"Content of all_gestures_labels.npy (unique labels): {unique_labels}")
logging.info(f"Total number of labels: {len(y)}")

# Validate data consistency
if len(unique_labels) != num_classes:
    logging.error(f"Error: Number of unique labels ({len(unique_labels)}) does not match gestures ({num_classes})")
    exit()

# Load and validate data
data_path = os.path.join(file_npy_dir, 'all_gestures_data.npy')
if not os.path.exists(data_path):
    logging.error(f"Error: Data file not found at {data_path}")
    exit()

X = np.load(data_path)

# Validate data
if X.shape[1] != 63:
    logging.error(f"Error: Expected 63 features, got {X.shape[1]}")
    exit()
if len(X) != len(y):
    logging.error(f"Error: Mismatch between data length ({len(X)}) and labels length ({len(y)})")
    exit()
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    logging.error("Error: Data contains NaN or infinite values")
    exit()

logging.info(f"Loaded data. Shape: {X.shape}, Min: {X.min()}, Max: {X.max()}")

# Chuẩn hóa dữ liệu với MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
np.save(os.path.join(models_scaler_dir, 'scaler_new.npy'), scaler)
logging.info(f"Saved scaler to {os.path.join(models_scaler_dir, 'scaler_new.npy')}")

# Data augmentation
def add_noise(data, noise_factor=0.005):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
    augmented_data = data + noise
    augmented_data = np.clip(augmented_data, 0, 1)  # Clip để giữ trong [0, 1]
    return augmented_data

X_augmented = add_noise(X, noise_factor=0.005)
X = np.concatenate([X, X_augmented], axis=0)
y = np.concatenate([y, y], axis=0)
logging.info(f"Data augmented. New shape: {X.shape}")

# Prepare data for LSTM
frames = 5
X_sequences = []
y_sequences = []
for i in range(0, len(X) - frames + 1, frames):
    X_sequences.append(X[i:i + frames])
    y_sequences.append(y[i + frames - 1])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

if X_sequences.shape[1:] != (frames, 63):
    logging.error(f"Error: Expected sequence shape (N, {frames}, 63), got {X_sequences.shape}")
    exit()

logging.info(f"Prepared sequences. Shape: {X_sequences.shape}, Labels shape: {y_sequences.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42,
                                                    stratify=y_sequences)
logging.info(f"Data split. Train: {X_train.shape}, Test: {X_test.shape}")
logging.info(f"Number of training samples: {len(X_train)}")
logging.info(f"Number of test samples: {len(X_test)}")
# Hiển thị số lượng mẫu cho mỗi lớp trong tập train và test
train_class_counts = np.bincount(y_train.astype(int))
test_class_counts = np.bincount(y_test.astype(int))
for i, gesture in enumerate(gestures):
    logging.info(f"Class {gesture} - Train samples: {train_class_counts[i] if i < len(train_class_counts) else 0}, "
                 f"Test samples: {test_class_counts[i] if i < len(test_class_counts) else 0}")

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).prefetch(tf.data.AUTOTUNE)

# Log input information to separate file
input_logger.info("Input Information:")
input_logger.info("-" * 80)
input_logger.info(f"{'Parameter':<25} {'Details':<55}")
input_logger.info("-" * 80)
input_logger.info(f"{'Input Description':<25} {f'Hand gesture features extracted from video data'}")
input_logger.info(f"{'Total Input Count':<25} {f'{X.shape[0]} samples'}")
input_logger.info(f"{'Input Format':<25} {f'NumPy array of shape ({X.shape[0]}, 63)'}")
input_logger.info("-" * 80)
input_logger.info("Table 1: Gesture Frame Counts")
input_logger.info("-" * 80)
input_logger.info(f"{'Parameter':<25} {'Details':<55}")
input_logger.info("-" * 80)
input_logger.info(f"{'Total Gesture Frames':<25} {f'{X_sequences.shape[0] * frames} frames'}")
input_logger.info(f"{'Train Gesture Frames':<25} {f'{X_train.shape[0] * frames} frames'}")
input_logger.info(f"{'Test Gesture Frames':<25} {f'{X_test.shape[0] * frames} frames'}")
input_logger.info("-" * 80)

# Log LSTM layer information to separate file
lstm_logger.info("LSTM Layer Information:")
lstm_logger.info("-" * 80)
lstm_logger.info(f"{'Layer':<25} {'Input Size':<25} {'Output Size':<25}")
lstm_logger.info("-" * 80)
lstm_logger.info(f"{'LSTM Layer 1':<25} {f'({frames}, 63)':<25} {'64':<25}")
lstm_logger.info(f"{'LSTM Layer 2':<25} {'(64)':<25} {'32':<25}")
lstm_logger.info(f"{'LSTM Layer 3':<25} {'(32)':<25} {'16':<25}")
lstm_logger.info(f"{'Dense Layer 1':<25} {'16':<25} {'32':<25}")
lstm_logger.info(f"{'Output Layer':<25} {'32':<25} {f'{num_classes}':<25}")
lstm_logger.info("-" * 80)

# Build LSTM model
def build_model(frames, feature_dim, num_classes):
    model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(frames, feature_dim),
             kernel_regularizer=l2(0.01)),
        LayerNormalization(),
        Dropout(0.3),
        LSTM(units=32, return_sequences=True, kernel_regularizer=l2(0.01)),
        LayerNormalization(),
        Dropout(0.3),
        LSTM(units=16, kernel_regularizer=l2(0.01)),
        LayerNormalization(),
        Dropout(0.3),
        Dense(units=32, activation='relu', kernel_regularizer=l2(0.01)),
        LayerNormalization(),
        Dropout(0.2),
        Dense(units=num_classes, activation='softmax')
    ])
    model.compile(optimizer=AdamW(learning_rate=0.0001, weight_decay=0.004),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train model
model = build_model(frames, X_sequences.shape[2], num_classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset, callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(test_dataset)
logging.info(f"Test accuracy: {accuracy * 100:.2f}%")

# Save model
model.save(os.path.join(models_h5_dir, 'hand_gesture_model_new.h5'))
logging.info(f"Saved model to {os.path.join(models_h5_dir, 'hand_gesture_model_new.h5')}")

# Convert to TFLite
logging.info("Converting model to TFLite with quantization...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.target_spec.supported_types = [tf.float16]

try:
    tflite_model = converter.convert()
    tflite_model_path = os.path.join(models_tflite_dir, 'hand_gesture_model_new.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    logging.info(f"Saved TFLite model to {tflite_model_path}")
except Exception as e:
    logging.error(f"Error converting model to TFLite: {e}")
    exit()

# Visualize results
def plot_and_save(data, labels, title, xlabel, ylabel, filepath, figsize=(12, 5)):
    plt.figure(figsize=figsize)
    for d, label in zip(data, labels):
        plt.plot(d, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    logging.info(f"Saved {title} plot to {filepath}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, gestures, filepath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gestures, yticklabels=gestures)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filepath)
    logging.info(f"Saved Confusion Matrix to {filepath}")
    plt.close()

    # Mô tả chi tiết confusion matrix
    logging.info("Confusion Matrix Description:")
    logging.info(
        "The confusion matrix shows the relationship between true labels (rows) and predicted labels (columns).")
    logging.info("Each cell [i,j] represents the number of samples with true label i predicted as label j.")
    logging.info("Diagonal elements [i,i] represent correct predictions (True Positives) for each class.")
    logging.info("Off-diagonal elements represent misclassifications.")
    logging.info("Detailed breakdown:")
    for i, gesture in enumerate(gestures):
        true_positives = cm[i, i]
        false_negatives = np.sum(cm[i, :]) - cm[i, i]
        false_positives = np.sum(cm[:, i]) - cm[i, i]
        logging.info(f"Class {gesture}:")
        logging.info(f"  True Positives (correct predictions): {true_positives}")
        logging.info(f"  False Negatives (missed predictions): {false_negatives}")
        logging.info(f"  False Positives (incorrectly predicted as {gesture}): {false_positives}")

def plot_metrics(y_true, y_pred, gestures, filepath):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    metrics = np.array([precision, recall, f1])
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(gestures))
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    for i, metric_name in enumerate(metrics_names):
        plt.bar(x + i * bar_width, metrics[i], bar_width, label=metric_name)
    plt.xlabel('Gestures')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score per Gesture')
    plt.xticks(x + bar_width, gestures, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(filepath)
    logging.info(f"Saved Precision, Recall, F1-Score plot to {filepath}")
    plt.close()

# Predict and visualize
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

plot_and_save([history.history['loss'], history.history['val_loss']],
              ['Training Loss', 'Validation Loss'],
              'Training and Validation Loss',
              'Epoch', 'Loss',
              os.path.join(output_images_dir, 'loss_plot.png'))

plot_and_save([history.history['accuracy'], history.history['val_accuracy']],
              ['Training Accuracy', 'Validation Accuracy'],
              'Training and Validation Accuracy',
              'Epoch', 'Accuracy',
              os.path.join(output_images_dir, 'accuracy_plot.png'))

plot_confusion_matrix(y_test, y_pred_classes, gestures, os.path.join(output_images_dir, 'confusion_matrix.png'))

logging.info("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=gestures))

plot_metrics(y_test, y_pred_classes, gestures, os.path.join(output_images_dir, 'metrics_plot.png'))