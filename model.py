import sys
import math
import datetime
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from sklearn.preprocessing import LabelEncoder
from finalDesign import Ui_MainWindow

class BehaviorMonitoringApp(QtWidgets.QMainWindow):
    def __init__(self, model, encoder, img_size=(216, 216), interval=15, window_size=20):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.model = model
        self.encoder = encoder
        self.img_size = img_size
        self.interval = interval
        self.window_size = window_size
        self.frame_buffer = []
        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_idx = 0
        self.last_pred_time = 0
        self.predicted_label = "Waiting..."
        self.seconds_elapsed = 1

        # Initialize CSV saving
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_folder = "monitoring_logs"
        os.makedirs(self.save_folder, exist_ok=True)
        self.monitor_csv_path = os.path.join(self.save_folder, f"monitoring_log_{timestamp}.csv")
        self.landmark_csv_path = os.path.join(self.save_folder, f"landmarks_log_{timestamp}.csv")
        self.logs = []
        self.landmark_logs = []

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_video)
        self.timer.start(10)

        self.second_timer = QtCore.QTimer()
        self.second_timer.timeout.connect(self.update_timer)
        self.second_timer.start(1000)

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_resized = cv2.resize(frame, self.img_size) / 255.0
        self.frame_buffer.append(frame_resized)

        frames_needed = int(self.interval * self.fps)
        if len(self.frame_buffer) > frames_needed:
            self.frame_buffer.pop(0)

        current_time = self.frame_idx / self.fps

        if current_time >= self.last_pred_time + self.interval:
            if len(self.frame_buffer) >= self.window_size:
                sampled_frames = np.array([
                    self.frame_buffer[math.floor(i * len(self.frame_buffer) / self.window_size)]
                    for i in range(self.window_size)
                ])
                input_data = np.expand_dims(sampled_frames, axis=0)

                classification_pred, landmarks_pred, motion_pred, velocity_pred = self.model.predict(input_data, verbose=0)

                # Classification
                class_idx = np.argmax(classification_pred[0])
                self.predicted_label = self.encoder.inverse_transform([class_idx])[0]

                self.ui.prediction_label.setText(f"Predicted Action: {self.predicted_label}")
                self.ui.prediction_label.setStyleSheet("color: green;")
                QtCore.QTimer.singleShot(1000, lambda: self.ui.prediction_label.setStyleSheet("color: black;"))

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp.split()[1]}] {self.predicted_label}"
                self.ui.prediction_log_box.setReadOnly(False)
                log_lines = self.ui.prediction_log_box.toPlainText().splitlines()
                if len(log_lines) >= 10:
                    text = '\n'.join(log_lines[1:] + [log_entry])
                else:
                    text = self.ui.prediction_log_box.toPlainText() + '\n' + log_entry
                self.ui.prediction_log_box.setPlainText(text.strip())
                self.ui.prediction_log_box.verticalScrollBar().setValue(
                    self.ui.prediction_log_box.verticalScrollBar().maximum()
                )
                self.ui.prediction_log_box.setReadOnly(True)

                # Save landmark logs
                landmarks = landmarks_pred[0].reshape(71, 2)
                for idx in range(self.window_size):
                    entry = {"Time": timestamp, "Frame_Index": idx}
                    for i, (x, y) in enumerate(landmarks):
                        entry[f"Landmark_{i}_X"] = x
                        entry[f"Landmark_{i}_Y"] = y
                    self.landmark_logs.append(entry)

                # Save monitoring logs
                motion = motion_pred[0]
                velocity = velocity_pred[0]
                log_entry = {"Time": timestamp, "Prediction": self.predicted_label}
                for i in range(len(motion)):
                    log_entry[f"Motion_{i}"] = motion[i]
                    log_entry[f"Velocity_{i}"] = velocity[i]
                self.logs.append(log_entry)

                self.last_pred_time += self.interval
                self.seconds_elapsed = 1

        # Update video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (600, 450))
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.ui.video_frame.setPixmap(pixmap)

        self.frame_idx += 1

    def update_timer(self):
        if self.seconds_elapsed < self.interval:
            self.seconds_elapsed += 1
        self.ui.timer_label.setText(f"Time since last prediction: {self.seconds_elapsed}s")

    def closeEvent(self, event):
        """Handle application close and save logs"""
        self.cap.release()
        cv2.destroyAllWindows()

        if self.logs:
            pd.DataFrame(self.logs).to_csv(self.monitor_csv_path, index=False)
            print(f"Monitoring logs saved to: {self.monitor_csv_path}")

        if self.landmark_logs:
            pd.DataFrame(self.landmark_logs).to_csv(self.landmark_csv_path, index=False)
            print(f"Landmark logs saved to: {self.landmark_csv_path}")

        event.accept()

    def run(self):
        self.show()

if __name__ == "__main__":
    model = tf.keras.models.load_model("models/examination_behavior-20F-216IS-v1.05.keras")
    CLASSES = ['Non-suspicious', 'Suspicious']
    encoder = LabelEncoder()
    encoder.fit(CLASSES)

    app = QtWidgets.QApplication(sys.argv)
    window = BehaviorMonitoringApp(model, encoder)
    window.run()
    sys.exit(app.exec_())
