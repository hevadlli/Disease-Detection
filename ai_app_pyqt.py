import sys
import os
import numpy as np
import json
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PIL import Image
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

# Kelas penyakit
disease_class = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Load model pre-trained
model = keras.models.load_model('E:/Code/zona_ai-main/zona_ai-main/model/sequential-Chicken Disease-97.03.h5')


def compute_zona_farm_vision_request(image_path):
    # Buka image lokal
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)

    # Normalisasi nilai pixel jika diperlukan oleh model
    image = image / 255.0  # Normalisasi

    # Tambahkan dimensi batch
    image = np.expand_dims(image, axis=0)

    # Prediksi dengan model
    result = model.predict(image)
    result = result[0]

    # Format output
    output = {
        disease_class[0]: float(result[0]),
        disease_class[1]: float(result[1]),
        disease_class[2]: float(result[2]),
        disease_class[3]: float(result[3])
    }

    return output


class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.is_running = True

    def run(self):
        try:
            self.progress.emit(10)
            results = compute_zona_farm_vision_request(self.image_path)
            self.progress.emit(100)
            if self.is_running:
                self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False


class ChickenDiseaseApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Chicken Disease Detection")
        self.resize(1280, 720)  # Resolusi 720p
        self.showMaximized()  # Mulai dalam kondisi maximize

        self.main_layout = QHBoxLayout()

        # Bagian kiri untuk menampilkan gambar
        self.left_layout = QVBoxLayout()
        self.image_label = QLabel("No Image Selected", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setFixedWidth(640)  # Setengah dari layar 720p
        self.image_label.setFixedHeight(720)
        self.left_layout.addWidget(self.image_label)

        self.open_button = QPushButton("Open Image", self)
        self.open_button.setFixedHeight(50)  # Membesarkan tombol
        self.open_button.setStyleSheet("font-size: 16px;")
        self.open_button.clicked.connect(self.open_file)
        self.left_layout.addWidget(self.open_button)

        self.analyze_button = QPushButton("Analyze Image", self)
        self.analyze_button.setFixedHeight(50)  # Membesarkan tombol
        self.analyze_button.setStyleSheet("font-size: 16px;")
        self.analyze_button.clicked.connect(self.analyze_image)
        self.left_layout.addWidget(self.analyze_button)

        self.cancel_button = QPushButton("Cancel Analysis", self)
        self.cancel_button.setFixedHeight(50)  # Membesarkan tombol
        self.cancel_button.setStyleSheet("font-size: 16px;")
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.cancel_button.setEnabled(False)
        self.left_layout.addWidget(self.cancel_button)

        self.main_layout.addLayout(self.left_layout)

        # Bagian kanan untuk hasil analisis
        self.right_layout = QVBoxLayout()
        self.label = QLabel("Analysis Results", self)
        self.label.setFont(QFont("Arial", 14))
        self.label.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.label)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet(
            "font-size: 18px; background-color: #f0f0f0; border: 1px solid #d3d3d3; padding: 10px;")
        self.right_layout.addWidget(self.text_edit)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.right_layout.addWidget(self.progress_bar)

        self.save_button = QPushButton("Save Results", self)
        self.save_button.setFixedHeight(50)  # Membesarkan tombol
        self.save_button.setStyleSheet("font-size: 16px;")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        self.right_layout.addWidget(
            self.save_button, alignment=Qt.AlignRight | Qt.AlignBottom)

        self.main_layout.addLayout(self.right_layout)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        self.results = None
        self.current_image_path = None
        self.analysis_thread = None

    def open_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)

        if file_path:
            try:
                # Tampilkan gambar di label
                pixmap = QPixmap(file_path)
                # Maksimalkan gambar hingga setengah layar
                pixmap = pixmap.scaled(640, 720, Qt.KeepAspectRatio)
                self.image_label.setPixmap(pixmap)
                self.image_label.setText("")
                self.current_image_path = file_path

                self.text_edit.clear()  # Kosongkan hasil analisis sebelum analisis baru dimulai
                self.save_button.setEnabled(False)

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load image:\n{str(e)}")

    def analyze_image(self):
        if self.current_image_path:
            self.open_button.setEnabled(False)
            self.analyze_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.progress_bar.setVisible(True)

            self.analysis_thread = AnalysisThread(self.current_image_path)
            self.analysis_thread.progress.connect(self.progress_bar.setValue)
            self.analysis_thread.finished.connect(self.on_analysis_finished)
            self.analysis_thread.error.connect(self.on_analysis_error)
            self.analysis_thread.start()

    def cancel_analysis(self):
        if self.analysis_thread:
            self.analysis_thread.stop()
            self.analysis_thread.quit()
            self.analysis_thread.wait()
            self.reset_buttons()
            QMessageBox.information(self, "Canceled", "Analysis canceled.")

    def on_analysis_finished(self, results):
        self.results = results
        formatted_results = "".join(
            [f"<div style='margin-bottom: 10px;'><table style='display: inline-block; width: 200px;'><tr><td><b>{key}</b></td><td>:</td><td>{value:.2f}</td></tr></table></div>" for key, value in self.results.items()])
        self.text_edit.setHtml(
            f"<div style='font-family: Arial; font-size: 18px; color: #333;'>{formatted_results}</div>")
        self.reset_buttons()
        self.save_button.setEnabled(True)

    def on_analysis_error(self, error_message):
        QMessageBox.critical(
            self, "Error", f"Failed to analyze image:\n{error_message}")
        self.reset_buttons()

    def reset_buttons(self):
        self.open_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

    def save_results(self):
        if self.results:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "", "CSV Files (*.csv)", options=options)

            if file_path:
                try:
                    df = pd.DataFrame([self.results])
                    df.to_csv(file_path, index=False)
                    QMessageBox.information(
                        self, "Success", "Results saved successfully!")
                except Exception as e:
                    QMessageBox.critical(
                        self, "Error", f"Failed to save results:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ChickenDiseaseApp()
    main_window.show()
    sys.exit(app.exec_())
