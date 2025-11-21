"""
Coordinate Testing Tool
Allows manual input of X,Y coordinates to visualize position on camera view
"""

import sys
import cv2
import numpy as np
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QLineEdit, QPushButton,
                              QGroupBox, QGridLayout)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
import PySpin


class CoordinateTester(QMainWindow):
    """Simple tool to test coordinates by showing dots on camera"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coordinate Testing Tool")
        self.setGeometry(100, 100, 1000, 700)

        # Load configuration
        self.config = self.load_config()

        # Camera
        self.system = None
        self.detection_camera = None

        # Calibration matrix
        self.H_camera_to_workspace = None
        self.H_workspace_to_camera = None  # Inverse matrix

        # Test points to display
        self.test_points = []  # List of (x_mm, y_mm) tuples

        self.init_ui()
        self.init_camera()
        self.load_calibration()

        # Start camera timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(33)  # ~30 FPS

    def load_config(self):
        """Load configuration from file"""
        config_path = "config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def load_calibration(self):
        """Load calibration matrix"""
        # Get script directory to find calibration file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calib_file = os.path.join(script_dir, "homography_auto.pkl")

        if os.path.exists(calib_file):
            import pickle
            with open(calib_file, 'rb') as f:
                self.H_camera_to_workspace = pickle.load(f)
                # Calculate inverse matrix
                self.H_workspace_to_camera = np.linalg.inv(self.H_camera_to_workspace)
            self.status_label.setText("Status: Calibration loaded ✓")
        else:
            self.status_label.setText(f"Status: No calibration found at {calib_file}")

    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("Coordinate Testing Tool")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Input group
        input_group = QGroupBox("Coordinate Input (Workspace mm)")
        input_layout = QGridLayout()

        # X coordinate
        input_layout.addWidget(QLabel("X (mm):"), 0, 0)
        self.x_input = QLineEdit()
        self.x_input.setPlaceholderText("Enter X coordinate")
        input_layout.addWidget(self.x_input, 0, 1)

        # Y coordinate
        input_layout.addWidget(QLabel("Y (mm):"), 1, 0)
        self.y_input = QLineEdit()
        self.y_input.setPlaceholderText("Enter Y coordinate")
        input_layout.addWidget(self.y_input, 1, 1)

        # Buttons
        btn_layout = QHBoxLayout()

        self.add_btn = QPushButton("Add Point")
        self.add_btn.clicked.connect(self.add_point)
        btn_layout.addWidget(self.add_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_points)
        btn_layout.addWidget(self.clear_btn)

        input_layout.addLayout(btn_layout, 2, 0, 1, 2)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Status label
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.status_label)

        # Camera view
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("border: 2px solid #555; background-color: #2a2a2a;")
        layout.addWidget(self.camera_label)

        # Info label
        self.info_label = QLabel("Enter coordinates and click 'Add Point' to visualize")
        self.info_label.setFont(QFont("Arial", 9))
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

    def init_camera(self):
        """Initialize FLIR camera"""
        try:
            self.system = PySpin.System.GetInstance()
            cam_list = self.system.GetCameras()

            if cam_list.GetSize() > 0:
                # Get first camera (detection camera)
                self.detection_camera = cam_list[0]
                self.detection_camera.Init()
                self.detection_camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
                self.detection_camera.BeginAcquisition()
                self.status_label.setText("Status: Camera connected ✓")
                print("[INFO] Camera initialized successfully")
            else:
                self.status_label.setText("Status: No camera detected")
                print("[WARNING] No cameras found")
                self.show_no_camera_message()
        except Exception as e:
            error_msg = str(e)
            self.status_label.setText(f"Status: Camera error - {error_msg[:50]}...")
            print(f"[ERROR] Camera initialization failed: {error_msg}")

            # Check if camera is in use
            if "in use" in error_msg.lower() or "already" in error_msg.lower():
                self.info_label.setText("⚠ Camera is in use by another application. Close other apps and restart.")

            self.show_no_camera_message()

    def show_no_camera_message(self):
        """Show message when camera is not available"""
        # Create a placeholder image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Camera Feed Available", (120, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder, "Make sure:", (200, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(placeholder, "1. Camera is connected", (180, 310),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(placeholder, "2. No other app is using camera", (180, 340),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(placeholder, "3. Restart this tool", (180, 370),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Display placeholder
        h, w, ch = placeholder.shape
        bytes_per_line = ch * w
        qt_image = QImage(placeholder.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image.rgbSwapped())
        self.camera_label.setPixmap(pixmap)

    def add_point(self):
        """Add a test point from input fields"""
        try:
            x_mm = float(self.x_input.text())
            y_mm = float(self.y_input.text())

            # Validate coordinates
            workspace_width = self.config.get("workspace", {}).get("width", 300)
            workspace_height = self.config.get("workspace", {}).get("height", 300)

            if not (0 <= x_mm <= workspace_width and 0 <= y_mm <= workspace_height):
                self.info_label.setText(f"⚠ Warning: Point ({x_mm}, {y_mm}) is outside workspace bounds")

            # Add to list
            self.test_points.append((x_mm, y_mm))
            self.info_label.setText(f"✓ Added point: ({x_mm:.1f}, {y_mm:.1f}) mm - Total points: {len(self.test_points)}")

            # Clear inputs
            self.x_input.clear()
            self.y_input.clear()
            self.x_input.setFocus()

        except ValueError:
            self.info_label.setText("✗ Error: Please enter valid numbers for X and Y")

    def clear_points(self):
        """Clear all test points"""
        self.test_points.clear()
        self.info_label.setText("All points cleared")

    def update_camera(self):
        """Update camera display with test points"""
        if not self.detection_camera:
            return

        try:
            # Get image from camera
            image_result = self.detection_camera.GetNextImage(1000)

            if image_result.IsIncomplete():
                return

            # Convert to OpenCV format
            width = image_result.GetWidth()
            height = image_result.GetHeight()
            image_data = image_result.GetNDArray()

            # Handle different pixel formats
            if len(image_data.shape) == 2:
                # Grayscale image
                frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
            elif len(image_data.shape) == 3:
                # Already RGB/BGR - make a writable copy
                frame = image_data.copy()
            else:
                # Unknown format
                image_result.Release()
                return

            image_result.Release()

            # Draw test points
            if self.H_workspace_to_camera is not None:
                for x_mm, y_mm in self.test_points:
                    # Convert workspace coordinates to camera pixel coordinates
                    workspace_pt = np.array([[[x_mm, y_mm]]], dtype=np.float32)
                    camera_pt = cv2.perspectiveTransform(workspace_pt, self.H_workspace_to_camera)

                    px = int(camera_pt[0][0][0])
                    py = int(camera_pt[0][0][1])

                    # Check if point is within image bounds
                    if 0 <= px < width and 0 <= py < height:
                        # Draw large red circle
                        cv2.circle(frame, (px, py), 15, (0, 0, 255), -1)
                        # Draw white outline
                        cv2.circle(frame, (px, py), 15, (255, 255, 255), 2)
                        # Draw crosshair
                        cv2.drawMarker(frame, (px, py), (255, 255, 255),
                                     cv2.MARKER_CROSS, 30, 2)

                        # Draw coordinate label
                        label = f"({x_mm:.1f}, {y_mm:.1f})"
                        cv2.putText(frame, label, (px + 20, py - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw workspace boundary if calibration exists
            if self.H_workspace_to_camera is not None:
                workspace_width = self.config.get("workspace", {}).get("width", 300)
                workspace_height = self.config.get("workspace", {}).get("height", 300)

                # Define workspace corners
                workspace_corners = np.array([
                    [[0, 0]],
                    [[workspace_width, 0]],
                    [[workspace_width, workspace_height]],
                    [[0, workspace_height]]
                ], dtype=np.float32)

                # Transform to camera coordinates
                camera_corners = cv2.perspectiveTransform(workspace_corners, self.H_workspace_to_camera)
                camera_corners = camera_corners.astype(np.int32)

                # Draw workspace boundary
                cv2.polylines(frame, [camera_corners], isClosed=True,
                            color=(0, 255, 255), thickness=2)

            # Convert to Qt format and display
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image.rgbSwapped())

            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.camera_label.size(),
                                         Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)

        except Exception as e:
            # Log errors but don't crash
            error_msg = str(e)
            if "Spinnaker" in error_msg or "timeout" in error_msg.lower():
                # Camera-specific errors - these are expected sometimes
                pass
            else:
                # Unexpected errors - print for debugging
                print(f"[ERROR] Camera update failed: {error_msg}")

    def closeEvent(self, event):
        """Clean up when closing"""
        self.timer.stop()

        if self.detection_camera:
            try:
                self.detection_camera.EndAcquisition()
                self.detection_camera.DeInit()
            except:
                pass

        if self.system:
            try:
                self.system.ReleaseInstance()
            except:
                pass

        event.accept()


def main():
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(palette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(palette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(palette)

    window = CoordinateTester()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
