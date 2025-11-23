"""
Coordinate Testing Tool
Allows manual input of X,Y coordinates to visualize position and send gripper to location
"""

# Standard library imports
import sys
import json
import os
import time
import struct
import pickle
from multiprocessing import shared_memory

# Third-party imports
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QLineEdit, QPushButton,
                              QGroupBox, QGridLayout, QSizePolicy)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor

# YOLO import - MUST be before PySpin to avoid DLL loading issues
from ultralytics import YOLO

# PySpin import - MUST be after YOLO
import PySpin


class ClickableLabel(QLabel):
    """Custom QLabel that emits click signals with scaled coordinates"""
    clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_image_size = None  # Store original image size for coordinate scaling
        self.displayed_image_size = None  # Store displayed (scaled) image size

    def mousePressEvent(self, event):
        """Handle mouse press events and emit scaled coordinates"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Get click position in widget coordinates
            click_pos = event.pos()
            widget_x = click_pos.x()
            widget_y = click_pos.y()

            # Scale coordinates to match original image size
            if self.original_image_size and self.displayed_image_size:
                # Calculate image offset (image is centered in label)
                label_w = self.width()
                label_h = self.height()
                img_w = self.displayed_image_size[0]
                img_h = self.displayed_image_size[1]

                offset_x = (label_w - img_w) / 2
                offset_y = (label_h - img_h) / 2

                # Adjust click position by offset
                img_click_x = widget_x - offset_x
                img_click_y = widget_y - offset_y

                # Check if click is within image bounds
                if 0 <= img_click_x < img_w and 0 <= img_click_y < img_h:
                    # Calculate scaling factors
                    scale_x = self.original_image_size[0] / self.displayed_image_size[0]
                    scale_y = self.original_image_size[1] / self.displayed_image_size[1]

                    # Scale click coordinates to original image size
                    orig_x = int(img_click_x * scale_x)
                    orig_y = int(img_click_y * scale_y)

                    # Emit signal with scaled coordinates
                    self.clicked.emit(orig_x, orig_y)
            else:
                # No scaling info, emit raw coordinates
                self.clicked.emit(widget_x, widget_y)


class ClickDataManager:
    """Manages shared memory for sending move coordinates to robot"""

    def __init__(self, name="ClickData", size=512):
        self.name = name
        self.size = size
        self.shm = None
        self._initialize_shared_memory()

    def _initialize_shared_memory(self):
        """Create or attach to existing shared memory."""
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            self._read_data()
        except FileNotFoundError:
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
                self._write_data({"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0})
            except Exception as e:
                print(f"[ERROR] Failed to create shared memory: {e}")
                raise

    def _write_data(self, data):
        """Write data to shared memory as JSON."""
        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')

            if len(json_bytes) > self.size - 4:
                print(f"[ERROR] Data too large: {len(json_bytes)} > {self.size-4}")
                return False

            self.shm.buf[:4] = struct.pack('I', len(json_bytes))
            self.shm.buf[4:4+len(json_bytes)] = json_bytes
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write click data: {e}")
            return False

    def _read_data(self):
        """Read data from shared memory."""
        try:
            length = struct.unpack('I', bytes(self.shm.buf[:4]))[0]
            if length == 0 or length > self.size - 4:
                return {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0}

            json_bytes = bytes(self.shm.buf[4:4+length])
            json_str = json_bytes.decode('utf-8')
            return json.loads(json_str)
        except Exception as e:
            print(f"[ERROR] Failed to read click data: {e}")
            return None

    def send_coordinate(self, x_mm, y_mm, angle=0.0):
        """Send coordinate to robot via shared memory"""
        data = {
            "click_x": float(x_mm),
            "click_y": float(y_mm),
            "timestamp": time.time(),
            "processed": False,
            "button": "left",
            "angle": float(angle)
        }
        return self._write_data(data)

    def cleanup(self):
        """Clean up shared memory."""
        if self.shm:
            try:
                self.shm.close()
            except:
                pass


class InspectDataManager:
    """Manages shared memory for sending inspection commands to robot"""

    def __init__(self, name="InspectData", size=512):
        self.name = name
        self.size = size
        self.shm = None
        self._initialize_shared_memory()

    def _initialize_shared_memory(self):
        """Create or attach to existing shared memory."""
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
        except FileNotFoundError:
            try:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
                # Get camera offset from config
                camera_offset = self.get_camera_offset_from_config()
                self._write_data({
                    "inspect": False,
                    "target_x": 0,
                    "target_y": 0,
                    "angle": 0.0,
                    "width": 0.0,
                    "height": 0.0,
                    "offset_x": camera_offset.get("offset_x", 7.0),
                    "offset_y": camera_offset.get("offset_y", 92.9),
                    "timestamp": 0,
                    "processed": True
                })
            except Exception as e:
                print(f"[ERROR] Failed to create inspect shared memory: {e}")
                raise

    def get_camera_offset_from_config(self):
        """Get camera offset from config.json"""
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r") as f:
                    config = json.load(f)
                    return config.get("camera_offset", {"offset_x": 7.0, "offset_y": 92.9, "offset_error": 0.0})
        except:
            pass
        return {"offset_x": 7.0, "offset_y": 92.9, "offset_error": 0.0}

    def _write_data(self, data):
        """Write data to shared memory as JSON."""
        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')

            if len(json_bytes) > self.size - 4:
                return False

            self.shm.buf[:4] = struct.pack('I', len(json_bytes))
            self.shm.buf[4:4+len(json_bytes)] = json_bytes
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write inspect data: {e}")
            return False

    def send_inspect_command(self, target_x, target_y, angle=0.0, width=0.0, height=0.0, offset_x=None, offset_y=None):
        """Send inspection command with target position and object angle."""
        camera_offset = self.get_camera_offset_from_config()
        # Use provided offsets or fall back to config
        if offset_x is None:
            offset_x = camera_offset.get("offset_x", 7.0)
        if offset_y is None:
            offset_y = camera_offset.get("offset_y", 92.9)

        data = {
            "inspect": True,
            "home": False,
            "target_x": float(target_x),
            "target_y": float(target_y),
            "angle": float(angle),
            "width": float(width),
            "height": float(height),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "offset_error": camera_offset.get("offset_error", 0.0),
            "timestamp": time.time(),
            "processed": False
        }
        return self._write_data(data)

    def send_home_command(self):
        """Send command to move robot to home position."""
        camera_offset = self.get_camera_offset_from_config()
        data = {
            "inspect": False,
            "home": True,
            "target_x": 0,
            "target_y": 0,
            "angle": 0.0,
            "width": 0.0,
            "height": 0.0,
            "offset_x": camera_offset.get("offset_x", 7.0),
            "offset_y": camera_offset.get("offset_y", 92.9),
            "offset_error": camera_offset.get("offset_error", 0.0),
            "timestamp": time.time(),
            "processed": False
        }
        return self._write_data(data)

    def cleanup(self):
        """Close and unlink shared memory."""
        if self.shm:
            try:
                self.shm.close()
            except:
                pass


class CoordinateTester(QMainWindow):
    """Simple tool to test coordinates by showing dots on camera"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coordinate Testing Tool with Robot Control")
        self.setGeometry(100, 100, 1400, 800)

        # Load configuration
        self.config = self.load_config()

        # Cameras
        self.system = None
        self.detection_camera = None
        self.inspection_camera = None

        # Calibration matrix
        self.H_camera_to_workspace = None
        self.H_workspace_to_camera = None  # Inverse matrix

        # Test points to display
        self.test_points = []  # List of (x_mm, y_mm) tuples

        # Click info panel tracking
        self.mouse_click_x = 0
        self.mouse_click_y = 0
        self.show_info_panel = False
        self.clicked_workspace_pos = None  # Store clicked position in workspace coordinates (x_mm, y_mm)

        # YOLO detection
        self.model = None
        self.detected_objects = []  # List of detected objects with their properties
        self.load_yolo_model()

        # Robot control - using InspectData for inspection commands
        self.inspect_data_mgr = None
        try:
            self.inspect_data_mgr = InspectDataManager(name="InspectData", size=512)
            print("[INFO] Connected to robot inspection control shared memory")
        except Exception as e:
            print(f"[WARNING] Could not connect to robot inspection control: {e}")

        self.init_ui()
        self.load_calibration()

        # Start camera timer - cameras will be initialized on first update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_cameras)
        self.timer.start(33)  # ~30 FPS

    def load_config(self, path="config.json"):
        """Load configuration from JSON file"""
        try:
            if not os.path.isabs(path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(base_dir, path)

            with open(path, "r") as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"[ERROR] Config load failed: {e}")
            return {}

    def load_yolo_model(self):
        """Load YOLO model from config"""
        model_path = self.config.get("yolo_model_path", "best.pt")
        self.model = YOLO(model_path)
        self.model.overrides['verbose'] = False

    def get_angle(self, obb_pts):
        """Calculate angle from OBB points (same as arm-app-v1.1.py)"""
        v1 = obb_pts[1] - obb_pts[0]
        v2 = obb_pts[2] - obb_pts[1]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        long_vec = v1 if len1 >= len2 else v2
        angle_rad = np.arctan2(long_vec[1], long_vec[0])
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 180
        return angle_deg

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using cv2.pointPolygonTest"""
        return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0

    def transform_points(self, points, matrix):
        """Transform points using homography matrix"""
        if len(points.shape) == 1:
            points = points.reshape(-1, 2)
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, matrix)
        return transformed.reshape(-1, 2)

    def is_inside_workspace(self, pts):
        """Check if points are inside workspace boundaries"""
        workspace_width = self.config.get("workspace", {}).get("width", 300)
        workspace_height = self.config.get("workspace", {}).get("height", 300)
        x, y = pts[:, 0], pts[:, 1]
        return np.all((x >= 0) & (x <= workspace_width) & (y >= 0) & (y <= workspace_height))

    def load_calibration(self):
        """Load calibration matrix from homography_auto.pkl"""
        # Get script directory to find calibration file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calib_file = os.path.join(script_dir, "homography_auto.pkl")

        if os.path.exists(calib_file):
            try:
                with open(calib_file, 'rb') as f:
                    self.H_camera_to_workspace = pickle.load(f)
                    # Calculate inverse matrix for camera display
                    self.H_workspace_to_camera = np.linalg.inv(self.H_camera_to_workspace)
                self.status_label.setText("Status: Calibration loaded ✓")
                print(f"[INFO] Loaded calibration: {calib_file}")
            except Exception as e:
                self.status_label.setText(f"Status: Error loading calibration - {str(e)[:30]}...")
                print(f"[ERROR] Failed to load calibration: {e}")
        else:
            self.status_label.setText(f"Status: No calibration found")
            print(f"[WARNING] Calibration file not found: {calib_file}")

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

        # Angle input
        input_layout.addWidget(QLabel("Angle (°):"), 2, 0)
        self.angle_input = QLineEdit()
        self.angle_input.setPlaceholderText("Enter angle (0-180)")
        self.angle_input.setText("0")  # Default to 0 degrees
        self.angle_input.textChanged.connect(self.on_angle_changed)  # Update display when angle changes
        input_layout.addWidget(self.angle_input, 2, 1)

        # Camera offset X input
        input_layout.addWidget(QLabel("Offset X (mm):"), 3, 0)
        self.offset_x_input = QLineEdit()
        self.offset_x_input.setPlaceholderText("Camera offset X")
        self.offset_x_input.setText("7.0")  # Default from config
        input_layout.addWidget(self.offset_x_input, 3, 1)

        # Camera offset Y input
        input_layout.addWidget(QLabel("Offset Y (mm):"), 4, 0)
        self.offset_y_input = QLineEdit()
        self.offset_y_input.setPlaceholderText("Camera offset Y")
        self.offset_y_input.setText("92.9")  # Default from config
        input_layout.addWidget(self.offset_y_input, 4, 1)

        # Buttons
        btn_layout = QHBoxLayout()

        self.add_btn = QPushButton("Add Point")
        self.add_btn.clicked.connect(self.add_point)
        btn_layout.addWidget(self.add_btn)

        self.send_robot_btn = QPushButton("Inspect Target")
        self.send_robot_btn.clicked.connect(self.send_to_robot)
        self.send_robot_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        btn_layout.addWidget(self.send_robot_btn)

        self.home_btn = QPushButton("Home")
        self.home_btn.clicked.connect(self.send_home)
        self.home_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_layout.addWidget(self.home_btn)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_points)
        btn_layout.addWidget(self.clear_btn)

        input_layout.addLayout(btn_layout, 5, 0, 1, 2)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Status label
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.status_label)

        # Camera views - side by side
        cameras_layout = QHBoxLayout()

        # Detection camera view
        det_group = QGroupBox("Detection Camera (Workspace View)")
        det_layout = QVBoxLayout()
        self.detection_label = ClickableLabel()  # Use ClickableLabel for click detection
        self.detection_label.setMinimumSize(400, 300)  # Smaller minimum for flexibility
        self.detection_label.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_label.setStyleSheet("border: 2px solid #555; background-color: #2a2a2a;")
        self.detection_label.clicked.connect(self.on_detection_click)  # Connect click signal
        self.detection_label.setScaledContents(False)  # Keep aspect ratio
        det_layout.addWidget(self.detection_label)
        det_group.setLayout(det_layout)
        cameras_layout.addWidget(det_group, 1)  # Stretch factor 1

        # Inspection camera view
        insp_group = QGroupBox("Inspection Camera (Gripper View)")
        insp_layout = QVBoxLayout()
        self.inspection_label = QLabel()
        self.inspection_label.setMinimumSize(400, 300)  # Smaller minimum for flexibility
        self.inspection_label.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self.inspection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inspection_label.setStyleSheet("border: 2px solid #555; background-color: #2a2a2a;")
        self.inspection_label.setScaledContents(False)  # Keep aspect ratio
        insp_layout.addWidget(self.inspection_label)
        insp_group.setLayout(insp_layout)
        cameras_layout.addWidget(insp_group, 1)  # Stretch factor 1

        layout.addLayout(cameras_layout, 1)  # Add stretch factor to make cameras expand

        # Info label
        self.info_label = QLabel("Enter coordinates and angle (0-180°) then click 'Inspect Target' to position inspection camera")
        self.info_label.setFont(QFont("Arial", 9))
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

    def init_cameras(self):
        """Initialize both detection and inspection cameras"""
        try:
            self.system = PySpin.System.GetInstance()
            cam_list = self.system.GetCameras()

            if cam_list.GetSize() >= 1:
                # Get first camera (detection camera)
                self.detection_camera = cam_list.GetByIndex(0)
                self.detection_camera.Init()
                self.detection_camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
                self.detection_camera.BeginAcquisition()
                print("[INFO] Detection camera initialized")

            if cam_list.GetSize() >= 2:
                # Get second camera (inspection camera)
                self.inspection_camera = cam_list.GetByIndex(1)
                self.inspection_camera.Init()
                self.inspection_camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
                self.inspection_camera.BeginAcquisition()
                print("[INFO] Inspection camera initialized")
                self.status_label.setText("Status: Both cameras connected ✓")
            elif cam_list.GetSize() == 1:
                self.status_label.setText("Status: Detection camera connected (inspection camera not found)")
            else:
                self.status_label.setText("Status: No cameras detected")
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

        # Display placeholder on both labels
        h, w, ch = placeholder.shape
        bytes_per_line = ch * w
        qt_image = QImage(placeholder.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image.rgbSwapped())
        self.detection_label.setPixmap(pixmap)
        self.inspection_label.setPixmap(pixmap)

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

    def send_to_robot(self):
        """Send inspection command to robot via shared memory"""
        try:
            x_mm = float(self.x_input.text())
            y_mm = float(self.y_input.text())
            angle = float(self.angle_input.text()) if self.angle_input.text() else 0.0
            offset_x = float(self.offset_x_input.text()) if self.offset_x_input.text() else 7.0
            offset_y = float(self.offset_y_input.text()) if self.offset_y_input.text() else 92.9

            if self.inspect_data_mgr is None:
                self.info_label.setText("✗ Error: Robot inspection control not connected")
                return

            # Send inspection command to robot with custom offset values
            # Robot will position gripper so inspection camera views the target at (x_mm, y_mm)
            # at inspection_height (103.4mm from config.json)
            if self.inspect_data_mgr.send_inspect_command(x_mm, y_mm, angle=angle, width=0.0, height=0.0,
                                                          offset_x=offset_x, offset_y=offset_y):
                self.info_label.setText(f"✓ Inspection sent: Target ({x_mm:.1f}, {y_mm:.1f}) mm, Angle: {angle:.1f}°, Offset: ({offset_x:.1f}, {offset_y:.1f})")
                print(f"[INFO] Sent inspection command: Target ({x_mm:.1f}, {y_mm:.1f}) mm, Angle: {angle:.1f}°, Offset: ({offset_x:.1f}, {offset_y:.1f})")
            else:
                self.info_label.setText("✗ Error: Failed to send inspection command")

        except ValueError:
            self.info_label.setText("✗ Error: Please enter valid numbers for X, Y, Angle, and Offsets")

    def send_home(self):
        """Send home command to robot via shared memory"""
        try:
            if self.inspect_data_mgr is None:
                self.info_label.setText("✗ Error: Robot inspection control not connected")
                return

            # Send home command to robot
            if self.inspect_data_mgr.send_home_command():
                self.info_label.setText("✓ Home command sent - Robot moving to home position")
                print(f"[INFO] Sent home command to robot")
                self.status_label.setText("Status: Robot moving to home position...")
            else:
                self.info_label.setText("✗ Error: Failed to send home command")

        except Exception as e:
            self.info_label.setText(f"✗ Error: {str(e)}")

    def on_detection_click(self, x, y):
        """Handle click on detection camera - show info panel with coordinates and auto-fill angle if clicked on object"""
        self.mouse_click_x = x
        self.mouse_click_y = y
        self.show_info_panel = True

        # Convert click position to workspace coordinates
        if self.H_camera_to_workspace is not None:
            click_pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
            workspace_coord = cv2.perspectiveTransform(click_pt, self.H_camera_to_workspace).reshape(-1, 2)
            click_x_mm = workspace_coord[0][0]
            click_y_mm = workspace_coord[0][1]
            self.clicked_workspace_pos = (click_x_mm, click_y_mm)

            # Auto-fill coordinates in input fields (use actual click position)
            self.x_input.setText(f"{click_x_mm:.1f}")
            self.y_input.setText(f"{click_y_mm:.1f}")

            # Check if clicked on any detected object to auto-fill angle
            clicked_on_object = False
            for obj_data in self.detected_objects:
                if self.point_in_polygon((x, y), obj_data['corners']):
                    clicked_on_object = True
                    # Auto-fill angle from detected object (coordinates already set to click position)
                    self.angle_input.setText(f"{obj_data['angle']:.0f}")
                    self.status_label.setText(f"Status: Clicked at ({click_x_mm:.1f}, {click_y_mm:.1f}) mm - Object {obj_data['id']}, Angle: {obj_data['angle']:.1f}° ✓")
                    break

            # If not clicked on object, just show click coordinates
            if not clicked_on_object:
                # Check if inside workspace
                workspace_width = self.config.get("workspace", {}).get("width", 300)
                workspace_height = self.config.get("workspace", {}).get("height", 300)

                if 0 <= click_x_mm <= workspace_width and 0 <= click_y_mm <= workspace_height:
                    self.status_label.setText(f"Status: Clicked at ({click_x_mm:.1f}, {click_y_mm:.1f}) mm - In workspace ✓")
                else:
                    self.status_label.setText(f"Status: Clicked at ({click_x_mm:.1f}, {click_y_mm:.1f}) mm - Outside workspace")
        else:
            self.clicked_workspace_pos = None
            self.status_label.setText(f"Status: Clicked at pixel ({x}, {y}) - No calibration")

    def on_angle_changed(self):
        """Called when angle input changes - no action needed, display updates automatically"""
        pass

    def draw_info_panel_on_frame(self, frame, x, y, info_lines, title="Info"):
        """Draw an information panel at specified position"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        padding = 10
        line_height = 20

        max_width = 0
        for line in info_lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, w)

        panel_width = max_width + 2 * padding
        panel_height = len(info_lines) * line_height + 2 * padding + 25

        # Adjust position if panel goes off screen
        if x + panel_width > frame.shape[1]:
            x = frame.shape[1] - panel_width - 10
        if y + panel_height > frame.shape[0]:
            y = frame.shape[0] - panel_height - 10

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (0, 255, 255), 2)

        # Draw title
        cv2.rectangle(frame, (x, y), (x + panel_width, y + 25), (0, 255, 255), -1)
        cv2.putText(frame, title, (x + padding, y + 18), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw info lines
        y_offset = y + 40
        for line in info_lines:
            cv2.putText(frame, line, (x + padding, y_offset), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y_offset += line_height

        return frame

    def update_cameras(self):
        """Update both camera displays"""
        # Initialize cameras on first update (lazy initialization to avoid DLL conflicts)
        if self.detection_camera is None:
            self.init_cameras()
            if self.detection_camera is None:
                # Failed to initialize, try again next time
                return

        # Update detection camera
        if self.detection_camera:
            try:
                image_result = self.detection_camera.GetNextImage(1000)
                if not image_result.IsIncomplete():
                    # Convert to OpenCV format
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    image_data = image_result.GetNDArray()

                    # Handle different pixel formats
                    if len(image_data.shape) == 2:
                        frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
                    elif len(image_data.shape) == 3:
                        frame = image_data.copy()
                    else:
                        image_result.Release()
                        return

                    image_result.Release()

                    # Run YOLO detection if model is loaded
                    self.detected_objects = []  # Clear previous detections
                    if self.model is not None:
                        try:
                            results = self.model(frame, conf=self.config.get("detection_confidence", 0.7))
                            if results and len(results) > 0 and hasattr(results[0], 'obb') and results[0].obb is not None:
                                obb_preds = results[0].obb

                                for i, obb in enumerate(obb_preds, 1):
                                    if hasattr(obb, "xyxyxyxy"):
                                        corners = obb.xyxyxyxy.cpu().numpy().reshape(-1, 2)
                                    elif hasattr(obb, "xyxy"):
                                        corners = obb.xyxy.cpu().numpy().reshape(-1, 2)
                                    else:
                                        continue

                                    # Transform corners to workspace coordinates
                                    if self.H_camera_to_workspace is not None:
                                        transformed = self.transform_points(corners, self.H_camera_to_workspace)
                                        is_inside = self.is_inside_workspace(transformed)

                                        if is_inside:
                                            # Calculate object properties in workspace coordinates
                                            center = np.mean(transformed, axis=0)
                                            angle = self.get_angle(transformed)

                                            side1 = np.linalg.norm(transformed[1] - transformed[0])
                                            side2 = np.linalg.norm(transformed[2] - transformed[1])
                                            width_mm = round(max(side1, side2), 1)
                                            height_mm = round(min(side1, side2), 1)

                                            x_mm, y_mm = round(center[0], 1), round(center[1], 1)
                                            angle_deg = round(angle, 1)

                                            # Store object data
                                            self.detected_objects.append({
                                                'id': i,
                                                'corners': corners.astype(int),
                                                'x_mm': x_mm,
                                                'y_mm': y_mm,
                                                'angle': angle_deg,
                                                'width': width_mm,
                                                'height': height_mm
                                            })

                                            # Draw bounding box (green for inside workspace)
                                            corners_int = corners.astype(int)
                                            cv2.polylines(frame, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)

                                            # Draw center point
                                            center_img = np.mean(corners, axis=0).astype(int)
                                            cv2.circle(frame, tuple(center_img), 5, (0, 0, 255), -1)
                                            cv2.circle(frame, tuple(center_img), 5, (255, 255, 255), 1)

                                            # Draw object label with angle
                                            label_pos = tuple(corners_int[0] - [0, 10])
                                            label_text = f"Obj {i}: {angle_deg:.0f}"
                                            cv2.putText(frame, label_text, label_pos,
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        except Exception as e:
                            print(f"[DEBUG] YOLO detection error: {e}")

                    # Draw test points
                    if self.H_workspace_to_camera is not None:
                        for x_mm, y_mm in self.test_points:
                            workspace_pt = np.array([[[x_mm, y_mm]]], dtype=np.float32)
                            camera_pt = cv2.perspectiveTransform(workspace_pt, self.H_workspace_to_camera)

                            px = int(camera_pt[0][0][0])
                            py = int(camera_pt[0][0][1])

                            if 0 <= px < width and 0 <= py < height:
                                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)
                                workspace_label = f"({x_mm:.1f}, {y_mm:.1f}) mm"
                                label_x = px + 10
                                label_y = py - 10
                                (w, h), _ = cv2.getTextSize(workspace_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(frame, (label_x - 2, label_y - h - 2),
                                             (label_x + w + 2, label_y + 2), (0, 0, 0), -1)
                                cv2.putText(frame, workspace_label, (label_x, label_y),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                        # Draw workspace boundary
                        workspace_width = self.config.get("workspace", {}).get("width", 300)
                        workspace_height = self.config.get("workspace", {}).get("height", 300)
                        workspace_corners = np.array([
                            [[0, 0]], [[workspace_width, 0]],
                            [[workspace_width, workspace_height]], [[0, workspace_height]]
                        ], dtype=np.float32)
                        camera_corners = cv2.perspectiveTransform(workspace_corners, self.H_workspace_to_camera)
                        camera_corners = camera_corners.astype(np.int32)
                        cv2.polylines(frame, [camera_corners], isClosed=True, color=(0, 255, 255), thickness=2)

                        # Draw angle indicator at current input position
                        try:
                            if self.x_input.text() and self.y_input.text() and self.angle_input.text():
                                target_x_mm = float(self.x_input.text())
                                target_y_mm = float(self.y_input.text())
                                angle_deg = float(self.angle_input.text())

                                # Convert workspace position to camera pixels
                                target_pt = np.array([[[target_x_mm, target_y_mm]]], dtype=np.float32)
                                target_px = cv2.perspectiveTransform(target_pt, self.H_workspace_to_camera)
                                center_x = int(target_px[0][0][0])
                                center_y = int(target_px[0][0][1])

                                if 0 <= center_x < width and 0 <= center_y < height:
                                    # Draw angle indicator line
                                    # Line length in pixels
                                    line_length = 50

                                    # Convert angle to radians (0° = horizontal right, counter-clockwise)
                                    angle_rad = np.radians(angle_deg)

                                    # Calculate end point of angle line
                                    end_x = int(center_x + line_length * np.cos(angle_rad))
                                    end_y = int(center_y - line_length * np.sin(angle_rad))  # Subtract because Y increases downward

                                    # Draw the angle indicator line
                                    cv2.line(frame, (center_x, center_y), (end_x, end_y), (255, 0, 255), 2, cv2.LINE_AA)

                                    # Draw arrowhead at end
                                    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.3)

                                    # Draw angle label
                                    label = f"{angle_deg:.0f}"
                                    label_pos = (center_x + 10, center_y - 10)
                                    cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
                        except (ValueError, AttributeError):
                            pass  # Ignore if inputs are invalid

                    # Draw info panel if showing coordinates
                    if self.show_info_panel and self.clicked_workspace_pos is not None:
                        workspace_width = self.config.get("workspace", {}).get("width", 300)
                        workspace_height = self.config.get("workspace", {}).get("height", 300)
                        click_x_mm, click_y_mm = self.clicked_workspace_pos
                        in_workspace = (0 <= click_x_mm <= workspace_width and 0 <= click_y_mm <= workspace_height)

                        info_lines = [
                            f"Pixel: ({self.mouse_click_x}, {self.mouse_click_y})",
                            f"Real: ({click_x_mm:.1f}, {click_y_mm:.1f}) mm",
                            f"In workspace: {'Yes' if in_workspace else 'No'}"
                        ]

                        frame = self.draw_info_panel_on_frame(
                            frame,
                            self.mouse_click_x + 10,
                            self.mouse_click_y + 10,
                            info_lines,
                            "Coordinates"
                        )

                        # Draw crosshair at click position
                        cv2.drawMarker(frame, (self.mouse_click_x, self.mouse_click_y),
                                      (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

                    # Display detection camera
                    h, w, ch = frame.shape

                    # Store original image size for ClickableLabel coordinate scaling
                    if isinstance(self.detection_label, ClickableLabel):
                        self.detection_label.original_image_size = (w, h)
                    qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image.rgbSwapped())

                    # Calculate displayed size for coordinate scaling
                    label_w = self.detection_label.width()
                    label_h = self.detection_label.height()
                    scale = min(label_w / w, label_h / h)
                    displayed_w = int(w * scale)
                    displayed_h = int(h * scale)

                    # Store displayed image size for ClickableLabel coordinate scaling
                    if isinstance(self.detection_label, ClickableLabel):
                        self.detection_label.displayed_image_size = (displayed_w, displayed_h)

                    scaled_pixmap = pixmap.scaled(self.detection_label.size(),
                                                 Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
                    self.detection_label.setPixmap(scaled_pixmap)
            except Exception as e:
                error_msg = str(e)
                if "Spinnaker" not in error_msg and "timeout" not in error_msg.lower():
                    print(f"[ERROR] Detection camera update failed: {error_msg}")

        # Update inspection camera
        if self.inspection_camera:
            try:
                image_result = self.inspection_camera.GetNextImage(1000)
                if not image_result.IsIncomplete():
                    image_data = image_result.GetNDArray()

                    # Handle different pixel formats
                    if len(image_data.shape) == 2:
                        frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
                    elif len(image_data.shape) == 3:
                        frame = image_data.copy()
                    else:
                        image_result.Release()
                        return

                    image_result.Release()

                    # Draw center point marker to show gripper aim point
                    h, w, ch = frame.shape
                    center_x = w // 2
                    center_y = h // 2

                    # Draw red dot at center
                    cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1, cv2.LINE_AA)

                    # Display inspection camera
                    qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image.rgbSwapped())
                    scaled_pixmap = pixmap.scaled(self.inspection_label.size(),
                                                 Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
                    self.inspection_label.setPixmap(scaled_pixmap)
            except Exception as e:
                error_msg = str(e)
                if "Spinnaker" not in error_msg and "timeout" not in error_msg.lower():
                    print(f"[ERROR] Inspection camera update failed: {error_msg}")

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop timer
        if hasattr(self, 'timer'):
            self.timer.stop()

        # Cleanup cameras
        if self.detection_camera:
            try:
                self.detection_camera.EndAcquisition()
                self.detection_camera.DeInit()
            except:
                pass

        if self.inspection_camera:
            try:
                self.inspection_camera.EndAcquisition()
                self.inspection_camera.DeInit()
            except:
                pass

        if self.system:
            try:
                cam_list = self.system.GetCameras()
                cam_list.Clear()
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
    app.exec()

    # Use os._exit() to avoid PyTorch/PySpin DLL conflict on exit
    os._exit(0)


if __name__ == "__main__":
    main()
