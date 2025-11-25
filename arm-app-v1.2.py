"""
Intelligent Robot Positioning GUI
PyQt6 application for calibration, dual camera visualization, and system monitoring
Communicates with xarm-motion via shared memory
"""

import sys
import cv2
import numpy as np
import pickle
import json
import os
from multiprocessing import shared_memory
import struct
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QTabWidget, QLabel, QLineEdit,
                              QPushButton, QGroupBox, QGridLayout, QTextEdit,
                              QCheckBox, QComboBox, QMessageBox, QSizePolicy)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from ultralytics import YOLO
import PySpin


class ClickableLabel(QLabel):
    """Custom QLabel that emits click signals with scaled coordinates"""
    clicked = pyqtSignal(int, int)  # Signal emits x, y in original image coordinates
    doubleClicked = pyqtSignal(int, int)  # Signal for double-click

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

    def mouseDoubleClickEvent(self, event):
        """Handle double-click events"""
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

                    # Emit double-click signal
                    self.doubleClicked.emit(orig_x, orig_y)


class DetectionDataManager:
    """Manages shared memory for detection data (writes object detections for xarm-motion to read)"""

    def __init__(self, name="DetectionData", size=4096):
        self.name = name
        self.size = size
        self.shm = None
        self._initialize_shared_memory()

    def _initialize_shared_memory(self):
        """Create or attach to existing shared memory."""
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            self._write_data({"status": "Not Ready", "timestamp": time.time(), "objects": {}})
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)

    def _write_data(self, data):
        """Write data to shared memory as JSON."""
        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode('utf-8')

            if len(json_bytes) > self.size - 4:
                print(f"[WARNING] Data too large for shared memory ({len(json_bytes)} > {self.size-4})")
                return False

            self.shm.buf[:4] = struct.pack('I', len(json_bytes))
            self.shm.buf[4:4+len(json_bytes)] = json_bytes
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write to shared memory: {e}")
            return False

    def update_status(self, status):
        """Update only the status field."""
        data = self._read_data()
        data["status"] = status
        data["timestamp"] = time.time()
        self._write_data(data)

    def update_object(self, object_id, x, y, angle, width, height):
        """Update a single object's data."""
        data = self._read_data()
        if "objects" not in data:
            data["objects"] = {}
        data["objects"][str(object_id)] = {
            "x": float(round(x, 1)),
            "y": float(round(y, 1)),
            "angle": float(round(angle, 1)),
            "width": float(round(width, 1)),
            "height": float(round(height, 1))
        }
        data["timestamp"] = time.time()
        self._write_data(data)

    def clear_object(self, object_id):
        """Remove an object from shared memory."""
        data = self._read_data()
        if str(object_id) in data["objects"]:
            del data["objects"][str(object_id)]
            data["timestamp"] = time.time()
            self._write_data(data)

    def clear_all_objects(self):
        """Clear all objects."""
        data = self._read_data()
        data["objects"] = {}
        data["timestamp"] = time.time()
        self._write_data(data)

    def _read_data(self):
        """Read data from shared memory."""
        try:
            length = struct.unpack('I', bytes(self.shm.buf[:4]))[0]
            if length == 0 or length > self.size - 4:
                return {"status": "Not Ready", "timestamp": time.time(), "objects": {}}

            json_bytes = bytes(self.shm.buf[4:4+length])
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)

            if "objects" not in data:
                data["objects"] = {}

            return data
        except Exception as e:
            print(f"[ERROR] Failed to read from shared memory: {e}")
            return {"status": "Not Ready", "timestamp": time.time(), "objects": {}}

    def get_data(self):
        """Public method to read current data."""
        return self._read_data()

    def cleanup(self):
        """Close and unlink shared memory."""
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception as e:
                print(f"[ERROR] Failed to cleanup shared memory: {e}")


class ClickDataManager:
    """Manages shared memory for mouse click data (sends click commands to xarm-motion)"""

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
                return {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0, "width": 0.0, "height": 0.0}

            json_bytes = bytes(self.shm.buf[4:4+length])
            json_str = json_bytes.decode('utf-8')
            data = json.loads(json_str)

            if "button" not in data:
                data["button"] = "left"
            if "angle" not in data:
                data["angle"] = 0.0
            if "width" not in data:
                data["width"] = 0.0
            if "height" not in data:
                data["height"] = 0.0

            return data
        except Exception as e:
            print(f"[ERROR] Failed to read click data: {e}")
            return {"click_x": 0, "click_y": 0, "timestamp": 0, "processed": True, "button": "none", "angle": 0.0, "width": 0.0, "height": 0.0}

    def write_click(self, x, y, button="left", angle=0.0, width=0.0, height=0.0):
        """Write a new click position with button type, object angle, and dimensions."""
        data = {
            "click_x": float(x),
            "click_y": float(y),
            "button": button,
            "angle": float(angle),
            "width": float(width),
            "height": float(height),
            "timestamp": time.time(),
            "processed": False
        }
        self._write_data(data)

    def cleanup(self):
        """Close and unlink shared memory."""
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception as e:
                print(f"[ERROR] Failed to cleanup: {e}")


class InspectDataManager:
    """Manages shared memory for inspection commands (sends inspection requests to xarm-motion)"""

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
            if not os.path.isabs("config.json"):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(base_dir, "config.json")
            else:
                config_path = "config.json"

            with open(config_path, "r") as f:
                config = json.load(f)
                return config.get("camera_offset", {"offset_x": 7.0, "offset_y": 92.9, "offset_error": 0.0})
        except:
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

    def write_inspect_command(self, target_x, target_y, angle=0.0, width=0.0, height=0.0):
        """Send inspection command with target position and object angle."""
        camera_offset = self.get_camera_offset_from_config()
        data = {
            "inspect": True,
            "target_x": float(target_x),
            "target_y": float(target_y),
            "angle": float(angle),
            "width": float(width),
            "height": float(height),
            "offset_x": camera_offset.get("offset_x", 7.0),
            "offset_y": camera_offset.get("offset_y", 92.9),
            "offset_error": camera_offset.get("offset_error", 0.0),
            "timestamp": time.time(),
            "processed": False
        }
        self._write_data(data)

    def cleanup(self):
        """Close and unlink shared memory."""
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception as e:
                print(f"[ERROR] Failed to cleanup: {e}")


class RobotVisionGUI(QMainWindow):
    """Main GUI application for robot vision system"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Robot Positioning")
        self.setGeometry(100, 100, 900, 850)  # Same size as before

        # Load configuration
        self.config = self.load_config()

        # Initialize variables
        self.detection_camera = None
        self.inspection_camera = None
        self.camera_system = None
        self.H_camera_to_workspace = None  # Camera → Workspace transformation
        self.H_workspace_to_robot = None   # Workspace → Robot transformation
        self.system_running = False

        # Object detection tracking
        self.detected_objects = []  # List of detected objects with their data
        self.selected_object = None  # Currently selected object
        self.mouse_click_x = 0  # Mouse click position in image coordinates
        self.mouse_click_y = 0
        self.show_info_panel = False  # Whether to show info panel
        self.clicked_workspace_pos = None  # Store clicked position in workspace coordinates (x_mm, y_mm)

        # Inspection visualization box (not camera overlay)
        self.inspection_box_visible = False
        self.inspection_box_x = 0  # Center position
        self.inspection_box_y = 0  # Center position
        self.inspection_box_width = 0  # Fixed size: 36mm x 36mm (converted to pixels)
        self.inspection_box_height = 0  # Fixed size: 36mm x 36mm (converted to pixels)
        self.inspection_box_angle = 0.0  # Rotation angle in degrees
        self.inspection_box_object_data = None  # Store object data for reference

        # Shared memory managers
        self.detection_data_mgr = None  # DetectionData shared memory manager (INFO ONLY - no control)
        self.click_data_mgr = None  # ClickData manager (initialized but not used for robot control)
        self.inspect_data_mgr = None  # InspectData manager (initialized but not used for robot control)

        # Detection status tracking (for shared memory updates)
        self.last_saved_status = None
        self.last_saved_objects = {}
        self.last_outputs = {}  # Track last output for each object
        self.last_change_time = {}  # Track last change time for each object

        # YOLO model
        model_path = self.config.get("yolo_model_path", "best.pt")
        self.model = YOLO(model_path)
        self.model.overrides['verbose'] = False

        # Setup UI
        self.init_ui()

        # Apply default theme on startup
        self.apply_dark_theme()
        
        # Auto-load existing homography files on startup
        self.auto_load_homography_files()
        
        # Auto-load last used profile on startup
        self.auto_load_last_profile()

        # Timer for camera updates
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_feeds)

        # Timer for info display updates (real-time timestamp)
        self.info_timer = QTimer()
        self.info_timer.timeout.connect(self.update_info_display)
        self.info_timer.start(1000)  # Update every 1 second

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
            return self.get_default_config()

    def get_default_config(self):
        """Return default configuration"""
        return {
            "yolo_model_path": "best.pt",
            "detection_confidence": 0.7,
            "camera_config": {
                "detection_camera_index": 0,
                "inspection_camera_index": 1
            },
            "workspace": {
                "width": 300,
                "height": 300
            }
        }

    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Add tabs
        tabs.addTab(self.create_calibration_tab(), "Calibration Setup")
        tabs.addTab(self.create_live_view_tab(), "Live Camera View")
        tabs.addTab(self.create_info_tab(), "System Info")
        tabs.addTab(self.create_settings_tab(), "Settings")

        # Add control panel at bottom
        self.control_panel = self.create_control_panel()
        main_layout.addWidget(self.control_panel)

        # Status bar
        self.statusBar().showMessage("Ready - Configure calibration and press START")

        # Install event filter on application to capture arrow keys globally
        QApplication.instance().installEventFilter(self)

    def create_calibration_tab(self):
        """Create calibration configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # === Calibration Status & Reload ===
        status_group = QGroupBox("Calibration Status")
        status_layout = QVBoxLayout()
        
        # Current profile display
        profile_info_layout = QHBoxLayout()
        profile_info_layout.addWidget(QLabel("Current Profile:"))
        self.current_profile_label = QLabel("(None)")
        self.current_profile_label.setStyleSheet("font-weight: bold; color: #0078d4;")
        profile_info_layout.addWidget(self.current_profile_label)
        profile_info_layout.addStretch()
        status_layout.addLayout(profile_info_layout)
        
        # Status labels
        status_info_layout = QHBoxLayout()
        status_info_layout.addWidget(QLabel("Camera → Workspace:"))
        self.cal_status_camera = QLabel("❌ Not Calibrated")
        self.cal_status_camera.setStyleSheet("font-weight: bold;")
        status_info_layout.addWidget(self.cal_status_camera)
        status_info_layout.addStretch()
        status_layout.addLayout(status_info_layout)
        
        status_info_layout2 = QHBoxLayout()
        status_info_layout2.addWidget(QLabel("Workspace → Robot:"))
        self.cal_status_robot = QLabel("❌ Not Calibrated")
        self.cal_status_robot.setStyleSheet("font-weight: bold;")
        status_info_layout2.addWidget(self.cal_status_robot)
        status_info_layout2.addStretch()
        status_layout.addLayout(status_info_layout2)
        
        # Reload button
        reload_btn = QPushButton("Reload Homography Files")
        reload_btn.clicked.connect(self.reload_homography_files)
        reload_btn.setToolTip("Load existing calibration files (homography_auto.pkl and homography_det_to_robot.pkl)")
        status_layout.addWidget(reload_btn)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # === Detection Camera Calibration ===
        detection_group = QGroupBox("Detection Camera Calibration (Camera → Workspace)")
        detection_layout = QVBoxLayout()

        # Dropdown for calibration mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Calibration Mode:")
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItems(["Auto Calibration (Circle Detection)", "Manual Calibration"])
        self.detection_mode_combo.currentIndexChanged.connect(self.toggle_detection_calibration_mode)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.detection_mode_combo)
        mode_layout.addStretch()
        detection_layout.addLayout(mode_layout)

        # === Auto Calibration Container ===
        self.auto_cal_container = QWidget()
        self.auto_cal_container.setObjectName("autoCalContainer")
        auto_cal_layout = QVBoxLayout(self.auto_cal_container)
        auto_cal_layout.setContentsMargins(0, 10, 0, 0)

        auto_cal_info = QLabel("Place 4 white circles (20-30mm) at workspace corners, then click calibrate.")
        auto_cal_info.setWordWrap(True)
        auto_cal_layout.addWidget(auto_cal_info)

        # Auto calibration button - fixed size right aligned
        auto_cal_btn = QPushButton("Calibrate")
        auto_cal_btn.clicked.connect(self.auto_calibrate_detection)
        auto_cal_btn.setFixedWidth(80)
        auto_cal_btn.setFixedHeight(30)
        auto_btn_layout = QHBoxLayout()
        auto_btn_layout.addStretch()
        auto_btn_layout.addWidget(auto_cal_btn)
        auto_cal_layout.addLayout(auto_btn_layout)

        detection_layout.addWidget(self.auto_cal_container)

        # === Manual Calibration Container ===
        self.manual_cal_container = QWidget()
        self.manual_cal_container.setObjectName("manualCalContainer")
        manual_cal_layout = QVBoxLayout(self.manual_cal_container)
        manual_cal_layout.setContentsMargins(0, 10, 0, 0)

        manual_detection_label = QLabel("Enter 4 corners (pixels → workspace mm):")
        manual_cal_layout.addWidget(manual_detection_label)

        self.detection_cal_inputs = {}
        corners = [
            ('BL', 'Bottom Left'),
            ('BR', 'Bottom Right'),
            ('TR', 'Top Right'),
            ('TL', 'Top Left')
        ]

        # Use grid layout for better alignment
        grid_layout = QGridLayout()
        grid_layout.setColumnStretch(1, 1)  # Pixel X
        grid_layout.setColumnStretch(2, 1)  # Pixel Y
        grid_layout.setColumnStretch(4, 1)  # Workspace X
        grid_layout.setColumnStretch(5, 1)  # Workspace Y

        for row, (corner_code, corner_name) in enumerate(corners):
            # Label
            grid_layout.addWidget(QLabel(f"{corner_name} ({corner_code}):"), row, 0)

            # Pixel inputs
            pixel_x = QLineEdit()
            pixel_x.setPlaceholderText("Pixel X")
            pixel_y = QLineEdit()
            pixel_y.setPlaceholderText("Pixel Y")
            grid_layout.addWidget(pixel_x, row, 1)
            grid_layout.addWidget(pixel_y, row, 2)

            # Arrow
            grid_layout.addWidget(QLabel("→"), row, 3)

            # Workspace inputs
            mm_x = QLineEdit()
            mm_x.setPlaceholderText("Workspace X (mm)")
            mm_y = QLineEdit()
            mm_y.setPlaceholderText("Workspace Y (mm)")
            grid_layout.addWidget(mm_x, row, 4)
            grid_layout.addWidget(mm_y, row, 5)

            self.detection_cal_inputs[corner_code] = {
                'pixel_x': pixel_x, 'pixel_y': pixel_y,
                'mm_x': mm_x, 'mm_y': mm_y
            }

        manual_cal_layout.addLayout(grid_layout)

        # Manual calibration button - fixed size right aligned
        manual_detection_btn = QPushButton("Calibrate")
        manual_detection_btn.clicked.connect(self.apply_manual_detection_calibration)
        manual_detection_btn.setFixedWidth(80)
        manual_detection_btn.setFixedHeight(30)
        manual_btn_layout = QHBoxLayout()
        manual_btn_layout.addStretch()
        manual_btn_layout.addWidget(manual_detection_btn)
        manual_cal_layout.addLayout(manual_btn_layout)

        detection_layout.addWidget(self.manual_cal_container)

        # Initially hide manual calibration (show auto by default)
        self.manual_cal_container.hide()

        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # === Robot Coordinate Transformation ===
        robot_group = QGroupBox("Robot Coordinate Transformation (Workspace → Robot)")
        robot_layout = QVBoxLayout()

        # Auto mode (2 points)
        auto_robot_label = QLabel("Auto Mode - Enter 2 corners (BL and TR):")
        robot_layout.addWidget(auto_robot_label)

        self.robot_auto_mode = QCheckBox("Auto Calculation (enter only BL and TR)")
        self.robot_auto_mode.setChecked(True)
        self.robot_auto_mode.stateChanged.connect(self.toggle_robot_calibration_mode)
        robot_layout.addWidget(self.robot_auto_mode)

        # Input fields for robot calibration
        self.robot_cal_inputs = {}
        robot_corners = [
            ('BL', 'Bottom Left'),
            ('BR', 'Bottom Right'),
            ('TR', 'Top Right'),
            ('TL', 'Top Left')
        ]

        # Use grid layout for better alignment
        robot_grid_layout = QGridLayout()
        robot_grid_layout.setColumnStretch(1, 1)  # Workspace X
        robot_grid_layout.setColumnStretch(2, 1)  # Workspace Y
        robot_grid_layout.setColumnStretch(4, 1)  # Robot X
        robot_grid_layout.setColumnStretch(5, 1)  # Robot Y

        for row, (corner_code, corner_name) in enumerate(robot_corners):
            # Label
            robot_grid_layout.addWidget(QLabel(f"{corner_name} ({corner_code}):"), row, 0)

            # Workspace inputs
            ws_x = QLineEdit()
            ws_x.setPlaceholderText("Workspace X (mm)")
            ws_y = QLineEdit()
            ws_y.setPlaceholderText("Workspace Y (mm)")
            robot_grid_layout.addWidget(ws_x, row, 1)
            robot_grid_layout.addWidget(ws_y, row, 2)

            # Arrow
            robot_grid_layout.addWidget(QLabel("→"), row, 3)

            # Robot inputs
            robot_x = QLineEdit()
            robot_x.setPlaceholderText("Robot X (mm)")
            robot_y = QLineEdit()
            robot_y.setPlaceholderText("Robot Y (mm)")
            robot_grid_layout.addWidget(robot_x, row, 4)
            robot_grid_layout.addWidget(robot_y, row, 5)

            self.robot_cal_inputs[corner_code] = {
                'ws_x': ws_x, 'ws_y': ws_y,
                'robot_x': robot_x, 'robot_y': robot_y
            }

            # Disable BR and TL in auto mode initially
            if corner_code in ['BR', 'TL']:
                ws_x.setEnabled(False)
                ws_y.setEnabled(False)
                robot_x.setEnabled(False)
                robot_y.setEnabled(False)

        robot_layout.addLayout(robot_grid_layout)

        # Pre-fill workspace coordinates
        self.robot_cal_inputs['BL']['ws_x'].setText("0")
        self.robot_cal_inputs['BL']['ws_y'].setText("0")
        self.robot_cal_inputs['BR']['ws_x'].setText("300")
        self.robot_cal_inputs['BR']['ws_y'].setText("0")
        self.robot_cal_inputs['TR']['ws_x'].setText("300")
        self.robot_cal_inputs['TR']['ws_y'].setText("300")
        self.robot_cal_inputs['TL']['ws_x'].setText("0")
        self.robot_cal_inputs['TL']['ws_y'].setText("300")

        # Calculate button - fixed size right aligned
        calculate_btn = QPushButton("Calculate")
        calculate_btn.clicked.connect(self.calculate_robot_transformation)
        calculate_btn.setFixedWidth(80)
        calculate_btn.setFixedHeight(30)
        calc_btn_layout = QHBoxLayout()
        calc_btn_layout.addStretch()
        calc_btn_layout.addWidget(calculate_btn)
        robot_layout.addLayout(calc_btn_layout)

        robot_group.setLayout(robot_layout)
        layout.addWidget(robot_group)

        # Add stretch to push everything to top
        layout.addStretch()

        return widget

    def create_live_view_tab(self):
        """Create live camera view tab"""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setSpacing(5)  # Reduce spacing between elements

        # Detection camera container with centered alignment
        camera_container = QHBoxLayout()
        camera_container.addStretch()  # Left spacer

        detection_container = QVBoxLayout()
        detection_container.setSpacing(5)  # Reduce spacing
        detection_title = QLabel("Detection Camera")
        detection_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detection_title.setStyleSheet("font-weight: bold; font-size: 30px;")
        detection_container.addWidget(detection_title)

        self.detection_label = ClickableLabel()  # Use ClickableLabel for click detection
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_label.setFixedSize(640, 480)  # Fixed size - won't expand
        self.detection_label.setStyleSheet("border: 2px solid black; background-color: #2b2b2b;")
        self.detection_label.clicked.connect(self.on_detection_click)  # Connect single click
        self.detection_label.doubleClicked.connect(self.on_detection_double_click)  # Connect double click
        detection_container.addWidget(self.detection_label)

        camera_container.addLayout(detection_container)
        camera_container.addStretch()  # Right spacer

        # Add centered camera to main layout
        main_layout.addLayout(camera_container)

        # Detection info display at bottom
        self.detection_info = QTextEdit()
        self.detection_info.setReadOnly(True)
        self.detection_info.setMaximumHeight(100)
        self.detection_info.setFont(QFont("Segoe UI", 9))
        main_layout.addWidget(self.detection_info)

        # Add stretch at bottom to push everything up
        main_layout.addStretch()

        return widget

    def create_info_tab(self):
        """Create system information tab with organized sections"""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        
        # Theme selection at top
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark Mode", "Light Mode"])
        self.theme_combo.currentIndexChanged.connect(self.toggle_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        main_layout.addLayout(theme_layout)
        
        # === Current Profile Section ===
        profile_group = QGroupBox("Active Profile")
        profile_layout = QGridLayout()
        profile_layout.addWidget(QLabel("Profile Name:"), 0, 0)
        self.info_profile_name = QLabel("(None)")
        self.info_profile_name.setStyleSheet("font-weight: bold;")
        profile_layout.addWidget(self.info_profile_name, 0, 1)
        profile_group.setLayout(profile_layout)
        main_layout.addWidget(profile_group)
        
        # === Vision System Section ===
        vision_group = QGroupBox("Vision System")
        vision_layout = QGridLayout()
        
        vision_layout.addWidget(QLabel("YOLO Model:"), 0, 0)
        self.info_model = QLabel("Loading...")
        vision_layout.addWidget(self.info_model, 0, 1)
        
        vision_layout.addWidget(QLabel("Detection Confidence:"), 1, 0)
        self.info_confidence = QLabel("0.7")
        vision_layout.addWidget(self.info_confidence, 1, 1)
        
        vision_layout.addWidget(QLabel("Detection Camera:"), 2, 0)
        self.info_det_cam = QLabel("Index 0")
        vision_layout.addWidget(self.info_det_cam, 2, 1)
        
        vision_layout.addWidget(QLabel("Inspection Camera:"), 3, 0)
        self.info_insp_cam = QLabel("Index 1")
        vision_layout.addWidget(self.info_insp_cam, 3, 1)
        
        vision_group.setLayout(vision_layout)
        main_layout.addWidget(vision_group)
        
        # === Robot System Section ===
        robot_group = QGroupBox("Robot System")
        robot_layout = QGridLayout()
        
        robot_layout.addWidget(QLabel("Robot IP:"), 0, 0)
        self.info_robot_ip = QLabel("192.168.1.151")
        robot_layout.addWidget(self.info_robot_ip, 0, 1)
        
        robot_layout.addWidget(QLabel("Workspace Size:"), 1, 0)
        self.info_workspace = QLabel("300 × 300 mm")
        robot_layout.addWidget(self.info_workspace, 1, 1)
        
        robot_layout.addWidget(QLabel("Camera Offset:"), 2, 0)
        self.info_cam_offset = QLabel("(0.0, 0.0) mm")
        robot_layout.addWidget(self.info_cam_offset, 2, 1)
        
        robot_layout.addWidget(QLabel("Inspection Height:"), 3, 0)
        self.info_insp_height = QLabel("103.4 mm")
        robot_layout.addWidget(self.info_insp_height, 3, 1)
        
        robot_group.setLayout(robot_layout)
        main_layout.addWidget(robot_group)
        
        # === Calibration Status Section ===
        calib_group = QGroupBox("Calibration Status")
        calib_layout = QGridLayout()
        
        calib_layout.addWidget(QLabel("Camera → Workspace:"), 0, 0)
        self.info_cal_camera = QLabel("❌ Not Configured")
        calib_layout.addWidget(self.info_cal_camera, 0, 1)
        
        calib_layout.addWidget(QLabel("Workspace → Robot:"), 1, 0)
        self.info_cal_robot = QLabel("❌ Not Configured")
        calib_layout.addWidget(self.info_cal_robot, 1, 1)
        
        calib_group.setLayout(calib_layout)
        main_layout.addWidget(calib_group)
        
        # === System Status Section ===
        status_group = QGroupBox("System Status")
        status_layout = QGridLayout()
        
        status_layout.addWidget(QLabel("Operational State:"), 0, 0)
        self.info_state = QLabel("STOPPED")
        self.info_state.setStyleSheet("font-weight: bold; color: #f44336;")
        status_layout.addWidget(self.info_state, 0, 1)
        
        status_layout.addWidget(QLabel("Detection Camera:"), 1, 0)
        self.info_det_status = QLabel("Disconnected")
        status_layout.addWidget(self.info_det_status, 1, 1)
        
        status_layout.addWidget(QLabel("Inspection Camera:"), 2, 0)
        self.info_insp_status = QLabel("Disconnected")
        status_layout.addWidget(self.info_insp_status, 2, 1)
        
        status_layout.addWidget(QLabel("Last Updated:"), 3, 0)
        self.info_timestamp = QLabel(time.strftime('%Y-%m-%d %H:%M:%S'))
        status_layout.addWidget(self.info_timestamp, 3, 1)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Add stretch to push everything to top
        main_layout.addStretch()
        
        # Initial update
        self.update_info_display()
        
        return widget

    def create_settings_tab(self):
        """Create settings configuration tab"""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        
        # === Detection Settings ===
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QGridLayout()
        
        # YOLO Model Path
        detection_layout.addWidget(QLabel("YOLO Model Path:"), 0, 0)
        self.setting_model_path = QLineEdit()
        self.setting_model_path.setText(self.config.get("yolo_model_path", ""))
        detection_layout.addWidget(self.setting_model_path, 0, 1, 1, 2)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model_path)
        detection_layout.addWidget(browse_btn, 0, 3)
        
        # Detection Confidence
        detection_layout.addWidget(QLabel("Detection Confidence:"), 1, 0)
        self.setting_confidence = QLineEdit()
        self.setting_confidence.setText(str(self.config.get("detection_confidence", 0.7)))
        self.setting_confidence.setPlaceholderText("0.0 - 1.0")
        detection_layout.addWidget(self.setting_confidence, 1, 1)
        
        # Camera Indices
        detection_layout.addWidget(QLabel("Detection Camera Index:"), 2, 0)
        self.setting_detection_cam = QLineEdit()
        cam_config = self.config.get("camera_config", {})
        self.setting_detection_cam.setText(str(cam_config.get("detection_camera_index", 0)))
        detection_layout.addWidget(self.setting_detection_cam, 2, 1)
        
        detection_layout.addWidget(QLabel("Inspection Camera Index:"), 3, 0)
        self.setting_inspection_cam = QLineEdit()
        self.setting_inspection_cam.setText(str(cam_config.get("inspection_camera_index", 1)))
        detection_layout.addWidget(self.setting_inspection_cam, 3, 1)
        
        detection_group.setLayout(detection_layout)
        main_layout.addWidget(detection_group)
        
        # === Inspection Settings ===
        inspection_group = QGroupBox("Inspection Settings")
        inspection_layout = QGridLayout()
        
        # Camera Offset
        inspection_layout.addWidget(QLabel("Camera Offset X (mm):"), 0, 0)
        self.setting_offset_x = QLineEdit()
        cam_offset = self.config.get("camera_offset", {})
        self.setting_offset_x.setText(str(cam_offset.get("offset_x", 0.0)))
        inspection_layout.addWidget(self.setting_offset_x, 0, 1)
        
        inspection_layout.addWidget(QLabel("Camera Offset Y (mm):"), 1, 0)
        self.setting_offset_y = QLineEdit()
        self.setting_offset_y.setText(str(cam_offset.get("offset_y", 0.0)))
        inspection_layout.addWidget(self.setting_offset_y, 1, 1)
        
        # Offset Error
        inspection_layout.addWidget(QLabel("Offset Error X (mm):"), 2, 0)
        self.setting_error_x = QLineEdit()
        self.setting_error_x.setText(str(cam_offset.get("offset_error_x", 0.0)))
        inspection_layout.addWidget(self.setting_error_x, 2, 1)
        
        inspection_layout.addWidget(QLabel("Offset Error Y (mm):"), 3, 0)
        self.setting_error_y = QLineEdit()
        self.setting_error_y.setText(str(cam_offset.get("offset_error_y", 0.0)))
        inspection_layout.addWidget(self.setting_error_y, 3, 1)
        
        # Inspection Height
        inspection_layout.addWidget(QLabel("Inspection Height (mm):"), 4, 0)
        self.setting_inspection_height = QLineEdit()
        click_control = self.config.get("click_control", {})
        self.setting_inspection_height.setText(str(click_control.get("inspection_height", 103.4)))
        inspection_layout.addWidget(self.setting_inspection_height, 4, 1)
        
        inspection_group.setLayout(inspection_layout)
        main_layout.addWidget(inspection_group)
        
        # === Profile Management ===
        profile_group = QGroupBox("Profile Management")
        profile_layout = QGridLayout()
        
        # Profile name input
        profile_layout.addWidget(QLabel("Profile Name:"), 0, 0)
        self.profile_name_input = QLineEdit()
        self.profile_name_input.setPlaceholderText("e.g., machine_A, station_1")
        profile_layout.addWidget(self.profile_name_input, 0, 1, 1, 2)
        
        # Profile selection dropdown
        profile_layout.addWidget(QLabel("Saved Profiles:"), 1, 0)
        self.profile_combo = QComboBox()
        self.refresh_profile_list()
        profile_layout.addWidget(self.profile_combo, 1, 1, 1, 2)
        
        # Profile action buttons
        save_profile_btn = QPushButton("Save Profile")
        save_profile_btn.clicked.connect(self.save_profile)
        profile_layout.addWidget(save_profile_btn, 2, 0)
        
        load_profile_btn = QPushButton("Load Profile")
        load_profile_btn.clicked.connect(self.load_profile)
        profile_layout.addWidget(load_profile_btn, 2, 1)
        
        delete_profile_btn = QPushButton("Delete Profile")
        delete_profile_btn.clicked.connect(self.delete_profile)
        profile_layout.addWidget(delete_profile_btn, 2, 2)
        
        profile_group.setLayout(profile_layout)
        main_layout.addWidget(profile_group)
        
        # === Action Buttons ===
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        load_btn = QPushButton("Load from Config")
        load_btn.clicked.connect(self.load_settings_from_config)
        button_layout.addWidget(load_btn)
        
        save_btn = QPushButton("Save to Config")
        save_btn.clicked.connect(self.save_settings_to_config)
        button_layout.addWidget(save_btn)
        
        apply_btn = QPushButton("Apply Settings")
        apply_btn.setStyleSheet("background-color: #0078d4; color: white; font-weight: bold;")
        apply_btn.clicked.connect(self.apply_settings)
        button_layout.addWidget(apply_btn)
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_settings)
        button_layout.addWidget(reset_btn)
        
        main_layout.addLayout(button_layout)
        
        # Add stretch to push everything to top
        main_layout.addStretch()
        
        return widget
    
    def browse_model_path(self):
        """Browse for YOLO model file"""
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "PyTorch Model (*.pt);;All Files (*.*)"
        )
        if file_path:
            self.setting_model_path.setText(file_path)
    
    def load_settings_from_config(self):
        """Load settings from config.json file"""
        try:
            # Reload config from file
            self.config = self.load_config()
            
            # Update detection settings
            self.setting_model_path.setText(self.config.get("yolo_model_path", ""))
            self.setting_confidence.setText(str(self.config.get("detection_confidence", 0.7)))
            
            cam_config = self.config.get("camera_config", {})
            self.setting_detection_cam.setText(str(cam_config.get("detection_camera_index", 0)))
            self.setting_inspection_cam.setText(str(cam_config.get("inspection_camera_index", 1)))
            
            # Update inspection settings
            cam_offset = self.config.get("camera_offset", {})
            self.setting_offset_x.setText(str(cam_offset.get("offset_x", 0.0)))
            self.setting_offset_y.setText(str(cam_offset.get("offset_y", 0.0)))
            self.setting_error_x.setText(str(cam_offset.get("offset_error_x", 0.0)))
            self.setting_error_y.setText(str(cam_offset.get("offset_error_y", 0.0)))
            
            click_control = self.config.get("click_control", {})
            self.setting_inspection_height.setText(str(click_control.get("inspection_height", 103.4)))
            
            QMessageBox.information(self, "Success", "Settings loaded from config.json")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load settings: {str(e)}")
    
    def save_settings_to_config(self):
        """Save current settings to config.json file"""
        try:
            # Validate inputs
            try:
                confidence = float(self.setting_confidence.text())
                if not (0.0 <= confidence <= 1.0):
                    raise ValueError("Confidence must be between 0.0 and 1.0")
                
                det_cam_idx = int(self.setting_detection_cam.text())
                insp_cam_idx = int(self.setting_inspection_cam.text())
                
                offset_x = float(self.setting_offset_x.text())
                offset_y = float(self.setting_offset_y.text())
                error_x = float(self.setting_error_x.text())
                error_y = float(self.setting_error_y.text())
                inspection_height = float(self.setting_inspection_height.text())
                
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", f"Please check your input values: {str(e)}")
                return
            
            # Update config dictionary
            self.config["yolo_model_path"] = self.setting_model_path.text()
            self.config["detection_confidence"] = confidence
            
            if "camera_config" not in self.config:
                self.config["camera_config"] = {}
            self.config["camera_config"]["detection_camera_index"] = det_cam_idx
            self.config["camera_config"]["inspection_camera_index"] = insp_cam_idx
            
            if "camera_offset" not in self.config:
                self.config["camera_offset"] = {}
            self.config["camera_offset"]["offset_x"] = offset_x
            self.config["camera_offset"]["offset_y"] = offset_y
            self.config["camera_offset"]["offset_error_x"] = error_x
            self.config["camera_offset"]["offset_error_y"] = error_y
            
            if "click_control" not in self.config:
                self.config["click_control"] = {}
            self.config["click_control"]["inspection_height"] = inspection_height
            
            # Save to file
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=4)
            
            QMessageBox.information(self, "Success", "Settings saved to config.json")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")
    
    def apply_settings(self):
        """Apply settings to running system (takes effect immediately)"""
        try:
            # Validate inputs
            try:
                confidence = float(self.setting_confidence.text())
                if not (0.0 <= confidence <= 1.0):
                    raise ValueError("Confidence must be between 0.0 and 1.0")
                
                offset_x = float(self.setting_offset_x.text())
                offset_y = float(self.setting_offset_y.text())
                error_x = float(self.setting_error_x.text())
                error_y = float(self.setting_error_y.text())
                inspection_height = float(self.setting_inspection_height.text())
                
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", f"Please check your input values: {str(e)}")
                return
            
            # Apply detection confidence
            self.config["detection_confidence"] = confidence
            print(f"[Settings] Detection confidence updated to: {confidence}")
            
            # Apply inspection settings
            if "camera_offset" not in self.config:
                self.config["camera_offset"] = {}
            self.config["camera_offset"]["offset_x"] = offset_x
            self.config["camera_offset"]["offset_y"] = offset_y
            self.config["camera_offset"]["offset_error_x"] = error_x
            self.config["camera_offset"]["offset_error_y"] = error_y
            
            if "click_control" not in self.config:
                self.config["click_control"] = {}
            self.config["click_control"]["inspection_height"] = inspection_height
            
            # Update inspection data manager if it exists
            if hasattr(self, 'inspect_data_manager') and self.inspect_data_manager:
                # The manager will read these values from config when writing commands
                print(f"[Settings] Inspection offset updated to: ({offset_x}, {offset_y})")
                print(f"[Settings] Inspection height updated to: {inspection_height}")
            
            QMessageBox.information(self, "Success", "Settings applied successfully!\n\nNote: Camera changes require system restart.")
            self.statusBar().showMessage(f"Settings applied - Confidence: {confidence}, Offset: ({offset_x}, {offset_y})")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {str(e)}")
    
    def reset_settings(self):
        """Reset settings to default values"""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Reset to defaults
            default_config = self.get_default_config()
            
            self.setting_model_path.setText(default_config.get("yolo_model_path", "best.pt"))
            self.setting_confidence.setText(str(default_config.get("detection_confidence", 0.7)))
            
            cam_config = default_config.get("camera_config", {})
            self.setting_detection_cam.setText(str(cam_config.get("detection_camera_index", 0)))
            self.setting_inspection_cam.setText(str(cam_config.get("inspection_camera_index", 1)))
            
            self.setting_offset_x.setText("0.0")
            self.setting_offset_y.setText("0.0")
            self.setting_error_x.setText("0.0")
            self.setting_error_y.setText("0.0")
            self.setting_inspection_height.setText("103.4")
            
            QMessageBox.information(self, "Success", "Settings reset to defaults")

    def refresh_profile_list(self):
        """Refresh the profile dropdown list"""
        self.profile_combo.clear()
        
        # Create profiles directory if it doesn't exist
        profiles_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        
        # List all .json files in profiles directory
        try:
            profiles = [f[:-5] for f in os.listdir(profiles_dir) if f.endswith('.json')]
            if profiles:
                self.profile_combo.addItems(sorted(profiles))
            else:
                self.profile_combo.addItem("(No profiles saved)")
        except Exception as e:
            print(f"[Profile] Error listing profiles: {e}")
            self.profile_combo.addItem("(Error loading profiles)")
    
    def save_profile(self):
        """Save current setup to a profile file"""
        profile_name = self.profile_name_input.text().strip()
        
        if not profile_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a profile name")
            return
        
        # Sanitize filename
        profile_name = "".join(c for c in profile_name if c.isalnum() or c in (' ', '_', '-')).strip()
        
        if not profile_name:
            QMessageBox.warning(self, "Invalid Name", "Profile name contains invalid characters")
            return
        
        try:
            # Collect all setup data
            profile_data = {
                "profile_name": profile_name,
                "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                
                # Detection calibration (manual 4 corners)
                "detection_calibration": {
                    "BL": {
                        "pixel_x": self.detection_cal_inputs['BL']['pixel_x'].text(),
                        "pixel_y": self.detection_cal_inputs['BL']['pixel_y'].text(),
                        "workspace_x": self.detection_cal_inputs['BL']['mm_x'].text(),
                        "workspace_y": self.detection_cal_inputs['BL']['mm_y'].text()
                    },
                    "BR": {
                        "pixel_x": self.detection_cal_inputs['BR']['pixel_x'].text(),
                        "pixel_y": self.detection_cal_inputs['BR']['pixel_y'].text(),
                        "workspace_x": self.detection_cal_inputs['BR']['mm_x'].text(),
                        "workspace_y": self.detection_cal_inputs['BR']['mm_y'].text()
                    },
                    "TR": {
                        "pixel_x": self.detection_cal_inputs['TR']['pixel_x'].text(),
                        "pixel_y": self.detection_cal_inputs['TR']['pixel_y'].text(),
                        "workspace_x": self.detection_cal_inputs['TR']['mm_x'].text(),
                        "workspace_y": self.detection_cal_inputs['TR']['mm_y'].text()
                    },
                    "TL": {
                        "pixel_x": self.detection_cal_inputs['TL']['pixel_x'].text(),
                        "pixel_y": self.detection_cal_inputs['TL']['pixel_y'].text(),
                        "workspace_x": self.detection_cal_inputs['TL']['mm_x'].text(),
                        "workspace_y": self.detection_cal_inputs['TL']['mm_y'].text()
                    }
                },
                
                # Robot coordinate transformation
                "robot_transformation": {
                    "BL": {
                        "workspace_x": self.robot_cal_inputs['BL']['ws_x'].text(),
                        "workspace_y": self.robot_cal_inputs['BL']['ws_y'].text(),
                        "robot_x": self.robot_cal_inputs['BL']['robot_x'].text(),
                        "robot_y": self.robot_cal_inputs['BL']['robot_y'].text()
                    },
                    "TR": {
                        "workspace_x": self.robot_cal_inputs['TR']['ws_x'].text(),
                        "workspace_y": self.robot_cal_inputs['TR']['ws_y'].text(),
                        "robot_x": self.robot_cal_inputs['TR']['robot_x'].text(),
                        "robot_y": self.robot_cal_inputs['TR']['robot_y'].text()
                    },
                    "auto_mode": self.robot_auto_mode.isChecked()
                },
                
                # Detection settings
                "detection_settings": {
                    "model_path": self.setting_model_path.text(),
                    "confidence": self.setting_confidence.text(),
                    "detection_camera_index": self.setting_detection_cam.text(),
                    "inspection_camera_index": self.setting_inspection_cam.text()
                },
                
                # Inspection settings
                "inspection_settings": {
                    "offset_x": self.setting_offset_x.text(),
                    "offset_y": self.setting_offset_y.text(),
                    "error_x": self.setting_error_x.text(),
                    "error_y": self.setting_error_y.text(),
                    "inspection_height": self.setting_inspection_height.text()
                }
            }
            
            # Save to file
            profiles_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
            os.makedirs(profiles_dir, exist_ok=True)
            
            profile_path = os.path.join(profiles_dir, f"{profile_name}.json")
            
            with open(profile_path, "w") as f:
                json.dump(profile_data, f, indent=4)
            
            print(f"[Profile] Saved: {profile_path}")
            
            # Refresh profile list
            self.refresh_profile_list()
            
            # Select the newly saved profile
            index = self.profile_combo.findText(profile_name)
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)
            
            QMessageBox.information(self, "Success", f"Profile '{profile_name}' saved successfully!")
            self.statusBar().showMessage(f"Profile '{profile_name}' saved")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save profile: {str(e)}")
            print(f"[Profile] Error saving: {e}")
    
    def load_profile(self):
        """Load setup from selected profile file"""
        profile_name = self.profile_combo.currentText()
        
        if profile_name == "(No profiles saved)" or profile_name == "(Error loading profiles)":
            QMessageBox.warning(self, "No Profile", "Please select a valid profile to load")
            return
        
        try:
            profiles_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
            profile_path = os.path.join(profiles_dir, f"{profile_name}.json")
            
            if not os.path.exists(profile_path):
                QMessageBox.warning(self, "Not Found", f"Profile file not found: {profile_name}")
                return
            
            # Load profile data
            with open(profile_path, "r") as f:
                profile_data = json.load(f)
            
            # Load detection calibration
            if "detection_calibration" in profile_data:
                cal_data = profile_data["detection_calibration"]
                for corner in ['BL', 'BR', 'TR', 'TL']:
                    if corner in cal_data:
                        self.detection_cal_inputs[corner]['pixel_x'].setText(str(cal_data[corner].get('pixel_x', '')))
                        self.detection_cal_inputs[corner]['pixel_y'].setText(str(cal_data[corner].get('pixel_y', '')))
                        self.detection_cal_inputs[corner]['mm_x'].setText(str(cal_data[corner].get('workspace_x', '')))
                        self.detection_cal_inputs[corner]['mm_y'].setText(str(cal_data[corner].get('workspace_y', '')))
            
            # Load robot transformation
            if "robot_transformation" in profile_data:
                robot_data = profile_data["robot_transformation"]
                
                # Set auto mode checkbox
                if "auto_mode" in robot_data:
                    self.robot_auto_mode.setChecked(robot_data["auto_mode"])
                
                # Load BL and TR
                for corner in ['BL', 'TR']:
                    if corner in robot_data:
                        self.robot_cal_inputs[corner]['ws_x'].setText(str(robot_data[corner].get('workspace_x', '')))
                        self.robot_cal_inputs[corner]['ws_y'].setText(str(robot_data[corner].get('workspace_y', '')))
                        self.robot_cal_inputs[corner]['robot_x'].setText(str(robot_data[corner].get('robot_x', '')))
                        self.robot_cal_inputs[corner]['robot_y'].setText(str(robot_data[corner].get('robot_y', '')))
            
            # Load detection settings
            if "detection_settings" in profile_data:
                det_data = profile_data["detection_settings"]
                self.setting_model_path.setText(str(det_data.get('model_path', '')))
                self.setting_confidence.setText(str(det_data.get('confidence', '0.7')))
                self.setting_detection_cam.setText(str(det_data.get('detection_camera_index', '0')))
                self.setting_inspection_cam.setText(str(det_data.get('inspection_camera_index', '1')))
            
            # Load inspection settings
            if "inspection_settings" in profile_data:
                insp_data = profile_data["inspection_settings"]
                self.setting_offset_x.setText(str(insp_data.get('offset_x', '0.0')))
                self.setting_offset_y.setText(str(insp_data.get('offset_y', '0.0')))
                self.setting_error_x.setText(str(insp_data.get('error_x', '0.0')))
                self.setting_error_y.setText(str(insp_data.get('error_y', '0.0')))
                self.setting_inspection_height.setText(str(insp_data.get('inspection_height', '103.4')))
            
            print(f"[Profile] Loaded: {profile_path}")
            
            # Save as last used profile in config
            self.config["last_used_profile"] = profile_name
            self.save_last_profile_to_config(profile_name)
            
            # Update current profile label
            self.current_profile_label.setText(profile_name)
            self.current_profile_label.setStyleSheet("font-weight: bold; color: #0078d4;")
            
            created_date = profile_data.get('created_date', 'Unknown')
            QMessageBox.information(self, "Success", f"Profile '{profile_name}' loaded successfully!\n\nCreated: {created_date}\n\nClick 'Apply Settings' and calibration buttons to activate.")
            self.statusBar().showMessage(f"Profile '{profile_name}' loaded - Click Apply to activate")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load profile: {str(e)}")
            print(f"[Profile] Error loading: {e}")
    
    def delete_profile(self):
        """Delete selected profile file"""
        profile_name = self.profile_combo.currentText()
        
        if profile_name == "(No profiles saved)" or profile_name == "(Error loading profiles)":
            QMessageBox.warning(self, "No Profile", "Please select a valid profile to delete")
            return
        
        reply = QMessageBox.question(
            self,
            "Delete Profile",
            f"Are you sure you want to delete profile '{profile_name}'?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                profiles_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
                profile_path = os.path.join(profiles_dir, f"{profile_name}.json")
                
                if os.path.exists(profile_path):
                    os.remove(profile_path)
                    print(f"[Profile] Deleted: {profile_path}")
                    
                    # Refresh profile list
                    self.refresh_profile_list()
                    
                    QMessageBox.information(self, "Success", f"Profile '{profile_name}' deleted successfully!")
                    self.statusBar().showMessage(f"Profile '{profile_name}' deleted")
                else:
                    QMessageBox.warning(self, "Not Found", f"Profile file not found: {profile_name}")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete profile: {str(e)}")
                print(f"[Profile] Error deleting: {e}")
    
    def save_last_profile_to_config(self, profile_name):
        """Save last used profile to config file"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
            
            # Read current config
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Update last used profile
            config["last_used_profile"] = profile_name
            
            # Save back to file
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            
            print(f"[Profile] Saved last used profile: {profile_name}")
            
        except Exception as e:
            print(f"[Profile] Error saving last profile to config: {e}")
    
    def auto_load_last_profile(self):
        """Auto-load last used profile on startup"""
        try:
            last_profile = self.config.get("last_used_profile", None)
            
            if not last_profile:
                print("[Startup] No last used profile found")
                self.current_profile_label.setText("(None)")
                self.current_profile_label.setStyleSheet("font-weight: bold; color: #888;")
                return
            
            # Check if profile still exists
            profiles_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
            profile_path = os.path.join(profiles_dir, f"{last_profile}.json")
            
            if not os.path.exists(profile_path):
                print(f"[Startup] Last profile '{last_profile}' not found")
                self.current_profile_label.setText("(None)")
                self.current_profile_label.setStyleSheet("font-weight: bold; color: #888;")
                return
            
            # Load the profile data
            with open(profile_path, "r") as f:
                profile_data = json.load(f)
            
            # Load detection calibration
            if "detection_calibration" in profile_data:
                cal_data = profile_data["detection_calibration"]
                for corner in ['BL', 'BR', 'TR', 'TL']:
                    if corner in cal_data:
                        self.detection_cal_inputs[corner]['pixel_x'].setText(str(cal_data[corner].get('pixel_x', '')))
                        self.detection_cal_inputs[corner]['pixel_y'].setText(str(cal_data[corner].get('pixel_y', '')))
                        self.detection_cal_inputs[corner]['mm_x'].setText(str(cal_data[corner].get('workspace_x', '')))
                        self.detection_cal_inputs[corner]['mm_y'].setText(str(cal_data[corner].get('workspace_y', '')))
            
            # Load robot transformation
            if "robot_transformation" in profile_data:
                robot_data = profile_data["robot_transformation"]
                
                # Set auto mode checkbox
                if "auto_mode" in robot_data:
                    self.robot_auto_mode.setChecked(robot_data["auto_mode"])
                
                # Load BL and TR
                for corner in ['BL', 'TR']:
                    if corner in robot_data:
                        self.robot_cal_inputs[corner]['ws_x'].setText(str(robot_data[corner].get('workspace_x', '')))
                        self.robot_cal_inputs[corner]['ws_y'].setText(str(robot_data[corner].get('workspace_y', '')))
                        self.robot_cal_inputs[corner]['robot_x'].setText(str(robot_data[corner].get('robot_x', '')))
                        self.robot_cal_inputs[corner]['robot_y'].setText(str(robot_data[corner].get('robot_y', '')))
            
            # Load detection settings
            if "detection_settings" in profile_data:
                det_data = profile_data["detection_settings"]
                self.setting_model_path.setText(str(det_data.get('model_path', '')))
                self.setting_confidence.setText(str(det_data.get('confidence', '0.7')))
                self.setting_detection_cam.setText(str(det_data.get('detection_camera_index', '0')))
                self.setting_inspection_cam.setText(str(det_data.get('inspection_camera_index', '1')))
            
            # Load inspection settings
            if "inspection_settings" in profile_data:
                insp_data = profile_data["inspection_settings"]
                self.setting_offset_x.setText(str(insp_data.get('offset_x', '0.0')))
                self.setting_offset_y.setText(str(insp_data.get('offset_y', '0.0')))
                self.setting_error_x.setText(str(insp_data.get('error_x', '0.0')))
                self.setting_error_y.setText(str(insp_data.get('error_y', '0.0')))
                self.setting_inspection_height.setText(str(insp_data.get('inspection_height', '103.4')))
            
            # Update current profile label
            self.current_profile_label.setText(last_profile)
            self.current_profile_label.setStyleSheet("font-weight: bold; color: #0078d4;")
            
            # Select in dropdown if exists
            if hasattr(self, 'profile_combo'):
                index = self.profile_combo.findText(last_profile)
                if index >= 0:
                    self.profile_combo.setCurrentIndex(index)
            
            print(f"[Startup] Auto-loaded last profile: {last_profile}")
            self.statusBar().showMessage(f"✅ Profile '{last_profile}' auto-loaded from last session")
            
        except Exception as e:
            print(f"[Startup] Error auto-loading last profile: {e}")
            self.current_profile_label.setText("(None)")
            self.current_profile_label.setStyleSheet("font-weight: bold; color: #888;")

    def toggle_theme(self, index):
        """Toggle between dark and light theme"""
        if index == 0:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

    def apply_light_theme(self):
        """Apply light theme"""
        self.setStyleSheet("""
            * { color: #000000; }
            QWidget { background-color: #f0f0f0; }
            QLabel, QCheckBox { background-color: transparent; }
            QWidget#autoCalContainer, QWidget#manualCalContainer { background-color: transparent; }
            QLineEdit, QTextEdit, QComboBox { background-color: #ffffff; border: 1px solid #c0c0c0; border-radius: 4px; padding: 2px; }
            QLineEdit:read-only, QTextEdit:read-only, QLineEdit:disabled, QTextEdit:disabled { background-color: #e8e8e8; }
            QPushButton { background-color: #ffffff; border: 1px solid #c0c0c0; border-radius: 4px; padding: 5px 15px; }
            QPushButton:hover { background-color: #e8e8e8; }
            QPushButton:pressed { background-color: #d0d0d0; }
            QGroupBox { background-color: #ffffff; border: 1px solid #c0c0c0; border-radius: 6px; margin-top: 6px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QTabWidget::pane { border: 1px solid #c0c0c0; border-radius: 4px; background-color: #ffffff; }
            QTabBar::tab { background-color: #e0e0e0; border: 1px solid #c0c0c0; border-top-left-radius: 4px; border-top-right-radius: 4px; padding: 3px 8px; }
            QTabBar::tab:selected { background-color: #ffffff; }
        """)

    def apply_dark_theme(self):
        """Apply dark theme"""
        self.setStyleSheet("""
            * { color: #ffffff; }
            QWidget { background-color: #353535; }
            QLabel, QCheckBox { background-color: transparent; }
            QWidget#autoCalContainer, QWidget#manualCalContainer { background-color: transparent; }
            QLineEdit, QTextEdit, QComboBox { background-color: #2b2b2b; border: 1px solid #555555; border-radius: 4px; padding: 2px; }
            QLineEdit:read-only, QTextEdit:read-only, QLineEdit:disabled, QTextEdit:disabled { background-color: #3c3c3c; }
            QPushButton { background-color: #3c3c3c; border: 1px solid #555555; border-radius: 4px; padding: 5px 15px; }
            QPushButton:hover { background-color: #454545; }
            QPushButton:pressed { background-color: #2b2b2b; }
            QGroupBox { background-color: #3c3c3c; border: 1px solid #555555; border-radius: 6px; margin-top: 6px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """)

    def create_control_panel(self):
        """Create control panel with START/STOP buttons"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # Set fixed size policy and height to prevent expansion
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        widget.setFixedHeight(50)  # Fixed height (not just maximum)

        self.start_btn = QPushButton("START System")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 4px;
                border: 2px solid #4CAF50;
            }
            QPushButton:hover {
                background-color: #66BB6A;
                border: 2px solid #66BB6A;
            }
            QPushButton:pressed {
                background-color: #2E7D32;
                border: 2px solid #2E7D32;
            }
        """)
        self.start_btn.clicked.connect(self.start_system)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP System")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                padding: 10px;
                border-radius: 4px;
                border: 2px solid #f44336;
            }
            QPushButton:hover {
                background-color: #ff6659;
                border: 2px solid #ff6659;
            }
            QPushButton:pressed {
                background-color: #c62828;
                border: 2px solid #c62828;
            }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_system)
        layout.addWidget(self.stop_btn)

        return widget

    def toggle_detection_calibration_mode(self, index):
        """Toggle between auto and manual detection calibration mode"""
        if index == 0:  # Auto Calibration
            self.auto_cal_container.show()
            self.manual_cal_container.hide()
        else:  # Manual Calibration
            self.auto_cal_container.hide()
            self.manual_cal_container.show()

    def toggle_robot_calibration_mode(self, state):
        """Toggle between auto and manual robot calibration mode"""
        auto_mode = (state == Qt.CheckState.Checked.value)

        # Enable/disable BR and TL inputs
        for corner in ['BR', 'TL']:
            for key in ['ws_x', 'ws_y', 'robot_x', 'robot_y']:
                self.robot_cal_inputs[corner][key].setEnabled(not auto_mode)
    
    def auto_load_homography_files(self):
        """Auto-load existing homography files on startup"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        camera_file = os.path.join(script_dir, "homography_auto.pkl")
        robot_file = os.path.join(script_dir, "homography_det_to_robot.pkl")
        
        camera_loaded = False
        robot_loaded = False
        
        # Try to load camera → workspace homography
        if os.path.exists(camera_file):
            try:
                with open(camera_file, "rb") as f:
                    self.H_camera_to_workspace = pickle.load(f)
                camera_loaded = True
                print(f"[Startup] Loaded camera→workspace homography from {camera_file}")
            except Exception as e:
                print(f"[Startup] Failed to load camera homography: {e}")
        
        # Try to load workspace → robot homography
        if os.path.exists(robot_file):
            try:
                with open(robot_file, "rb") as f:
                    self.H_workspace_to_robot = pickle.load(f)
                robot_loaded = True
                print(f"[Startup] Loaded workspace→robot homography from {robot_file}")
            except Exception as e:
                print(f"[Startup] Failed to load robot homography: {e}")
        
        # Update status labels if they exist
        if hasattr(self, 'cal_status_camera'):
            if camera_loaded:
                self.cal_status_camera.setText("✅ Loaded from file")
                self.cal_status_camera.setStyleSheet("font-weight: bold; color: #4CAF50;")
            else:
                self.cal_status_camera.setText("❌ Not Calibrated")
                self.cal_status_camera.setStyleSheet("font-weight: bold; color: #f44336;")
        
        if hasattr(self, 'cal_status_robot'):
            if robot_loaded:
                self.cal_status_robot.setText("✅ Loaded from file")
                self.cal_status_robot.setStyleSheet("font-weight: bold; color: #4CAF50;")
            else:
                self.cal_status_robot.setText("❌ Not Calibrated")
                self.cal_status_robot.setStyleSheet("font-weight: bold; color: #f44336;")
        
        # Show warning if files are missing
        if not camera_loaded or not robot_loaded:
            missing = []
            if not camera_loaded:
                missing.append("Camera → Workspace calibration")
            if not robot_loaded:
                missing.append("Workspace → Robot transformation")
            
            warning_msg = "⚠️ Warning: Calibration files not found!\n\n"
            warning_msg += "Missing:\n"
            for item in missing:
                warning_msg += f"  • {item}\n"
            warning_msg += "\nPlease calibrate the system before starting."
            
            print(f"[Startup] Warning: {', '.join(missing)} not found")
            self.statusBar().showMessage("⚠️ Calibration required - See Calibration Setup tab")
        else:
            self.statusBar().showMessage("✅ Calibration loaded successfully")
    
    def reload_homography_files(self):
        """Manually reload homography files from disk"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        camera_file = os.path.join(script_dir, "homography_auto.pkl")
        robot_file = os.path.join(script_dir, "homography_det_to_robot.pkl")
        
        camera_loaded = False
        robot_loaded = False
        messages = []
        
        # Try to load camera → workspace homography
        if os.path.exists(camera_file):
            try:
                with open(camera_file, "rb") as f:
                    self.H_camera_to_workspace = pickle.load(f)
                camera_loaded = True
                messages.append("✅ Camera → Workspace: Loaded")
                print(f"[Reload] Loaded camera→workspace homography")
            except Exception as e:
                messages.append(f"❌ Camera → Workspace: Error - {str(e)}")
                print(f"[Reload] Failed to load camera homography: {e}")
        else:
            messages.append("❌ Camera → Workspace: File not found (homography_auto.pkl)")
        
        # Try to load workspace → robot homography
        if os.path.exists(robot_file):
            try:
                with open(robot_file, "rb") as f:
                    self.H_workspace_to_robot = pickle.load(f)
                robot_loaded = True
                messages.append("✅ Workspace → Robot: Loaded")
                print(f"[Reload] Loaded workspace→robot homography")
            except Exception as e:
                messages.append(f"❌ Workspace → Robot: Error - {str(e)}")
                print(f"[Reload] Failed to load robot homography: {e}")
        else:
            messages.append("❌ Workspace → Robot: File not found (homography_det_to_robot.pkl)")
        
        # Update status labels
        if camera_loaded:
            self.cal_status_camera.setText("✅ Loaded from file")
            self.cal_status_camera.setStyleSheet("font-weight: bold; color: #4CAF50;")
        else:
            self.cal_status_camera.setText("❌ Not Calibrated")
            self.cal_status_camera.setStyleSheet("font-weight: bold; color: #f44336;")
        
        if robot_loaded:
            self.cal_status_robot.setText("✅ Loaded from file")
            self.cal_status_robot.setStyleSheet("font-weight: bold; color: #4CAF50;")
        else:
            self.cal_status_robot.setText("❌ Not Calibrated")
            self.cal_status_robot.setStyleSheet("font-weight: bold; color: #f44336;")
        
        # Show result message
        result_msg = "Homography Files Reload Result:\n\n" + "\n".join(messages)
        
        if camera_loaded and robot_loaded:
            QMessageBox.information(self, "Success", result_msg)
            self.statusBar().showMessage("✅ All calibration files reloaded successfully")
        elif camera_loaded or robot_loaded:
            QMessageBox.warning(self, "Partial Success", result_msg)
            self.statusBar().showMessage("⚠️ Some calibration files missing")
        else:
            QMessageBox.critical(self, "Failed", result_msg + "\n\nPlease calibrate the system.")
            self.statusBar().showMessage("❌ No calibration files found")
        
        # Update info display
        self.update_info_display()

    def auto_calibrate_detection(self):
        """Auto calibrate detection camera using circle detection"""
        try:
            self.statusBar().showMessage("Starting auto calibration...")

            # Initialize camera if not already done
            if self.detection_camera is None:
                if not self.initialize_cameras():
                    QMessageBox.warning(self, "Error", "Failed to initialize camera")
                    return

            # Get frame
            frame = self.get_detection_frame()
            if frame is None:
                QMessageBox.warning(self, "Error", "Failed to capture frame")
                return

            # Perform auto calibration (circle detection)
            H = self.perform_circle_calibration(frame)

            if H is not None:
                self.H_camera_to_workspace = H

                # Save homography
                script_dir = os.path.dirname(os.path.abspath(__file__))
                homography_file = os.path.join(script_dir, "homography_auto.pkl")
                with open(homography_file, "wb") as f:
                    pickle.dump(H, f)

                QMessageBox.information(self, "Success", "Auto calibration completed!")
                self.cal_status_camera.setText("✅ Calibrated (Auto)")
                self.cal_status_camera.setStyleSheet("font-weight: bold; color: #4CAF50;")
                self.statusBar().showMessage("Auto calibration successful")
                self.update_info_display()
            else:
                QMessageBox.warning(self, "Failed", "Could not detect calibration circles")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Calibration failed: {str(e)}")
            print(f"[ERROR] Auto calibration: {e}")

    def perform_circle_calibration(self, frame):
        """Perform circle-based calibration (from your existing code)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=30
        )

        if circles is None:
            return None

        filtered = []
        for (x, y, r) in np.round(circles[0]).astype("int"):
            if 10 <= r <= 50:
                filtered.append((x, y, r))

        if len(filtered) < 4:
            return None

        # Get 4 corners
        image_points = np.array([[x, y] for (x, y, _) in filtered[:4]], dtype=np.float32)
        image_points = sorted(image_points, key=lambda pt: (pt[1], pt[0]))
        top = sorted(image_points[:2], key=lambda pt: pt[0])
        bottom = sorted(image_points[2:], key=lambda pt: pt[0])
        sorted_img_pts = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

        # Real world points (workspace)
        workspace_width = self.config.get("workspace", {}).get("width", 300)
        workspace_height = self.config.get("workspace", {}).get("height", 300)

        real_pts = np.array([
            [0, workspace_height],
            [workspace_width, workspace_height],
            [workspace_width, 0],
            [0, 0]
        ], dtype=np.float32)

        H, _ = cv2.findHomography(sorted_img_pts, real_pts)
        return H

    def apply_manual_detection_calibration(self):
        """Apply manual detection camera calibration"""
        try:
            # Collect input points
            image_points = []
            real_points = []

            corners = ['BL', 'BR', 'TR', 'TL']
            for corner in corners:
                inputs = self.detection_cal_inputs[corner]

                try:
                    px = float(inputs['pixel_x'].text())
                    py = float(inputs['pixel_y'].text())
                    mx = float(inputs['mm_x'].text())
                    my = float(inputs['mm_y'].text())

                    image_points.append([px, py])
                    real_points.append([mx, my])
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", f"Please fill all fields for {corner}")
                    return

            # Calculate homography
            image_points = np.array(image_points, dtype=np.float32)
            real_points = np.array(real_points, dtype=np.float32)

            H, _ = cv2.findHomography(image_points, real_points)

            if H is not None:
                self.H_camera_to_workspace = H

                # Save homography
                script_dir = os.path.dirname(os.path.abspath(__file__))
                homography_file = os.path.join(script_dir, "homography_auto.pkl")
                with open(homography_file, "wb") as f:
                    pickle.dump(H, f)

                QMessageBox.information(self, "Success", "Manual calibration applied!")
                self.cal_status_camera.setText("✅ Calibrated (Manual)")
                self.cal_status_camera.setStyleSheet("font-weight: bold; color: #4CAF50;")
                self.update_info_display()
            else:
                QMessageBox.warning(self, "Failed", "Could not calculate homography")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Manual calibration failed: {str(e)}")

    def calculate_robot_transformation(self):
        """Calculate workspace to robot coordinate transformation"""
        try:
            auto_mode = self.robot_auto_mode.isChecked()

            # Collect points
            workspace_points = []
            robot_points = []

            if auto_mode:
                # Get BL and TR only
                corners_to_use = ['BL', 'TR']

                for corner in corners_to_use:
                    inputs = self.robot_cal_inputs[corner]
                    try:
                        ws_x = float(inputs['ws_x'].text())
                        ws_y = float(inputs['ws_y'].text())
                        robot_x = float(inputs['robot_x'].text())
                        robot_y = float(inputs['robot_y'].text())

                        workspace_points.append([ws_x, ws_y])
                        robot_points.append([robot_x, robot_y])
                    except ValueError:
                        QMessageBox.warning(self, "Invalid Input", f"Please fill all fields for {corner}")
                        return

                # Auto-calculate BR and TL using simple logic
                bl_ws = np.array(workspace_points[0])
                tr_ws = np.array(workspace_points[1])
                bl_robot = np.array(robot_points[0])
                tr_robot = np.array(robot_points[1])

                # Simple calculation:
                # BR takes X from BL, Y from TR
                # TL takes X from TR, Y from BL
                br_robot = np.array([bl_robot[0], tr_robot[1]])  # X from BL, Y from TR
                tl_robot = np.array([tr_robot[0], bl_robot[1]])  # X from TR, Y from BL

                # Add to lists
                workspace_points.append([300, 0])  # BR workspace
                robot_points.append(br_robot.tolist())

                workspace_points.append([0, 300])  # TL workspace
                robot_points.append(tl_robot.tolist())

                # Update display fields
                self.robot_cal_inputs['BR']['robot_x'].setText(f"{br_robot[0]:.1f}")
                self.robot_cal_inputs['BR']['robot_y'].setText(f"{br_robot[1]:.1f}")
                self.robot_cal_inputs['TL']['robot_x'].setText(f"{tl_robot[0]:.1f}")
                self.robot_cal_inputs['TL']['robot_y'].setText(f"{tl_robot[1]:.1f}")

            else:
                # Manual mode - use all 4 points
                corners = ['BL', 'BR', 'TR', 'TL']
                for corner in corners:
                    inputs = self.robot_cal_inputs[corner]
                    try:
                        ws_x = float(inputs['ws_x'].text())
                        ws_y = float(inputs['ws_y'].text())
                        robot_x = float(inputs['robot_x'].text())
                        robot_y = float(inputs['robot_y'].text())

                        workspace_points.append([ws_x, ws_y])
                        robot_points.append([robot_x, robot_y])
                    except ValueError:
                        QMessageBox.warning(self, "Invalid Input", f"Please fill all fields for {corner}")
                        return

            # Calculate homography
            workspace_points = np.array(workspace_points, dtype=np.float32)
            robot_points = np.array(robot_points, dtype=np.float32)

            H, _ = cv2.findHomography(workspace_points, robot_points)

            if H is not None:
                self.H_workspace_to_robot = H

                # Save homography
                script_dir = os.path.dirname(os.path.abspath(__file__))
                homography_file = os.path.join(script_dir, "homography_det_to_robot.pkl")
                with open(homography_file, "wb") as f:
                    pickle.dump(H, f)

                QMessageBox.information(self, "Success", "Robot transformation calculated!")
                self.cal_status_robot.setText("✅ Calibrated")
                self.cal_status_robot.setStyleSheet("font-weight: bold; color: #4CAF50;")
                self.update_info_display()
            else:
                QMessageBox.warning(self, "Failed", "Could not calculate transformation")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Transformation calculation failed: {str(e)}")
            print(f"[ERROR] Robot transformation: {e}")

    def initialize_cameras(self):
        """Initialize both cameras"""
        try:
            self.camera_system = PySpin.System.GetInstance()
            cam_list = self.camera_system.GetCameras()

            if cam_list.GetSize() < 1:
                print("[ERROR] No cameras found")
                return False

            # Initialize detection camera
            self.detection_camera = cam_list.GetByIndex(0)
            self.detection_camera.Init()
            self.detection_camera.BeginAcquisition()

            # Initialize inspection camera if available
            if cam_list.GetSize() >= 2:
                self.inspection_camera = cam_list.GetByIndex(1)
                self.inspection_camera.Init()
                self.inspection_camera.BeginAcquisition()

            return True

        except Exception as e:
            print(f"[ERROR] Camera initialization: {e}")
            return False

    def get_detection_frame(self):
        """Get frame from detection camera"""
        if self.detection_camera is None:
            return None

        try:
            image = self.detection_camera.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                return None

            img_array = image.GetNDArray()
            image.Release()

            if len(img_array.shape) == 2:
                return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            return img_array
        except:
            return None

    def get_inspection_frame(self):
        """Get frame from inspection camera"""
        if self.inspection_camera is None:
            return None

        try:
            image = self.inspection_camera.GetNextImage()
            if image.IsIncomplete():
                image.Release()
                return None

            img_array = image.GetNDArray()
            image.Release()

            if len(img_array.shape) == 2:
                return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            return img_array
        except:
            return None

    def transform_points(self, points, H):
        """Transform points using homography matrix"""
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H)
        return warped.reshape(-1, 2)

    def is_inside_workspace(self, pts, width=None, height=None):
        """Check if points are inside workspace boundaries"""
        if width is None:
            width = self.config.get("workspace", {}).get("width", 300)
        if height is None:
            height = self.config.get("workspace", {}).get("height", 300)

        x, y = pts[:, 0], pts[:, 1]
        return np.all((x >= 0) & (x <= width) & (y >= 0) & (y <= height))

    def get_angle(self, obb_pts):
        """Calculate angle from OBB points"""
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

    def update_camera_feeds(self):
        """Update camera display (called by timer)"""
        # Get detection frame
        detection_frame = self.get_detection_frame()
        if detection_frame is not None:
            # Run YOLO detection
            results = self.model(detection_frame, conf=self.config.get("detection_confidence", 0.7))

            # Draw detections and update display
            annotated = self.draw_detections(detection_frame, results)

            # Draw inspection visualization box if visible
            if self.inspection_box_visible:
                annotated = self.draw_inspection_box(annotated)

            self.display_frame(annotated, self.detection_label)

        # Get inspection frame (still initialize camera but don't display)
        inspection_frame = self.get_inspection_frame()

    def draw_detections(self, frame, results):
        """Draw YOLO detections on frame with color based on workspace position"""
        annotated = frame.copy()

        # Clear previous detected objects
        self.detected_objects = []

        # Draw workspace boundary if calibrated
        if self.H_camera_to_workspace is not None:
            H_inv = np.linalg.inv(self.H_camera_to_workspace)
            self.draw_workspace_boundary(annotated, H_inv)

        # Draw detection results
        obb_preds = results[0].obb

        for i, obb in enumerate(obb_preds, 1):
            if hasattr(obb, "xyxyxyxy"):
                corners = obb.xyxyxyxy.cpu().numpy().reshape(-1, 2)
            elif hasattr(obb, "xyxy"):
                corners = obb.xyxy.cpu().numpy().reshape(-1, 2)
            else:
                continue

            # Determine color based on workspace position
            if self.H_camera_to_workspace is not None:
                # Transform corners to workspace coordinates
                transformed = self.transform_points(corners, self.H_camera_to_workspace)
                is_inside = self.is_inside_workspace(transformed)

                if is_inside:
                    color = (0, 255, 0)  # Green - inside workspace

                    # Calculate object properties
                    center = np.mean(transformed, axis=0)
                    angle = self.get_angle(transformed)

                    side1 = np.linalg.norm(transformed[1] - transformed[0])
                    side2 = np.linalg.norm(transformed[2] - transformed[1])
                    width_mm = round(max(side1, side2), 1)
                    height_mm = round(min(side1, side2), 1)

                    x_mm, y_mm = round(center[0], 1), round(center[1], 1)
                    angle_deg = round(angle, 2)

                    # Store object data for click detection
                    self.detected_objects.append({
                        'id': i,
                        'corners': corners.astype(int),
                        'x_mm': x_mm,
                        'y_mm': y_mm,
                        'angle': angle_deg,
                        'width': width_mm,
                        'height': height_mm
                    })

                    # Track changes for shared memory updates (like yolo-mouse-v2.py)
                    current_time = time.time()
                    prev = self.last_outputs.get(i, (None, None, None, None, None))
                    if (
                        prev[0] is None or
                        abs(prev[0] - x_mm) >= 2.0 or
                        abs(prev[1] - y_mm) >= 2.0 or
                        abs(prev[2] - angle_deg) >= 2.0 or
                        abs(prev[3] - width_mm) >= 2.0 or
                        abs(prev[4] - height_mm) >= 2.0
                    ):
                        self.last_outputs[i] = (x_mm, y_mm, angle_deg, width_mm, height_mm)
                        self.last_change_time[i] = current_time
                        # Write to shared memory immediately when object changes
                        if self.detection_data_mgr:
                            self.detection_data_mgr.update_object(i, x_mm, y_mm, angle_deg, width_mm, height_mm)

                    # Draw center point
                    center_img = np.mean(corners, axis=0).astype(int)
                    cv2.circle(annotated, tuple(center_img), 5, (0, 0, 255), -1)
                    cv2.circle(annotated, tuple(center_img), 5, (255, 255, 255), 1)
                    cv2.drawMarker(annotated, tuple(center_img), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
                else:
                    color = (0, 0, 255)  # Red - outside workspace
            else:
                color = (128, 128, 128)  # Gray - no calibration

            # Draw border
            corners_int = corners.astype(int)
            cv2.polylines(annotated, [corners_int], isClosed=True, color=color, thickness=2)

            # Draw object label (like yolo-mouse-v2.py)
            label_pos = tuple(corners_int[0] - [0, 10])
            cv2.putText(annotated, f"Object {i}", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Status tracking and shared memory updates (like yolo-mouse-v2.py lines 987-1020)
        if self.detection_data_mgr:
            current_time = time.time()
            detected_ids = set([obj['id'] for obj in self.detected_objects])

            # Calculate status
            if len(detected_ids) <= 1:
                status_text = "Not Ready"
            elif len(detected_ids) >= 2 and all(current_time - self.last_change_time.get(obj_id, 0) >= 1.0 for obj_id in detected_ids):
                status_text = "Ready"
            else:
                status_text = "Detect"

            # Update shared memory status
            if status_text == "Ready":
                if status_text != self.last_saved_status:
                    self.detection_data_mgr.update_status(status_text)
                    self.last_saved_status = status_text

                # Update all detected objects in shared memory
                for obj in self.detected_objects:
                    obj_id = obj['id']
                    prev_obj = self.last_saved_objects.get(obj_id)
                    curr_obj = (obj['x_mm'], obj['y_mm'], obj['angle'], obj['width'], obj['height'])
                    if prev_obj != curr_obj:
                        self.detection_data_mgr.update_object(obj_id, obj['x_mm'], obj['y_mm'], obj['angle'], obj['width'], obj['height'])
                        self.last_saved_objects[obj_id] = curr_obj

            elif status_text == "Not Ready":
                if status_text != self.last_saved_status:
                    self.detection_data_mgr.update_status(status_text)
                    self.last_saved_status = status_text

                self.detection_data_mgr.clear_all_objects()
                self.last_saved_objects.clear()

            else:  # Detect
                if status_text != self.last_saved_status:
                    self.detection_data_mgr.update_status(status_text)
                    self.last_saved_status = status_text

            # Draw status on frame
            if status_text == "Not Ready":
                status_color = (0, 100, 200)
            elif status_text == "Detect":
                status_color = (200, 150, 0)
            elif status_text == "Ready":
                status_color = (0, 150, 0)

            cv2.putText(
                annotated,
                f"Status: {status_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                2,
                cv2.LINE_AA
            )

        # Draw info panel if showing coordinates (object or empty space)
        if self.show_info_panel:
            if self.selected_object:
                # Object clicked - highlight and show object info
                cv2.polylines(annotated, [self.selected_object['corners']],
                             isClosed=True, color=(255, 255, 0), thickness=3)

                # Draw info panel near click position
                info_lines = [
                    f"Object ID: {self.selected_object['id']}",
                    f"Center: ({self.selected_object['x_mm']:.1f}, {self.selected_object['y_mm']:.1f}) mm",
                ]

                # Add clicked position if available
                if self.clicked_workspace_pos is not None:
                    click_x_mm, click_y_mm = self.clicked_workspace_pos
                    info_lines.append(f"Clicked: ({click_x_mm:.1f}, {click_y_mm:.1f}) mm")
                else:
                    info_lines.append(f"Clicked: N/A (no calibration)")

                info_lines.extend([
                    f"Angle: {self.selected_object['angle']:.1f} degrees",
                    f"Width: {self.selected_object['width']:.1f} mm",
                    f"Height: {self.selected_object['height']:.1f} mm"
                ])

                annotated = self.draw_info_panel_on_frame(
                    annotated,
                    self.mouse_click_x + 10,
                    self.mouse_click_y + 10,
                    info_lines,
                    f"Object {self.selected_object['id']}"
                )
            elif self.clicked_workspace_pos is not None:
                # Empty space clicked - show workspace coordinates
                workspace_width = self.config.get("workspace", {}).get("width", 300)
                workspace_height = self.config.get("workspace", {}).get("height", 300)

                click_x_mm, click_y_mm = self.clicked_workspace_pos
                in_workspace = (0 <= click_x_mm <= workspace_width and 0 <= click_y_mm <= workspace_height)

                info_lines = [
                    f"Pixel: ({self.mouse_click_x}, {self.mouse_click_y})",
                    f"Real: ({click_x_mm:.1f}, {click_y_mm:.1f}) mm",
                    f"In workspace: {'Yes' if in_workspace else 'No'}"
                ]

                annotated = self.draw_info_panel_on_frame(
                    annotated,
                    self.mouse_click_x + 10,
                    self.mouse_click_y + 10,
                    info_lines,
                    "Coordinates"
                )

                # Draw crosshair at click position
                cv2.drawMarker(annotated, (self.mouse_click_x, self.mouse_click_y),
                              (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        return annotated

    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def draw_info_panel_on_frame(self, frame, x, y, info_lines, title="Info"):
        """Draw an information panel at specified position (like yolo-mouse-v2.py)"""
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

    def draw_inspection_crosshair(self, frame, offset_error=0):
        """Draw inspection crosshair with error tolerance circle (like yolo-mouse-v2.py)"""
        h, w = frame.shape[:2]
        x, y = w // 2, h // 2  # Center of frame

        # Red crosshair
        line_length = 40
        thickness = 3
        color = (0, 0, 255)

        cv2.line(frame, (x - line_length, y), (x + line_length, y), color, thickness)
        cv2.line(frame, (x, y - line_length), (x, y + line_length), color, thickness)

        # Green center circle
        cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)

        # Error tolerance circle (yellow, dashed appearance)
        error_radius_px = 15
        for angle in range(0, 360, 30):
            angle_rad = np.radians(angle)
            x1 = int(x + error_radius_px * np.cos(angle_rad))
            y1 = int(y + error_radius_px * np.sin(angle_rad))
            x2 = int(x + error_radius_px * np.cos(angle_rad + np.radians(15)))
            y2 = int(y + error_radius_px * np.sin(angle_rad + np.radians(15)))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        return frame

    def on_detection_click(self, x, y):
        """Handle click on detection camera - find which object was clicked or close inspection box"""
        # Check if clicked outside inspection box (to close it)
        if self.inspection_box_visible:
            # Create rotated rectangle points to check if click is inside
            box_pts = self.get_rotated_box_points(
                self.inspection_box_x,
                self.inspection_box_y,
                self.inspection_box_width,
                self.inspection_box_height,
                self.inspection_box_angle
            )

            # If click is outside box, close it and continue with normal click handling
            if not self.point_in_polygon((x, y), box_pts.astype(int)):
                self.inspection_box_visible = False
                # Don't return - allow normal click handling to proceed

        # Normal single-click handling
        self.mouse_click_x = x
        self.mouse_click_y = y

        # Check if clicked on any detected object
        clicked_on_object = False
        for obj_data in self.detected_objects:
            # Test if point is in polygon
            result = self.point_in_polygon((x, y), obj_data['corners'])

            if result:
                self.selected_object = obj_data
                self.show_info_panel = True
                clicked_on_object = True

                # Convert click position to workspace coordinates
                if self.H_camera_to_workspace is not None:
                    click_pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
                    workspace_coord = cv2.perspectiveTransform(click_pt, self.H_camera_to_workspace).reshape(-1, 2)
                    click_x_mm = workspace_coord[0][0]
                    click_y_mm = workspace_coord[0][1]
                    self.clicked_workspace_pos = (click_x_mm, click_y_mm)
                else:
                    self.clicked_workspace_pos = None

                # Update detection info text
                info_text = f"OBJECT DETECTION | ID: {obj_data['id']}\n"
                info_text += f"Position (X, Y): {obj_data['x_mm']:.1f} mm, {obj_data['y_mm']:.1f} mm\n"
                info_text += f"Orientation: {obj_data['angle']:.1f}° | "
                info_text += f"Dimensions: {obj_data['width']:.1f} mm × {obj_data['height']:.1f} mm"
                self.detection_info.setText(info_text)
                break

        # If clicked on empty space
        if not clicked_on_object:
            self.selected_object = None
            self.show_info_panel = True  # Show info panel for empty space too

            # Convert click position to workspace coordinates and display
            if self.H_camera_to_workspace is not None:
                click_pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
                workspace_coord = cv2.perspectiveTransform(click_pt, self.H_camera_to_workspace).reshape(-1, 2)
                click_x_mm = workspace_coord[0][0]
                click_y_mm = workspace_coord[0][1]

                # Store clicked workspace position
                self.clicked_workspace_pos = (click_x_mm, click_y_mm)

                # Check if inside workspace
                workspace_width = self.config.get("workspace", {}).get("width", 300)
                workspace_height = self.config.get("workspace", {}).get("height", 300)

                if 0 <= click_x_mm <= workspace_width and 0 <= click_y_mm <= workspace_height:
                    # Update detection info text
                    info_text = f"WORKSPACE COORDINATES\n"
                    info_text += f"Position (X, Y): {click_x_mm:.1f} mm, {click_y_mm:.1f} mm\n"
                    info_text += f"Pixel Coordinates: ({x}, {y})"
                    self.detection_info.setText(info_text)
                else:
                    self.detection_info.clear()
            else:
                self.clicked_workspace_pos = None
                self.detection_info.clear()

    def on_detection_double_click(self, x, y):
        """Handle double-click on detection camera - show inspection visualization box"""
        # Check if double-clicked on any detected object
        for obj_data in self.detected_objects:
            if self.point_in_polygon((x, y), obj_data['corners']):
                # Use fixed inspection box size (36mm x 36mm)
                pixel_per_mm = 2.0  # Approximate ratio, adjust if needed
                box_size_mm = 36  # Fixed 36mm inspection area
                box_size = int(box_size_mm * pixel_per_mm)  # Convert to pixels

                # Store inspection box data
                self.inspection_box_visible = True
                self.inspection_box_x = x  # Center at click position
                self.inspection_box_y = y
                self.inspection_box_width = box_size
                self.inspection_box_height = box_size
                self.inspection_box_angle = obj_data['angle']  # Start with object's angle
                self.inspection_box_object_data = obj_data
                break

    def get_rotated_box_points(self, cx, cy, width, height, angle):
        """Calculate the 4 corner points of a rotated rectangle"""
        # Convert angle to radians (negate for image coordinates where Y is down)
        angle_rad = np.radians(-angle)

        # Half dimensions
        w2 = width / 2
        h2 = height / 2

        # Calculate corners relative to center
        corners = np.array([
            [-w2, -h2],
            [w2, -h2],
            [w2, h2],
            [-w2, h2]
        ], dtype=np.float32)

        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # Rotate corners
        rotated_corners = corners @ rotation_matrix.T

        # Translate to center position
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy

        return rotated_corners

    def draw_inspection_box(self, frame):
        """Draw transparent inspection visualization box with rotation"""
        # Get rotated box corners
        box_pts = self.get_rotated_box_points(
            self.inspection_box_x,
            self.inspection_box_y,
            self.inspection_box_width,
            self.inspection_box_height,
            self.inspection_box_angle
        ).astype(int)

        # Create overlay for transparency
        overlay = frame.copy()

        # Fill the box with black color (will be made transparent)
        cv2.fillPoly(overlay, [box_pts], (0, 0, 0))  # Black fill

        # Blend with original frame (50% transparent = 50% opacity)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Draw cyan border (thickness 2)
        cv2.polylines(frame, [box_pts], isClosed=True, color=(255, 255, 0), thickness=2)

        # Draw center crosshair
        cv2.drawMarker(frame, (self.inspection_box_x, self.inspection_box_y),
                      (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Draw angle text near the box
        if self.inspection_box_object_data:
            text = f"Angle: {self.inspection_box_angle:.1f}"
            text_pos = (self.inspection_box_x + 10, self.inspection_box_y - 10)
            cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 0), 2, cv2.LINE_AA)

        return frame

    def draw_workspace_boundary(self, frame, H_inv):
        """Draw workspace boundary box"""
        workspace_width = self.config.get("workspace", {}).get("width", 300)
        workspace_height = self.config.get("workspace", {}).get("height", 300)

        box_real = np.array([
            [0, 0],
            [workspace_width, 0],
            [workspace_width, workspace_height],
            [0, workspace_height]
        ], dtype=np.float32).reshape(-1, 1, 2)

        box_img = cv2.perspectiveTransform(box_real, H_inv).reshape(-1, 2).astype(int)
        cv2.polylines(frame, [box_img], isClosed=True, color=(255, 255, 255), thickness=3)

    def display_frame(self, frame, label):
        """Display OpenCV frame in QLabel"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Store original image size for click coordinate scaling
        h, w, ch = rgb_frame.shape
        if isinstance(label, ClickableLabel):
            label.original_image_size = (w, h)

        # Resize to fit label
        label_w = label.width()
        label_h = label.height()

        # Calculate scaling
        scale = min(label_w / w, label_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Store displayed image size for click coordinate scaling
        if isinstance(label, ClickableLabel):
            label.displayed_image_size = (new_w, new_h)

        resized = cv2.resize(rgb_frame, (new_w, new_h))

        # Convert to QImage
        bytes_per_line = ch * new_w
        qt_image = QImage(resized.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)

        # Display
        label.setPixmap(QPixmap.fromImage(qt_image))

    def update_info_display(self):
        """Update system information display in organized sections"""
        from datetime import datetime
        
        # Update Active Profile
        if hasattr(self, 'current_profile_label'):
            profile_name = self.current_profile_label.text()
            self.info_profile_name.setText(profile_name)
            if profile_name == "(None)":
                self.info_profile_name.setStyleSheet("font-weight: bold; color: #888;")
            else:
                self.info_profile_name.setStyleSheet("font-weight: bold; color: #0078d4;")
        
        # Update Vision System
        model_path = self.config.get("yolo_model_path", "")
        if model_path:
            # Show only filename
            model_name = os.path.basename(model_path)
            self.info_model.setText(model_name)
        else:
            self.info_model.setText("Not configured")
        
        confidence = self.config.get("detection_confidence", 0.7)
        self.info_confidence.setText(f"{confidence}")
        
        cam_config = self.config.get("camera_config", {})
        det_cam_idx = cam_config.get("detection_camera_index", 0)
        insp_cam_idx = cam_config.get("inspection_camera_index", 1)
        self.info_det_cam.setText(f"Index {det_cam_idx}")
        self.info_insp_cam.setText(f"Index {insp_cam_idx}")
        
        # Update Robot System
        robot_ip = self.config.get("robot_ip", "192.168.1.151")
        self.info_robot_ip.setText(robot_ip)
        
        workspace = self.config.get("workspace", {})
        ws_width = workspace.get("width", 300)
        ws_height = workspace.get("height", 300)
        self.info_workspace.setText(f"{ws_width} × {ws_height} mm")
        
        cam_offset = self.config.get("camera_offset", {})
        offset_x = cam_offset.get("offset_x", 0.0)
        offset_y = cam_offset.get("offset_y", 0.0)
        self.info_cam_offset.setText(f"({offset_x}, {offset_y}) mm")
        
        click_control = self.config.get("click_control", {})
        insp_height = click_control.get("inspection_height", 103.4)
        self.info_insp_height.setText(f"{insp_height} mm")
        
        # Update Calibration Status
        if self.H_camera_to_workspace is not None:
            self.info_cal_camera.setText("✅ Configured")
            self.info_cal_camera.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.info_cal_camera.setText("❌ Not Configured")
            self.info_cal_camera.setStyleSheet("color: #f44336; font-weight: bold;")
        
        if self.H_workspace_to_robot is not None:
            self.info_cal_robot.setText("✅ Configured")
            self.info_cal_robot.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.info_cal_robot.setText("❌ Not Configured")
            self.info_cal_robot.setStyleSheet("color: #f44336; font-weight: bold;")
        
        # Update System Status
        if self.system_running:
            self.info_state.setText("RUNNING")
            self.info_state.setStyleSheet("font-weight: bold; color: #4CAF50;")
        else:
            self.info_state.setText("STOPPED")
            self.info_state.setStyleSheet("font-weight: bold; color: #f44336;")
        
        if self.detection_camera:
            self.info_det_status.setText("✅ Connected")
            self.info_det_status.setStyleSheet("color: #4CAF50;")
        else:
            self.info_det_status.setText("❌ Disconnected")
            self.info_det_status.setStyleSheet("color: #f44336;")
        
        if self.inspection_camera:
            self.info_insp_status.setText("✅ Connected")
            self.info_insp_status.setStyleSheet("color: #4CAF50;")
        else:
            self.info_insp_status.setText("❌ Disconnected")
            self.info_insp_status.setStyleSheet("color: #f44336;")
        
        # Update timestamp
        self.info_timestamp.setText(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def start_system(self):
        """Start the vision system"""
        try:
            # Check calibrations
            if self.H_camera_to_workspace is None:
                QMessageBox.warning(self, "Not Ready", "Please calibrate detection camera first!")
                return

            if self.H_workspace_to_robot is None:
                QMessageBox.warning(self, "Not Ready", "Please configure robot transformation first!")
                return

            # Initialize cameras
            if self.detection_camera is None:
                if not self.initialize_cameras():
                    QMessageBox.critical(self, "Error", "Failed to initialize cameras")
                    return

            # Initialize shared memory
            self.initialize_shared_memory()

            # Start camera timer
            self.camera_timer.start(33)  # ~30 FPS

            # Update UI
            self.system_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.statusBar().showMessage("System RUNNING")
            self.update_info_display()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start system: {str(e)}")
            print(f"[ERROR] Start system: {e}")

    def stop_system(self):
        """Stop the vision system"""
        try:
            # Stop camera timer
            self.camera_timer.stop()

            # Cleanup shared memory
            self.cleanup_shared_memory()

            # Update UI
            self.system_running = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.statusBar().showMessage("System STOPPED")
            self.update_info_display()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop system: {str(e)}")

    def initialize_shared_memory(self):
        """Initialize shared memory for communication with xarm-motion (INFO ONLY MODE)"""
        try:
            # Detection data shared memory (writes detection results for monitoring)
            self.detection_data_mgr = DetectionDataManager(name="DetectionData", size=4096)

            # Click and Inspect managers (initialized but commands are NOT sent to robot)
            self.click_data_mgr = ClickDataManager(name="ClickData", size=512)
            self.inspect_data_mgr = InspectDataManager(name="InspectData", size=512)

        except Exception as e:
            print(f"[ERROR] SharedMemory: {e}")

    def cleanup_shared_memory(self):
        """Cleanup shared memory"""
        try:
            if self.detection_data_mgr:
                self.detection_data_mgr.cleanup()
            if self.click_data_mgr:
                self.click_data_mgr.cleanup()
            if self.inspect_data_mgr:
                self.inspect_data_mgr.cleanup()
        except:
            pass

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop system if running
        if self.system_running:
            self.stop_system()

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

        if self.camera_system:
            try:
                cam_list = self.camera_system.GetCameras()
                cam_list.Clear()
                self.camera_system.ReleaseInstance()
            except:
                pass

        event.accept()

    def eventFilter(self, obj, event):
        """Event filter to intercept arrow keys globally when inspection box is visible"""
        from PyQt6.QtCore import QEvent

        if event.type() == QEvent.Type.KeyPress and self.inspection_box_visible:
            key = event.key()
            if key == Qt.Key.Key_Left:
                # Rotate counter-clockwise (decrease angle)
                self.inspection_box_angle -= 1.0
                if self.inspection_box_angle < 0:
                    self.inspection_box_angle += 360
                return True  # Event handled, block from all widgets

            elif key == Qt.Key.Key_Right:
                # Rotate clockwise (increase angle)
                self.inspection_box_angle += 1.0
                if self.inspection_box_angle >= 360:
                    self.inspection_box_angle -= 360
                return True  # Event handled, block from all widgets

            elif key == Qt.Key.Key_Escape:
                # Close inspection box
                self.inspection_box_visible = False
                return True  # Event handled

        # Let the event pass through normally
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """Handle keyboard events for inspection box rotation"""
        from PyQt6.QtCore import Qt

        if self.inspection_box_visible:
            if event.key() == Qt.Key.Key_Left:
                # Rotate counter-clockwise (decrease angle)
                self.inspection_box_angle -= 1.0
                if self.inspection_box_angle < 0:
                    self.inspection_box_angle += 360
                event.accept()  # Consume event to prevent tab navigation

            elif event.key() == Qt.Key.Key_Right:
                # Rotate clockwise (increase angle)
                self.inspection_box_angle += 1.0
                if self.inspection_box_angle >= 360:
                    self.inspection_box_angle -= 360
                event.accept()  # Consume event to prevent tab navigation

            elif event.key() == Qt.Key.Key_Escape:
                # Close inspection box
                self.inspection_box_visible = False
                event.accept()  # Consume event
            else:
                # Pass other events to parent
                super().keyPressEvent(event)
        else:
            # Pass event to parent if inspection box not visible
            super().keyPressEvent(event)

    def changeEvent(self, event):
        """Handle window state changes (maximize/restore)"""
        if event.type() == event.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMaximized:
                # When window is MAXIMIZED - use BIGGER fixed size (850x720)
                self.detection_label.setFixedSize(850, 720)
            else:
                # When window is RESTORED/Normal - resize window and set smaller label size
                self.detection_label.setFixedSize(640, 480)
                # Use QTimer to delay resize so layout calculates first
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(10, lambda: self._restore_normal_size())

        super().changeEvent(event)

    def _restore_normal_size(self):
        """Helper to restore normal window size and update layout"""
        self.resize(900, 850)
        # Force control panel to update its geometry
        if hasattr(self, 'control_panel'):
            self.control_panel.updateGeometry()
        self.centralWidget().layout().invalidate()
        self.centralWidget().layout().activate()
        self.centralWidget().adjustSize()
        self.adjustSize()

        # Center window on screen after restore
        screen = QApplication.primaryScreen().geometry()
        window_size = self.frameGeometry()
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2
        self.move(x, y)


def main():
    app = QApplication(sys.argv)
    window = RobotVisionGUI()

    # Center window on screen
    screen = app.primaryScreen().geometry()
    window_size = window.frameGeometry()
    x = (screen.width() - window_size.width()) // 2
    y = (screen.height() - window_size.height()) // 2
    window.move(x, y)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()