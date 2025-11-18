# ARM-APP Error Report & Fix Guide

## **Diagnostic Results Summary**

| Check | Status | Severity |
|-------|--------|----------|
| Python Dependencies | ❌ FAIL | 🔴 CRITICAL |
| Configuration Paths | ❌ FAIL | 🔴 CRITICAL |
| Calibration Files | ✅ PASS | ✅ OK |
| Shared Memory | ✅ PASS | ✅ OK |
| Code File Paths | ❌ FAIL | ⚠️ WARNING |

---

## **🔴 CRITICAL ERRORS (Must Fix)**

### **Error 1: Missing Python Dependencies**

**Problem:**
All required Python packages are not installed:
- opencv-python (cv2)
- numpy
- PyQt6
- ultralytics (YOLO)
- PySpin (FLIR camera SDK)
- xarm-python-sdk

**Fix:**

```bash
# Step 1: Install basic dependencies
pip install opencv-python numpy PyQt6 ultralytics

# Step 2: Install xArm SDK
pip install xarm-python-sdk

# Step 3: Install PySpin (FLIR Camera SDK)
# This requires manual download from:
# https://www.flir.com/products/spinnaker-sdk/
# Download the Python wrapper for your OS and install it
```

**Verification:**
```bash
python3 -c "import cv2, numpy, PyQt6, ultralytics; print('✅ Core packages installed')"
python3 -c "import PySpin; print('✅ PySpin installed')"
python3 -c "from xarm.wrapper import XArmAPI; print('✅ xArm SDK installed')"
```

---

### **Error 2: Windows Path Format in config.json**

**Problem:**
```json
"yolo_model_path": "D:\\2. yolo\\train30\\weights\\best.pt"
```
This Windows path won't work on Linux/Mac systems.

**Fix:**

**Option A: Use relative path (recommended)**
```json
"yolo_model_path": "./models/best.pt"
```

**Option B: Use absolute Linux path**
```json
"yolo_model_path": "/home/user/yolo/train30/weights/best.pt"
```

**Updated config.json:**
```json
{
    "robot_ip": "192.168.1.151",
    "tcp_speed": 300,
    "tcp_acc": 1000,
    "angle_speed": 20,
    "angle_acc": 500,
    "calibration_position": {
        "x": -55.3,
        "y": 207.3,
        "z": 92.7,
        "roll": 179.7,
        "pitch": -0.2,
        "yaw": -2.4
    },
    "click_control": {
        "safe_height": 150,
        "pick_height": -2,
        "inspection_height": 103.4,
        "workspace_min_x": 0,
        "workspace_max_x": 300,
        "workspace_min_y": 0,
        "workspace_max_y": 300,
        "coordinate_offset_x": 0,
        "coordinate_offset_y": -150
    },
    "yolo_model_path": "./models/best.pt",  <!-- FIXED -->
    "detection_confidence": 0.8,
    "camera_config": {
        "detection_camera_index": 0,
        "inspection_camera_index": 1
    },
    "camera_offset": {
        "offset_x": 7,
        "offset_y": 92.9,
        "offset_error": 0
    },
    "window_config": {
        "detection_width": 960,
        "detection_height": 720,
        "detection_pos_x": 950,
        "detection_pos_y": 50,
        "inspection_width": 960,
        "inspection_height": 720,
        "inspection_pos_x": 0,
        "inspection_pos_y": 50,
        "calibration_width": 960,
        "calibration_height": 720,
        "calibration_pos_x": 480,
        "calibration_pos_y": 180
    },
    "workspace": {
        "width": 300,
        "height": 300
    }
}
```

**Create models directory:**
```bash
mkdir -p models
# Place your YOLO model file in models/best.pt
```

---

## **⚠️ WARNINGS (Should Fix)**

### **Warning 1: Hardcoded Camera Offset Values**

**Problem:**
`xarm-motion-v1.7.py` lines 24-26 have hardcoded values that don't match config.json:

```python
CAMERA_OFFSET_X = 92.9   # Hardcoded
CAMERA_OFFSET_Y = -1.35  # Hardcoded
CAMERA_OFFSET_ERROR = 0.4  # Hardcoded
```

But config.json has:
```json
"offset_x": 7,
"offset_y": 92.9,
"offset_error": 0
```

**Fix:**
The code should load from config instead. However, since the robot controller loads from config in the class initialization, this is just a default fallback. To fix properly, update lines 24-27 in `xarm-motion-v1.7.py`:

```python
# OLD (lines 24-26):
CAMERA_OFFSET_X = 92.9
CAMERA_OFFSET_Y = -1.35
CAMERA_OFFSET_ERROR = 0.4

# NEW (better approach - load from config):
# These will be loaded from config.json in XArmController.__init__()
# Defaults here are just fallbacks
CAMERA_OFFSET_X = 7
CAMERA_OFFSET_Y = 92.9
CAMERA_OFFSET_ERROR = 0
```

---

### **Warning 2: Windows Path in yolo-mouse-v2.py**

**Problem:**
Lines 33 and 68 in `yolo-mouse-v2.py` have Windows default path:

```python
"yolo_model_path": "D:\\2. yolo\\train30\\weights\\best.pt",
model_path = CONFIG.get("yolo_model_path", "D:\\2. yolo\\train30\\weights\\best.pt")
```

**Fix:**
Update the default fallback path to use relative path:

```python
# Line 33:
"yolo_model_path": "./models/best.pt",

# Line 68:
model_path = CONFIG.get("yolo_model_path", "./models/best.pt")
```

---

## **ℹ️ CONFIGURATION VERIFICATION**

### **Verify Robot IP Address**
```bash
# Test if robot is reachable
ping 192.168.1.151

# If not reachable, update config.json with correct IP
```

### **Verify Camera Indices**
Run this to check connected cameras:

```python
import PySpin

system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
print(f"Found {cam_list.GetSize()} cameras")

for i in range(cam_list.GetSize()):
    cam = cam_list.GetByIndex(i)
    cam.Init()
    serial = cam.TLDevice.DeviceSerialNumber.GetValue()
    print(f"Camera {i}: Serial {serial}")
    cam.DeInit()

cam_list.Clear()
system.ReleaseInstance()
```

If cameras are reversed, update config.json:
```json
"camera_config": {
    "detection_camera_index": 1,  // Swap if needed
    "inspection_camera_index": 0   // Swap if needed
}
```

---

## **📋 Complete Fix Checklist**

Run these commands in order:

```bash
# 1. Install Python dependencies
pip install opencv-python numpy PyQt6 ultralytics xarm-python-sdk

# 2. Download and install PySpin from FLIR website
# https://www.flir.com/products/spinnaker-sdk/

# 3. Create models directory
mkdir -p models

# 4. Get a YOLO model (train your own or download pre-trained)
# Place it in: ./models/best.pt

# 5. Update config.json with correct YOLO path
sed -i 's|D:\\\\2. yolo\\\\train30\\\\weights\\\\best.pt|./models/best.pt|g' config.json

# 6. Verify robot IP is reachable
ping 192.168.1.151

# 7. Run diagnostic check
python3 diagnostic_check.py

# 8. If all checks pass, start the system
python3 xarm-motion-v1.7.py &
python3 yolo-mouse-v2.py
```

---

## **🚀 Starting the System (After Fixes)**

**Terminal 1: Start Robot Controller**
```bash
python3 xarm-motion-v1.7.py
```

**Terminal 2: Start Vision System**
```bash
python3 yolo-mouse-v2.py
```

**OR: Start GUI Application**
```bash
python3 arm-app-v1.py
```

---

## **🔧 Troubleshooting**

### **If you see: "FileNotFoundError: best.pt"**
```bash
# Download a YOLO model or train one
# Place it in ./models/best.pt
# Update config.json "yolo_model_path" to point to the correct location
```

### **If you see: "Failed to connect to robot"**
```bash
# Check robot IP
ping 192.168.1.151

# Verify robot is powered on
# Check network connection
# Update config.json with correct IP if needed
```

### **If you see: "No cameras found"**
```bash
# Check camera connections
# Verify PySpin is installed correctly
# Run camera index verification script above
```

### **If you see: "FileExistsError: shared memory"**
```bash
# Clean up leftover shared memory
python3 -c "
from multiprocessing import shared_memory
for name in ['ClickData', 'InspectData', 'DetectionData']:
    try:
        shm = shared_memory.SharedMemory(name)
        shm.close()
        shm.unlink()
        print(f'Cleaned {name}')
    except: pass
"
```

---

## **✅ Success Indicators**

You'll know the system is working when you see:

```
[Robot] ✅ Connected successfully!
[Robot] ✅ Initialization complete!
[Homography] ✅ Loaded camera-to-workspace transformation
[Homography] ✅ Loaded workspace-to-robot transformation
[YOLO] Model loaded from: ./models/best.pt
[INFO] Number of cameras detected: 2
[INFO] Detection camera initialized
[INFO] Inspection camera initialized
```

---

## **Need More Help?**

Run the diagnostic script to see current status:
```bash
python3 diagnostic_check.py
```

This will show you exactly which issues remain to be fixed.
