# ARM-APP - Robotic Vision System

A sophisticated robotic vision system integrating YOLO object detection with xArm robotic arm control for automated pick-and-place operations with dual-camera inspection capabilities.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Fixed](https://img.shields.io/badge/Critical_Errors-Fixed-green.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Current Status](#current-status)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This system enables a robotic arm to:
- **Detect objects** using YOLOv8 OBB (Oriented Bounding Box) detection
- **Pick and place** objects with automatic gripper angle adjustment
- **Inspect objects** using a gripper-mounted camera
- **Auto-calibrate** camera-to-robot coordinate transformations
- **Interact** via GUI or mouse clicks on live camera feed

### Hardware Requirements
- **Robot**: xArm Lite 6 (6-DOF robotic arm)
- **Cameras**: 2× FLIR cameras (PySpin compatible)
  - Detection camera (top-down view)
  - Inspection camera (gripper-mounted)
- **Network**: Ethernet connection to robot (default IP: 192.168.1.151)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────┐
│           User Interface Layer              │
│  ┌────────────────┐    ┌─────────────────┐ │
│  │  arm-app-v1.py │    │ yolo-mouse-v2.py│ │
│  │  (PyQt6 GUI)   │    │ (OpenCV Mouse)  │ │
│  └────────┬───────┘    └────────┬────────┘ │
└───────────┼──────────────────────┼──────────┘
            │                      │
    ┌───────▼──────────────────────▼────────┐
    │     Shared Memory (IPC)               │
    │  • ClickData (512B)                   │
    │  • InspectData (512B)                 │
    │  • DetectionData (4096B)              │
    └───────┬───────────────────────────────┘
            │
    ┌───────▼────────────────────────────┐
    │  xarm-motion-v1.7.py               │
    │  (Robot Controller)                │
    │  • Coordinate transformation       │
    │  • Pick/Place/Inspect sequences    │
    │  • Safety boundary checks          │
    └───────┬────────────────────────────┘
            │
    ┌───────▼─────────────┐
    │  xArm Lite 6 Robot  │
    │  192.168.1.151      │
    └─────────────────────┘
```

---

## ✅ Current Status

### Recent Fixes (2025-11-19)

| Component | Status | Details |
|-----------|--------|---------|
| **INFO ONLY Mode** | ✅ Added | arm-app-v1.py displays info but doesn't send robot commands |
| **On-screen Info Panel** | ✅ Added | Click/inspect info overlay on detection display |
| **Arrow Key Rotation** | ✅ Fixed | Inspection box rotates with arrow keys (no tab switching) |
| **Click Detection** | ✅ Fixed | Accurate click coordinates in maximized window |
| **Inspection Box Angle** | ✅ Fixed | Box follows object angle correctly |
| **Window Centering** | ✅ Added | Centers on startup and after restore from maximized |
| **Control Panel Layout** | ✅ Fixed | START/STOP buttons maintain size after maximize/restore |

### Previous Fixes (2025-11-18)

| Component | Status | Details |
|-----------|--------|---------|
| **Code Syntax** | ✅ Valid | No syntax errors |
| **Configuration** | ✅ Fixed | Windows paths converted to relative paths |
| **File Paths** | ✅ Fixed | All hardcoded paths removed |
| **Calibration Files** | ✅ Present | homography_auto.pkl, homography_det_to_robot.pkl |
| **Shared Memory** | ✅ Clean | No lingering memory blocks |
| **Git Ignore** | ✅ Added | Python cache and build artifacts excluded |

### Action Required

⚠️ **Before running the system, you must:**

1. **Install Python dependencies** (see [Installation](#installation))
2. **Add YOLO model** to `./models/best.pt`
3. **Verify robot IP** matches your hardware (config.json)

**Overall Readiness:** 🟡 **60%** (Core fixes complete, dependencies needed)

---

## 📦 Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/Harismuqri/arm-app.git
cd arm-app
```

### Step 2: Install Python Dependencies

```bash
# Install core packages
pip install opencv-python numpy PyQt6 ultralytics xarm-python-sdk

# Install PySpin (FLIR Camera SDK) - Manual installation required
# Download from: https://www.flir.com/products/spinnaker-sdk/
# Choose your OS version and install the Python wrapper
```

### Step 3: Add YOLO Model

```bash
# Option A: Use your trained model
cp /path/to/your/best.pt ./models/best.pt

# Option B: Train a new model with YOLOv8 OBB
# Follow: https://docs.ultralytics.com/tasks/obb/

# Option C: Download a pre-trained model
# (Ensure it's an OBB model for oriented bounding boxes)
```

### Step 4: Verify Installation

```bash
python3 diagnostic_check.py
```

Expected output:
```
✅ All core dependencies installed
✅ config.json is valid JSON
✅ YOLO model file found
✅ Calibration files present
✅ No hardcoded Windows paths
```

---

## 🚀 Quick Start

### Method 1: Command Line (Dual Process)

**Terminal 1 - Robot Controller:**
```bash
python3 xarm-motion-v1.7.py
```

**Terminal 2 - Vision System:**
```bash
python3 yolo-mouse-v2.py
```

### Method 2: GUI Application

```bash
python3 arm-app-v1.py
```

### First Run Calibration

The system will automatically:
1. Move robot to calibration position
2. Wait for you to start `yolo-mouse-v2.py`
3. Detect 4 white calibration circles
4. Create homography transformation matrices
5. Move robot to home position

**Calibration Setup:**
- Place 4 white circles (20-30mm diameter) at workspace corners
- Ensure good lighting and contrast
- Workspace: 300×300mm

---

## ✨ Features

### Object Detection
- **YOLOv8 OBB** for oriented bounding box detection
- Real-time position, angle, width, height calculation
- Workspace boundary filtering
- Confidence threshold: 0.8 (configurable)

### Pick & Place Operations
- **Left Click**: Move to position
- **Right Click**: Pick/Place toggle
  - First click: PICK with auto gripper angle
  - Second click: PLACE at position
- **Smart gripper angle**: Automatically grips narrower dimension
- Safety height: 150mm, Pick height: -2mm

### Inspection Mode
- **T Key**: Inspect selected object
- Smart workspace-aware positioning
- Tests 4 approach angles (0°, 90°, 180°, 270°)
- Accounts for camera offset (7mm, 92.9mm)
- Live inspection camera view with crosshair

### Coordinate Transformation
- **Two-step homography**:
  1. Camera pixels → Workspace (0-300mm)
  2. Workspace → Robot coordinates (88.9-382mm X, 14.7-312mm Y)
- Handles ~90° rotation between detection and robot frames
- Auto-calibration with circle detection

### Safety Features
- Workspace boundary checks
- Emergency stop (Ctrl+C)
- State machine prevents pick-pick without place
- TCP load updates for gripper protection

---

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
  "robot_ip": "192.168.1.151",          // Robot network address
  "tcp_speed": 300,                      // TCP movement speed (mm/s)
  "tcp_acc": 1000,                       // TCP acceleration (mm/s²)

  "click_control": {
    "safe_height": 150,                  // Safe Z height (mm)
    "pick_height": -2,                   // Pick Z height (mm)
    "inspection_height": 103.4,          // Inspection camera Z (mm)
    "workspace_min_x": 0,                // Workspace bounds
    "workspace_max_x": 300,
    "workspace_min_y": 0,
    "workspace_max_y": 300
  },

  "camera_offset": {
    "offset_x": 7,                       // Camera offset from gripper (mm)
    "offset_y": 92.9,
    "offset_error": 0                    // Tolerance (mm)
  },

  "yolo_model_path": "./models/best.pt", // Path to YOLO model
  "detection_confidence": 0.8             // Detection threshold
}
```

---

## 🔧 Troubleshooting

### System Won't Start

**Run diagnostic:**
```bash
python3 diagnostic_check.py
```

### Common Issues

#### "No module named 'cv2'"
```bash
pip install opencv-python numpy PyQt6 ultralytics xarm-python-sdk
```

#### "FileNotFoundError: best.pt"
```bash
# Add your YOLO model to:
./models/best.pt
```

#### "Failed to connect to robot"
```bash
# Check robot IP
ping 192.168.1.151

# Update config.json if needed
```

#### "No cameras found"
```bash
# Verify PySpin installation
python3 -c "import PySpin; print('PySpin OK')"

# Check camera connections and indices
```

#### "FileExistsError: shared memory"
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

### Detailed Troubleshooting

See `ERRORS_AND_FIXES.md` for comprehensive troubleshooting guide.

---

## 📚 Documentation

- **ERRORS_AND_FIXES.md** - Complete fix guide and troubleshooting
- **ERROR_ANALYSIS.md** - System status and error summary
- **diagnostic_check.py** - Automated error detection
- **quick_fix.sh** - Automated path fixes

---

## 🔄 Workflow Example

1. **System Startup**
   ```bash
   python3 xarm-motion-v1.7.py &
   python3 yolo-mouse-v2.py
   ```

2. **Calibration** (automatic on first run)
   - Robot moves to calibration position
   - Camera detects 4 white circles
   - Homography matrices created
   - Robot returns home

3. **Pick & Place**
   - YOLO detects objects (green = in workspace)
   - Right-click on object → PICK (auto-adjusts angle)
   - Right-click on target location → PLACE

4. **Inspection**
   - Click on object to select
   - Press 'T' key → Inspection camera positions
   - View close-up in inspection window

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Robot | xArm Lite 6 (6-DOF) |
| Vision | FLIR cameras + PySpin SDK |
| Object Detection | YOLOv8 OBB (Ultralytics) |
| GUI | PyQt6 |
| IPC | Python multiprocessing shared_memory |
| Calibration | OpenCV homography |
| Config | JSON |

---

## 📝 Files Overview

### Main Applications
- `arm-app-v1.py` - PyQt6 GUI with single camera + inspection box (INFO ONLY mode - displays detection data without sending robot commands)
- `arm-app.py` - PyQt6 GUI with dual camera side-by-side
- `yolo-mouse-v2.py` - OpenCV-based mouse interaction system
- `xarm-motion-v1.7.py` - Robot controller with smart positioning

### Configuration & Data
- `config.json` - System configuration
- `homography_auto.pkl` - Camera → Workspace transformation
- `homography_det_to_robot.pkl` - Workspace → Robot transformation

### Tools & Documentation
- `diagnostic_check.py` - Automated error checker
- `quick_fix.sh` - Path fix automation
- `ERRORS_AND_FIXES.md` - Troubleshooting guide
- `ERROR_ANALYSIS.md` - Error summary

---

## 🤝 Contributing

When modifying code:
1. Run `diagnostic_check.py` before committing
2. Test calibration with 4 white circles
3. Verify workspace boundaries are respected
4. Update documentation if adding features

---

## ⚠️ Safety Notes

- System checks workspace boundaries before all movements
- Emergency stop available (Ctrl+C)
- Safe height movements prevent collisions
- TCP load updates prevent gripper damage
- State machine prevents unsafe pick/place sequences

**Workspace Limits:**
- X: 88.9 - 382.0 mm
- Y: 14.7 - 312.0 mm
- Z: -2 mm (pick) to 150 mm (safe)

---

## 📄 License

See repository for license details.

---

## 🆘 Support

For issues or questions:
1. Check `ERRORS_AND_FIXES.md`
2. Run `diagnostic_check.py`
3. Review error logs in terminal output
4. Ensure all dependencies are installed

---

**Last Updated:** 2025-11-19
**Status:** ✅ Core fixes complete, ready for deployment after dependency installation
