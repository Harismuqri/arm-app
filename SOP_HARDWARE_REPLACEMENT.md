# Standard Operating Procedure (SOP)
## Hardware Replacement and System Startup

**Document Version:** 1.0
**Last Updated:** 2025-11-24
**System:** IROPO (Intelligent Robot Positioning System)

---

## Table of Contents
1. [Pre-Replacement Checklist](#pre-replacement-checklist)
2. [Hardware Replacement Procedures](#hardware-replacement-procedures)
3. [System Configuration](#system-configuration)
4. [Calibration Procedures](#calibration-procedures)
5. [System Startup](#system-startup)
6. [Verification and Testing](#verification-and-testing)
7. [Troubleshooting](#troubleshooting)

---

## Pre-Replacement Checklist

### Safety First
- [ ] **EMERGENCY STOP** accessible and tested
- [ ] Disconnect power to robot arm
- [ ] Ensure workspace is clear
- [ ] Backup current configuration files:
  ```bash
  cp config.json config.json.backup
  cp homography_auto.pkl homography_auto.pkl.backup
  cp homography_det_to_robot.pkl homography_det_to_robot.pkl.backup
  ```

### Document Current Setup
- [ ] Take photos of current hardware setup
- [ ] Note all cable connections
- [ ] Record current IP addresses
- [ ] Note camera serial numbers (if FLIR cameras)
- [ ] Save current robot position:
  ```bash
  # Run this in robot controller
  # Record displayed position
  ```

---

## Hardware Replacement Procedures

### Option A: Replacing Robot Arm

#### Step 1: Physical Disconnection
1. **Power down the robot**
   - Turn off robot power switch
   - Wait 30 seconds for complete shutdown

2. **Disconnect cables**
   - Ethernet cable from robot base
   - Power cable
   - Label all cables for reconnection

3. **Remove robot from mounting**
   - Unbolt robot from work surface
   - Store in safe location

#### Step 2: Install New Robot
1. **Mount new robot**
   - Bolt securely to work surface
   - Ensure stable and level mounting

2. **Connect cables**
   - Connect power cable
   - Connect Ethernet cable to same network switch

3. **Configure network**
   - Power on robot
   - Use xArm Studio or button on robot to set IP address
   - **Default IP to use:** `192.168.1.151`
   - If different IP needed, note for configuration step

4. **Initialize robot**
   - Use xArm Studio to:
     - Clear any warnings
     - Enable servo motors
     - Test basic movement
     - Set motion mode to position mode

#### Step 3: Update Configuration
Go to [System Configuration - Robot Settings](#updating-robot-settings)

---

### Option B: Replacing Cameras

#### Step 1: Identify Camera Type
- **Current System:** FLIR cameras (PySpin SDK)
- **If replacing with FLIR:** Skip to Step 2
- **If replacing with different brand:** See [Changing Camera Type](#changing-camera-type)

#### Step 2: Physical Replacement (FLIR Cameras)

**Detection Camera (Top-down view):**
1. Power off camera or disconnect USB
2. Unmount camera from overhead position
3. Note mounting height and angle
4. Install new camera in **exact same position**
5. Connect USB cable
6. Connect power (if using external power)

**Inspection Camera (Gripper-mounted):**
1. Power off camera
2. Remove from gripper mount
3. Install new camera in **exact same position and orientation**
4. Secure all cables to prevent interference with robot motion
5. Connect USB cable

#### Step 3: Verify Camera Detection
```bash
# Test camera detection
python3 -c "import PySpin; system = PySpin.System.GetInstance(); cam_list = system.GetCameras(); print(f'Found {cam_list.GetSize()} cameras'); cam_list.Clear(); system.ReleaseInstance()"
```

Expected output: `Found 2 cameras`

#### Step 4: Identify Camera Indices
```bash
# Run camera test
python3 << 'EOF'
import PySpin
import cv2

system = PySpin.System.GetInstance()
cam_list = system.GetCameras()

for i, cam in enumerate(cam_list):
    print(f"\nCamera Index {i}:")
    cam.Init()
    nodemap = cam.GetNodeMap()
    node_device_serial = PySpin.CStringPtr(nodemap.GetNode('DeviceSerialNumber'))
    print(f"  Serial: {node_device_serial.GetValue()}")

    # Test capture
    cam.BeginAcquisition()
    image_result = cam.GetNextImage(1000)
    print(f"  Image size: {image_result.GetWidth()}x{image_result.GetHeight()}")
    image_result.Release()
    cam.EndAcquisition()
    cam.DeInit()

cam_list.Clear()
system.ReleaseInstance()
EOF
```

Note which camera is which:
- Camera 0 should be **detection camera** (top-down)
- Camera 1 should be **inspection camera** (gripper-mounted)

If reversed, update `config.json` camera indices.

#### Step 5: Update Configuration
Go to [System Configuration - Camera Settings](#updating-camera-settings)

---

### Option C: Changing Camera Type

**WARNING:** Changing from FLIR to another camera type requires code modifications.

#### Supported Alternatives:
- USB webcams (OpenCV compatible)
- Other industrial cameras with Python SDK

#### Code Modifications Required:
1. **Modify:** `/home/user/arm-app/yolo-mouse-v2.py`
   - Replace PySpin initialization (lines ~689-730)
   - Replace camera capture method (lines in main loop)

2. **Example for USB webcam:**
   ```python
   # Replace PySpin initialization with:
   detection_cam = cv2.VideoCapture(0)  # Index 0 for detection
   inspection_cam = cv2.VideoCapture(1)  # Index 1 for inspection

   # Set resolution
   detection_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
   detection_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
   ```

3. **Update requirements:**
   - Remove PySpin dependency
   - Ensure opencv-python installed

---

## System Configuration

### Configuration File Location
**File:** `/home/user/arm-app/config.json`

### Updating Robot Settings

1. **Open configuration file:**
   ```bash
   cd /home/user/arm-app
   nano config.json
   ```

2. **Update robot IP address:**
   ```json
   {
     "robot": {
       "ip": "192.168.1.151",  // Change if different
       ...
     }
   }
   ```

3. **Verify network connectivity:**
   ```bash
   ping 192.168.1.151
   ```
   Should see replies. Press Ctrl+C to stop.

4. **Test robot connection:**
   ```bash
   python3 << 'EOF'
   from xarm.wrapper import XArmAPI

   arm = XArmAPI('192.168.1.151')
   arm.connect()
   print(f"Connected: {arm.connected}")
   print(f"State: {arm.state}")
   arm.disconnect()
   EOF
   ```

### Updating Camera Settings

1. **Open configuration file:**
   ```bash
   nano config.json
   ```

2. **Update camera indices if needed:**
   ```json
   {
     "camera": {
       "detection_index": 0,      // Top-down camera
       "inspection_index": 1,     // Gripper camera
       ...
     }
   }
   ```

3. **Adjust camera window positions (optional):**
   ```json
   "detection_window": {
     "width": 960,
     "height": 720,
     "x_position": 950,
     "y_position": 50
   }
   ```

### Updating Workspace Settings

**Only change if workspace physical dimensions changed!**

```json
{
  "workspace": {
    "x_min": 0,
    "x_max": 300,     // Detection workspace in mm
    "y_min": 0,
    "y_max": 300,
    "safe_height": 150,      // Safe Z height
    "pick_height": -2,       // Z when picking
    "inspection_height": 103.4  // Inspection camera height
  }
}
```

---

## Calibration Procedures

### When Calibration is Required

**MUST recalibrate if:**
- Detection camera moved or replaced
- Robot arm replaced or repositioned
- Workspace layout changed
- Mounting height changed

**NOT required if:**
- Only inspection camera replaced (same position)
- Software reinstalled
- Config values changed (not hardware)

### Preparation for Calibration

#### Materials Needed:
- 4 white circular markers (20-30mm diameter)
  - Can use: white stickers, bottle caps, printed circles
- Dark/contrasting background surface
- Good lighting (no harsh shadows)

#### Step 1: Delete Old Calibration
```bash
cd /home/user/arm-app
rm homography_auto.pkl
rm homography_det_to_robot.pkl
```

#### Step 2: Place Calibration Markers

**CRITICAL:** Marker placement defines your workspace!

1. **Measure 300mm x 300mm workspace**
2. **Place markers at corners:**
   ```
   Marker 4                    Marker 3
   (0, 300)                    (300, 300)
        +-------------------------+
        |                         |
        |    Detection Area       |
        |       300x300mm         |
        |                         |
        +-------------------------+
   Marker 1                    Marker 2
   (0, 0)                      (300, 0)
   ```

3. **Ensure markers are:**
   - Clearly visible to camera
   - High contrast with background
   - Flat and circular
   - Not overlapping or touching workspace edges

#### Step 3: Verify Robot Corner Mapping

The robot will map these corners to physical coordinates:
```
Detection Corner  →  Robot Coordinate
[0, 0]           →  [88.9, 312.0]    (Bottom Left)
[300, 0]         →  [88.9, 14.7]     (Bottom Right)
[300, 300]       →  [382.0, 14.7]    (Top Right)
[0, 300]         →  [382.0, 312.0]   (Top Left)
```

Ensure robot can physically reach these positions.

### Running Auto-Calibration

Calibration runs automatically on first startup after deletion of calibration files.

**The system will:**
1. Detect the 4 white circles
2. Create coordinate transformation matrices
3. Save `homography_auto.pkl` (camera → workspace)
4. Save `homography_det_to_robot.pkl` (workspace → robot)

**Success indicators:**
- Console shows: "Calibration completed"
- Both .pkl files created
- Robot moves to home position

**If calibration fails:**
- See [Calibration Troubleshooting](#calibration-fails)

---

## System Startup

### Startup Options

The system can be started in **two different modes:**

**Option A: GUI Application (Recommended for Setup/Calibration)**
- User-friendly graphical interface
- Built-in calibration wizard
- Visual feedback and monitoring
- **INFO ONLY mode** - shows detection data but does NOT control robot
- Best for: Initial setup, calibration, testing, monitoring

**Option B: Command-Line Mode (Recommended for Production)**
- Full robot control capability
- Pick and place operations
- Inspection mode
- Best for: Active production, automated operations

Choose **Option A** if you need to calibrate the system or just monitor detections.
Choose **Option B** if you need full robot control for pick-and-place operations.

---

### Prerequisites Check

**Before starting either mode, verify:**
- [ ] Robot powered on and connected to network
- [ ] Both cameras connected via USB
- [ ] Configuration file updated (config.json)
- [ ] Calibration markers in place (if new calibration needed)
- [ ] Workspace clear of obstacles
- [ ] Emergency stop accessible

---

### Option A: GUI Application Startup

**Use this method for calibration, setup, and monitoring.**

#### Step 1: Start GUI Application

```bash
cd /home/user/arm-app
python3 arm-app-v1.1.py
```

#### Step 2: Configure Calibration (First Time Setup)

**Tab 1: Calibration Setup**

1. **Detection Camera Calibration:**
   - Select "Auto Calibration (Circle Detection)" from dropdown
   - Ensure 4 white circles (20-30mm) are visible at workspace corners
   - Click "Calibrate" button
   - Wait for confirmation message
   - Calibration file saved: `homography_auto.pkl`

2. **Robot Coordinate Transformation:**
   - Leave "Auto Calculation" checkbox enabled
   - Enter only Bottom Left (BL) and Top Right (TR) robot coordinates:
     - **BL:** Workspace (0, 0) → Robot (88.9, 312.0)
     - **TR:** Workspace (300, 300) → Robot (382.0, 14.7)
   - Click "Calculate" button
   - System auto-calculates BR and TL coordinates
   - Calibration file saved: `homography_det_to_robot.pkl`

#### Step 3: Start Live Detection

**Tab 2: Live Camera View**

1. Click **"START System"** button at bottom
2. Detection camera feed appears
3. YOLO detection starts automatically
4. Objects inside workspace show in **green**
5. Objects outside workspace show in **red**
6. Detection status shown: "Not Ready" / "Detect" / "Ready"

#### Step 4: Monitor and Test

**Features Available:**

- **Single Click on Object:**
  - Shows object information panel
  - Displays: ID, position, angle, dimensions
  - Shows workspace coordinates

- **Double Click on Object:**
  - Opens inspection visualization box (36mm × 36mm)
  - Use arrow keys to rotate: ← (CCW) / → (CW)
  - Press ESC to close inspection box

- **Single Click on Empty Space:**
  - Shows clicked pixel coordinates
  - Shows workspace mm coordinates
  - Verifies calibration accuracy

**Tab 3: System Info**

- Shows system status and readiness percentage
- Camera connection status
- Calibration status
- Theme selection (Dark/Light mode)

#### Step 5: Stop System

Click **"STOP System"** button when finished.

**Important Notes:**
- GUI runs in **INFO ONLY** mode - it does NOT send commands to the robot
- It shares detection data via shared memory for monitoring
- Robot controller (xarm-motion-v1.7.py) must be running separately for robot movement
- Use this mode for calibration, testing, and verification

---

### Option B: Command-Line Startup (Full Robot Control)

**Use this method for production operations with robot control.**

#### Step-by-Step Startup Sequence

#### Step 1: Open Two Terminal Windows

**Terminal 1:** Robot Controller
**Terminal 2:** Vision System

#### Step 2: Start Robot Controller (Terminal 1)

```bash
cd /home/user/arm-app
python3 xarm-motion-v1.7.py
```

**Expected sequence:**
```
Loading configuration...
Connecting to robot at 192.168.1.151...
✓ Connected
✓ Motion enabled
✓ Errors cleared
✓ Mode set to position control
Loading homography matrices...
Moving to calibration position...
Waiting for YOLO calibration...
```

**Robot will:**
1. Connect and initialize
2. Move to calibration position: [-55.3, 207.3, 92.7, 179.7, -0.2, -2.4]
3. Wait for vision system calibration

**Leave this terminal running!**

#### Step 3: Start Vision System (Terminal 2)

**Wait for robot to reach calibration position first!**

```bash
cd /home/user/arm-app
python3 yolo-mouse-v2.py
```

**Expected sequence:**
```
Loading configuration...
Initializing PySpin system...
Found 2 cameras
Camera 0: Detection (Serial: XXXXX)
Camera 1: Inspection (Serial: XXXXX)
Starting acquisition...
```

**If calibration needed:**
```
Starting auto-calibration...
Detecting circles...
Found 4 circles
Calculating homography...
✓ Calibration completed
Saved homography_auto.pkl
Saved homography_det_to_robot.pkl
```

**Detection window will open showing live camera feed**

#### Step 4: Verify Robot Completion (Terminal 1)

Watch Terminal 1 for:
```
✓ Calibration files updated
Moving to home position...
✓ Ready for operation
```

**Robot will move to home:** [1.5, 6.3, 45.5, 0, 39.2, 3.2]

#### Step 5: System Ready

**Both terminals running:**
- Terminal 1: Robot controller (monitoring commands)
- Terminal 2: Vision system (detection window visible)

**Indicators of successful startup:**
- Detection window shows live camera feed
- YOLO bounding boxes appear on detected objects
- No error messages in either terminal
- Robot at home position

---

## Verification and Testing

### Test 1: Camera Detection

**In detection window:**
- Place an object in workspace
- Verify YOLO bounding box appears
- Check angle and dimensions display correctly

### Test 2: Coordinate Transformation

**Test tool:**
```bash
python3 coordinate_tester.py
```

**Follow prompts to test:**
1. Click position in camera view
2. Verify transformed robot coordinates
3. Ensure coordinates within robot boundaries:
   - X: 88.9 to 382.0 mm
   - Y: 14.7 to 312.0 mm

### Test 3: Robot Movement (LEFT CLICK)

**WARNING:** Start with object at safe distance from robot

1. **Left-click on object** in detection window
2. **Verify:**
   - Robot moves smoothly
   - Reaches correct position
   - No collision warnings
   - Returns to safe height

### Test 4: Pick and Place (RIGHT CLICK)

1. **Place lightweight test object** in workspace
2. **Right-click on object**
3. **Verify sequence:**
   - Robot moves to position
   - Lowers to pick height
   - Gripper closes at correct angle
   - Picks up object
   - Returns to safe height

4. **Right-click target location**
5. **Verify:**
   - Robot moves to location
   - Lowers to place height
   - Gripper opens
   - Returns to safe height

### Test 5: Inspection Mode (T KEY)

1. **Click object** to select
2. **Press T key**
3. **Verify:**
   - Inspection window opens
   - Robot tests multiple angles (0°, 90°, 180°, 270°)
   - Inspection camera shows close-up view
   - Crosshair centered on object

### Test 6: Emergency Stop

**CRITICAL SAFETY TEST:**

1. **Initiate robot movement**
2. **Press Ctrl+C in Terminal 1**
3. **Verify:**
   - Robot stops immediately
   - Both programs exit cleanly
   - No error states on robot

4. **Recovery:**
   - Restart both programs
   - Verify normal operation

---

## Troubleshooting

### Robot Won't Connect

**Error:** `Failed to connect to robot at 192.168.1.151`

**Solutions:**
1. **Check network connection:**
   ```bash
   ping 192.168.1.151
   ```
   If no reply:
   - Verify Ethernet cable connected
   - Check robot power
   - Verify IP address in config.json
   - Check computer and robot on same subnet

2. **Check robot state:**
   - Use xArm Studio to connect
   - Clear any errors
   - Enable servos
   - Disconnect xArm Studio before running Python script

3. **Verify IP address:**
   - Robot display shows correct IP
   - Update config.json if changed

### Cameras Not Detected

**Error:** `Found 0 cameras`

**Solutions:**
1. **Check USB connections:**
   ```bash
   lsusb
   ```
   Look for FLIR or Point Grey devices

2. **Check PySpin installation:**
   ```bash
   python3 -c "import PySpin; print('PySpin OK')"
   ```

3. **Restart USB:**
   ```bash
   # Unplug and replug USB cables
   # Or restart udev:
   sudo udevadm control --reload-rules
   ```

4. **Check permissions:**
   ```bash
   # Add user to video group
   sudo usermod -a -G video $USER
   # Logout and login
   ```

### Calibration Fails

**Error:** `Could not detect 4 circles`

**Solutions:**
1. **Check marker visibility:**
   - Ensure good lighting
   - Increase contrast (dark background)
   - Verify markers are circular and flat
   - Check camera focus

2. **Adjust detection parameters** (if persistent):
   Edit `/home/user/arm-app/yolo-mouse-v2.py` line ~619:
   ```python
   circles = cv2.HoughCircles(
       gray,
       cv2.HOUGH_GRADIENT,
       dp=1.2,
       minDist=100,
       param1=100,
       param2=30,      # Try lowering to 20
       minRadius=10,   # Try adjusting
       maxRadius=50    # Try adjusting
   )
   ```

3. **Manual verification:**
   - Run vision system alone
   - Check if circles are highlighted
   - Adjust marker size/position

### YOLO Model Missing

**Error:** `Could not load model from ./models/best.pt`

**Solution:**
```bash
# Create models directory
mkdir -p /home/user/arm-app/models

# Add your trained YOLO model
# Copy best.pt to models/
cp /path/to/your/best.pt /home/user/arm-app/models/
```

### Coordinates Out of Bounds

**Error:** `Target coordinates outside robot workspace`

**Cause:** Calibration mismatch or workspace definition incorrect

**Solution:**
1. **Verify workspace dimensions** in config.json
2. **Recalibrate system:**
   - Delete calibration files
   - Restart with correct marker placement
3. **Test with coordinate_tester.py**

### Robot Moves to Wrong Position

**Cause:** Calibration error or coordinate transformation issue

**Solution:**
1. **Recalibrate immediately:**
   ```bash
   rm homography_auto.pkl homography_det_to_robot.pkl
   ```
   Restart system

2. **Verify marker placement:**
   - Must be exactly at workspace corners
   - Check measurements

3. **Test transformation:**
   ```bash
   python3 coordinate_tester.py
   ```

### Inspection Camera Shows Nothing

**Solutions:**
1. **Check camera index** in config.json:
   ```json
   "inspection_index": 1
   ```

2. **Verify camera mounting:**
   - Camera facing downward
   - Proper offset configured

3. **Check inspection height** in config.json:
   ```json
   "inspection_height": 103.4  // Adjust if needed
   ```

### Shared Memory Errors

**Error:** `Could not create shared memory`

**Solution:**
```bash
# Clean up shared memory
rm /dev/shm/DetectionData
rm /dev/shm/ClickData
rm /dev/shm/InspectData

# Restart system
```

---

## Additional Resources

### Documentation Files
- **Main README:** `/home/user/arm-app/README.md`
- **Error Guide:** `/home/user/arm-app/Guide/ERRORS_AND_FIXES.md`
- **Diagnostic Tool:** `/home/user/arm-app/Guide/diagnostic_check.py`

### Running Diagnostics
```bash
cd /home/user/arm-app/Guide
python3 diagnostic_check.py
```

### Contact Information
For additional support:
- Check GitHub issues
- Review ERRORS_AND_FIXES.md
- Run diagnostic_check.py

---

## Revision History

| Version | Date       | Changes                          | Author |
|---------|------------|----------------------------------|--------|
| 1.0     | 2025-11-24 | Initial SOP creation            | Claude |

---

## Appendix: Quick Reference

### Startup Commands

**Option A: GUI Application (Calibration/Monitoring)**
```bash
cd /home/user/arm-app && python3 arm-app-v1.1.py
```

**Option B: Command-Line (Full Robot Control)**
```bash
# Terminal 1
cd /home/user/arm-app && python3 xarm-motion-v1.7.py

# Terminal 2 (wait for robot calibration position)
cd /home/user/arm-app && python3 yolo-mouse-v2.py
```

### Emergency Stop
```
Ctrl+C in either terminal
```

### Recalibration
```bash
rm homography_auto.pkl homography_det_to_robot.pkl
# Restart system
```

### File Locations
```
Configuration:     /home/user/arm-app/config.json
Calibration:       /home/user/arm-app/homography*.pkl
GUI Application:   /home/user/arm-app/arm-app-v1.1.py
Robot Controller:  /home/user/arm-app/xarm-motion-v1.7.py
Vision System:     /home/user/arm-app/yolo-mouse-v2.py
Documentation:     /home/user/arm-app/README.md
```

---

**END OF SOP**
