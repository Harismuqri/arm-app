#!/bin/bash
# Quick Fix Script for arm-app errors
# This script automates some of the common fixes

echo "============================================================"
echo "ARM-APP QUICK FIX SCRIPT"
echo "============================================================"
echo ""

# 1. Create models directory
echo "1. Creating models directory..."
mkdir -p models
echo "   ✅ Created ./models/"

# 2. Fix config.json path (create backup first)
echo ""
echo "2. Fixing YOLO model path in config.json..."
if [ -f "config.json" ]; then
    cp config.json config.json.backup
    echo "   📋 Backup created: config.json.backup"

    # Replace Windows path with relative path
    sed -i 's|D:\\\\2\. yolo\\\\train30\\\\weights\\\\best\.pt|./models/best.pt|g' config.json
    echo "   ✅ Updated config.json"
else
    echo "   ❌ config.json not found!"
fi

# 3. Fix yolo-mouse-v2.py default path
echo ""
echo "3. Fixing default path in yolo-mouse-v2.py..."
if [ -f "yolo-mouse-v2.py" ]; then
    cp yolo-mouse-v2.py yolo-mouse-v2.py.backup
    echo "   📋 Backup created: yolo-mouse-v2.py.backup"

    # Fix line 33
    sed -i 's|"yolo_model_path": "D:\\\\2\. yolo\\\\train30\\\\weights\\\\best\.pt"|"yolo_model_path": "./models/best.pt"|g' yolo-mouse-v2.py

    # Fix line 68
    sed -i 's|"D:\\\\2\. yolo\\\\train30\\\\weights\\\\best\.pt"|"./models/best.pt"|g' yolo-mouse-v2.py

    echo "   ✅ Updated yolo-mouse-v2.py"
else
    echo "   ❌ yolo-mouse-v2.py not found!"
fi

# 4. Clean up any leftover shared memory
echo ""
echo "4. Cleaning up shared memory..."
python3 -c "
from multiprocessing import shared_memory
cleaned = []
for name in ['ClickData', 'InspectData', 'DetectionData']:
    try:
        shm = shared_memory.SharedMemory(name)
        shm.close()
        shm.unlink()
        cleaned.append(name)
    except FileNotFoundError:
        pass
if cleaned:
    print('   ✅ Cleaned: ' + ', '.join(cleaned))
else:
    print('   ✅ No shared memory to clean')
"

# 5. Check if dependencies are installed
echo ""
echo "5. Checking Python dependencies..."
python3 -c "
missing = []
try:
    import cv2
except ImportError:
    missing.append('opencv-python')
try:
    import numpy
except ImportError:
    missing.append('numpy')
try:
    import PyQt6
except ImportError:
    missing.append('PyQt6')
try:
    from ultralytics import YOLO
except ImportError:
    missing.append('ultralytics')
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    missing.append('xarm-python-sdk')

if missing:
    print('   ⚠️  Missing packages:', ', '.join(missing))
    print('   Run: pip install ' + ' '.join(missing))
else:
    print('   ✅ All core dependencies installed')
"

echo ""
echo "============================================================"
echo "QUICK FIX COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Install missing dependencies (see above)"
echo "2. Place your YOLO model in ./models/best.pt"
echo "3. Verify robot IP in config.json (currently: 192.168.1.151)"
echo "4. Run diagnostic: python3 diagnostic_check.py"
echo ""
echo "Backups created:"
echo "  - config.json.backup"
echo "  - yolo-mouse-v2.py.backup"
echo ""
