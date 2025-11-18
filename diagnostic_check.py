#!/usr/bin/env python3
"""
Diagnostic script for arm-app error checking
Run this to identify all potential issues before starting the system
"""

import sys
import os
import json
from pathlib import Path

def check_dependencies():
    """Check if all required Python packages are installed"""
    print("\n" + "="*60)
    print("1. CHECKING PYTHON DEPENDENCIES")
    print("="*60)

    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PyQt6': 'PyQt6',
        'ultralytics': 'ultralytics',
        'PySpin': 'PySpin (manual install from FLIR)',
        'xarm': 'xarm-python-sdk'
    }

    missing = []
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)

    return len(missing) == 0

def check_config():
    """Check configuration file for issues"""
    print("\n" + "="*60)
    print("2. CHECKING CONFIG.JSON")
    print("="*60)

    issues = []

    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("✅ config.json is valid JSON")

        # Check YOLO model path
        yolo_path = config.get('yolo_model_path', '')
        if yolo_path.startswith('D:\\'):
            print(f"⚠️  YOLO model path uses Windows format: {yolo_path}")
            print(f"   This won't work on Linux/Mac")
            issues.append("Windows path format")
        elif not os.path.exists(yolo_path):
            print(f"❌ YOLO model file not found: {yolo_path}")
            issues.append("Missing YOLO model")
        else:
            print(f"✅ YOLO model path: {yolo_path}")

        # Check robot IP
        robot_ip = config.get('robot_ip', '')
        print(f"ℹ️  Robot IP: {robot_ip} (verify this is correct)")

        # Check camera offset consistency
        camera_offset = config.get('camera_offset', {})
        print(f"ℹ️  Camera offset: X={camera_offset.get('offset_x')}, Y={camera_offset.get('offset_y')}")

    except FileNotFoundError:
        print("❌ config.json not found")
        issues.append("Missing config.json")
    except json.JSONDecodeError as e:
        print(f"❌ config.json is invalid: {e}")
        issues.append("Invalid JSON")

    return len(issues) == 0

def check_calibration_files():
    """Check if calibration files exist"""
    print("\n" + "="*60)
    print("3. CHECKING CALIBRATION FILES")
    print("="*60)

    files = ['homography_auto.pkl', 'homography_det_to_robot.pkl']
    all_exist = True

    for filename in files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ {filename} ({size} bytes)")
        else:
            print(f"⚠️  {filename} not found (will be created on first calibration)")
            all_exist = False

    return True  # Not critical, files will be created

def check_shared_memory():
    """Check for lingering shared memory"""
    print("\n" + "="*60)
    print("4. CHECKING SHARED MEMORY")
    print("="*60)

    from multiprocessing import shared_memory

    memory_blocks = ['ClickData', 'InspectData', 'DetectionData']
    found = []

    for name in memory_blocks:
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            print(f"⚠️  {name} exists (left over from previous run)")
            found.append(name)
            shm.close()
        except FileNotFoundError:
            print(f"✅ {name} clean")

    if found:
        print("\nTo clean up, run:")
        print("python3 -c \"")
        print("from multiprocessing import shared_memory")
        for name in found:
            print(f"shm = shared_memory.SharedMemory('{name}'); shm.close(); shm.unlink()")
        print("\"")

    return True  # Not critical

def check_file_paths():
    """Check for hardcoded Windows paths"""
    print("\n" + "="*60)
    print("5. CHECKING FOR WINDOWS PATHS IN CODE")
    print("="*60)

    issues = []
    py_files = ['arm-app.py', 'arm-app-v1.py', 'xarm-motion-v1.7.py', 'yolo-mouse-v2.py']

    for filename in py_files:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
                if 'D:\\' in content or 'C:\\' in content:
                    print(f"⚠️  {filename} contains Windows paths")
                    issues.append(filename)

    if not issues:
        print("✅ No hardcoded Windows paths in Python files")

    return len(issues) == 0

def main():
    print("\n" + "="*60)
    print("ARM-APP DIAGNOSTIC CHECK")
    print("="*60)

    results = {
        'Dependencies': check_dependencies(),
        'Configuration': check_config(),
        'Calibration Files': check_calibration_files(),
        'Shared Memory': check_shared_memory(),
        'File Paths': check_file_paths()
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✅ All checks passed! System is ready to run.")
        return 0
    else:
        print("\n⚠️  Some issues found. Please fix the errors above before running.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
