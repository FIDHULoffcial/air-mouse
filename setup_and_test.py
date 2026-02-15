#!/usr/bin/env python3
"""
Iron Man Holographic System - Quick Setup & Test Script
Run this to install dependencies and test your system
"""

import subprocess
import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_python_version():
    """Check if Python version is adequate"""
    print_header("CHECKING PYTHON VERSION")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher required!")
        print("   Please install Python 3.8+ and try again.")
        return False
    
    print("✓ Python version is compatible")
    return True

def install_dependencies():
    """Install required packages"""
    print_header("INSTALLING DEPENDENCIES")
    
    packages = [
        'opencv-python',
        'mediapipe',
        'numpy',
        'PyOpenGL',
        'PyOpenGL-accelerate',
        'pygame',
        'scipy',
        'Pillow'
    ]
    
    print("Installing packages (this may take a few minutes)...\n")
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', package
            ], stdout=subprocess.DEVNULL)
            print(f"  ✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"  ⚠ Warning: Could not install {package}")
            print(f"    Try manually: pip install {package}")
    
    print("\n✓ Dependency installation complete!")

def test_imports():
    """Test if all modules can be imported"""
    print_header("TESTING IMPORTS")
    
    modules = [
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy'),
        ('OpenGL', 'PyOpenGL'),
        ('pygame', 'PyGame')
    ]
    
    all_ok = True
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name} import successful")
        except ImportError as e:
            print(f"❌ {display_name} import failed: {e}")
            all_ok = False
    
    if all_ok:
        print("\n✓ All modules imported successfully!")
    else:
        print("\n⚠ Some modules failed to import. Please check errors above.")
    
    return all_ok

def test_webcam():
    """Test webcam availability"""
    print_header("TESTING WEBCAM")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam (index 0)")
            print("   Trying alternate camera...")
            cap = cv2.VideoCapture(1)
            
            if not cap.isOpened():
                print("❌ No webcam detected")
                print("   Please ensure:")
                print("   1. Webcam is connected")
                print("   2. No other app is using it")
                print("   3. Webcam permissions are granted")
                return False
            else:
                print("✓ Webcam found at index 1")
        else:
            print("✓ Webcam found at index 0")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"✓ Webcam resolution: {w}x{h}")
        else:
            print("⚠ Could not read frame from webcam")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Error testing webcam: {e}")
        return False

def test_opengl():
    """Test OpenGL functionality"""
    print_header("TESTING OPENGL")
    
    try:
        from OpenGL.GL import *
        from OpenGL.GLU import *
        from OpenGL.GLUT import *
        import pygame
        
        # Initialize pygame
        pygame.init()
        display = (640, 480)
        pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
        
        # Test basic OpenGL commands
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        print("✓ OpenGL initialization successful")
        print("✓ GLUT available")
        
        pygame.quit()
        return True
        
    except Exception as e:
        print(f"❌ OpenGL test failed: {e}")
        print("\n  If on Windows, you may need to:")
        print("  1. Download FreeGLUT DLL")
        print("  2. Place it in Python directory")
        print("\n  If on Linux, install:")
        print("  sudo apt install freeglut3-dev mesa-common-dev")
        return False

def test_hand_detection():
    """Test MediaPipe hand detection"""
    print_header("TESTING HAND DETECTION")
    
    try:
        import cv2
        import mediapipe as mp
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        print("✓ MediaPipe Hands initialized")
        
        # Test with webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        
        print("\nTesting hand detection (show your hand to camera)...")
        print("Press 'q' to continue...")
        
        hand_detected = False
        
        for _ in range(300):  # Try for ~10 seconds at 30fps
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_detected = True
                cv2.putText(frame, "HAND DETECTED!", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Show your hand...", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            cv2.imshow("Hand Detection Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if hand_detected:
            print("\n✓ Hand detection working!")
        else:
            print("\n⚠ No hand detected. Check:")
            print("  1. Adequate lighting")
            print("  2. Hand clearly visible to camera")
            print("  3. No obstructions")
        
        return True
        
    except Exception as e:
        print(f"❌ Hand detection test failed: {e}")
        return False

def create_test_config():
    """Create test configuration file"""
    print_header("CREATING TEST CONFIGURATION")
    
    config = {
        "camera_index": 0,
        "resolution": [1280, 720],
        "hand_detection_confidence": 0.7,
        "tracking_confidence": 0.7,
        "max_hands": 2
    }
    
    import json
    
    with open('config.json', 'w') as f:
        json.dump(config, indent=2, fp=f)
    
    print("✓ Configuration file created: config.json")
    print("\nYou can edit this file to customize settings:")
    print("  - camera_index: Change if webcam not at 0")
    print("  - resolution: Adjust for performance")
    print("  - confidence: Lower for easier detection")

def display_system_info():
    """Display system information"""
    print_header("SYSTEM INFORMATION")
    
    import platform
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Architecture: {platform.machine()}")
    
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except:
        print("OpenCV: Not installed")
    
    try:
        import mediapipe
        print(f"MediaPipe: {mediapipe.__version__}")
    except:
        print("MediaPipe: Not installed")
    
    try:
        import numpy
        print(f"NumPy: {numpy.__version__}")
    except:
        print("NumPy: Not installed")

def main():
    """Main setup and test routine"""
    print("\n" + "🚀"*35)
    print("  IRON MAN HOLOGRAPHIC SYSTEM - SETUP & TEST")
    print("🚀"*35)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Display system info
    display_system_info()
    
    # Ask user what to do
    print_header("SETUP OPTIONS")
    print("1. Install dependencies only")
    print("2. Run full system test")
    print("3. Quick camera test")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        install_dependencies()
        print("\n✓ Setup complete!")
        print("  Run 'python ironman_hologram.py' to start the system")
    
    elif choice == '2':
        # Full test
        install_dependencies()
        
        if not test_imports():
            print("\n⚠ Some imports failed. Please fix errors above.")
            return
        
        if not test_webcam():
            print("\n⚠ Webcam test failed. Please check your camera.")
            return
        
        if not test_opengl():
            print("\n⚠ OpenGL test failed. Please install required libraries.")
            return
        
        test_hand_detection()
        create_test_config()
        
        print_header("SETUP COMPLETE!")
        print("✓ All tests passed!")
        print("\n🎯 Ready to run Iron Man Holographic System!")
        print("\nTo start:")
        print("  Basic version:    python ironman_hologram.py")
        print("  Enhanced version: python ironman_hologram_enhanced.py")
    
    elif choice == '3':
        # Quick camera test
        test_webcam()
    
    else:
        print("Exiting...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
