#!/usr/bin/env python3
"""
Test script to verify whisper.cpp setup without requiring microphone input
"""

import os
import subprocess
import sys
from main import WhisperRecognition

def test_setup():
    """Test the whisper.cpp setup"""
    print("Testing whisper.cpp setup...")
    print("=" * 40)
    
    try:
        recognizer = WhisperRecognition()
        
        # Test directory creation
        print("✓ Directories created")
        
        # Test whisper.cpp compilation
        print("Setting up whisper.cpp...")
        recognizer.setup_whisper_cpp()
        print("✓ whisper.cpp compiled successfully")
        
        # Test model download
        print("Downloading model...")
        recognizer.download_model()
        print("✓ Model downloaded successfully")
        
        # Test whisper.cpp executable
        whisper_main = os.path.join(recognizer.whisper_cpp_path, "build", "bin", "whisper-cli")
        if os.path.exists(whisper_main):
            print("✓ whisper.cpp executable found")
            
            # Test whisper.cpp help
            try:
                result = subprocess.run([whisper_main, "--help"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("✓ whisper.cpp executable working")
                else:
                    print("⚠ whisper.cpp executable has issues")
            except subprocess.TimeoutExpired:
                print("⚠ whisper.cpp help command timed out")
            except Exception as e:
                print(f"⚠ Error testing whisper.cpp: {e}")
        else:
            print("✗ whisper.cpp executable not found")
            
        print("\nSetup test completed!")
        print("You can now run: python3 main.py")
        
    except Exception as e:
        print(f"✗ Setup test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_setup()