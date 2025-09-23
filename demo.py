#!/usr/bin/env python3
"""
Demo script for Russian Speech Recognition with whisper.cpp
Creates a test audio file instead of recording from microphone
"""

import os
import sys
import numpy as np
import wave
from main import WhisperRecognition

def create_test_audio(filename="test_audio.wav", duration=3, frequency=440):
    """Create a simple test audio file (sine wave)"""
    print(f"Creating test audio file: {filename}")
    
    # Audio parameters
    sample_rate = 16000
    samples = int(sample_rate * duration)
    
    # Generate sine wave
    t = np.linspace(0, duration, samples, False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✓ Test audio created: {filename}")

def demo_recognition():
    """Demo the recognition system"""
    print("Russian Speech Recognition Demo")
    print("=" * 40)
    
    try:
        recognizer = WhisperRecognition()
        
        # Setup
        print("Setting up whisper.cpp...")
        recognizer.setup_whisper_cpp()
        
        print("Downloading model...")
        recognizer.download_model()
        
        # Create test audio
        test_audio = "test_audio.wav"
        create_test_audio(test_audio)
        
        # Temporarily replace audio file for testing
        original_audio = recognizer.audio_file
        recognizer.audio_file = test_audio
        
        print("Running recognition on test audio...")
        text = recognizer.recognize_speech()
        
        if text:
            print(f"✓ Recognition result: {text}")
        else:
            print("⚠ No text recognized (expected for sine wave)")
        
        # Cleanup
        if os.path.exists(test_audio):
            os.remove(test_audio)
            print(f"✓ Cleaned up {test_audio}")
        
        print("\nDemo completed!")
        print("To use with real microphone, run: python3 main.py")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    demo_recognition()