#!/usr/bin/env python3
"""
Russian Speech Recognition using whisper.cpp
Records audio from microphone and performs offline speech recognition.
"""

import os
import sys
import subprocess
import pyaudio
import wave
import urllib.request
import hashlib
from pathlib import Path
import tempfile
import shutil

class WhisperRecognition:
    def __init__(self):
        self.model_path = "models/ggml-tiny.bin"
        self.audio_file = "audio.wav"
        self.whisper_cpp_path = "whisper.cpp"
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs("models", exist_ok=True)
        os.makedirs("whisper.cpp", exist_ok=True)
        
    def download_model(self):
        """Download ggml-tiny.bin model if not present"""
        if os.path.exists(self.model_path):
            print(f"Model already exists: {self.model_path}")
            return
            
        print("Downloading ggml-tiny.bin model...")
        model_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
        
        try:
            urllib.request.urlretrieve(model_url, self.model_path)
            print(f"Model downloaded successfully: {self.model_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            sys.exit(1)
    
    def setup_whisper_cpp(self):
        """Setup whisper.cpp if not present"""
        if os.path.exists(os.path.join(self.whisper_cpp_path, "build", "bin", "whisper-cli")):
            print("whisper.cpp already compiled")
            return
            
        print("Setting up whisper.cpp...")
        
        # Clone whisper.cpp repository
        if not os.path.exists(os.path.join(self.whisper_cpp_path, ".git")):
            subprocess.run([
                "git", "clone", 
                "https://github.com/ggerganov/whisper.cpp.git", 
                self.whisper_cpp_path
            ], check=True)
        
        # Compile whisper.cpp
        os.chdir(self.whisper_cpp_path)
        subprocess.run(["make"], check=True)
        os.chdir("..")
        
        print("whisper.cpp compiled successfully")
    
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        print(f"Recording audio for {duration} seconds...")
        print("Speak now...")
        
        # Audio parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        frames = []
        
        # Record for specified duration
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("Recording finished")
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save audio to file
        wf = wave.open(self.audio_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setrate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Audio saved to {self.audio_file}")
    
    def recognize_speech(self):
        """Run speech recognition using whisper.cpp"""
        print("Running speech recognition...")
        
        # Path to whisper.cpp main executable
        whisper_main = os.path.join(self.whisper_cpp_path, "build", "bin", "whisper-cli")
        
        # Run whisper.cpp with Russian language
        cmd = [
            whisper_main,
            "--model", self.model_path,
            "--file", self.audio_file,
            "--language", "ru",  # Russian language
            "--threads", "4",   # Number of threads
            "--no-timestamps"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            recognized_text = result.stdout.strip()
            
            if recognized_text:
                print(f"\nRecognized Russian text: {recognized_text}")
                return recognized_text
            else:
                print("No speech detected or recognition failed")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"Error running whisper.cpp: {e}")
            print(f"stderr: {e.stderr}")
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.audio_file):
            os.remove(self.audio_file)
            print(f"Cleaned up {self.audio_file}")

def main():
    """Main function"""
    print("Russian Speech Recognition using whisper.cpp")
    print("=" * 50)
    
    # Check if running on Linux
    if sys.platform != "linux":
        print("Warning: This script is optimized for Linux. Some features may not work on other platforms.")
    
    try:
        # Initialize recognizer
        recognizer = WhisperRecognition()
        
        # Setup whisper.cpp
        recognizer.setup_whisper_cpp()
        
        # Download model
        recognizer.download_model()
        
        # Record audio
        recognizer.record_audio(duration=5)
        
        # Recognize speech
        text = recognizer.recognize_speech()
        
        if text:
            print(f"\nFinal result: {text}")
        else:
            print("Recognition failed. Please try again.")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'recognizer' in locals():
            recognizer.cleanup()

if __name__ == "__main__":
    main()