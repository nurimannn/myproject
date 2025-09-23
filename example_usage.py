#!/usr/bin/env python3
"""
Example usage of the Russian Speech Recognition system
"""

from main import WhisperRecognition
import time

def example_usage():
    """Example of how to use the WhisperRecognition class"""
    print("Russian Speech Recognition - Example Usage")
    print("=" * 50)
    
    # Initialize the recognizer
    recognizer = WhisperRecognition()
    
    # Setup (this will download model and compile whisper.cpp if needed)
    print("Setting up...")
    recognizer.setup_whisper_cpp()
    recognizer.download_model()
    
    print("\nExample 1: Record and recognize speech")
    print("This will record 5 seconds of audio from your microphone")
    print("Press Enter to start recording...")
    input()
    
    # Record audio
    recognizer.record_audio(duration=5)
    
    # Recognize speech
    text = recognizer.recognize_speech()
    
    if text:
        print(f"Recognized text: {text}")
    else:
        print("No speech detected")
    
    # Cleanup
    recognizer.cleanup()
    
    print("\nExample 2: Process existing audio file")
    print("You can also process existing audio files by modifying the audio_file path")
    print("recognizer.audio_file = 'path/to/your/audio.wav'")
    print("text = recognizer.recognize_speech()")

if __name__ == "__main__":
    example_usage()