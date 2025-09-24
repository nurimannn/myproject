#!/usr/bin/env python3
"""
Test script for Kazakh speech transcription
"""

import os
import numpy as np
import soundfile as sf
from main import KazakhSpeechTranscriber

def create_test_audio():
    """Create a simple test audio file with some sine wave"""
    # Generate a simple sine wave (440 Hz for 3 seconds)
    sample_rate = 16000
    duration = 3.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, audio.shape)
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Save as WAV file
    test_file = "test_audio.wav"
    sf.write(test_file, audio, sample_rate)
    
    return test_file

def test_transcription():
    """Test the transcription functionality"""
    print("Testing Kazakh Speech Transcription...")
    
    # Create test audio
    print("Creating test audio file...")
    test_audio = create_test_audio()
    
    try:
        # Initialize transcriber
        print("Initializing transcriber...")
        transcriber = KazakhSpeechTranscriber()
        
        # Test transcription
        print("Testing transcription...")
        result = transcriber.transcribe_kazakh(test_audio)
        
        print(f"Transcription result: '{result}'")
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_audio):
            os.remove(test_audio)
            print(f"Cleaned up test file: {test_audio}")

if __name__ == "__main__":
    test_transcription()