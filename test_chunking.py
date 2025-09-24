#!/usr/bin/env python3
"""
Test script to demonstrate audio chunking functionality for long files
"""

import os
import numpy as np
import soundfile as sf
from main import KazakhSpeechTranscriber

def create_long_test_audio():
    """Create a test audio file longer than 30 seconds"""
    # Generate a longer sine wave (35 seconds)
    sample_rate = 16000
    duration = 35.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some variation to make it more interesting
    audio += 0.1 * np.sin(2 * np.pi * frequency * 2 * t)
    
    # Add noise
    noise = np.random.normal(0, 0.05, audio.shape)
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Save as WAV file
    test_file = "long_test_audio.wav"
    sf.write(test_file, audio, sample_rate)
    
    print(f"Created test audio file: {test_file}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    
    return test_file

def test_chunking():
    """Test the chunking functionality"""
    print("Testing Audio Chunking for Long Files...")
    
    # Create long test audio
    test_audio = create_long_test_audio()
    
    try:
        # Initialize transcriber
        print("\nInitializing transcriber...")
        transcriber = KazakhSpeechTranscriber()
        
        # Test transcription with chunking
        print("Testing transcription with automatic chunking...")
        result = transcriber.transcribe_kazakh(test_audio)
        
        print(f"\nTranscription result:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        print("Chunking test completed successfully!")
        
    except Exception as e:
        print(f"Chunking test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(test_audio):
            os.remove(test_audio)
            print(f"\nCleaned up test file: {test_audio}")

if __name__ == "__main__":
    test_chunking()