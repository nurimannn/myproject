#!/usr/bin/env python3
"""
Example usage of the Kazakh speech transcription functionality
"""

from main import KazakhSpeechTranscriber

def example_transcription():
    """Example of how to use the transcribe_kazakh function"""
    
    # Initialize the transcriber
    print("Initializing Kazakh Speech Transcriber...")
    transcriber = KazakhSpeechTranscriber()
    
    # Example 1: Transcribe a Kazakh audio file
    audio_file = "kazakh_audio.wav"  # Replace with your audio file path
    
    try:
        print(f"Transcribing audio file: {audio_file}")
        transcription = transcriber.transcribe_kazakh(audio_file)
        
        print(f"\nTranscription Result:")
        print("-" * 50)
        print(transcription)
        print("-" * 50)
        
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file}")
        print("Please provide a valid audio file path.")
        
    except Exception as e:
        print(f"Error during transcription: {e}")

def example_with_custom_model():
    """Example using a different Whisper model"""
    
    print("\nExample with custom model...")
    
    # You can use other Whisper models
    custom_transcriber = KazakhSpeechTranscriber(model_name="openai/whisper-base")
    
    print("Custom model transcriber initialized successfully!")
    print("Note: You can use any Hugging Face Whisper model.")

if __name__ == "__main__":
    example_transcription()
    example_with_custom_model()