#!/usr/bin/env python3
"""
Kazakh Speech Recognition using Hugging Face Whisper Model
Integrates akuzdeuov/whisper-base.kk for Kazakh speech transcription
"""

import argparse
import os
import sys
import wave
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import logging

import torch
import torchaudio
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import logging as transformers_logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress transformers warnings
transformers_logging.set_verbosity_error()


class KazakhSpeechTranscriber:
    """Kazakh speech transcription using Whisper model"""
    
    def __init__(self, model_name: str = "akuzdeuov/whisper-base.kk"):
        """Initialize the transcriber with the specified model"""
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_model(self) -> None:
        """Load the Whisper model and processor"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load model {self.model_name}: {e}")
    
    def validate_audio_file(self, audio_path: str) -> None:
        """Validate that the audio file exists and is in a supported format"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check file extension
        supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        file_ext = Path(audio_path).suffix.lower()
        if file_ext not in supported_formats:
            raise ValueError(f"Unsupported audio format: {file_ext}. Supported formats: {supported_formats}")
        
        # Try to load the file to check if it's valid
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            if len(audio) == 0:
                raise ValueError("Audio file is empty or corrupted")
            logger.info(f"Audio file validated: {len(audio)} samples, {sr} Hz")
        except Exception as e:
            raise ValueError(f"Invalid audio file: {e}")
    
    def load_audio(self, audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file"""
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # Ensure audio is mono
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")
    
    def chunk_audio(self, audio: np.ndarray, sr: int, chunk_duration: float = 30.0) -> List[np.ndarray]:
        """Split audio into chunks for processing long files"""
        chunk_samples = int(chunk_duration * sr)
        chunks = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks
    
    def transcribe_chunk(self, audio_chunk: np.ndarray, sr: int, language: str = "kk") -> str:
        """Transcribe a single audio chunk"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Process audio with the processor
            inputs = self.processor(audio_chunk, sampling_rate=sr, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    max_length=448,
                    num_beams=5,
                    language=language,
                    task="transcribe"
                )
            
            # Decode the transcription
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
            return ""
    
    def transcribe_kazakh(self, audio_path: str) -> str:
        """
        Main function to transcribe Kazakh speech from audio file
        
        Args:
            audio_path (str): Path to the audio file (.wav, .mp3, .flac, .m4a, .ogg)
            
        Returns:
            str: Transcribed text in Kazakh
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Validate audio file
            self.validate_audio_file(audio_path)
            
            # Load audio
            audio, sr = self.load_audio(audio_path)
            duration = len(audio) / sr
            
            logger.info(f"Processing audio: {duration:.2f} seconds")
            
            # Check if audio is longer than 30 seconds
            if duration > 30:
                logger.info("Audio longer than 30 seconds, using chunking")
                chunks = self.chunk_audio(audio, sr)
                transcriptions = []
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    transcription = self.transcribe_chunk(chunk, sr, language="kk")
                    if transcription:
                        transcriptions.append(transcription)
                
                # Merge transcriptions
                final_transcription = " ".join(transcriptions)
            else:
                # Process entire audio at once
                final_transcription = self.transcribe_chunk(audio, sr, language="kk")
            
            return final_transcription
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise


class AudioRecorder:
    """Simple audio recorder for microphone input"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
    
    def record_audio(self, duration: float = 10.0, output_path: str = "recorded_audio.wav") -> str:
        """Record audio from microphone"""
        try:
            import pyaudio
        except ImportError:
            raise ImportError("pyaudio is required for microphone recording. Install with: pip install pyaudio")
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info(f"Recording for {duration} seconds...")
            frames = []
            
            for _ in range(int(self.sample_rate * duration / self.chunk_size)):
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # Save to file
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            logger.info(f"Audio saved to: {output_path}")
            return output_path
            
        finally:
            p.terminate()


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Kazakh Speech Recognition using Whisper")
    parser.add_argument("--audio", "-a", type=str, help="Path to audio file to transcribe")
    parser.add_argument("--lang", "-l", choices=["kk", "ru"], default="kk", 
                       help="Language for transcription (kk=Kazakh, ru=Russian)")
    parser.add_argument("--record", "-r", action="store_true", 
                       help="Record audio from microphone")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                       help="Recording duration in seconds (default: 10)")
    parser.add_argument("--output", "-o", type=str, default="output.txt",
                       help="Output file for transcription (default: output.txt)")
    parser.add_argument("--model", "-m", type=str, default="akuzdeuov/whisper-base.kk",
                       help="Hugging Face model name (default: akuzdeuov/whisper-base.kk)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        transcriber = KazakhSpeechTranscriber(model_name=args.model)
        
        if args.record:
            # Record audio from microphone
            recorder = AudioRecorder()
            audio_path = recorder.record_audio(duration=args.duration)
            logger.info(f"Recording completed: {audio_path}")
        elif args.audio:
            # Use provided audio file
            audio_path = args.audio
        else:
            logger.error("Please provide either --audio file path or --record to record from microphone")
            sys.exit(1)
        
        # Transcribe audio
        logger.info(f"Starting transcription (language: {args.lang})")
        transcription = transcriber.transcribe_kazakh(audio_path)
        
        # Output results
        print(f"\nTranscription ({args.lang}):")
        print("-" * 50)
        print(transcription)
        print("-" * 50)
        
        # Save to file
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        logger.info(f"Transcription saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()