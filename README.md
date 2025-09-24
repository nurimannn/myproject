# Kazakh Speech Recognition

This project integrates the Hugging Face model `akuzdeuov/whisper-base.kk` for Kazakh speech recognition using OpenAI's Whisper architecture.

## Features

- **Kazakh Speech Recognition**: Transcribe Kazakh speech from audio files
- **Language Selection**: Support for both Kazakh (`kk`) and Russian (`ru`) languages
- **Audio Chunking**: Automatic handling of long audio files (>30 seconds)
- **Multiple Audio Formats**: Support for WAV, MP3, FLAC, M4A, OGG files
- **Microphone Recording**: Record audio directly from microphone
- **Comprehensive Error Handling**: Robust error handling for various edge cases

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

#### Transcribe an audio file:
```bash
python main.py --audio path/to/audio.wav
```

#### Transcribe with specific language:
```bash
python main.py --audio path/to/audio.wav --lang kk  # Kazakh (default)
python main.py --audio path/to/audio.wav --lang ru  # Russian
```

#### Record from microphone and transcribe:
```bash
python main.py --record --duration 10  # Record for 10 seconds
```

#### Save transcription to file:
```bash
python main.py --audio path/to/audio.wav --output transcription.txt
```

#### Use different model:
```bash
python main.py --audio path/to/audio.wav --model openai/whisper-base
```

#### Verbose logging:
```bash
python main.py --audio path/to/audio.wav --verbose
```

### Programmatic Usage

```python
from main import KazakhSpeechTranscriber

# Initialize transcriber
transcriber = KazakhSpeechTranscriber()

# Transcribe audio file
result = transcriber.transcribe_kazakh("path/to/audio.wav")
print(f"Transcription: {result}")
```

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- M4A (.m4a)
- OGG (.ogg)

## Command Line Options

- `--audio, -a`: Path to audio file to transcribe
- `--lang, -l`: Language for transcription (`kk` for Kazakh, `ru` for Russian)
- `--record, -r`: Record audio from microphone
- `--duration, -d`: Recording duration in seconds (default: 10)
- `--output, -o`: Output file for transcription (default: output.txt)
- `--model, -m`: Hugging Face model name (default: akuzdeuov/whisper-base.kk)
- `--verbose, -v`: Enable verbose logging

## Error Handling

The implementation includes comprehensive error handling for:

- **Missing model**: Automatically downloads model if not found
- **Invalid audio files**: Validates file existence and format
- **Corrupted audio**: Checks for empty or corrupted audio data
- **Unsupported formats**: Validates audio format support
- **Network issues**: Handles model download failures
- **Memory issues**: Optimized for large audio files with chunking

## Audio Chunking

For audio files longer than 30 seconds, the system automatically:

1. Splits the audio into 30-second chunks
2. Transcribes each chunk separately
3. Merges the transcriptions into a final result

This ensures optimal performance and memory usage for long audio files.

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 4GB+ RAM (8GB+ recommended for large files)

## Dependencies

- transformers>=4.35.0
- datasets>=2.14.0
- torch>=2.0.0
- torchaudio>=2.0.0
- librosa>=0.10.0
- soundfile>=0.12.0
- numpy>=1.24.0
- scipy>=1.10.0

## Testing

Run the test script to verify the installation:

```bash
python test_transcription.py
```

## Examples

### Basic transcription:
```bash
python main.py --audio sample.wav
```

### Record and transcribe Kazakh speech:
```bash
python main.py --record --lang kk --duration 15
```

### Transcribe long audio file:
```bash
python main.py --audio long_audio.mp3 --output result.txt
```

## Troubleshooting

### Common Issues:

1. **Model download fails**: Ensure internet connection and sufficient disk space
2. **Audio file not found**: Check file path and permissions
3. **Memory errors**: Use shorter audio files or increase system RAM
4. **CUDA errors**: Install CUDA-compatible PyTorch or use CPU-only version

### Performance Tips:

- Use GPU acceleration when available
- For very long files, consider pre-splitting into smaller chunks
- Ensure audio quality is good for better transcription accuracy
- Use appropriate sample rates (16kHz recommended)

## License

This project uses the Hugging Face transformers library and the akuzdeuov/whisper-base.kk model. Please check their respective licenses for usage terms.