# Russian Speech Recognition with whisper.cpp

This project provides offline Russian speech recognition using whisper.cpp. It records audio from your microphone and converts it to text using the ggml-tiny.bin model.

## Features

- **Offline Recognition**: Works completely offline using whisper.cpp
- **Russian Language Support**: Optimized for Russian speech recognition
- **Easy to Use**: Simple `python main.py` command to run
- **Automatic Setup**: Downloads model and compiles whisper.cpp automatically
- **Microphone Recording**: Records audio directly from your microphone

## Requirements

- Linux (tested on Ubuntu/Debian)
- Python 3.7+
- Git
- Make
- Audio system (ALSA/PulseAudio)
- Microphone

## Installation

1. **Clone or download this project**:
   ```bash
   git clone <repository-url>
   cd russian-speech-recognition
   ```

2. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3-pip python3-dev portaudio19-dev git make build-essential
   
   # For other distributions, install equivalent packages
   ```

3. **Install Python dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

## Usage

### Quick Start

Simply run the main script:

```bash
python3 main.py
```

The script will:
1. Automatically download the ggml-tiny.bin model (if not present)
2. Compile whisper.cpp (if not already compiled)
3. Record 5 seconds of audio from your microphone
4. Process the audio and display the recognized Russian text

### Testing Setup

Before using with a microphone, you can test the setup:

```bash
# Test whisper.cpp compilation and model download
python3 test_setup.py

# Demo with generated test audio (no microphone needed)
python3 demo.py
```

### Manual Setup

If you prefer to set up manually:

```bash
# Run the automated setup script
./setup.sh

# Or install dependencies manually
sudo apt install python3-pyaudio python3-numpy git make build-essential
```

## How it Works

1. **Audio Recording**: Uses PyAudio to record 16kHz mono audio from your microphone
2. **Model Processing**: Uses whisper.cpp with the ggml-tiny.bin model for fast inference
3. **Language Detection**: Configured specifically for Russian language recognition
4. **Text Output**: Displays the recognized text in the terminal

## File Structure

```
.
├── main.py              # Main application script
├── demo.py              # Demo script with test audio
├── test_setup.py        # Setup verification script
├── setup.sh             # Automated setup script
├── requirements.txt     # System dependencies list
├── README.md           # This file
├── models/             # Downloaded models (auto-created)
│   └── ggml-tiny.bin   # Whisper model
├── whisper.cpp/        # Compiled whisper.cpp (auto-created)
└── audio.wav          # Temporary audio file (auto-created)
```

## Troubleshooting

### Audio Issues
- **No microphone detected**: Check your audio system and microphone permissions
- **Permission denied**: Run with `sudo` or add your user to the audio group:
  ```bash
  sudo usermod -a -G audio $USER
  ```

### Compilation Issues
- **Make not found**: Install build tools: `sudo apt install build-essential`
- **Git not found**: Install git: `sudo apt install git`

### Python Issues
- **PyAudio installation fails**: Install portaudio development headers first
- **Permission errors**: Use `pip3 install --user -r requirements.txt`

## Model Information

- **Model**: ggml-tiny.bin (39 MB)
- **Language**: Russian (ru)
- **Accuracy**: Good for general speech, optimized for speed
- **Speed**: Very fast inference on CPU

## Customization

You can modify the following parameters in `main.py`:

- **Recording duration**: Change the `duration` parameter in `record_audio()`
- **Audio quality**: Modify `RATE`, `CHUNK`, `FORMAT` in `record_audio()`
- **Model**: Change `model_path` to use a different model
- **Language**: Modify the `-l` parameter in the whisper.cpp command

## License

This project uses whisper.cpp which is licensed under MIT License.