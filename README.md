## Whisper.cpp Russian speech-to-text (ggml-tiny)

This small project records audio from your microphone to `audio.wav`, then runs transcription using `whisper.cpp` with the `ggml-tiny.bin` model and prints recognized Russian text.

### Prerequisites
- Python 3.10+
- Build tools: `git`, `make`, a C/C++ compiler (e.g. `gcc`, `clang`)
- Microphone input configured on your system

On Debian/Ubuntu you may need:
```bash
sudo apt update && sudo apt install -y git build-essential python3-dev portaudio19-dev
```

### Install Python dependencies
```bash
python -m pip install -r requirements.txt
```

If `sounddevice` complains about PortAudio, install `portaudio`/`portaudio19-dev` per your distro.

### Usage
Run without args to start recording and stop with Ctrl+C:
```bash
python main.py
```

Or pass duration in seconds to auto-stop recording, e.g. 8 seconds:
```bash
python main.py 8
```

The first run will:
- Clone and build `whisper.cpp` locally
- Download `models/ggml-tiny.bin` automatically

Output shows the `whisper.cpp` build logs the first time, then the recognized Russian text.

Audio is saved to `audio.wav` in the project root.

### Notes
- The script forces 16 kHz mono recording which is suitable for Whisper.
- You can change the model by editing `MODEL_FILENAME` in `main.py` after the first run.
- If building `whisper.cpp` fails, ensure `make` and a C/C++ toolchain are installed.

