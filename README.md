# Whisper.cpp Russian Transcriber (ggml-tiny)

This project records audio from your microphone to `audio.wav` and transcribes it to Russian text using `whisper.cpp` with the `ggml-tiny.bin` model.

## Quick Start

1) Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

2) Run the script. It will automatically:
- Clone and build `whisper.cpp` if missing
- Download `ggml-tiny.bin` if missing
- Record audio from your microphone
- Print recognized Russian text

```bash
python main.py
```

By default it records for 5 seconds at 16 kHz mono into `audio.wav`.

## Options

```bash
python main.py \
  --duration 8 \
  --samplerate 16000 \
  --language ru \
  --model tiny \
  --output audio.wav
```

You can also use environment variables: `DURATION_SECONDS`, `SAMPLE_RATE`, `LANGUAGE`, `MODEL`, `OUTPUT_WAV`.

## Requirements

- Linux with microphone access
- `git` and `make` available in PATH to build `whisper.cpp`

If `git`/`make` are unavailable, install them via your distro package manager.

## Notes

- Models are stored in `whisper.cpp/models/`. We attempt to download from Hugging Face first and fall back to the repo scripts.
- To force rebuild of `whisper.cpp`, delete the `whisper.cpp` directory and rerun.
