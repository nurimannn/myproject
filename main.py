import os
import sys
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


WORKSPACE_DIR = Path(__file__).resolve().parent
WHISPER_CPP_REPO = WORKSPACE_DIR / "whisper.cpp"
WHISPER_MAIN_BIN = WHISPER_CPP_REPO / "main"
MODELS_DIR = WHISPER_CPP_REPO / "models"
MODEL_FILENAME = "ggml-tiny.bin"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
AUDIO_PATH = WORKSPACE_DIR / "audio.wav"


def run(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and stream its output live to the console."""
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output live
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    ret = process.wait()
    if check and ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return subprocess.CompletedProcess(cmd, ret)


def ensure_whisper_cpp() -> None:
    """Clone and build whisper.cpp if not already built."""
    if not WHISPER_CPP_REPO.exists():
        print("Cloning whisper.cpp repository...")
        run(["git", "clone", "https://github.com/ggerganov/whisper.cpp.git", str(WHISPER_CPP_REPO)])

    if not WHISPER_MAIN_BIN.exists():
        print("Building whisper.cpp (this may take a few minutes)...")
        # Prefer plain make which builds the default targets including 'main'
        run(["make", "-j"], cwd=WHISPER_CPP_REPO)

    if not WHISPER_MAIN_BIN.exists():
        raise RuntimeError("Failed to build whisper.cpp 'main' binary.")


def download_file(url: str, dest_path: Path) -> None:
    import requests

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"Downloading model: {pct}%\r", end="")
    print("")


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    print(f"Downloading model {MODEL_FILENAME}...")
    # Official mirror for whisper.cpp models on Hugging Face
    url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin?download=true"
    download_file(url, MODEL_PATH)
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 10_000_000:
        raise RuntimeError("Model download seems incomplete. Please re-run.")


def record_audio_to_wav(output_path: Path, samplerate: int = 16000, channels: int = 1, duration_sec: Optional[float] = None) -> None:
    """
    Record microphone input to a WAV file.
    - If duration_sec is provided, records for that many seconds.
    - Otherwise, press Ctrl+C to stop recording.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using input devices:")
    try:
        devices = sd.query_devices()
        default_device = sd.default.device
        for idx, dev in enumerate(devices):
            mark = " (default)" if isinstance(default_device, (list, tuple)) and idx == default_device[0] else ""
            print(f" - [{idx}] {dev['name']}{mark}")
    except Exception:
        pass

    print(f"Recording at {samplerate} Hz, {channels} channel(s)")

    if duration_sec is not None:
        num_frames = int(duration_sec * samplerate)
        print(f"Recording for {duration_sec} seconds...")
        audio = sd.rec(frames=num_frames, samplerate=samplerate, channels=channels, dtype="float32")
        sd.wait()
        sf.write(str(output_path), audio, samplerate)
        print(f"Saved to {output_path}")
        return

    print("Recording... Press Ctrl+C to stop.")
    recorded_chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):  # type: ignore[no-redef]
        if status:
            print(status, file=sys.stderr)
        recorded_chunks.append(indata.copy())

    try:
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype="float32", callback=callback):
            while not stop_event.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    if recorded_chunks:
        audio = np.concatenate(recorded_chunks, axis=0)
        sf.write(str(output_path), audio, samplerate)
        print(f"Saved to {output_path}")
    else:
        print("No audio captured.")


def run_recognition_ru(audio_path: Path) -> str:
    ensure_whisper_cpp()
    ensure_model()

    cmd = [
        str(WHISPER_MAIN_BIN),
        "-m", str(MODEL_PATH),
        "-f", str(audio_path),
        "-l", "ru",
        "-nt",  # no timestamps in output lines
    ]

    print("Running recognition with whisper.cpp...")
    result = subprocess.run(cmd, cwd=str(WHISPER_CPP_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    output = result.stdout
    # The 'main' binary prints recognized text lines prefixed by timestamps unless -nt.
    # Just print the raw output; also try to extract the final line as recognized text.
    print(output)

    # Heuristic: last non-empty line is the transcript.
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return ""


def parse_duration_arg() -> Optional[float]:
    if len(sys.argv) >= 2:
        try:
            return float(sys.argv[1])
        except ValueError:
            pass
    return None


def main() -> None:
    duration = parse_duration_arg()
    if duration is None:
        print("Tip: pass duration in seconds, e.g. 'python main.py 8' to record 8 seconds.")
        print("Otherwise, press Ctrl+C to stop recording.")

    # Record
    record_audio_to_wav(AUDIO_PATH, samplerate=16000, channels=1, duration_sec=duration)
    if not AUDIO_PATH.exists() or AUDIO_PATH.stat().st_size == 0:
        print("No audio to transcribe.")
        sys.exit(1)

    # Recognize
    transcript = run_recognition_ru(AUDIO_PATH)
    if transcript:
        print("Recognized (RU):")
        print(transcript)
    else:
        print("No text recognized.")


if __name__ == "__main__":
    main()

