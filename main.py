import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import requests
import sounddevice as sd
from scipy.io.wavfile import write as wav_write


WHISPER_REPO = "https://github.com/ggerganov/whisper.cpp.git"
MODEL_FILE_MAP = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v1": "ggml-large-v1.bin",
    "large-v2": "ggml-large-v2.bin",
    "large-v3": "ggml-large-v3.bin",
}

# Primary and fallback URLs for model downloads
MODEL_URLS = [
    # Hugging Face canonical
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model_file}",
    # ggerganov ggml mirror (filenames differ for tiny/base)
    "https://ggml.ggerganov.com/{model_file}",
]


def run_command(command: List[str], cwd: Path | None = None) -> None:
    """Run a system command, streaming its output, raising on failure."""
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {' '.join(command)}")


def ensure_whisper_repo(repo_dir: Path) -> None:
    if repo_dir.exists() and (repo_dir / ".git").exists():
        return
    print("Cloning whisper.cpp repo...")
    run_command(["git", "clone", "--depth", "1", WHISPER_REPO, str(repo_dir)])


def ensure_whisper_built(repo_dir: Path) -> Path:
    binary_path = repo_dir / "main"
    if binary_path.exists():
        return binary_path
    print("Building whisper.cpp (this may take a minute)...")
    run_command(["make", "-j"], cwd=repo_dir)
    if not binary_path.exists():
        raise RuntimeError("Failed to build whisper.cpp 'main' binary")
    return binary_path


def download_file(url: str, dest_path: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100 // total
                            print(f"\rDownloading {dest_path.name}: {percent}%", end="")
            if total:
                print("\r", end="")
        print(f"Downloaded {dest_path}")
        return True
    except Exception as e:
        print(f"Download failed from {url}: {e}")
        return False


def ensure_model(repo_dir: Path, model_size: str) -> Path:
    if model_size not in MODEL_FILE_MAP:
        raise ValueError(f"Unsupported model '{model_size}'. Choose from: {', '.join(MODEL_FILE_MAP)}")
    model_file = MODEL_FILE_MAP[model_size]
    models_dir = repo_dir / "models"
    model_path = models_dir / model_file
    if model_path.exists():
        return model_path

    print(f"Model {model_file} not found. Attempting download...")
    # Try Python download first
    for url_template in MODEL_URLS:
        url = url_template.format(model_file=model_file)
        if download_file(url, model_path):
            return model_path

    # Fallback to whisper.cpp's download script if available
    script_sh = models_dir / "download-ggml-model.sh"
    script_ps1 = models_dir / "download-ggml-model.ps1"
    if script_sh.exists() or script_ps1.exists():
        print("Falling back to whisper.cpp download script...")
        size_arg = model_size.split("-")[0] if model_size.startswith("large") else model_size
        try:
            run_command(["bash", str(script_sh), size_arg], cwd=models_dir)
        except Exception:
            pass
        if model_path.exists():
            return model_path

    raise RuntimeError(
        f"Could not download model file {model_file}. Check your network and try again."
    )


def record_audio(
    output_wav: Path,
    duration_seconds: float,
    sample_rate_hz: int = 16000,
    channels: int = 1,
) -> None:
    print(f"Recording {duration_seconds:.1f}s of audio @ {sample_rate_hz} Hz, mono...")
    try:
        frames = int(duration_seconds * sample_rate_hz)
        audio = sd.rec(frames, samplerate=sample_rate_hz, channels=channels, dtype="int16")
        sd.wait()
    except Exception as e:
        raise RuntimeError(
            "Microphone recording failed. Ensure your system audio input is configured and accessible."
        ) from e

    wav_write(str(output_wav), sample_rate_hz, audio)
    print(f"Saved recording to {output_wav}")


def transcribe(
    whisper_binary: Path,
    model_path: Path,
    audio_wav: Path,
    language: str = "ru",
) -> None:
    if not audio_wav.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_wav}")
    print("Running transcription with whisper.cpp...")
    # Use -np to reduce progress spam; -l to set language
    command = [
        str(whisper_binary),
        "-m",
        str(model_path),
        "-f",
        str(audio_wav),
        "-l",
        language,
        "-np",
    ]
    run_command(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record audio and transcribe with whisper.cpp")
    parser.add_argument("--duration", type=float, default=float(os.environ.get("DURATION_SECONDS", 5)), help="Recording duration in seconds")
    parser.add_argument("--samplerate", type=int, default=int(os.environ.get("SAMPLE_RATE", 16000)), help="Recording sample rate (Hz)")
    parser.add_argument("--language", type=str, default=os.environ.get("LANGUAGE", "ru"), help="Target language code (e.g., ru, en)")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL", "tiny"), help="Model size: tiny, base, small, medium, large-v3")
    parser.add_argument("--output", type=str, default=os.environ.get("OUTPUT_WAV", "audio.wav"), help="Output WAV filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    whisper_dir = project_root / "whisper.cpp"

    # 1) Ensure repo cloned and built
    ensure_whisper_repo(whisper_dir)
    whisper_binary = ensure_whisper_built(whisper_dir)

    # 2) Ensure model file present (auto-download)
    model_path = ensure_model(whisper_dir, args.model)

    # 3) Record audio
    output_wav = project_root / args.output
    record_audio(output_wav=output_wav, duration_seconds=args.duration, sample_rate_hz=args.samplerate)

    # 4) Transcribe in Russian (default)
    transcribe(whisper_binary=whisper_binary, model_path=model_path, audio_wav=output_wav, language=args.language)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

