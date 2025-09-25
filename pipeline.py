from __future__ import annotations

from typing import Literal, Optional

from postprocess import postprocess_text_ru, postprocess_text_kk


Language = Literal["ru", "kk", "en", "other"]


def postprocess_transcription(text: str, lang: Language, use_lm: bool = False) -> str:
    if not text:
        return text
    if lang == "ru":
        return postprocess_text_ru(text, use_language_model=use_lm)
    if lang == "kk":
        return postprocess_text_kk(text, use_language_model=use_lm)
    return text


def save_transcription(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def process_and_save(transcript: str, lang: Language, out_path: str, use_lm: bool = False) -> str:
    processed = postprocess_transcription(transcript, lang, use_lm=use_lm)
    save_transcription(processed, out_path)
    return processed

