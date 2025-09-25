import re
from typing import Dict, Iterable, List


def normalize_whitespace(text: str) -> str:
    if not text:
        return text
    return re.sub(r"\s+", " ", text).strip()


def apply_replacements(text: str, replacements: Dict[str, str]) -> str:
    if not text:
        return text
    if not replacements:
        return text
    # Whole-word replacements first
    def repl_word(match: re.Match) -> str:
        word = match.group(0)
        lower = word.lower()
        if lower in replacements:
            replacement = replacements[lower]
            if word.istitle():
                return replacement.capitalize()
            if word.isupper():
                return replacement.upper()
            return replacement
        return word

    text = re.sub(r"\b\w+\b", repl_word, text, flags=re.UNICODE)
    # Phrase-level replacements (case-insensitive)
    for wrong, right in replacements.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE | re.UNICODE)
    return text


def top_candidate(candidates: Iterable[str]) -> str:
    for item in candidates:
        return item
    return ""


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

