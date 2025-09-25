from __future__ import annotations

import os
import regex as re
from typing import Dict, List, Optional, Tuple

try:
    from symspellpy import SymSpell, Verbosity
except Exception:  # pragma: no cover - optional
    SymSpell = None  # type: ignore
    Verbosity = None  # type: ignore


_SYM_SPELL_KK: Optional["SymSpell"] = None


_COMMON_REPLACEMENTS_KK: Dict[str, str] = {
    # Examples of Latin to Cyrillic often-mistyped or ASR-like errors
    "bugyn": "бүгін",
    "maktap": "мектеп",
    "sagat": "сағат",
    "balalar": "балалар",
    "joly": "жолы",
}


_WORD_RE = re.compile(r"\p{L}+[\p{L}\-']*")


def _match_case_like(sample: str, target: str) -> str:
    if sample.isupper():
        return target.upper()
    if sample[:1].isupper():
        return target[:1].upper() + target[1:]
    return target.lower()


def _replace_common_errors(text: str, replacements: Dict[str, str]) -> str:
    for wrong, right in replacements.items():
        pattern = re.compile(rf"(?<!\p{{L}}){re.escape(wrong)}(?!\p{{L}})", flags=re.IGNORECASE)

        def _sub(m: re.Match) -> str:
            return _match_case_like(m.group(0), right)

        text = pattern.sub(_sub, text)
    return text


def _load_symspell_kk(dictionary_path: Optional[str] = None, max_edit_distance: int = 2, prefix_length: int = 7) -> Optional["SymSpell"]:
    global _SYM_SPELL_KK
    if _SYM_SPELL_KK is not None:
        return _SYM_SPELL_KK

    if SymSpell is None:
        return None

    dictionary_path = (
        dictionary_path
        or os.getenv("KK_SYMSPELL_DICT")
        or os.path.join(os.path.dirname(__file__), "dicts", "kk_frequency_dictionary.txt")
    )

    sym = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=prefix_length)
    if os.path.exists(dictionary_path):
        loaded = sym.load_dictionary(dictionary_path, term_index=0, count_index=1, separator=" ")
        if not loaded:
            return None
        _SYM_SPELL_KK = sym
        return sym
    return None


def _symspell_candidates(word: str, sym: "SymSpell", max_edit_distance: int) -> List[str]:
    suggestions = sym.lookup(word.lower(), Verbosity.TOP, max_edit_distance=max_edit_distance, include_unknown=True)
    seen: set[str] = set()
    out: List[str] = []
    for s in suggestions:
        term = str(s.term)
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out[:3]


def _correct_with_symspell(text: str, sym: "SymSpell", max_edit_distance: int = 2, min_len: int = 3) -> Tuple[str, List[Tuple[int, List[str]]]]:
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    for m in _WORD_RE.finditer(text):
        tokens.append(m.group(0))
        spans.append(m.span())

    if not tokens:
        return text, []

    text_chars = list(text)
    ambiguous: List[Tuple[int, List[str]]] = []

    for idx, (token, (start, end)) in enumerate(zip(tokens, spans)):
        letters_only = re.fullmatch(_WORD_RE, token) is not None
        if not letters_only or len(token) < min_len:
            continue

        candidates = _symspell_candidates(token, sym, max_edit_distance)
        if not candidates:
            continue
        best = candidates[0]
        if best.lower() != token.lower():
            replacement = _match_case_like(token, best)
            text_chars[start:end] = list(replacement)
            ambiguous.append((idx, candidates))

    return "".join(text_chars), ambiguous


def _build_sentence_candidates(base_text: str, tokens_and_spans: List[Tuple[str, Tuple[int, int]]], ambiguous: List[Tuple[int, List[str]]], limit: int = 5) -> List[str]:
    if not ambiguous:
        return [base_text]
    variants: List[str] = [base_text]
    text_chars = list(base_text)
    for idx, candidates in ambiguous[:3]:
        if len(candidates) < 2:
            continue
        token, (start, end) = tokens_and_spans[idx]
        alt = _match_case_like(token, candidates[1])
        new_chars = text_chars.copy()
        new_chars[start:end] = list(alt)
        variants.append("".join(new_chars))
        if len(variants) >= limit:
            break
    seen: set[str] = set()
    uniq: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def postprocess_text_kk(text: str, *, dictionary_path: Optional[str] = None, enable_lm_rerank: Optional[bool] = None, lm_model_name: Optional[str] = None, max_edit_distance: int = 2) -> str:
    """Postprocess Kazakh text from ASR.

    Steps:
      1) Apply common replacements (incl. Latin-to-Cyrillic frequent forms)
      2) Spell-correct via SymSpell if dictionary available
      3) Optional LM reranking with a Kazakh-capable model

    Env:
      - KK_SYMSPELL_DICT for frequency dictionary
      - ENABLE_LM_RERANK_KK = '1' to enable LM rerank
      - KK_LM_MODEL_NAME model id/path for LM rerank
    """
    if not text:
        return text

    text = _replace_common_errors(text, _COMMON_REPLACEMENTS_KK)

    sym = _load_symspell_kk(dictionary_path=dictionary_path, max_edit_distance=max_edit_distance)
    candidates: List[str] = []
    if sym is not None:
        tokens: List[str] = []
        spans: List[Tuple[int, int]] = []
        for m in _WORD_RE.finditer(text):
            tokens.append(m.group(0))
            spans.append(m.span())

        corrected, ambiguous = _correct_with_symspell(text, sym, max_edit_distance=max_edit_distance)
        candidates.append(corrected)
        tokens_and_spans = list(zip(tokens, spans))
        candidates.extend(_build_sentence_candidates(corrected, tokens_and_spans, ambiguous))
    else:
        candidates.append(text)

    if text not in candidates:
        candidates.insert(0, text)
    seen: set[str] = set()
    unique_candidates: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    if enable_lm_rerank is None:
        enable_lm_rerank = os.getenv("ENABLE_LM_RERANK_KK", "0") == "1"

    if enable_lm_rerank and len(unique_candidates) > 1:
        try:
            from .lm import rank_by_causal_lm
        except Exception:
            return unique_candidates[0]
        model_name = lm_model_name or os.getenv("KK_LM_MODEL_NAME", "kaznlp/kazbert")
        best = rank_by_causal_lm(unique_candidates, model_name)
        return best

    return unique_candidates[0]

