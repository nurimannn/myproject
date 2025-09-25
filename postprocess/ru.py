from __future__ import annotations

from typing import Dict, List, Optional

from .utils import apply_replacements, normalize_whitespace, tokenize_words


DEFAULT_RU_REPLACEMENTS: Dict[str, str] = {
    "славо": "слово",
    "превет": "привет",
    "зделать": "сделать",
    "щас": "сейчас",
    "чё": "что",
}


class RussianSpellChecker:
    def __init__(self) -> None:
        self._method = None
        self._spell = None
        self._symspell = None
        self._morph = None
        # Try pyspellchecker
        try:
            from spellchecker import SpellChecker  # type: ignore

            self._spell = SpellChecker(language=None, case_sensitive=False)
            # Load a minimal Russian frequency dictionary if available
            # Users can add words via add() at runtime
            self._method = "pyspellchecker"
        except Exception:
            self._spell = None
        # Try symspellpy
        if self._spell is None:
            try:
                from symspellpy.symspellpy import SymSpell, Verbosity  # type: ignore

                sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                self._symspell = sym
                self._method = "symspell"
                self._verbosity = Verbosity.CLOSEST
            except Exception:
                self._symspell = None
        # Optional pymorphy2 for lemma-informed suggestions
        try:
            import pymorphy2  # type: ignore

            self._morph = pymorphy2.MorphAnalyzer()
        except Exception:
            self._morph = None

    def correct_tokens(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return tokens
        corrected: List[str] = []
        for token in tokens:
            if not token.isalpha() or len(token) <= 2:
                corrected.append(token)
                continue
            lower = token.lower()
            suggestion: Optional[str] = None
            if self._method == "pyspellchecker" and self._spell is not None:
                try:
                    candidates = list(self._spell.candidates(lower))
                    if candidates:
                        suggestion = candidates[0]
                except Exception:
                    suggestion = None
            elif self._method == "symspell" and self._symspell is not None:
                try:
                    suggestions = self._symspell.lookup(lower, self._verbosity, max_edit_distance=2)
                    if suggestions:
                        suggestion = suggestions[0].term
                except Exception:
                    suggestion = None

            if suggestion is None:
                corrected.append(token)
                continue

            # Use pymorphy2 to keep consistent casing and valid lemma if possible
            if self._morph is not None:
                try:
                    parse = self._morph.parse(suggestion)
                    if parse:
                        suggestion = parse[0].normal_form
                except Exception:
                    pass

            if token.istitle():
                corrected.append(suggestion.capitalize())
            elif token.isupper():
                corrected.append(suggestion.upper())
            else:
                corrected.append(suggestion)
        return corrected


def _optional_russian_lm_rerank(text: str) -> str:
    try:
        from transformers import pipeline  # type: ignore

        # Small, widely available Russian model for fill-mask/re-ranking context
        nlp = pipeline("fill-mask", model="cointegrated/rubert-tiny2")
        # Simple heuristic: if there is no [MASK], we skip LM step
        if "[MASK]" not in text:
            return text
        preds = nlp(text)
        if isinstance(preds, list) and preds:
            return preds[0]["sequence"].replace("<s>", "").replace("</s>", "").strip()
    except Exception:
        return text
    return text


def postprocess_text_ru(
    text: str,
    custom_replacements: Optional[Dict[str, str]] = None,
    use_language_model: bool = False,
) -> str:
    if not text:
        return text
    repl = dict(DEFAULT_RU_REPLACEMENTS)
    if custom_replacements:
        repl.update({k.lower(): v for k, v in custom_replacements.items()})
    text = apply_replacements(text, repl)
    tokens = tokenize_words(text)
    checker = RussianSpellChecker()
    tokens = checker.correct_tokens(tokens)
    text = "".join(tokens)
    text = normalize_whitespace(text)
    if use_language_model:
        text = _optional_russian_lm_rerank(text)
        text = normalize_whitespace(text)
    return text

