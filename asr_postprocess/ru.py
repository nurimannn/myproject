from typing import Dict, List, Optional, Tuple, Callable

from .utils import apply_replacements, rerank_with_language_model


# Minimal default replacement dictionary for Russian common ASR mistakes
DEFAULT_RU_REPLACEMENTS: Dict[str, str] = {
    "славо": "слово",
    "канешна": "конечно",
    "шась": "сейчас",
    # Keep single-token entries to align with token-wise replacer
}


def _try_load_spellchecker():
    """Lazily try to load a Russian spellchecker. Prefer symspellpy if available, fallback to pyspellchecker.
    Returns a function correct(text) -> corrected_text, or None if unavailable.
    """
    # Try symspellpy
    try:
        from symspellpy import SymSpell, Verbosity

        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # The dictionary paths should be configured by the host app. We load minimal frequency data if present.
        # Users can call set_symspell_dictionary() to provide their own.
        def correct(text: str) -> str:
            words = text.split()
            corrected: List[str] = []
            for w in words:
                suggestions = sym_spell.lookup(w, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
                if suggestions:
                    corrected.append(suggestions[0].term)
                else:
                    corrected.append(w)
            return " ".join(corrected)

        # Attach a way to set dictionary externally
        def set_symspell_dictionary(dictionary_path: str, bigram_path: Optional[str] = None) -> None:
            sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            if bigram_path:
                sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

        correct.set_symspell_dictionary = set_symspell_dictionary  # type: ignore[attr-defined]
        return correct
    except Exception:
        pass

    # Fallback: pyspellchecker
    try:
        from spellchecker import SpellChecker

        ru_spell = SpellChecker(language="ru")

        def correct(text: str) -> str:
            tokens = text.split()
            out: List[str] = []
            for t in tokens:
                if t.isalpha():
                    corr = ru_spell.correction(t)
                    out.append(corr or t)
                else:
                    out.append(t)
            return " ".join(out)

        return correct
    except Exception:
        pass

    # pymorphy2 based correction is not a spellchecker by itself; skipping here for simplicity.
    return None


def postprocess_text_ru(
    text: str,
    custom_replacements: Optional[Dict[str, str]] = None,
    enable_spellcheck: bool = True,
    lm_generate_candidates_fn: Optional[Callable[[str], List[Tuple[str, float]]]] = None,
    lm_score_fn: Optional[Callable[[str], float]] = None,
) -> str:
    """
    Postprocess Russian text with replacements, spellcheck, and optional LM reranking.

    - custom_replacements: merged over defaults; pass {} to disable defaults.
    - enable_spellcheck: if False, skip spellchecking.
    - lm_generate_candidates_fn/lm_score_fn: optional interfaces for external LMs (e.g., ruGPT-3/ruBERT).
    """
    if not text:
        return text

    replacements = dict(DEFAULT_RU_REPLACEMENTS)
    if custom_replacements is not None:
        replacements.update(custom_replacements)
    # Step 1: dictionary replacements
    out = apply_replacements(text, replacements)

    # Step 2: spellcheck
    if enable_spellcheck:
        correct_fn = _try_load_spellchecker()
        if correct_fn is not None:
            try:
                out = correct_fn(out)
            except Exception:
                pass

    # Step 3: optional LM rerank
    if lm_generate_candidates_fn is not None:
        out = rerank_with_language_model(out, lm_generate_candidates_fn, lm_score_fn)

    return out

