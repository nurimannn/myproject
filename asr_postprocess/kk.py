from typing import Dict, List, Optional, Tuple, Callable

from .utils import apply_replacements, rerank_with_language_model


# Minimal default replacement dictionary for Kazakh common ASR mistakes (Latin -> Cyrillic etc.)
DEFAULT_KK_REPLACEMENTS: Dict[str, str] = {
    "bugyn": "бүгін",
    "bugin": "бүгін",
    "maktap": "мектеп",
    "mektep": "мектеп",
    "sagalak": "сағалық",
}


def _try_load_spellchecker():
    """Try to load a Kazakh spellchecker. Prefer symspellpy if user provides dicts; otherwise try hunspell.
    Returns function correct(text) -> corrected_text, or None if unavailable.
    """
    # symspellpy (requires user-supplied dictionaries)
    try:
        from symspellpy import SymSpell, Verbosity

        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

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

        def set_symspell_dictionary(dictionary_path: str, bigram_path: Optional[str] = None) -> None:
            sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            if bigram_path:
                sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

        correct.set_symspell_dictionary = set_symspell_dictionary  # type: ignore[attr-defined]
        return correct
    except Exception:
        pass

    # hunspell with Kazakh dictionary (requires system dictionaries available)
    try:
        import hunspell  # type: ignore

        try:
            hobj = hunspell.HunSpell("/usr/share/hunspell/kk_KZ.dic", "/usr/share/hunspell/kk_KZ.aff")
        except Exception:
            # Try common alternative path
            hobj = hunspell.HunSpell("/usr/share/hunspell/kk.dic", "/usr/share/hunspell/kk.aff")

        def correct(text: str) -> str:
            tokens = text.split()
            out: List[str] = []
            for t in tokens:
                if t and any(ch.isalpha() for ch in t):
                    if not hobj.spell(t):
                        sugg = hobj.suggest(t)
                        out.append(sugg[0] if sugg else t)
                    else:
                        out.append(t)
                else:
                    out.append(t)
            return " ".join(out)

        return correct
    except Exception:
        pass

    return None


def postprocess_text_kk(
    text: str,
    custom_replacements: Optional[Dict[str, str]] = None,
    enable_spellcheck: bool = True,
    lm_generate_candidates_fn: Optional[Callable[[str], List[Tuple[str, float]]]] = None,
    lm_score_fn: Optional[Callable[[str], float]] = None,
) -> str:
    """
    Postprocess Kazakh text with replacements, spellcheck, and optional LM reranking.
    - custom_replacements: merged over defaults; pass {} to disable defaults.
    - enable_spellcheck: if False, skip spellchecking.
    - lm_generate_candidates_fn/lm_score_fn: optional LM interfaces (e.g., kaznlp/kazbert).
    """
    if not text:
        return text

    replacements = dict(DEFAULT_KK_REPLACEMENTS)
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

