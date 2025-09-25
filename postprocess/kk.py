from __future__ import annotations

from typing import Dict, List, Optional

from .utils import apply_replacements, normalize_whitespace, tokenize_words


DEFAULT_KK_REPLACEMENTS: Dict[str, str] = {
    "bugyn": "бүгін",
    "maktap": "мектеп",
    "audarma": "аударма",
    "kuni": "күні",
}


class KazakhSpellChecker:
    def __init__(self) -> None:
        self._symspell = None
        self._hunspell = None
        # Try symspellpy
        try:
            from symspellpy.symspellpy import SymSpell, Verbosity  # type: ignore

            sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            # Expect user to load Kazakh dictionary externally if available
            self._symspell = (sym, Verbosity.CLOSEST)
        except Exception:
            self._symspell = None
        # Try hunspell with KK dictionary if present
        try:
            import hunspell  # type: ignore

            # This expects system dictionaries installed, e.g., kk_KZ
            self._hunspell = hunspell.HunSpell("/usr/share/hunspell/kk_KZ.dic", "/usr/share/hunspell/kk_KZ.aff")
        except Exception:
            self._hunspell = None

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
            # Prefer Hunspell if available for Kazakh morphology
            if self._hunspell is not None:
                try:
                    if self._hunspell.spell(lower):
                        suggestion = lower
                    else:
                        suggests = self._hunspell.suggest(lower)
                        if suggests:
                            suggestion = suggests[0]
                except Exception:
                    suggestion = None
            # Fallback to SymSpell
            if suggestion is None and self._symspell is not None:
                try:
                    sym, verbosity = self._symspell
                    results = sym.lookup(lower, verbosity, max_edit_distance=2)
                    if results:
                        suggestion = results[0].term
                except Exception:
                    suggestion = None

            if suggestion is None:
                corrected.append(token)
                continue
            if token.istitle():
                corrected.append(suggestion.capitalize())
            elif token.isupper():
                corrected.append(suggestion.upper())
            else:
                corrected.append(suggestion)
        return corrected


def _optional_kazakh_lm_rerank(text: str) -> str:
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore
        import torch  # type: ignore

        model_name = "kaznlp/kazbert-tiny"  # small if available; otherwise user can change
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        if "[MASK]" not in text:
            return text
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_token_index.numel() == 0:
            return text
        mask_token_index = mask_token_index[0, 1]
        probs = logits[0, mask_token_index]
        top_id = int(torch.argmax(probs).item())
        predicted = tokenizer.decode([top_id])
        return text.replace(tokenizer.mask_token, predicted)
    except Exception:
        return text


def postprocess_text_kk(
    text: str,
    custom_replacements: Optional[Dict[str, str]] = None,
    use_language_model: bool = False,
) -> str:
    if not text:
        return text
    repl = dict(DEFAULT_KK_REPLACEMENTS)
    if custom_replacements:
        repl.update({k.lower(): v for k, v in custom_replacements.items()})
    text = apply_replacements(text, repl)
    tokens = tokenize_words(text)
    checker = KazakhSpellChecker()
    tokens = checker.correct_tokens(tokens)
    text = "".join(tokens)
    text = normalize_whitespace(text)
    if use_language_model:
        text = _optional_kazakh_lm_rerank(text)
        text = normalize_whitespace(text)
    return text

