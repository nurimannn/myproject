import re
from typing import Dict, List, Tuple, Callable, Optional


WORD_RE = re.compile(r"\w+", re.UNICODE)


def split_tokens_with_delimiters(text: str) -> List[str]:
    """
    Split text into a list preserving delimiters. Words (\w+) and non-words are separate tokens.
    Example: "Привет, мир!" -> ["Привет", ", ", "мир", "!"]
    """
    if not text:
        return []
    tokens: List[str] = []
    idx = 0
    for match in WORD_RE.finditer(text):
        start, end = match.span()
        if start > idx:
            tokens.append(text[idx:start])
        tokens.append(text[start:end])
        idx = end
    if idx < len(text):
        tokens.append(text[idx:])
    return tokens


def is_word(token: str) -> bool:
    return bool(WORD_RE.fullmatch(token))


def apply_replacements(
    text: str,
    replacements: Dict[str, str],
    case_insensitive: bool = True,
) -> str:
    """
    Apply dictionary replacements token-wise, preserving capitalization of the original token.
    Only exact token matches are replaced.
    """
    if not text or not replacements:
        return text

    tokens = split_tokens_with_delimiters(text)
    normalized_map = { (k.lower() if case_insensitive else k): v for k, v in replacements.items() }

    def preserve_case(src: str, dst: str) -> str:
        if not src:
            return dst
        # Titlecase (First letter uppercase, rest lowercase)
        if src[:1].isupper() and src[1:].islower():
            return dst[:1].upper() + dst[1:].lower()
        # All caps
        if src.isupper():
            return dst.upper()
        # Lowercase or mixed
        if src.islower():
            return dst.lower()
        # Mixed case: fallback to dst as-is
        return dst

    out: List[str] = []
    for t in tokens:
        if is_word(t):
            key = t.lower() if case_insensitive else t
            if key in normalized_map:
                out.append(preserve_case(t, normalized_map[key]))
            else:
                out.append(t)
        else:
            out.append(t)
    return "".join(out)


def simple_detokenize(tokens: List[str]) -> str:
    return "".join(tokens)


def rerank_with_language_model(
    text: str,
    generate_candidates_fn: Callable[[str], List[Tuple[str, float]]],
    score_with_lm_fn: Optional[Callable[[str], float]] = None,
    top_k: int = 3,
) -> str:
    """
    Rerank candidate corrections using a provided LM scoring function.
    - generate_candidates_fn: returns list of (candidate_text, candidate_score)
    - score_with_lm_fn: returns language model score (higher is better). If None, returns the best candidate.
    """
    try:
        candidates = generate_candidates_fn(text)
    except Exception:
        return text
    if not candidates:
        return text
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
    if score_with_lm_fn is None:
        return candidates[0][0]
    best_text = text
    best_score = float("-inf")
    for cand_text, _ in candidates:
        try:
            s = score_with_lm_fn(cand_text)
        except Exception:
            continue
        if s > best_score:
            best_score = s
            best_text = cand_text
    return best_text

