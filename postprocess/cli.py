from __future__ import annotations

import argparse
import os
import sys

from . import postprocess_text_ru, postprocess_text_kk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Postprocess ASR text for RU/KK")
    parser.add_argument("text", nargs="?", help="Input text to postprocess; if omitted, read stdin")
    parser.add_argument("--lang", "-l", choices=["ru", "kk"], required=True)
    parser.add_argument("--dict", dest="dictionary_path", default=None, help="Optional SymSpell frequency dictionary path")
    parser.add_argument("--lm", dest="enable_lm", action="store_true", help="Enable LM reranking")
    parser.add_argument("--lm-model", dest="lm_model", default=None, help="Transformers model id for reranking")
    args = parser.parse_args(argv)

    text = args.text
    if text is None:
        text = sys.stdin.read()

    if args.lang == "ru":
        out = postprocess_text_ru(
            text,
            dictionary_path=args.dictionary_path,
            enable_lm_rerank=args.enable_lm,
            lm_model_name=args.lm_model,
        )
    else:
        out = postprocess_text_kk(
            text,
            dictionary_path=args.dictionary_path,
            enable_lm_rerank=args.enable_lm,
            lm_model_name=args.lm_model,
        )

    sys.stdout.write(out)
    if not out.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

