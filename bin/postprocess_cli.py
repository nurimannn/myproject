#!/usr/bin/env python3
import argparse
import sys

from asr_postprocess import postprocess_text_ru, postprocess_text_kk


def main() -> int:
    parser = argparse.ArgumentParser(description="Postprocess ASR text for RU or KK.")
    parser.add_argument("lang", choices=["ru", "kk"], help="Language code")
    parser.add_argument("text", nargs="?", default=None, help="Input text (optional, else stdin)")
    parser.add_argument("--no-spell", action="store_true", help="Disable spellcheck")
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()

    if args.lang == "ru":
        out = postprocess_text_ru(text, enable_spellcheck=(not args.no_spell))
    else:
        out = postprocess_text_kk(text, enable_spellcheck=(not args.no_spell))

    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

