Postprocessing for Russian and Kazakh ASR
=========================================

This repository includes utilities to postprocess ASR outputs for Russian (ru) and Kazakh (kk).

Features
--------
- Common error replacements per language
- Spell correction using SymSpell frequency dictionaries (optional but recommended)
- Optional language model reranking using Hugging Face transformers

Installation
------------
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

SymSpell dictionaries
---------------------
Place frequency dictionaries under `postprocess/dicts/`:
- `ru_frequency_dictionary.txt`
- `kk_frequency_dictionary.txt`

Format: one entry per line: `term count` (space separated).

You can also set environment variables to custom paths:
- `RU_SYMSPELL_DICT=/path/to/ru.txt`
- `KK_SYMSPELL_DICT=/path/to/kk.txt`

Language models (optional)
--------------------------
Install `transformers` and `torch` (already in `requirements.txt`). You may set:
- `ENABLE_LM_RERANK_RU=1` to enable LM reranking for Russian
- `ENABLE_LM_RERANK_KK=1` to enable LM reranking for Kazakh
- `RU_LM_MODEL_NAME=ai-forever/rugpt3small_based_on_gpt2` (default)
- `KK_LM_MODEL_NAME=kaznlp/kazbert` (default)

CLI usage
---------
```bash
python -m postprocess.cli --lang ru "славо eto horosho"
python -m postprocess.cli --lang kk "bugyn maktap baramyn"

# With dictionaries and LM rerank
python -m postprocess.cli --lang ru --dict postprocess/dicts/ru_frequency_dictionary.txt --lm --lm-model ai-forever/rugpt3small_based_on_gpt2 "счас я пажалуйста приду"
```

Programmatic usage
------------------
```python
from postprocess import postprocess_text_ru, postprocess_text_kk

text_ru = postprocess_text_ru("славо eto horosho")
text_kk = postprocess_text_kk("bugyn maktap baramyn")
```

Integration guidance
--------------------
Ensure that all Russian and Kazakh transcriptions pass through these functions before display/save:
- Russian branch: `postprocess.postprocess_text_ru(text)`
- Kazakh branch: `postprocess.postprocess_text_kk(text)`

Russian and Kazakh ASR Postprocessing

This repository provides postprocessing functions for Russian and Kazakh transcriptions that apply:
- dictionary-based replacements for common ASR errors,
- spell checking via symspellpy/pyspellchecker (RU) or symspellpy/hunspell (KK), and
- optional language model reranking hooks.

Install

```bash
pip install -r requirements.txt
```

Usage (Python)

```python
from asr_postprocess import postprocess_text_ru, postprocess_text_kk

ru_text = postprocess_text_ru("Это славо неверно распознано")
kk_text = postprocess_text_kk("bugyn men maktapka bardym")
```

CLI

```bash
python bin/postprocess_cli.py ru "Это славо неверно распознано"
python bin/postprocess_cli.py kk "bugyn men maktapka bardym"
```

Integrating into ASR pipeline

Ensure that all Russian and Kazakh recognition branches call postprocessing before display or save:

```python
if lang == "ru":
    text = postprocess_text_ru(text)
elif lang == "kk":
    text = postprocess_text_kk(text)
```

Providing dictionaries for symspell

For better results, load frequency and bigram dictionaries (see symspellpy docs). Both `postprocess_text_ru` and `postprocess_text_kk` expose an internal corrector that supports setting dictionaries via `set_symspell_dictionary` attribute if detected. Alternatively, you can fork these functions to inject your own `SymSpell` instance.

Language Model reranking

Pass `lm_generate_candidates_fn` and `lm_score_fn` to the postprocessors to enable reranking with an external model such as ruGPT-3, ruBERT, or kazbert.


