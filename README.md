Russian and Kazakh ASR Postprocessing
====================================

This workspace provides language-specific postprocessing for Russian (ru) and Kazakh (kk) transcriptions.

Features
--------
- Custom replacement dictionaries for common ASR errors
- Spell checking: ru via pyspellchecker/symspellpy (+optional pymorphy2), kk via hunspell or symspellpy
- Optional language model re-ranking using Hugging Face models

Usage
-----
```bash
pip install -r requirements.txt
```

Then in Python:
```python
from pipeline import process_and_save

processed = process_and_save("превет мир", "ru", "/tmp/out_ru.txt", use_lm=False)
print(processed)

processed_kk = process_and_save("bugyn jaksy kun", "kk", "/tmp/out_kk.txt", use_lm=False)
print(processed_kk)
```

Notes
-----
- Hunspell Kazakh dictionaries may need to be installed on your system (e.g., kk_KZ). If unavailable, the system falls back gracefully.
- Language model usage is optional and guarded; if models are missing, text is returned unchanged after spellcheck.


