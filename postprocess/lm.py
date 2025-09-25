from __future__ import annotations

from typing import List


def rank_by_causal_lm(candidates: List[str], model_name: str) -> str:
    """Rank candidate sentences using a Causal LM (e.g., ruGPT-3 or GPT-2-like).

    We import transformers lazily to avoid heavy imports when not needed.
    Ranking is performed by average negative log-likelihood per token; lower NLL is better.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        # transformers not installed or no GPU/CPU support; fall back to first
        return candidates[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    def avg_nll(text: str) -> float:
        with torch.no_grad():
            enc = tokenizer(text, return_tensors="pt")
            input_ids = enc.input_ids.to(device)
            attn = enc.attention_mask.to(device)
            # Shift for next-token prediction
            outputs = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            # loss: mean cross-entropy per token
            loss = outputs.loss
            return float(loss.item())

    best_text = candidates[0]
    best_score = avg_nll(best_text)
    for cand in candidates[1:]:
        score = avg_nll(cand)
        if score < best_score:
            best_text = cand
            best_score = score
    return best_text

