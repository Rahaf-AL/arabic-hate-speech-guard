"""
Single-Text Inference

Shared classification helper used by the interactive app (src/app.py) and
the classroom monitoring demo (src/monitor.py), so both apply the exact
same loading, cleaning, and inference steps.
"""
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evaluate import load_model  # noqa: E402
from preprocessing import clean_text  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def load_classifier(model_dir):
    """Loads the tokenizer and a trained model (local path or Hugging Face
    Hub repo id) ready for inference."""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = load_model(model_dir)
    model.eval()
    return tokenizer, model


def classify_text(tokenizer, model, text: str) -> dict:
    """Cleans and classifies a single piece of text, returning a
    {label: probability} dict."""
    if not text or not text.strip():
        return {config.ID2LABEL[0]: 1.0, config.ID2LABEL[1]: 0.0}

    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    return {config.ID2LABEL[i]: probs[i] for i in range(len(probs))}
