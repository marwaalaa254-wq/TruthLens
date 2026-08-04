#!/usr/bin/env python
"""TruthLens real-time inference module and CLI.

Pipeline (matches the notebook's Phase 9 "Real-Time Inference", with the
model swapped to the best-performing CNN-LSTM Hybrid -- see note below):

    1. Validate raw input (reject empty text).
    2. Apply preprocess_text() from src.preprocessing (the SAME function used
       during training -- no separate/duplicated cleaning logic here).
    3. Load artifacts/tokenizer.pkl (fitted during training; never refit here).
    4. Convert cleaned text to integer token IDs.
    5. Post-pad / post-truncate to the trained sequence length (400).
    6. Convert to torch.LongTensor.
    7. Build CNNLSTMHybridNet from src.models and load models/best_cnn_lstm.pt.
    8. model.eval() + torch.no_grad().
    9. Apply sigmoid exactly once to the raw logit.
   10. Apply the decision threshold (default 0.5).
   11. Return a structured prediction dict.

IMPORTANT BUG FIX: the notebook's own ``predict_news`` demo (Phase 9) calls
``bilstm_model.predict(...)`` -- the BiLSTM+Attention model -- even though the
notebook's own benchmark (Phase 8) identifies CNN-LSTM Hybrid as the best
performer (Accuracy 99.22%, F1 99.28%, ROC-AUC 99.94% vs BiLSTM+Attention's
98.55%/98.67%/99.88%). This script intentionally loads the CNN-LSTM Hybrid
checkpoint instead, per the project's own stated "best model" conclusion.

Usage:
    python inference.py --text "Paste an English news article here"
    python inference.py --file article.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models import get_model
from src.preprocessing import MAX_LEN, encode_texts, load_tokenizer, preprocess_text
from src.utils import get_device, load_checkpoint, setup_logger

logger = setup_logger("truthlens.inference")

DEFAULT_CHECKPOINT = Path("models/best_cnn_lstm.pt")
DEFAULT_TOKENIZER = Path("artifacts/tokenizer.pkl")
DEFAULT_MODEL_NAME = "cnn_lstm_hybrid"

_model_cache: dict[str, torch.nn.Module] = {}
_tokenizer_cache: dict[str, object] = {}


def _load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    cache_key = str(checkpoint_path.resolve())
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    ckpt = load_checkpoint(checkpoint_path, device=device)
    model_config = ckpt.get("model_config", {})
    model = get_model(
        ckpt.get("model_name", DEFAULT_MODEL_NAME),
        vocab_size=ckpt.get("max_words", model_config.get("vocab_size", 50_000)),
        embed_dim=ckpt.get("embed_dim", model_config.get("embed_dim", 128)),
        max_len=ckpt.get("max_len", MAX_LEN),
        **{k: v for k, v in model_config.items() if k not in {"vocab_size", "embed_dim"}},
    ).to(device)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    logger.info(f"Loaded {checkpoint_path.name} with strict=True: {result}")
    model.eval()
    _model_cache[cache_key] = model
    return model


def _load_tokenizer_cached(tokenizer_path: Path):
    cache_key = str(tokenizer_path.resolve())
    if cache_key not in _tokenizer_cache:
        _tokenizer_cache[cache_key] = load_tokenizer(tokenizer_path)
    return _tokenizer_cache[cache_key]


@torch.no_grad()
def predict_news(
    text: str,
    threshold: float = 0.5,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    max_len: int = MAX_LEN,
    device: Optional[torch.device] = None,
) -> dict:
    """Classifies a news article as FAKE or REAL using the CNN-LSTM Hybrid model.

    Args:
        text: raw article text (title and/or body).
        threshold: decision threshold applied to P(REAL). Default 0.5.
        checkpoint_path: path to the standardized CNN-LSTM Hybrid checkpoint.
        tokenizer_path: path to the fitted tokenizer pickle.
        max_len: sequence length used for padding/truncation (must match training).
        device: torch device; auto-selected (CUDA if available) when omitted.

    Returns:
        A dict with ``label``, ``class_id``, ``probability_real``,
        ``probability_fake``, ``confidence``, ``threshold``, and ``cleaned_text``.

    Raises:
        ValueError: if ``text`` is empty or not a string.
    """
    if not isinstance(text, str):
        raise ValueError(f"predict_news expected a string, got {type(text)!r}")
    if not text.strip():
        raise ValueError("predict_news received empty input text.")

    device = device or get_device()

    cleaned = preprocess_text(text)
    tokenizer = _load_tokenizer_cached(Path(tokenizer_path))
    sequence = encode_texts(tokenizer, [cleaned], max_len=max_len)
    x = torch.as_tensor(sequence, dtype=torch.long, device=device)

    model = _load_model(Path(checkpoint_path), device)
    logit = model(x)
    prob_real = torch.sigmoid(logit).item()
    prob_fake = 1.0 - prob_real

    label = "REAL" if prob_real >= threshold else "FAKE"
    class_id = 1 if label == "REAL" else 0
    confidence = prob_real if label == "REAL" else prob_fake

    return {
        "label": label,
        "class_id": class_id,
        "probability_real": prob_real,
        "probability_fake": prob_fake,
        "confidence": confidence,
        "threshold": threshold,
        "cleaned_text": cleaned,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TruthLens real-time inference on a news article.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw article text to classify.")
    group.add_argument("--file", type=Path, help="Path to a UTF-8 text file containing the article.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--vocabulary", type=Path, default=Path("artifacts/vocabulary.pkl"),
                         help="Not required for inference itself; accepted for interface "
                              "consistency with export_artifacts.py.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.file is not None:
        text = args.file.read_text(encoding="utf-8")
    else:
        text = args.text

    device = torch.device(args.device) if args.device else None

    result = predict_news(
        text,
        threshold=args.threshold,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=device,
    )

    print(f"\nVerdict     : {result['label']}")
    print(f"Confidence  : {result['confidence'] * 100:.2f}%")
    print(f"P(REAL)     : {result['probability_real']:.4f}")
    print(f"P(FAKE)     : {result['probability_fake']:.4f}")
    print(f"Threshold   : {result['threshold']}")


if __name__ == "__main__":
    main()
