#!/usr/bin/env python
"""TruthLens standalone evaluation script.

Loads a trained checkpoint and a fitted tokenizer (never refits), reproduces
the original stratified test split (or scores a separately supplied test
file), and reports accuracy / precision / recall / F1 / ROC-AUC / confusion
matrix / classification report.

Positive class = REAL (1). Negative class = FAKE (0).
A false positive means a FAKE article was classified REAL (the more
consequential error for a credibility system); a false negative means a REAL
article was classified FAKE.

Example:
    python evaluate.py --true-data data/True.csv --fake-data data/Fake.csv \\
        --checkpoint models/best_cnn_lstm.pt --tokenizer artifacts/tokenizer.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models import get_model
from src.preprocessing import (
    MAX_LEN,
    RANDOM_SEED,
    download_nltk_resources,
    encode_texts,
    load_and_merge_data,
    load_tokenizer,
    preprocess_dataframe,
    split_dataset,
)
from src.utils import calculate_metrics, get_device, load_checkpoint, save_metrics, set_seed, setup_logger

logger = setup_logger("truthlens.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TruthLens checkpoint.")
    parser.add_argument("--true-data", type=Path, default=Path("data/True.csv"))
    parser.add_argument("--fake-data", type=Path, default=Path("data/Fake.csv"))
    parser.add_argument("--test-file", type=Path, default=None,
                         help="Optional separate CSV with 'text' and 'label' columns to "
                              "evaluate instead of reproducing the original split.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_cnn_lstm.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.pkl"))
    parser.add_argument("--model", type=str, default="cnn_lstm_hybrid")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def predict_probs(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.long)
    outputs = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i : i + batch_size].to(device)
        logits = model(xb)
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.no_cuda)
    logger.info(f"Device: {device}")

    logger.info(f"Loading tokenizer (read-only, no fitting) from {args.tokenizer} ...")
    tokenizer = load_tokenizer(args.tokenizer)

    if args.test_file is not None:
        logger.info(f"Loading supplied test file: {args.test_file}")
        test_df = pd.read_csv(args.test_file)
        X_test = test_df["text"]
        y_test = test_df["label"]
    else:
        download_nltk_resources()
        logger.info("Reproducing the original 70/10/20 stratified split (seed=%d)...", args.seed)
        df = load_and_merge_data(args.true_data, args.fake_data)
        df = preprocess_dataframe(df)
        _, _, X_test, _, _, y_test = split_dataset(df, random_state=args.seed)

    logger.info(f"Test set size: {len(X_test):,}")
    X_te_seq = encode_texts(tokenizer, X_test, max_len=args.max_len)

    logger.info(f"Loading checkpoint from {args.checkpoint} ...")
    ckpt = load_checkpoint(args.checkpoint, device=device)
    model_config = ckpt.get("model_config", {})
    model = get_model(
        ckpt.get("model_name", args.model),
        vocab_size=ckpt.get("max_words", model_config.get("vocab_size", 50_000)),
        embed_dim=ckpt.get("embed_dim", model_config.get("embed_dim", 128)),
        max_len=ckpt.get("max_len", args.max_len),
        **{k: v for k, v in model_config.items() if k not in {"vocab_size", "embed_dim"}},
    ).to(device)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    logger.info(f"Checkpoint loaded with strict=True: {result}")
    model.eval()

    threshold = ckpt.get("threshold", args.threshold)
    probs = predict_probs(model, X_te_seq, device, batch_size=args.batch_size)
    preds = (probs >= threshold).astype(int)
    y_true = np.asarray(y_test)

    metrics = calculate_metrics(y_true, probs, threshold=threshold)
    report = classification_report(y_true, preds, target_names=["FAKE", "REAL"], output_dict=True)
    report_text = classification_report(y_true, preds, target_names=["FAKE", "REAL"])
    cm = confusion_matrix(y_true, preds).tolist()

    print("\n" + "=" * 60)
    print("TruthLens Evaluation Results")
    print("Positive class = REAL (1)  |  Negative class = FAKE (0)")
    print("=" * 60)
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1-score   : {metrics['f1']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix ([[TN, FP], [FN, TP]]):")
    print(np.array(cm))
    print("\nClassification Report:")
    print(report_text)
    print("=" * 60)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, args.results_dir / "metrics.json")
    save_metrics(report, args.results_dir / "classification_report.json")
    save_metrics({"confusion_matrix": cm, "labels": ["FAKE", "REAL"]}, args.results_dir / "confusion_matrix.json")
    logger.info(f"Saved results to {args.results_dir}")


if __name__ == "__main__":
    main()
