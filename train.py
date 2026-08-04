#!/usr/bin/env python
"""TruthLens training script.

Reproduces the notebook's Phase 4-7 pipeline for a single selected model:
load True.csv/Fake.csv, clean, split 70/10/20 (stratified, seed=42), fit a
tokenizer on the training split only, encode/pad to MAX_LEN, train with
early stopping + LR reduction on plateau, checkpoint the best epoch, then
evaluate the restored-best model on the untouched test set.

Example:
    python train.py --true-data data/True.csv --fake-data data/Fake.csv \\
        --model cnn_lstm_hybrid --epochs 10 --batch-size 64 --learning-rate 0.0005
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models import get_model
from src.preprocessing import (
    MAX_LEN,
    MAX_WORDS,
    EMBED_DIM,
    RANDOM_SEED,
    build_tokenizer,
    download_nltk_resources,
    encode_texts,
    load_and_merge_data,
    preprocess_dataframe,
    save_tokenizer,
    save_vocabulary,
    split_dataset,
)
from src.utils import (
    calculate_metrics,
    count_trainable_parameters,
    create_dataloader,
    create_directories,
    get_device,
    save_checkpoint,
    save_metrics,
    set_seed,
    setup_logger,
    EarlyStopping,
    TrainingHistory,
)

logger = setup_logger("truthlens.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TruthLens fake-news classifier.")
    parser.add_argument("--true-data", type=Path, default=Path("data/True.csv"), help="Path to True.csv")
    parser.add_argument("--fake-data", type=Path, default=Path("data/Fake.csv"), help="Path to Fake.csv")
    parser.add_argument(
        "--model", type=str, default="cnn_lstm_hybrid",
        choices=["bilstm", "bilstm_attention", "cnn", "cnn_lstm_hybrid"],
        help="Which architecture to train (DistilBERT is not supported by this script).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience (epochs).")
    parser.add_argument("--lr-patience", type=int, default=2, help="ReduceLROnPlateau patience (epochs).")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Learning-rate reduction factor.")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU training even if CUDA is available.")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        probs = torch.sigmoid(logits)
        total_loss += loss.item() * xb.size(0)
        total_correct += ((probs >= 0.5).float() == yb).sum().item()
        total_n += xb.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def predict_probs(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.long)
    outputs = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i : i + batch_size].to(device)
        outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.no_cuda)
    logger.info(f"Device: {device}")

    create_directories(args.models_dir, args.artifacts_dir, args.results_dir)

    download_nltk_resources()
    logger.info("Loading and merging datasets...")
    df = load_and_merge_data(args.true_data, args.fake_data)
    logger.info(f"Merged dataset shape: {df.shape}")

    logger.info("Cleaning dataset (Phase 4 pipeline)...")
    df = preprocess_dataframe(df)
    logger.info(f"Cleaned dataset shape: {df.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, random_state=args.seed)
    logger.info(
        f"Split sizes -- train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}"
    )

    logger.info("Fitting tokenizer on training data only...")
    tokenizer = build_tokenizer(X_train, max_words=args.max_words)
    vocab_size_fitted = len(tokenizer.word_index)
    logger.info(f"Fitted vocabulary size: {vocab_size_fitted:,} (capped at {args.max_words:,} for embedding)")

    tokenizer_path = args.artifacts_dir / "tokenizer.pkl"
    vocabulary_path = args.artifacts_dir / "vocabulary.pkl"
    save_tokenizer(tokenizer, tokenizer_path)
    save_vocabulary(tokenizer.word_index, vocabulary_path)
    logger.info(f"Saved tokenizer -> {tokenizer_path}")
    logger.info(f"Saved vocabulary -> {vocabulary_path}")

    X_tr_seq = encode_texts(tokenizer, X_train, max_len=args.max_len)
    X_vl_seq = encode_texts(tokenizer, X_val, max_len=args.max_len)
    X_te_seq = encode_texts(tokenizer, X_test, max_len=args.max_len)

    train_loader = create_dataloader(X_tr_seq, y_train.values, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(X_vl_seq, y_val.values, batch_size=args.batch_size, shuffle=False)

    model = get_model(
        args.model, vocab_size=args.max_words, embed_dim=args.embed_dim, max_len=args.max_len
    ).to(device)
    n_params = count_trainable_parameters(model)
    logger.info(f"Model: {args.model}  |  Trainable parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True, verbose=1
    )
    lr_wait = 0
    lr_best = np.inf

    history = TrainingHistory()
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        logs = {"loss": train_loss, "accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc}
        history.append(logs)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
        )

        # ReduceLROnPlateau (monitor=val_loss, mode=min)
        if val_loss < lr_best:
            lr_best = val_loss
            lr_wait = 0
        else:
            lr_wait += 1
            if lr_wait >= args.lr_patience:
                lr_wait = 0
                for group in optimizer.param_groups:
                    new_lr = max(group["lr"] * args.lr_factor, args.min_lr)
                    if new_lr < group["lr"]:
                        logger.info(f"ReduceLROnPlateau: reducing learning rate to {new_lr:.2e}")
                    group["lr"] = new_lr

        early_stopping.on_epoch_end(epoch, logs, model)
        if early_stopping.should_stop():
            break

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f}s. Best val_loss: {early_stopping.best:.4f}")

    best_epoch = history.best_epoch(monitor="val_loss", mode="min")

    # Save the standardized best checkpoint.
    model_config = {
        "vocab_size": args.max_words,
        "embed_dim": args.embed_dim,
        "padding_idx": 0,
        "kernel_sizes": [3, 5, 7],
        "lstm_hidden": 128,
        "attn_heads": 4,
    }
    checkpoint_path = args.models_dir / "best_cnn_lstm.pt"

    # Evaluate the restored-best model on the held-out test set.
    test_probs = predict_probs(model, X_te_seq, device, batch_size=args.batch_size)
    test_metrics = calculate_metrics(y_test.values, test_probs, threshold=0.5)
    test_preds = (test_probs >= 0.5).astype(int)

    report = classification_report(
        y_test.values, test_preds, target_names=["FAKE", "REAL"], output_dict=True
    )
    cm = confusion_matrix(y_test.values, test_preds).tolist()

    metrics_payload = {
        "model": args.model,
        "trainable_parameters": n_params,
        "dataset_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "vocabulary_size": vocab_size_fitted,
        "best_epoch": best_epoch + 1,
        "training_time_seconds": train_time,
        "train_loss": history.history["loss"][best_epoch],
        "val_loss": history.history["val_loss"][best_epoch],
        "train_accuracy": history.history["accuracy"][best_epoch],
        "val_accuracy": history.history["val_accuracy"][best_epoch],
        "test": test_metrics,
        "history": history.to_dict(),
    }

    save_checkpoint(
        checkpoint_path,
        model=model,
        model_name=args.model,
        model_config=model_config,
        max_words=args.max_words,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        padding_idx=0,
        oov_index=1,
        label_mapping={"FAKE": 0, "REAL": 1},
        threshold=0.5,
        metrics=metrics_payload,
        random_seed=args.seed,
        source_checkpoint=None,
    )
    logger.info(f"Saved best checkpoint -> {checkpoint_path}")

    save_metrics(metrics_payload, args.results_dir / "metrics.json")
    save_metrics(report, args.results_dir / "classification_report.json")
    save_metrics({"confusion_matrix": cm, "labels": ["FAKE", "REAL"]}, args.results_dir / "confusion_matrix.json")

    print("\n" + "=" * 60)
    print(f"Device               : {device}")
    print(f"Trainable parameters : {n_params:,}")
    print(f"Dataset sizes        : train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")
    print(f"Vocabulary size      : {vocab_size_fitted:,}")
    print(f"Best epoch           : {best_epoch + 1}")
    print(f"Train loss           : {metrics_payload['train_loss']:.4f}")
    print(f"Val loss             : {metrics_payload['val_loss']:.4f}")
    print(f"Val accuracy         : {metrics_payload['val_accuracy']:.4f}")
    print(f"Test accuracy        : {test_metrics['accuracy']:.4f}")
    print(f"Test precision       : {test_metrics['precision']:.4f}")
    print(f"Test recall          : {test_metrics['recall']:.4f}")
    print(f"Test F1-score        : {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC         : {test_metrics['roc_auc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
