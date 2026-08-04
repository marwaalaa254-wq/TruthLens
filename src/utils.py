"""Shared training/evaluation/inference helpers for the TruthLens project.

Ported from the notebook's "Shared PyTorch Infrastructure" cell (Phase 7.2):
device selection, EarlyStopping/ReduceLROnPlateau-style callbacks, checkpoint
save/load, metrics computation, and small JSON/logging utilities.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PathLike = Union[str, os.PathLike]


def set_seed(seed: int = 42) -> None:
    """Fixes Python/NumPy/PyTorch RNG state for reproducibility (matches the notebook's
    per-model "Fix Random Seed for Reproducibility" cells)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Returns a CUDA device if available and requested, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_directories(*paths: PathLike) -> None:
    """Creates each directory (and parents) if it does not already exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dataloader(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    """Builds a DataLoader of (LongTensor sequences[, FloatTensor labels])."""
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.long)
    if y is not None:
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        dataset: torch.utils.data.Dataset = TensorDataset(X_t, y_t)
    else:
        dataset = TensorDataset(X_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
# Checkpoint save/load
# --------------------------------------------------------------------------- #

def save_checkpoint(
    path: PathLike,
    model: nn.Module,
    model_name: str,
    model_config: dict[str, Any],
    max_words: int,
    max_len: int,
    embed_dim: int,
    padding_idx: int = 0,
    oov_index: int = 1,
    label_mapping: Optional[dict[str, int]] = None,
    threshold: float = 0.5,
    metrics: Optional[dict[str, Any]] = None,
    random_seed: int = 42,
    source_checkpoint: Optional[str] = None,
) -> None:
    """Saves a standardized TruthLens checkpoint dictionary.

    Always contains ``model_state_dict`` plus enough metadata to rebuild the
    exact architecture and reproduce inference (padding index, OOV index,
    label mapping, decision threshold) without relying on hidden globals.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "model_config": model_config,
        "max_words": max_words,
        "max_len": max_len,
        "embed_dim": embed_dim,
        "padding_idx": padding_idx,
        "oov_index": oov_index,
        "label_mapping": label_mapping or {"FAKE": 0, "REAL": 1},
        "threshold": threshold,
        "metrics": metrics or {},
        "random_seed": random_seed,
        "source_checkpoint": source_checkpoint,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: PathLike, device: Optional[torch.device] = None) -> dict[str, Any]:
    """Loads a checkpoint file, supporting both raw formats found in this project:

    1. A raw ``state_dict`` (an ``OrderedDict`` of tensors) — the format produced by
       the notebook's ``ModelCheckpoint`` callback (``torch.save(model.state_dict(), path)``).
    2. A standardized dict checkpoint containing a ``model_state_dict`` key plus metadata
       (the format written by :func:`save_checkpoint` / ``export_artifacts.py``).

    Returns a dict that always has a ``model_state_dict`` key, with any other keys the
    checkpoint contained (empty dict for a raw state_dict).
    """
    device = device or get_device()
    obj = torch.load(path, map_location=device, weights_only=False)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj

    # Raw state_dict: every value is a tensor.
    if isinstance(obj, dict):
        return {"model_state_dict": obj}

    raise ValueError(f"Unrecognized checkpoint format at {path!s}: {type(obj)!r}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Computes accuracy/precision/recall/F1/ROC-AUC for binary predictions.

    Positive class = REAL (1). ``y_prob`` must be P(REAL).
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


# --------------------------------------------------------------------------- #
# JSON I/O
# --------------------------------------------------------------------------- #

def save_json(data: Any, path: PathLike) -> None:
    """Writes ``data`` as pretty-printed JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: PathLike) -> Any:
    """Reads and parses a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics(metrics: dict[str, Any], path: PathLike) -> None:
    """Alias of :func:`save_json` for readability at call sites."""
    save_json(metrics, path)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def setup_logger(name: str = "truthlens", level: int = logging.INFO) -> logging.Logger:
    """Returns a configured stdout logger, idempotent across repeated calls."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# --------------------------------------------------------------------------- #
# Training history + early stopping
# --------------------------------------------------------------------------- #

@dataclass
class TrainingHistory:
    """Mimics the notebook's Keras-style ``History`` object: a dict of per-epoch lists."""

    history: dict[str, list[float]] = field(
        default_factory=lambda: {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    )

    def append(self, logs: dict[str, float]) -> None:
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)

    def best_epoch(self, monitor: str = "val_loss", mode: str = "min") -> int:
        """Returns the 0-indexed epoch with the best monitored value."""
        values = self.history[monitor]
        return int(np.argmin(values)) if mode == "min" else int(np.argmax(values))

    def to_dict(self) -> dict[str, list[float]]:
        return self.history


class EarlyStopping:
    """Stops training when ``monitor`` has not improved for ``patience`` epochs.

    Matches the notebook's EarlyStopping callback: optionally restores the best
    weights seen so far when triggered.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 3,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: int = 0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best = np.inf if mode == "min" else -np.inf
        self.wait = 0
        self.best_state: Optional[dict[str, torch.Tensor]] = None
        self.stopped_epoch: Optional[int] = None
        self._stop = False

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best - self.min_delta
        return current > self.best + self.min_delta

    def on_epoch_end(self, epoch: int, logs: dict[str, float], model: nn.Module) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return
        if self._is_improvement(current):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._stop = True
                self.stopped_epoch = epoch
                if self.verbose:
                    print(
                        f"\nEarlyStopping: no improvement in {self.monitor} for "
                        f"{self.patience} epochs — stopping at epoch {epoch + 1}."
                    )
                if self.restore_best_weights and self.best_state is not None:
                    model.load_state_dict(self.best_state)

    def should_stop(self) -> bool:
        return self._stop
