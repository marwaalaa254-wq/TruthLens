"""PyTorch model architectures extracted from the TruthLens research notebook.

All classes return raw logits for binary classification (FAKE=0, REAL=1).
Sigmoid must be applied by the caller (evaluation/inference code), never inside
these modules, so that ``torch.nn.BCEWithLogitsLoss`` can be used safely during
training.

Architectures are ported 1:1 from ``notebooks/TruthLens_Notebook.ipynb``
(Phase 7 — Deep Learning Models). Dimensions, kernel sizes, dropout rates and
activation order are preserved exactly so that the trained checkpoints in
``models/`` load with ``strict=True``.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout1D(nn.Module):
    """Drops entire embedding channels across all timesteps (matches Keras SpatialDropout1D)."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, time, channels)
        if not self.training or self.p == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


class LSTMNet(nn.Module):
    """Embedding -> stacked BiLSTM -> Dense head. Notebook Model 1 (BiLSTM baseline)."""

    def __init__(self, vocab_size: int = 50_000, embed_dim: int = 128, padding_idx: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.spatial_dropout = SpatialDropout1D(0.5)
        self.lstm1 = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.spatial_dropout(self.embedding(x))
        out, _ = self.lstm1(emb)
        _, (h_n, _) = self.lstm2(out)
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        x = F.relu(self.fc1(h))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


class BiLSTMAttentionNet(nn.Module):
    """Embedding -> stacked BiLSTM -> Multi-Head Self-Attention (+residual) -> Dense head.

    Notebook Model 2 (BiLSTM + Attention).
    """

    def __init__(self, vocab_size: int = 50_000, embed_dim: int = 128, padding_idx: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.spatial_dropout = SpatialDropout1D(0.4)
        self.lstm1 = nn.LSTM(embed_dim, 64, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(128)
        self.lstm2 = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.spatial_dropout(self.embedding(x))
        out, _ = self.lstm1(emb)
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm2(out)

        attn_out, _ = self.attn(out, out, out)
        x = self.ln(out + attn_out)

        x = x.mean(dim=1)  # GlobalAveragePooling1D
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x).squeeze(-1)


class CNNTextClassifierNet(nn.Module):
    """Embedding -> Conv1D -> BatchNorm -> GlobalMaxPool -> Dense head. Notebook Model 3 (CNN)."""

    def __init__(self, vocab_size: int = 50_000, embed_dim: int = 128, padding_idx: int = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.spatial_dropout = SpatialDropout1D(0.4)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.spatial_dropout(self.embedding(x)).transpose(1, 2)  # (B, E, T) for Conv1d
        c = F.relu(self.conv(emb))
        c = self.bn(c)
        pooled = torch.amax(c, dim=2)  # GlobalMaxPooling1D
        x = F.relu(self.fc1(pooled))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x).squeeze(-1)


class CNNLSTMHybridNet(nn.Module):
    """Multi-filter Conv1D branches -> BiLSTM -> Multi-Head Attention (+residual) ->
    dual global pooling -> Dense head.

    This is the final, best-performing model (Notebook Model 4, ``best_cnn_lstm_hybrid.pt``):
    Accuracy 99.22%, Precision 99.45%, Recall 99.12%, F1 99.28%, ROC-AUC 99.94% on the
    held-out test split.

    Returns raw logits — apply ``torch.sigmoid`` exactly once, only at evaluation/inference time.
    """

    def __init__(
        self,
        vocab_size: int = 50_000,
        embed_dim: int = 128,
        padding_idx: int = 0,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        lstm_hidden: int = 128,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # 1. Multi-Filter CNN Block
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, 128, kernel_size=fs, padding=fs // 2), nn.ReLU(),
                nn.Conv1d(128, 64, kernel_size=fs, padding=fs // 2), nn.ReLU(),
            )
            for fs in kernel_sizes
        ])
        merged_dim = 64 * len(self.branches)  # 192

        # 2. Bidirectional LSTM Block
        self.lstm = nn.LSTM(merged_dim, lstm_hidden, batch_first=True, bidirectional=True)
        attn_dim = lstm_hidden * 2  # 256

        # 3. Multi-Head Attention Block
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True)
        self.ln = nn.LayerNorm(attn_dim)

        # 4/5. Dual Global Pooling + Dense Classification Head
        self.fc1 = nn.Linear(attn_dim * 2, 128)
        self.drop1 = nn.Dropout(0.40)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.30)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)  # (B, E, T)
        merged = torch.cat([branch(emb) for branch in self.branches], dim=1).transpose(1, 2)  # (B, T, 192)

        lstm_out, _ = self.lstm(merged)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        x = self.ln(lstm_out + attn_out)

        # Dual Global Pooling Block
        pooled = torch.cat([x.mean(dim=1), torch.amax(x, dim=1)], dim=1)

        x = self.drop1(F.relu(self.fc1(pooled)))
        x = self.drop2(F.relu(self.fc2(x)))
        return self.fc3(x).squeeze(-1)


def load_distilbert_classifier(num_labels: int = 2) -> Any:
    """Loads the pretrained DistilBERT sequence-classification head used for Model 5.

    Requires the ``transformers`` package. Included only for reproducibility of the
    notebook's Model 5 (DistilBERT fine-tuning); the final CNN-LSTM Hybrid inference
    path in ``inference.py`` does NOT use this function.
    """
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )


_MODEL_BUILDERS = {
    "bilstm": LSTMNet,
    "bilstm_attention": BiLSTMAttentionNet,
    "cnn": CNNTextClassifierNet,
    "cnn_lstm_hybrid": CNNLSTMHybridNet,
}


def get_model(
    model_name: str,
    vocab_size: int = 50_000,
    embed_dim: int = 128,
    max_len: int = 400,
    **kwargs: Any,
) -> nn.Module:
    """Model factory used by train.py / evaluate.py / inference.py.

    Args:
        model_name: one of "bilstm", "bilstm_attention", "cnn", "cnn_lstm_hybrid", "distilbert".
        vocab_size: embedding vocabulary size (MAX_WORDS in the notebook).
        embed_dim: embedding dimension.
        max_len: sequence length (unused by the architectures themselves — they are
            length-agnostic — but accepted for interface consistency and validation).
        **kwargs: forwarded to the underlying model constructor (e.g. padding_idx).

    Raises:
        ValueError: if model_name is not recognized.
    """
    del max_len  # architectures here are sequence-length agnostic; kept for API symmetry
    name = model_name.lower().strip()

    if name == "distilbert":
        return load_distilbert_classifier(**kwargs)

    if name not in _MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model_name={model_name!r}. "
            f"Expected one of {list(_MODEL_BUILDERS) + ['distilbert']}."
        )

    builder = _MODEL_BUILDERS[name]
    return builder(vocab_size=vocab_size, embed_dim=embed_dim, **kwargs)
