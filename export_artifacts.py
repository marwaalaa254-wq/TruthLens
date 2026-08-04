#!/usr/bin/env python
"""TruthLens artifact export & validation utility.

Converts the original notebook-era artifacts into the standardized project
layout (``models/best_cnn_lstm.pt``, ``artifacts/tokenizer.pkl``,
``artifacts/vocabulary.pkl``), validating every saved file by reloading it.

Background
----------
The uploaded project contains a genuine trained checkpoint
(``best_cnn_lstm_hybrid.pt``, a raw ``state_dict`` produced by the notebook's
``ModelCheckpoint`` callback) but the notebook never pickled its in-memory
``SimpleTokenizer`` to disk -- no ``tokenizer.pkl`` / ``vocabulary.pkl`` exists
anywhere in the source project. Because ``SimpleTokenizer.fit_on_texts`` is
fully deterministic given the same input order (Python's ``Counter.most_common``
breaks ties by insertion order) and the original ``True.csv`` / ``Fake.csv``
files are available, this script can deterministically RECONSTRUCT the exact
tokenizer the checkpoint was trained with by re-running the real, extracted
notebook pipeline (``src.preprocessing``) end-to-end with ``random_state=42``.

This is not fabrication: the reconstruction is validated below by scoring the
checkpoint on the reproduced test split and confirming it reaches the
notebook's own reported test accuracy (0.9922) -- a mismatched/fabricated
tokenizer would instead produce near-random (~0.50) accuracy, as was
empirically demonstrated when testing this exact checkpoint against a
differently-preprocessed (but equally deterministic) tokenizer variant. See
``ARTIFACT_AUDIT.md`` for the full investigation.

If ``--true-data``/``--fake-data`` are not supplied and no existing
``tokenizer.pkl``/``vocabulary.pkl`` are given via ``--tokenizer``/``--vocabulary``,
this script fails with a clear error rather than fabricating a tokenizer.

Usage:
    python export_artifacts.py --discover
    python export_artifacts.py \\
        --checkpoint path/to/best_cnn_lstm_hybrid.pt \\
        --true-data path/to/True.csv --fake-data path/to/Fake.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models import CNNLSTMHybridNet
from src.preprocessing import (
    MAX_LEN,
    MAX_WORDS,
    EMBED_DIM,
    RANDOM_SEED,
    build_tokenizer,
    download_nltk_resources,
    encode_texts,
    load_and_merge_data,
    load_tokenizer,
    load_vocabulary,
    preprocess_dataframe,
    save_tokenizer,
    save_vocabulary,
    split_dataset,
)
from src.utils import calculate_metrics, get_device, save_checkpoint, set_seed, setup_logger

logger = setup_logger("truthlens.export_artifacts")

# Known reported metrics for the shipped CNN-LSTM Hybrid checkpoint, taken
# verbatim from the notebook's own printed benchmark (Phase 8, cells 82/91).
# Recorded here as provenance metadata -- NOT recomputed/fabricated -- and
# overwritten with freshly measured metrics whenever a live evaluation runs
# below (i.e. whenever --true-data/--fake-data are supplied).
REPORTED_METRICS = {
    "accuracy": 0.9922,
    "precision": 0.9945,
    "recall": 0.9912,
    "f1": 0.9928,
    "roc_auc": 0.9994,
    "source": "notebook Phase 8 benchmark (cells 82/91), reported not recomputed",
}

DISCOVERY_CANDIDATE_DIRS = [Path("."), Path(".."), Path("../.."), Path("../Fake & real")]
DISCOVERY_CHECKPOINT_NAMES = ["best_cnn_lstm_hybrid.pt", "best_cnn_lstm.pt"]
DISCOVERY_CSV_NAMES = {"true": ["True.csv"], "fake": ["Fake.csv"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standardized TruthLens artifacts.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                         help="Path to the original raw state_dict checkpoint "
                              "(e.g. best_cnn_lstm_hybrid.pt).")
    parser.add_argument("--tokenizer", type=Path, default=None,
                         help="Path to an existing fitted tokenizer.pkl, if already available.")
    parser.add_argument("--vocabulary", type=Path, default=None,
                         help="Path to an existing vocabulary.pkl, if already available.")
    parser.add_argument("--true-data", type=Path, default=None,
                         help="Path to True.csv, used to deterministically rebuild the "
                              "tokenizer/vocabulary if not already available.")
    parser.add_argument("--fake-data", type=Path, default=None,
                         help="Path to Fake.csv (see --true-data).")
    parser.add_argument("--discover", action="store_true",
                         help="Scan common relative locations for source artifacts.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def discover_sources() -> dict[str, Optional[Path]]:
    """Scans DISCOVERY_CANDIDATE_DIRS for a checkpoint + True.csv/Fake.csv."""
    found: dict[str, Optional[Path]] = {"checkpoint": None, "true_data": None, "fake_data": None}
    for d in DISCOVERY_CANDIDATE_DIRS:
        if not d.exists():
            continue
        if found["checkpoint"] is None:
            for name in DISCOVERY_CHECKPOINT_NAMES:
                candidate = d / name
                if candidate.exists():
                    found["checkpoint"] = candidate
                    break
        if found["true_data"] is None:
            for name in DISCOVERY_CSV_NAMES["true"]:
                candidate = d / name
                if candidate.exists():
                    found["true_data"] = candidate
                    break
        if found["fake_data"] is None:
            for name in DISCOVERY_CSV_NAMES["fake"]:
                candidate = d / name
                if candidate.exists():
                    found["fake_data"] = candidate
                    break
    return found


def fmt_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.discover:
        logger.info("Discovery mode: scanning common relative locations...")
        found = discover_sources()
        for key, value in found.items():
            logger.info(f"  {key}: {value if value else 'NOT FOUND'}")
        args.checkpoint = args.checkpoint or found["checkpoint"]
        args.true_data = args.true_data or found["true_data"]
        args.fake_data = args.fake_data or found["fake_data"]

    if args.checkpoint is None or not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            "No genuine trained checkpoint found. Pass --checkpoint pointing to the "
            "original best_cnn_lstm_hybrid.pt, or use --discover. Refusing to fabricate "
            "random untrained weights."
        )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load and validate the source checkpoint against the real architecture --- #
    logger.info(f"Loading source checkpoint: {args.checkpoint} ({fmt_size(Path(args.checkpoint))})")
    raw_state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(raw_state_dict, dict):
        raise ValueError(f"Unexpected checkpoint type {type(raw_state_dict)!r}; expected a state_dict.")

    model = CNNLSTMHybridNet(vocab_size=args.max_words, embed_dim=args.embed_dim)
    load_result = model.load_state_dict(raw_state_dict, strict=True)
    logger.info(f"strict=True load result: {load_result}")
    embedding_shape = tuple(raw_state_dict["embedding.weight"].shape)
    logger.info(f"embedding.weight shape: {embedding_shape}")
    if embedding_shape != (args.max_words, args.embed_dim):
        raise ValueError(
            f"Embedding shape {embedding_shape} does not match expected "
            f"({args.max_words}, {args.embed_dim}). Aborting export."
        )

    # --- 2. Tokenizer / vocabulary: reuse existing or deterministically rebuild --- #
    metrics_payload = dict(REPORTED_METRICS)

    if args.tokenizer and Path(args.tokenizer).exists():
        logger.info(f"Using existing tokenizer: {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)
    elif args.true_data and args.fake_data:
        logger.info(
            "No existing tokenizer.pkl found in the source project -- "
            "deterministically reconstructing it from the original datasets "
            "(same code path + random_state=42 as the notebook)."
        )
        download_nltk_resources()
        df = load_and_merge_data(args.true_data, args.fake_data)
        df = preprocess_dataframe(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, random_state=args.seed)
        logger.info(f"Reconstructed split sizes -- train:{len(X_train)} val:{len(X_val)} test:{len(X_test)}")

        tokenizer = build_tokenizer(X_train, max_words=args.max_words)
        logger.info(f"Reconstructed vocabulary size: {len(tokenizer.word_index):,}")

        # Validate the reconstruction: score the checkpoint on the reproduced
        # test split and confirm it matches the notebook's own reported accuracy.
        X_te_seq = encode_texts(tokenizer, X_test, max_len=args.max_len)
        model.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(X_te_seq, dtype=torch.long, device=device)
            probs = torch.sigmoid(model(x_t)).cpu().numpy()
        measured = calculate_metrics(y_test.values, probs, threshold=0.5)
        logger.info(f"Measured test metrics on reconstructed split: {measured}")
        if abs(measured["accuracy"] - REPORTED_METRICS["accuracy"]) > 0.01:
            raise RuntimeError(
                f"Reconstructed tokenizer does NOT reproduce the reported accuracy "
                f"(measured {measured['accuracy']:.4f} vs reported {REPORTED_METRICS['accuracy']:.4f}). "
                "Refusing to export a mismatched tokenizer -- investigate preprocessing drift."
            )
        logger.info("Reconstruction validated: measured accuracy matches the reported notebook result.")
        metrics_payload = {**measured, "source": "measured on reconstructed test split via export_artifacts.py"}
    else:
        raise FileNotFoundError(
            "No tokenizer.pkl found anywhere in the source project, and --true-data/--fake-data "
            "were not supplied to deterministically rebuild one. Refusing to fabricate a tokenizer. "
            "Re-run with --true-data/--fake-data pointing at the original ISOT CSV files."
        )

    tokenizer_path = args.artifacts_dir / "tokenizer.pkl"
    vocabulary_path = args.artifacts_dir / "vocabulary.pkl"
    save_tokenizer(tokenizer, tokenizer_path)
    save_vocabulary(tokenizer.word_index, vocabulary_path)

    # --- 3. Save the standardized checkpoint --- #
    checkpoint_path = args.models_dir / "best_cnn_lstm.pt"
    model_config = {
        "vocab_size": args.max_words,
        "embed_dim": args.embed_dim,
        "padding_idx": 0,
        "kernel_sizes": [3, 5, 7],
        "lstm_hidden": 128,
        "attn_heads": 4,
    }
    save_checkpoint(
        checkpoint_path,
        model=model,
        model_name="cnn_lstm_hybrid",
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
        source_checkpoint=str(args.checkpoint),
    )

    # --- 4. Reload and validate every saved artifact --- #
    logger.info("Validating saved artifacts by reloading them...")
    reloaded_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validation_model = CNNLSTMHybridNet(vocab_size=args.max_words, embed_dim=args.embed_dim)
    reload_result = validation_model.load_state_dict(reloaded_ckpt["model_state_dict"], strict=True)
    logger.info(f"Reloaded checkpoint strict=True result: {reload_result}")

    reloaded_tokenizer = load_tokenizer(tokenizer_path)
    reloaded_vocab = load_vocabulary(vocabulary_path)
    assert len(reloaded_tokenizer.word_index) == len(reloaded_vocab), "Tokenizer/vocabulary size mismatch!"

    validation_model.eval()
    with torch.no_grad():
        dummy = torch.randint(0, args.max_words, (1, args.max_len), dtype=torch.long)
        out = validation_model(dummy)
    assert out.shape == (1,), f"Unexpected model output shape: {out.shape}"

    print("\n" + "=" * 60)
    print("TruthLens Artifact Export -- Summary")
    print("=" * 60)
    for path in [checkpoint_path, tokenizer_path, vocabulary_path]:
        print(f"  {path}  ({fmt_size(path)})")
    print(f"\nstrict=True load result : {reload_result}")
    print(f"Vocabulary size         : {len(reloaded_vocab):,}")
    print(f"Metrics                 : {metrics_payload}")
    print("All artifacts validated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
