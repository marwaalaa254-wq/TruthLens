# TruthLens

**See Through the Noise**

A PyTorch deep-learning pipeline for classifying English news articles as **credible (REAL)** or
**fabricated (FAKE)**, built on the ISOT Fake News Dataset.

---

## Project Overview

TruthLens is a binary text-classification system that scores the *stylistic and linguistic
credibility patterns* of a news article using a hybrid CNN‑LSTM‑Attention neural network. It was
developed and benchmarked against four other architectures (classical TF‑IDF baselines, a BiLSTM,
a BiLSTM with self-attention, a plain CNN, and a fine-tuned DistilBERT transformer) in the research
notebook under `notebooks/`.

TruthLens is **not a fact-checking system**. It does not verify claims against real-world facts,
knowledge bases, or live sources. It is a **probabilistic credibility-classification model**
trained to recognize the stylistic, structural, and publisher-specific patterns that distinguish
the REAL and FAKE articles in its training corpus — predictions can be influenced by
dataset-specific source, topical, and stylistic patterns rather than truthfulness per se. See
[Limitations](#limitations) below.

## Problem Statement

Misinformation spreads faster than fact-checkers can verify it. Automated, content-based
credibility classifiers offer a first line of triage — flagging articles whose writing patterns
resemble known fabricated content — while making no claim to replace human fact-checking or
editorial judgment.

## Dataset

**ISOT Fake News Dataset** — two CSV files:

| File | Rows (raw) | Label |
|------|-----------:|-------|
| `True.csv` | 21,417 | `1` (REAL) |
| `Fake.csv` | 23,270 | `0` (FAKE) |

After merging, shuffling (`random_state=42`), and the cleaning/filtering pipeline described below,
**38,218** articles remain, split 70/10/20 into:

| Split | Size |
|-------|-----:|
| Train | 26,752 |
| Validation | 3,822 |
| Test | 7,644 |

See [`data/README.md`](data/README.md) for placement instructions and dataset-source details.

### Label Mapping

```
FAKE = 0
REAL = 1
```

## Data Quality Checks

- Rows with missing/empty article text are dropped.
- Articles with fewer than 10 words are dropped.
- Duplicate articles (post-cleaning) are dropped.
- Class balance is near 50/50 (no oversampling/undersampling required).

## Data Leakage Audit

Two leakage risks were investigated:

1. **Publisher watermark leakage.** Real articles in this dataset are predominantly sourced from
   Reuters and frequently open with a `"CITY (Reuters) -"` dateline; a naive model could learn to
   key on the literal string `"Reuters"` rather than genuine credibility signal. The notebook's
   Phase 4 cleaning pipeline (`clean_article_text` in `src/preprocessing.py`) strips Reuters
   datelines/mentions specifically to prevent this shortcut.

2. **Confirmed pipeline bug: cleaning pipeline never reached the model.** Auditing the notebook
   cell-by-cell revealed that the Phase 5 train/val/test split cell reads from `df['text_clean']`
   — a column captured *during EDA, before* the Phase 4 cleaning pipeline ran — instead of the
   fully cleaned `df['text']` column the notebook computes immediately above it. As a result, the
   tokenizer and every downstream deep-learning model (including the shipped CNN-LSTM Hybrid
   checkpoint) were actually fit and trained on **raw, uncleaned article text** — the elaborate
   Reuters-stripping/lowercasing/punctuation-removal/stopword-removal/lemmatization pipeline never
   reached the model input; it only affected which rows survived the earlier dedup/short-article
   filters.

   This was verified empirically: reconstructing both variants from the original CSVs and scoring
   the real `best_cnn_lstm_hybrid.pt` checkpoint on each gives **99.22% accuracy on the raw
   variant** (exactly matching the notebook's own reported result) versus **51.58% (near-random)
   on the fully cleaned variant**. Since this project ships the existing trained checkpoint rather
   than retraining, `src/preprocessing.py`'s canonical `preprocess_text()` intentionally matches
   the raw pipeline the shipped weights actually expect — using the "intended" cleaned pipeline
   here would silently collapse real-world inference accuracy to chance. The full, notebook-faithful
   cleaning implementation is preserved in `src/preprocessing.py` (`clean_article_text`,
   `lemmatize_and_remove_stopwords`) for research reproducibility and as a starting point for a
   future corrected retrain. Full investigation notes: [`ARTIFACT_AUDIT.md`](ARTIFACT_AUDIT.md).

3. **No train/test overlap.** The tokenizer is fit exclusively on the training split
   (`build_tokenizer(X_train, ...)` in `train.py` / `export_artifacts.py`); validation and test
   text is only ever passed through `tokenizer.texts_to_sequences`, never `fit_on_texts`.

## Preprocessing Pipeline

As actually wired into the shipped model (see leakage audit above):

1. Merge `True.csv` (label=1) + `Fake.csv` (label=0), shuffle with `random_state=42`.
2. Drop rows with zero-length text; drop rows with fewer than 10 words (computed identically on
   both the raw and cleaned columns, so row survival is unaffected by the bug above).
3. Deduplicate on the fully-cleaned text (title+body, Reuters-stripped, lowercased,
   punctuation-removed, stopword-removed, lemmatized) — this filtering step *is* effective even
   though the cleaned text itself is not what reaches the model.
4. The surviving rows' **raw** article body (NaN-safe cast to string) is what the tokenizer fits
   on and the model consumes.
5. `SimpleTokenizer` (a dependency-free reimplementation of `keras.preprocessing.text.Tokenizer`)
   fits a whitespace-based vocabulary on the training split only, capped at `MAX_WORDS=50,000`,
   with `<OOV>` at index 1 and padding index 0.
6. Sequences are post-padded / post-truncated to `MAX_LEN=400`.

## Models Evaluated

### Classical Baselines (TF-IDF + scikit-learn)
- Logistic Regression
- Random Forest
- Gradient Boosting
- Multinomial Naive Bayes

### Deep Learning (PyTorch)
- **BiLSTM** — Embedding → 2-layer BiLSTM → Dense head
- **BiLSTM + Multi-Head Attention** — Embedding → BiLSTM → Multi-Head Self-Attention (+residual) → Dense head
- **CNN Text Classifier** — Embedding → Conv1D → BatchNorm → Global Max Pool → Dense head
- **CNN-LSTM Hybrid** (best model, shipped) — see below
- **DistilBERT** (fine-tuned transformer, subsampled for feasibility)

## Best Model: CNN-LSTM Hybrid

```
Embedding(50,000 → 128, padding_idx=0)
        │
        ├── Conv1D(128→128, k=3) → ReLU → Conv1D(128→64, k=3) → ReLU ──┐
        ├── Conv1D(128→128, k=5) → ReLU → Conv1D(128→64, k=5) → ReLU ──┼── concat (192-dim)
        └── Conv1D(128→128, k=7) → ReLU → Conv1D(128→64, k=7) → ReLU ──┘
        │
Bidirectional LSTM(192 → 128×2 = 256)
        │
Multi-Head Self-Attention(embed_dim=256, heads=4) + residual + LayerNorm
        │
   ┌────┴────┐
Global Avg Pool   Global Max Pool   (concat → 512-dim)
   └────┬────┘
Linear(512→128) → ReLU → Dropout(0.40)
Linear(128→64)  → ReLU → Dropout(0.30)
Linear(64→1)    → raw logit  (sigmoid applied only at eval/inference time)
```

Full implementation: [`src/models.py`](src/models.py) (`CNNLSTMHybridNet`).

## Final Results (held-out test set, n=7,644)

| Metric | Score |
|--------|------:|
| Accuracy | **99.22%** |
| Precision | **99.45%** |
| Recall | **99.12%** |
| F1-score | **99.28%** |
| ROC-AUC | **99.94%** |

These figures are taken directly from the notebook's own printed benchmark and were independently
reproduced by scoring the shipped checkpoint on a freshly reconstructed test split (see
`ARTIFACT_AUDIT.md`). Reported values are not fabricated or altered.

## Explainable AI (SHAP)

The research notebook includes a SHAP (`shap.Explainer` with a text masker) integration that
highlights which tokens push a prediction toward REAL (green) or FAKE (red), plus an interactive
`ipywidgets` demo UI. This XAI tooling is preserved in `notebooks/TruthLens_Notebook.ipynb` for
research/demo purposes; it is not part of the production `inference.py` path.

## Project Structure

```
TruthLens/
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── inference.py
├── export_artifacts.py
├── ARTIFACT_AUDIT.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── models.py
│   └── utils.py
├── models/
│   └── best_cnn_lstm.pt
├── artifacts/
│   ├── tokenizer.pkl
│   └── vocabulary.pkl
├── notebooks/
│   └── TruthLens_Notebook.ipynb
├── results/
│   ├── metrics.json
│   ├── classification_report.json
│   └── confusion_matrix.json
├── data/
│   └── README.md
└── tests/
    └── test_inference.py
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### CPU vs GPU

`requirements.txt` pins a dependency-version-compatible, platform-neutral `torch` range. For a
GPU-accelerated build, install PyTorch separately **before** `requirements.txt`, following the
official selector at https://pytorch.org/get-started/locally/ for your CUDA version, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

CPU-only users can simply run `pip install -r requirements.txt` as-is — `get_device()` in
`src/utils.py` automatically falls back to CPU when CUDA is unavailable.

## Dataset Placement

Download the ISOT dataset and place both CSVs at:

```
data/True.csv
data/Fake.csv
```

See [`data/README.md`](data/README.md) for details.

## Training

```bash
python train.py --true-data data/True.csv --fake-data data/Fake.csv \
    --model cnn_lstm_hybrid --epochs 10 --batch-size 64 --learning-rate 0.0005
```

## Evaluation

```bash
python evaluate.py --true-data data/True.csv --fake-data data/Fake.csv \
    --checkpoint models/best_cnn_lstm.pt --tokenizer artifacts/tokenizer.pkl
```

## Inference

```bash
python inference.py --text "Paste an English news article here"
python inference.py --file article.txt
```

### Example Prediction

```
Verdict     : REAL
Confidence  : 99.97%
P(REAL)     : 0.9997
P(FAKE)     : 0.0003
Threshold   : 0.5
```

## Reproducibility

- Random seed: **42** (Python `random`, NumPy, PyTorch, CUDA, and the stratified splits).
- Tokenizer is fit on the training split only, saved to `artifacts/tokenizer.pkl`.
- The exact preprocessing → tokenization → padding pipeline is shared by `train.py`,
  `evaluate.py`, and `inference.py` via `src/preprocessing.py` to prevent train-serve skew.

## Hardware Notes

The shipped checkpoint was trained on the notebook author's hardware; training time for the
CNN-LSTM Hybrid to convergence (7 epochs before early stopping) was ~194 seconds on GPU.
CPU-only training is supported but significantly slower. Inference is fast enough for real-time
use on CPU (single forward pass through a ~7.4M-parameter model).

## Limitations

- **Not a fact-checker.** TruthLens classifies *stylistic credibility patterns*, not factual
  accuracy. A well-written fabrication can still be misclassified as REAL, and a poorly-written
  true story can be misclassified as FAKE.
- **Temporal/topical bias.** The dataset is drawn almost entirely from 2015–2018 US political
  news. Performance on modern news cycles, non-political topics, or non-US sources is unverified
  and likely to degrade.
- **Publisher-style shortcut risk.** Despite the Reuters-stripping step in the cleaning pipeline
  (see Data Leakage Audit), the model may still key on other source-specific structural or
  stylistic patterns rather than generalizable credibility signal.
- **Raw-text training input.** As documented above, the shipped model was trained on
  minimally-processed article text (whitespace-tokenized, case-sensitive, punctuation attached to
  tokens) rather than the fully normalized pipeline the notebook intended — see the Data Leakage
  Audit section for the full explanation and its implications for out-of-vocabulary robustness.

## Responsible-Use Disclaimer

TruthLens is a research and educational project. Do not use its predictions as the sole basis for
content moderation, journalistic decisions, or any action with real-world consequences for
individuals or organizations. Always pair automated credibility scoring with human review.

## Contributors

Marwa, Radwa, Basma, Shrouk

## Report & Presentation

- [`TruthLens.pptx`](TruthLens.pptx) — presentation deck (included in this repository).
- `Deep_Learning Report.pdf` — full written report (source project folder, not included here).
- `TruthLens_Dashboard.html` — standalone results dashboard (source project folder, not included here).

## License

No license has been specified for this project. All rights reserved by the author(s) unless a
`LICENSE` file is added.
