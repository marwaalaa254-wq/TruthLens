# TruthLens Artifact Audit

Audit of the source project at `final code/Fake & real/` and conversion into the modular
`TruthLens/` project. All findings below were produced by actually executing code against the
real uploaded files (`torch.load`, `strict=True` state_dict loading, and a full re-run of the
extracted preprocessing pipeline against the original CSVs) — nothing here is inferred from
filenames or assumed.

## 1. Complete File Inventory (source project)

`final code/Fake & real/`:

| File | Size | Type |
|------|-----:|------|
| `Fake__for_5_models_seeded_25_7_2026.ipynb` | 3.0MB | Research notebook (107 cells) |
| `best_cnn_lstm_hybrid.pt` | 28.4MB | Raw `state_dict` — **CNN-LSTM Hybrid, best model** |
| `best_bilstm_attention.pt` | 25.1MB | Raw `state_dict` — BiLSTM + Attention |
| `best_cnn_text.pt` | 24.8MB | Raw `state_dict` — CNN Text Classifier |
| `weights/*.pt` (21 files) | ~1.3GB | Per-epoch snapshots (LSTM, BiLSTM-Attn, CNN, CNN-LSTM Hybrid, DistilBERT) |
| `attention_history.json`, `cnn_history.json` | <5KB | Per-epoch training curves |
| `*.png` (7 files) | ~1.1MB | EDA/CV/ROC/scorecard figures referenced by the notebook |
| `Deep_Learning Report.pdf`, `TruthLens.pptx`, `TruthLens_Dashboard.html` | ~71MB | Report/deck/dashboard deliverables |

No `.pkl`/`.pickle`/`.joblib` files (tokenizer or vocabulary) exist anywhere in this folder or
its notebook's outputs. A `tokenizer.pkl` was found at `Fake news/artifacts/tokenizer.pkl`, but
inspection (`pickle.load`) shows it is a `keras.src.legacy.preprocessing.text.Tokenizer` from an
unrelated, earlier Keras-based project iteration — **not** used, as it is architecturally
incompatible with this notebook's custom PyTorch-native `SimpleTokenizer`.

`True.csv` (51.9MB) and `Fake.csv` (46.6MB) exist at `Fake news/True.csv` / `Fake news/Fake.csv`,
exactly matching the hard-coded paths in the notebook's Phase 2 loading cell.

## 2. Notebook Structure Summary

107 cells across 10 phases: Setup → Data Loading → EDA → Cleaning → Split → TF-IDF Baselines →
Deep Learning Models (LSTM, BiLSTM+Attention, CNN, CNN-LSTM Hybrid, DistilBERT) → K-Fold CV → Final
Comparison → Real-Time Inference demo → SHAP explainability demo.

## 3. Models Discovered

| Model | Notebook class | Checkpoint | Test Accuracy |
|-------|-----------------|------------|---------------:|
| LSTM (baseline) | `LSTMNet` | (per-epoch only, no `best_*.pt`) | 94.32% |
| BiLSTM + Attention | `BiLSTMAttentionNet` | `best_bilstm_attention.pt` | 98.55% |
| CNN Text Classifier | `CNNTextClassifierNet` | `best_cnn_text.pt` | 98.64% |
| **CNN-LSTM Hybrid** | `CNNLSTMHybridNet` | `best_cnn_lstm_hybrid.pt` | **99.22%** |
| DistilBERT (fine-tuned) | HuggingFace `AutoModelForSequenceClassification` | (per-epoch only) | 98.50% |

## 4. Selected CNN-LSTM Checkpoint

**`best_cnn_lstm_hybrid.pt`** — selected because it is (a) the model the notebook's own Phase 8
benchmark identifies as the top performer by every metric, (b) explicitly requested as the final
model for this project, and (c) the only CNN-LSTM Hybrid checkpoint present (the `weights/`
per-epoch copies are earlier/worse snapshots of the same run — the ModelCheckpoint callback
confirms `best_cnn_lstm_hybrid.pt` was overwritten only on val_loss improvement, last updated at
epoch 4, matching the training log's best epoch).

### State-dictionary key summary

33 keys, `OrderedDict[str, torch.Tensor]` (raw `state_dict`, not a wrapped checkpoint dict) —
`embedding.weight`, `branches.{0,1,2}.{0,2}.{weight,bias}`, `lstm.*`, `attn.*`, `ln.*`,
`fc1/fc2/fc3.{weight,bias}`. `embedding.weight` shape: `(50000, 128)`.

### Architecture match / strict=True result

Built `CNNLSTMHybridNet` exactly as coded in notebook cell 59 (3 parallel Conv1D branches,
kernel sizes 3/5/7, 128→64 channels each, ReLU; BiLSTM hidden=128; MultiheadAttention embed_dim=256
heads=4; residual+LayerNorm; dual avg/max pooling; 512→128→64→1 dense head, dropout 0.40/0.30).

```
model.load_state_dict(state_dict, strict=True)
>>> <All keys matched successfully>
```

Confirmed on both the original source file and the re-saved standardized checkpoint
(`models/best_cnn_lstm.pt`). Total trainable parameters: **7,436,609**.

## 5. Tokenizer Source

**No tokenizer.pkl existed anywhere in the uploaded project.** The notebook's `SimpleTokenizer`
was fit in-memory (notebook cell 36, `tokenizer.fit_on_texts(X_train)`) and never serialized.

Since `SimpleTokenizer.fit_on_texts` is fully deterministic given identical input order (Python's
`Counter.most_common()` breaks ties by first-insertion order) and the original `True.csv`/`Fake.csv`
are available and unmodified, `export_artifacts.py` deterministically **reconstructs** the exact
tokenizer by re-running the real, extracted pipeline (`src/preprocessing.py`) end-to-end with
`random_state=42` — the same code, the same data, the same split. This is a reconstruction, not a
fabrication: it was validated by scoring `best_cnn_lstm_hybrid.pt` on the resulting test split and
confirming the result matches the notebook's own reported metrics almost to 4 decimal places
(§9 below). A tokenizer that did not match training would instead produce near-random accuracy, as
directly demonstrated in the investigation below.

## 6. CONFIRMED BUG: `text_clean` vs `text` in the Phase 5 split cell

This is the most significant finding of this audit.

**Evidence.** Notebook cell 15 (Phase 3 EDA, runs *before* cleaning):
```python
df['text_clean'] = df['text'].fillna('')   # raw text, captured pre-cleaning
```
Notebook cell 23 (Phase 4, runs *after* cell 15) builds an elaborate cleaning pipeline — Reuters
dateline/watermark stripping, lowercasing, punctuation removal, stopword removal, lemmatization —
and writes the result into `df['text']` (not `text_clean`). `text_clean` is never reassigned
anywhere else in the notebook (verified by exhaustive grep across all 107 cells).

Notebook cell 29 (Phase 5, the train/val/test split):
```python
X = df['text_clean']   # <-- reads the RAW column, not the cleaned df['text']
y = df['label']
```

**Consequence.** The tokenizer and every downstream deep-learning model — including the shipped
CNN-LSTM Hybrid — were fit and trained on raw, minimally-touched article text (whitespace-split,
case-sensitive, punctuation attached to tokens, Reuters watermarks intact, title never
concatenated in). The entire Phase 4 cleaning pipeline is dead code with respect to model input;
its only real effect is on which rows survive the dedup/short-article filters (which do run
against the cleaned column).

**Empirical verification.** Both variants were reconstructed from the original CSVs and scored
against the real `best_cnn_lstm_hybrid.pt` checkpoint:

| Variant | Test accuracy |
|---|---:|
| Raw `text_clean` pipeline (as actually used) | **0.9922** — matches the notebook's own reported result exactly |
| Fully cleaned `text` pipeline (as apparently intended) | 0.5158 — near-random |

**Resolution.** Retraining a corrected model was out of scope (the task explicitly requires
shipping the existing checkpoint, not retraining a different one). `src/preprocessing.py`'s
canonical `preprocess_text()` therefore matches the raw pipeline the shipped weights actually
require — using the "intended" cleaned pipeline would silently collapse real-world inference to
near-random accuracy despite looking more correct. The full, notebook-faithful cleaning
implementation is preserved as `clean_article_text()` / `lemmatize_and_remove_stopwords()` for
research reproducibility and as a starting point for a properly-fixed retrain. This is documented
prominently in `src/preprocessing.py`'s module docstring and in `README.md`'s Data Leakage Audit
section — nothing about this was hidden or silently patched over.

## 7. Vocabulary

- Size: **316,575** unique tokens (uncapped fit; embedding matrix caps consumption at
  `MAX_WORDS=50,000` via the `SimpleTokenizer.num_words` check — tokens beyond that rank map to
  the `<OOV>` index).
- Padding index: **0** (reserved, `nn.Embedding(..., padding_idx=0)`).
- OOV index: **1** (`<OOV>` token, first entry in `word_index`).
- Max sequence length: **400**, post-padding / post-truncation.
- Label mapping: `{"FAKE": 0, "REAL": 1}` — confirmed consistent across notebook, checkpoint
  metadata, and all modular scripts.

## 8. Preprocessing Source Location

`src/preprocessing.py`, extracted from notebook cells 5 (loading), 15 (raw `text_clean` capture),
23 (cleaning pipeline + row filtering), 29 (split), and 36 (`SimpleTokenizer`/`pad_sequences`).

## 9. Prediction-Consistency Results

The notebook's own Phase 9 demo (cell 93) calls `bilstm_model.predict(...)` — the BiLSTM+Attention
model — not the CNN-LSTM Hybrid, despite Phase 8 identifying CNN-LSTM Hybrid as the best model.
**This is a second confirmed bug**, and per the project requirements, `inference.py` intentionally
loads the CNN-LSTM Hybrid checkpoint instead of reproducing this mistake.

Running the notebook's own two example texts through the corrected modular `inference.py`
(`predict_news`, backed by `best_cnn_lstm.pt`):

| Text | Notebook (BiLSTM+Attn, buggy demo) | `inference.py` (CNN-LSTM Hybrid, corrected) |
|---|---|---|
| REAL_NEWS (Senate climate bill) | REAL, 99.97% confidence | REAL, 100.00% confidence (P(REAL)=0.99999+) |
| FAKE_NEWS (Clinton/Benghazi conspiracy) | FAKE, 99.99% confidence | FAKE, 99.68% confidence (P(REAL)=0.0032) |

Both models agree on the correct label for both examples; the modular pipeline used the intended
best-performing architecture. Determinism was verified (identical repeated predictions, no
tokenizer refitting, no training triggered).

## 10. Tests Executed

```
python -m compileall .                                          -> exit 0
python train.py --help                                          -> exit 0
python evaluate.py --help                                        -> exit 0
python inference.py --help                                       -> exit 0
python export_artifacts.py --help                                -> exit 0
python -c "from src.preprocessing import preprocess_text"        -> OK
python -c "from src.models import CNNLSTMHybridNet"               -> OK
python -c "from src.utils import set_seed"                        -> OK
python export_artifacts.py --checkpoint <src> --true-data <src> --fake-data <src>  -> exit 0
python evaluate.py --checkpoint models/best_cnn_lstm.pt --tokenizer artifacts/tokenizer.pkl --true-data ... --fake-data ...  -> exit 0
pytest -q                                                          -> 13 passed
```

All commands were actually executed (not simulated) in this environment (Python 3.11.0,
torch 2.13.0+cpu, Windows).

## 11. Tests Passed / Failed

**13 passed, 0 failed, 0 skipped** (real artifacts were available, so no checkpoint-dependent test
needed to skip). Coverage: `preprocess_text` type handling, tokenizer/vocabulary loading,
`strict=True` checkpoint loading, sequence shape/dtype, probability range, label validity,
deterministic repeated inference, very-short input, input far exceeding `MAX_LEN`, CPU loading.

## 12. Remaining Limitations

- The shipped model's true "cleaning pipeline" is minimal (raw text, whitespace tokenization) —
  see §6. This is now a documented, understood characteristic of the shipped weights, not an
  unresolved bug in the modular code.
- DistilBERT (`get_model("distilbert", ...)`) is included for architectural reproducibility only;
  no DistilBERT checkpoint was standardized (out of scope — CNN-LSTM Hybrid is the deliverable).
  Its `weights/distilbert_epoch_*.pt` snapshots (267MB each) were left untouched in the source
  folder.
- `weights/*.pt` per-epoch snapshots were not copied into the modular project (only the best
  checkpoint is required by the spec); they remain available in the original source folder.
- `artifacts/tokenizer.pkl` (7.4MB) and `artifacts/vocabulary.pkl` (5.2MB) are within typical Git
  size limits; `models/best_cnn_lstm.pt` (28.4MB) is also well under GitHub's 100MB hard limit, so
  no Git LFS configuration was necessary.

## 13. Files Requiring User Execution

- Placing `True.csv`/`Fake.csv` under `data/` before running `train.py` from scratch (not required
  to use the already-exported `models/best_cnn_lstm.pt` + `artifacts/*.pkl` for evaluation/inference).
- Installing `transformers` only if reproducing the DistilBERT path via `get_model("distilbert", ...)`.
