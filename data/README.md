# Data Directory

TruthLens is trained on the **ISOT Fake News Dataset**, distributed as two CSV files:

| File | Description | Label |
|------|-------------|-------|
| `True.csv` | Genuine news articles (mostly Reuters) | `1` (REAL) |
| `Fake.csv` | Fabricated / unreliable news articles | `0` (FAKE) |

## Expected columns

Both files must contain at least:

| Column | Type | Notes |
|--------|------|-------|
| `title` | string | Article headline |
| `text` | string | Article body |
| `subject` | string | Source category (used only for EDA) |
| `date` | string | Publication date (used only for EDA) |

## Where to place the files

Place both CSVs directly in this `data/` directory:

```
data/True.csv
data/Fake.csv
```

Then run training with:

```bash
python train.py --true-data data/True.csv --fake-data data/Fake.csv --model cnn_lstm_hybrid
```

## Label mapping

```
Fake.csv -> label = 0 (FAKE)
True.csv -> label = 1 (REAL)
```

This mapping is fixed throughout the entire project (preprocessing, training, evaluation,
inference) and must never be changed, since the trained checkpoint in `models/best_cnn_lstm.pt`
was produced with this exact convention.

## Why raw datasets are excluded from Git

The combined dataset is ~98 MB (`True.csv` ≈ 51 MB, `Fake.csv` ≈ 47 MB), which is impractical
to version in a Git repository and may be subject to redistribution restrictions depending on
the dataset's original license terms. `.gitignore` excludes `data/*.csv` by default — download
or copy the files locally before training.

## Dataset source

ISOT Fake News Dataset, University of Victoria — Information Security and Object Technology
(ISOT) Research Lab. Commonly distributed via Kaggle as "Fake and Real News Dataset".

## Data-license considerations

Confirm the redistribution terms of the ISOT dataset (or your specific Kaggle mirror) before
publishing the raw CSVs alongside this repository. This project ships only code and, where a
genuine trained checkpoint exists, model weights — never the raw dataset itself.
