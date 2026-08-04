"""Data loading, cleaning, tokenization and encoding pipeline for TruthLens.

Extracted from ``notebooks/TruthLens_Notebook.ipynb`` (Phase 2 "Load Dataset",
Phase 4 "Data Cleaning and Preprocessing", Phase 5 "Train/Val/Test Split", and
Phase 7.1 "Tokenization and Padding").

IMPORTANT -- CONFIRMED NOTEBOOK BUG (documented, not silently hidden)
----------------------------------------------------------------------
The notebook's Phase 4 cell builds an elaborate cleaning pipeline
(``clean_text`` + ``lemmatize_and_remove_stopwords``): it strips Reuters
datelines/watermarks, lowercases, removes punctuation, removes stopwords, and
lemmatizes -- and overwrites it into ``df['text']``.

However, the Phase 5 split cell reads from ``df['text_clean']`` instead of
``df['text']``. ``text_clean`` was assigned *earlier*, during Phase 3 EDA
(before Phase 4 cleaning ever ran), as a raw, minimally-touched copy of the
original article body (``df['text'].fillna('')``) -- title is not even
concatenated in. Because of this variable mix-up, the tokenizer, and every
downstream deep-learning model including the shipped CNN-LSTM Hybrid
checkpoint (``models/best_cnn_lstm.pt``), were actually fit and trained on
this raw, uncleaned text -- the entire Phase 4 cleaning pipeline is dead code
as far as model input is concerned (it only affects which rows survive via
dedup/short-article filtering, since those filters run against the cleaned
column).

This was verified empirically by reconstructing both variants from the
original ``True.csv`` / ``Fake.csv`` files and scoring the real
``best_cnn_lstm_hybrid.pt`` checkpoint on each:

    * raw ``text_clean`` pipeline  -> test accuracy 0.9922 (matches the
      notebook's own reported result exactly, see ``results/metrics.json``)
    * fully cleaned pipeline       -> test accuracy 0.5158 (near-random)

Since this project ships the *existing* trained checkpoint rather than
retraining (retraining a corrected pipeline is out of scope and was not
requested), ``preprocess_text`` below reproduces the raw pipeline that the
shipped weights actually expect -- using the fully cleaned pipeline here
would silently break inference (near-random predictions) despite "looking"
more correct. The full, notebook-faithful cleaning implementation is still
preserved and exposed as ``clean_article_text`` /
``lemmatize_and_remove_stopwords`` for research reproducibility, documentation,
and as a starting point should the project be retrained with the bug fixed.
See ``README.md`` -> "Data Leakage Audit" and ``ARTIFACT_AUDIT.md`` for details.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

MAX_WORDS: int = 50_000
MAX_LEN: int = 400
EMBED_DIM: int = 128
RANDOM_SEED: int = 42
OOV_TOKEN: str = "<OOV>"
PADDING_IDX: int = 0
LABEL_MAPPING: dict[str, int] = {"FAKE": 0, "REAL": 1}


def download_nltk_resources() -> None:
    """Downloads the NLTK resources used by the cleaning pipeline (idempotent)."""
    import nltk

    for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
        nltk.download(pkg, quiet=True)


# --------------------------------------------------------------------------- #
# Phase 2 -- Data loading & merging
# --------------------------------------------------------------------------- #

def load_and_merge_data(true_path: PathLike, fake_path: PathLike) -> pd.DataFrame:
    """Loads True.csv / Fake.csv, assigns labels, merges and shuffles.

    Args:
        true_path: path to the real-news CSV (assigned label=1 / REAL).
        fake_path: path to the fake-news CSV (assigned label=0 / FAKE).

    Returns:
        A shuffled DataFrame (``random_state=42``) with ``label`` (0/1),
        ``source`` ("Fake"/"Real") and ``label_name`` columns added.

    Raises:
        FileNotFoundError: if either path does not exist.
    """
    true_path, fake_path = Path(true_path), Path(fake_path)
    if not true_path.exists():
        raise FileNotFoundError(f"True-news CSV not found: {true_path}")
    if not fake_path.exists():
        raise FileNotFoundError(f"Fake-news CSV not found: {fake_path}")

    file_true = pd.read_csv(true_path, engine="python", encoding="utf-8", on_bad_lines="warn")
    file_fake = pd.read_csv(fake_path, engine="python", encoding="utf-8", on_bad_lines="warn")

    file_true["label"] = 1
    file_true["source"] = "Real"
    file_fake["label"] = 0
    file_fake["source"] = "Fake"

    df = pd.concat([file_true, file_fake], axis=0, ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df["label_name"] = df["label"].map({1: "Real", 0: "Fake"})
    return df


# --------------------------------------------------------------------------- #
# Phase 4 -- Cleaning (notebook-faithful; see module docstring re: the
# text_clean/text bug -- these functions are preserved for completeness and
# are NOT what the shipped checkpoint's tokenizer was actually fit on).
# --------------------------------------------------------------------------- #

def clean_article_text(text: str) -> str:
    """Notebook ``clean_text``: strips Reuters datelines/watermarks, lowercases,
    drops bracketed scraping artifacts, and collapses non-word characters to spaces.
    """
    if not isinstance(text, str):
        return ""

    # Prevent leak: remove Reuters source patterns to stop shortcut learning.
    text = re.sub(r"^.*?Reuters\s*-\s*", "", text)
    text = re.sub(r"\bReuters\b[^\w]*", "", text, flags=re.IGNORECASE)

    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)  # bracketed scraping artifacts / image credits
    text = re.sub(r"\W", " ", text)  # punctuation, URLs, symbols -> whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_and_remove_stopwords(text: str, stopwords_set: set[str], lemmatizer) -> str:
    """Notebook ``preprocess``: removes stopwords and short tokens (len<=2), lemmatizes the rest."""
    if not isinstance(text, str) or not text.strip():
        return ""
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords_set and len(t) > 2]
    return " ".join(tokens)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the notebook's Phase 3/4 row-level cleaning and filtering.

    Reproduces, in order:
        1. (Phase 3) ``text_clean`` = raw ``text`` (fillna only) -- captured
           *before* cleaning; this becomes the actual model-input column.
        2. Drop rows with ``word_count == 0`` on the raw text.
        3. (Phase 4) Drop rows with missing ``text``; fill missing ``title``.
        4. Drop articles with fewer than 10 words (on raw text).
        5. Combine ``title`` + ``text`` into a temporary ``content`` column.
        6. Apply :func:`clean_article_text` then
           :func:`lemmatize_and_remove_stopwords`, overwriting ``df['text']``
           (this cleaned column is used only for dedup/short-article
           filtering below and is retained for research purposes -- it is
           NOT the column the trained tokenizer/model actually consume).
        7. Drop rows left empty after cleaning; drop duplicate cleaned articles.

    Returns:
        The filtered DataFrame with both ``text_clean`` (raw model input) and
        ``text`` (fully cleaned, research/EDA only) columns present.
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    df = df.copy()

    # Phase 3: raw snapshot taken before any cleaning -- this is what the
    # shipped model's tokenizer/embedding were actually fit on.
    df["text_clean"] = df["text"].fillna("")
    df["title_clean"] = df["title"].fillna("")
    df["word_count"] = df["text_clean"].str.split().str.len()
    df = df[df["word_count"] > 0]

    # Phase 4: cleaning pipeline (kept for row-filtering fidelity + research reuse).
    stopwords_set = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df = df.dropna(subset=["text"]).copy()
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df = df[df["text"].str.split().str.len() >= 10].reset_index(drop=True)
    df["content"] = df["title"] + " " + df["text"]

    df["content"] = df["content"].apply(clean_article_text)
    df["text"] = df["content"].apply(lambda t: lemmatize_and_remove_stopwords(t, stopwords_set, lemmatizer))
    df = df.drop(columns=["content"])

    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Alias of :func:`clean_dataframe` (public name matching the project spec)."""
    return clean_dataframe(df)


def preprocess_text(text: str) -> str:
    """Canonical single-text preprocessing function used by train/evaluate/inference.

    This MUST match whatever the shipped checkpoint's tokenizer was actually
    fit on to avoid train-serve skew. As documented in the module docstring,
    that is the raw article body (NaN-safe cast to string) -- NOT the fully
    cleaned/lemmatized pipeline in :func:`clean_article_text` /
    :func:`lemmatize_and_remove_stopwords`, which the notebook computed but
    never actually routed into training due to a confirmed variable-reference
    bug (``text_clean`` vs ``text`` in the Phase 5 split cell).

    Args:
        text: raw input string (a news article, or article body).

    Returns:
        The string coerced to a safe ``str`` (empty string for ``None``/NaN/
        non-string input). Tokenization (whitespace split) happens later in
        :func:`encode_texts`, matching ``SimpleTokenizer.texts_to_sequences``.

    Raises:
        TypeError: if ``text`` is not a string and not ``None``.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"preprocess_text expected str or None, got {type(text)!r}")
    return text


# --------------------------------------------------------------------------- #
# Phase 5 -- Train / Val / Test split (70 / 10 / 20, stratified)
# --------------------------------------------------------------------------- #

def split_dataset(
    df: pd.DataFrame, random_state: int = RANDOM_SEED
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Reproduces the notebook's stratified 70/10/20 train/val/test split.

    Uses ``df['text_clean']`` as the feature column -- see module docstring:
    this is the raw text the shipped model was actually trained on.

    Returns:
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    from sklearn.model_selection import train_test_split

    X = df["text_clean"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=2 / 3, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------------------------------- #
# Phase 7.1 -- Tokenizer + padding
# --------------------------------------------------------------------------- #

class SimpleTokenizer:
    """PyTorch-native stand-in for ``keras.preprocessing.text.Tokenizer``.

    Same ``word_index`` / ``texts_to_sequences`` interface. Text is
    whitespace-split (matching the notebook exactly).
    """

    def __init__(self, num_words: Optional[int] = None, oov_token: Optional[str] = OOV_TOKEN) -> None:
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index: dict[str, int] = {}
        self.word_counts: Counter = Counter()

    def fit_on_texts(self, texts) -> None:
        """Fits the vocabulary on an iterable of strings. Call on TRAINING data only."""
        for text in texts:
            self.word_counts.update(str(text).split())

        idx = 1
        if self.oov_token is not None:
            self.word_index[self.oov_token] = idx
            idx += 1
        for word, _ in self.word_counts.most_common():
            self.word_index[word] = idx
            idx += 1

    def texts_to_sequences(self, texts) -> list[list[int]]:
        """Converts an iterable of strings into integer-ID sequences.

        Words absent from the vocabulary, or whose index falls outside the
        ``num_words`` cap, map to the OOV index (or are dropped if no OOV
        token was configured).
        """
        oov_idx = self.word_index.get(self.oov_token) if self.oov_token else None
        sequences = []
        for text in texts:
            seq = []
            for word in str(text).split():
                word_idx = self.word_index.get(word)
                if word_idx is not None and (self.num_words is None or word_idx < self.num_words):
                    seq.append(word_idx)
                elif oov_idx is not None:
                    seq.append(oov_idx)
            sequences.append(seq)
        return sequences


def pad_sequences(
    sequences: list[list[int]],
    maxlen: Optional[int] = None,
    padding: str = "post",
    truncating: str = "post",
    value: int = PADDING_IDX,
) -> np.ndarray:
    """PyTorch-native stand-in for ``keras.preprocessing.sequence.pad_sequences``.

    Applies post-padding and post-truncation by default (matching the notebook).
    """
    if maxlen is None:
        maxlen = max((len(s) for s in sequences), default=0)
    padded = np.full((len(sequences), maxlen), value, dtype="int64")
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        trunc = seq[-maxlen:] if truncating == "pre" else seq[:maxlen]
        if padding == "post":
            padded[i, : len(trunc)] = trunc
        else:
            padded[i, -len(trunc):] = trunc
    return padded


def build_tokenizer(train_texts, max_words: int = MAX_WORDS) -> SimpleTokenizer:
    """Fits a new :class:`SimpleTokenizer` on training texts ONLY.

    Never call this with validation/test data -- doing so would leak
    vocabulary information into evaluation.
    """
    tokenizer = SimpleTokenizer(num_words=max_words, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def encode_texts(tokenizer: SimpleTokenizer, texts, max_len: int = MAX_LEN) -> np.ndarray:
    """Converts texts to a padded/truncated ``(N, max_len)`` int array via ``tokenizer``."""
    sequences = tokenizer.texts_to_sequences(list(texts))
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")


def save_tokenizer(tokenizer: SimpleTokenizer, output_path: PathLike) -> None:
    """Pickles a fitted tokenizer to ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(input_path: PathLike) -> SimpleTokenizer:
    """Loads a pickled :class:`SimpleTokenizer` from ``input_path``.

    Raises:
        FileNotFoundError: if the tokenizer file does not exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {input_path}. Run train.py or export_artifacts.py first."
        )
    with open(input_path, "rb") as f:
        return pickle.load(f)


def save_vocabulary(vocabulary: dict[str, int], output_path: PathLike) -> None:
    """Pickles a word-to-index vocabulary mapping to ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(vocabulary, f)


def load_vocabulary(input_path: PathLike) -> dict[str, int]:
    """Loads a pickled word-to-index vocabulary mapping from ``input_path``.

    Raises:
        FileNotFoundError: if the vocabulary file does not exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Vocabulary not found at {input_path}. Run train.py or export_artifacts.py first."
        )
    with open(input_path, "rb") as f:
        return pickle.load(f)
