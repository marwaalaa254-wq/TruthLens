"""Smoke tests for the TruthLens preprocessing/model/inference pipeline.

Checkpoint-dependent tests are skipped with a clear reason when the real
trained artifacts (models/best_cnn_lstm.pt, artifacts/tokenizer.pkl,
artifacts/vocabulary.pkl) are not present -- they are never faked as passing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import encode_texts, load_tokenizer, load_vocabulary, preprocess_text
from src.models import CNNLSTMHybridNet, get_model
from src.utils import load_checkpoint, set_seed

CHECKPOINT_PATH = ROOT / "models" / "best_cnn_lstm.pt"
TOKENIZER_PATH = ROOT / "artifacts" / "tokenizer.pkl"
VOCABULARY_PATH = ROOT / "artifacts" / "vocabulary.pkl"

ARTIFACTS_AVAILABLE = CHECKPOINT_PATH.exists() and TOKENIZER_PATH.exists() and VOCABULARY_PATH.exists()
SKIP_REASON = (
    "Real trained artifacts not found under models/ and artifacts/ -- "
    "run export_artifacts.py first. Skipping rather than faking a pass."
)


# --------------------------------------------------------------------------- #
# preprocess_text
# --------------------------------------------------------------------------- #

def test_preprocess_text_accepts_string() -> None:
    result = preprocess_text("Some news article text.")
    assert isinstance(result, str)


def test_preprocess_text_none_returns_empty_string() -> None:
    assert preprocess_text(None) == ""


def test_preprocess_text_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        preprocess_text(12345)


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not TOKENIZER_PATH.exists(), reason=SKIP_REASON)
def test_tokenizer_loads() -> None:
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    assert hasattr(tokenizer, "word_index")
    assert len(tokenizer.word_index) > 0


@pytest.mark.skipif(not VOCABULARY_PATH.exists(), reason=SKIP_REASON)
def test_vocabulary_loads() -> None:
    vocabulary = load_vocabulary(VOCABULARY_PATH)
    assert isinstance(vocabulary, dict)
    assert len(vocabulary) > 0


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason=SKIP_REASON)
def test_checkpoint_loads_strict() -> None:
    device = torch.device("cpu")
    ckpt = load_checkpoint(CHECKPOINT_PATH, device=device)
    model_config = ckpt.get("model_config", {})
    model = get_model(
        ckpt.get("model_name", "cnn_lstm_hybrid"),
        vocab_size=ckpt.get("max_words", 50_000),
        embed_dim=ckpt.get("embed_dim", 128),
        **{k: v for k, v in model_config.items() if k not in {"vocab_size", "embed_dim"}},
    )
    result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0


# --------------------------------------------------------------------------- #
# End-to-end model behavior
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not ARTIFACTS_AVAILABLE, reason=SKIP_REASON)
class TestEndToEndInference:
    @classmethod
    def setup_class(cls) -> None:
        set_seed(42)
        cls.device = torch.device("cpu")
        cls.tokenizer = load_tokenizer(TOKENIZER_PATH)
        ckpt = load_checkpoint(CHECKPOINT_PATH, device=cls.device)
        model_config = ckpt.get("model_config", {})
        cls.model = get_model(
            ckpt.get("model_name", "cnn_lstm_hybrid"),
            vocab_size=ckpt.get("max_words", 50_000),
            embed_dim=ckpt.get("embed_dim", 128),
            **{k: v for k, v in model_config.items() if k not in {"vocab_size", "embed_dim"}},
        )
        cls.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        cls.model.eval()
        cls.max_len = ckpt.get("max_len", 400)

    def _predict_prob(self, text: str) -> float:
        cleaned = preprocess_text(text)
        seq = encode_texts(self.tokenizer, [cleaned], max_len=self.max_len)
        assert seq.shape == (1, self.max_len)
        x = torch.as_tensor(seq, dtype=torch.long)
        assert x.dtype == torch.long
        with torch.no_grad():
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()
        assert 0.0 <= prob <= 1.0
        return prob

    def test_sequence_length_and_dtype(self) -> None:
        self._predict_prob("A short test article about the economy.")

    def test_probability_in_valid_range(self) -> None:
        prob = self._predict_prob("Scientists announce a new discovery in renewable energy research.")
        assert 0.0 <= prob <= 1.0

    def test_returned_label_is_fake_or_real(self) -> None:
        prob = self._predict_prob("Breaking news about the local city council meeting.")
        label = "REAL" if prob >= 0.5 else "FAKE"
        assert label in {"FAKE", "REAL"}

    def test_deterministic_repeated_predictions(self) -> None:
        text = "The stock market closed higher today amid positive earnings reports."
        prob_1 = self._predict_prob(text)
        prob_2 = self._predict_prob(text)
        assert prob_1 == pytest.approx(prob_2, abs=1e-6)

    def test_very_short_input(self) -> None:
        self._predict_prob("News.")

    def test_long_input_exceeding_max_len(self) -> None:
        long_text = " ".join(["word"] * (self.max_len * 3))
        self._predict_prob(long_text)

    def test_cpu_loading_works(self) -> None:
        ckpt = load_checkpoint(CHECKPOINT_PATH, device=torch.device("cpu"))
        assert "model_state_dict" in ckpt
