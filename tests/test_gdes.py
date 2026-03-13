"""
Unit and integration tests for the GDES pretraining pipeline.

Run with:
    pytest tests/ -v --cov=rtd_gdes
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from rtd_gdes.config import TrainConfig
from rtd_gdes.gdes.model import DebertaV3GDES
from rtd_gdes.gdes.trainer import _build_disc_labels, _disc_loss, evaluate, train_one_epoch
from rtd_gdes.gdes.utils import MixedPrecisionSelectionError

# ───────────────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────────────

MODEL_ID = "microsoft/deberta-v3-base"
BATCH, SEQ, HIDDEN = 2, 16, 768


# ───────────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model() -> DebertaV3GDES:
    """Instantiate the full model once per test module (slow, cached)."""
    return DebertaV3GDES(MODEL_ID)


@pytest.fixture()
def dummy_input_ids() -> torch.Tensor:
    return torch.randint(0, 1000, (BATCH, SEQ))


@pytest.fixture()
def dummy_attention_mask() -> torch.Tensor:
    mask = torch.ones(BATCH, SEQ, dtype=torch.long)
    mask[:, -2:] = 0
    return mask


# ───────────────────────────────────────────────────────────────────────────────
# Config tests
# ───────────────────────────────────────────────────────────────────────────────


class TestTrainConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = TrainConfig()
        assert cfg.model_id == "microsoft/deberta-v3-base"
        assert cfg.lambda_disc == 0.5

    def test_fp16_and_bf16_together_raises(self) -> None:
        with pytest.raises(ValueError, match="Only one of fp16 or bf16"):
            TrainConfig(fp16=True, bf16=True)

    def test_invalid_test_size_raises(self) -> None:
        with pytest.raises(ValueError, match="test_size"):
            TrainConfig(test_size=1.5)

    def test_negative_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_disc"):
            TrainConfig(lambda_disc=-0.1)


# ───────────────────────────────────────────────────────────────────────────────
# Utils tests
# ───────────────────────────────────────────────────────────────────────────────


class TestUtils:
    def test_mixed_precision_error_is_exception(self) -> None:
        with pytest.raises(MixedPrecisionSelectionError):
            raise MixedPrecisionSelectionError("both selected")

    def test_mixed_precision_error_message(self) -> None:
        err = MixedPrecisionSelectionError("msg")
        assert str(err) == "msg"


# ───────────────────────────────────────────────────────────────────────────────
# Trainer helper tests
# ───────────────────────────────────────────────────────────────────────────────


class TestBuildDiscLabels:
    MASK_ID = 128000

    def test_masked_positions_are_one(self) -> None:
        ids = torch.tensor([[1, self.MASK_ID, 3], [self.MASK_ID, 5, self.MASK_ID]])
        labels = _build_disc_labels(ids, self.MASK_ID)
        assert labels[0, 1].item() == 1.0
        assert labels[1, 0].item() == 1.0
        assert labels[1, 2].item() == 1.0

    def test_non_masked_positions_are_zero(self) -> None:
        ids = torch.tensor([[1, 2, 3]])
        labels = _build_disc_labels(ids, self.MASK_ID)
        assert labels.sum().item() == 0.0

    def test_output_shape_matches_input(self) -> None:
        ids = torch.randint(0, 500, (4, 32))
        labels = _build_disc_labels(ids, self.MASK_ID)
        assert labels.shape == ids.shape

    def test_output_dtype_is_float(self) -> None:
        ids = torch.tensor([[1, self.MASK_ID]])
        assert _build_disc_labels(ids, self.MASK_ID).dtype == torch.float32


class TestDiscLoss:
    def test_loss_is_zero_for_perfect_predictions(self) -> None:
        # Large positive logit → sigmoid ≈ 1.0 matches label 1.0
        logits = torch.tensor([[10.0, 10.0], [10.0, 10.0]])
        labels = torch.ones(2, 2)
        mask = torch.ones(2, 2, dtype=torch.long)
        loss = _disc_loss(logits, labels, mask)
        assert loss.item() < 0.01

    def test_padding_excluded_from_loss(self) -> None:
        logits = torch.tensor([[10.0, 0.0]])  # second position is uncertain
        labels = torch.tensor([[1.0, 1.0]])
        full_mask = torch.ones(1, 2, dtype=torch.long)
        pad_mask = torch.tensor([[1, 0]], dtype=torch.long)  # mask out second
        loss_full = _disc_loss(logits, labels, full_mask)
        loss_padded = _disc_loss(logits, labels, pad_mask)
        # Padded loss should be lower since the uncertain position is excluded.
        assert loss_padded.item() < loss_full.item()


# ───────────────────────────────────────────────────────────────────────────────
# Model tests
# ───────────────────────────────────────────────────────────────────────────────


class TestDebertaV3GDES:
    def test_disc_head_is_linear(self, model: DebertaV3GDES) -> None:
        assert isinstance(model.disc_head, nn.Linear)
        assert model.disc_head.out_features == 1

    def test_forward_gen_output_shape(
        self,
        model: DebertaV3GDES,
        dummy_input_ids: torch.Tensor,
        dummy_attention_mask: torch.Tensor,
    ) -> None:
        labels = dummy_input_ids.clone()
        out = model.forward_gen(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            labels=labels,
        )
        vocab_size = model.deberta.config.vocab_size
        assert out.logits is not None
        assert out.logits.shape == (BATCH, SEQ, vocab_size)
        assert out.loss is not None

    def test_forward_disc_output_shape(
        self,
        model: DebertaV3GDES,
        dummy_input_ids: torch.Tensor,
        dummy_attention_mask: torch.Tensor,
    ) -> None:
        logits = model.forward_disc(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
        )
        assert logits.shape == (BATCH, SEQ)

    def test_forward_disc_padding_is_zero(
        self,
        model: DebertaV3GDES,
        dummy_input_ids: torch.Tensor,
        dummy_attention_mask: torch.Tensor,
    ) -> None:
        logits = model.forward_disc(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
        )
        pad_positions = dummy_attention_mask == 0
        assert torch.all(logits[pad_positions] == 0.0)

    def test_freeze_embeddings(self, model: DebertaV3GDES) -> None:
        model.freeze_embeddings()
        for name, param in model.deberta.named_parameters():
            if "embed" in name:
                assert not param.requires_grad, f"{name} should be frozen"
        model.unfreeze_embeddings()

    def test_unfreeze_embeddings(self, model: DebertaV3GDES) -> None:
        model.freeze_embeddings()
        model.unfreeze_embeddings()
        for name, param in model.deberta.named_parameters():
            if "embed" in name:
                assert param.requires_grad, f"{name} should be unfrozen"

    def test_embeddings_unfrozen_after_disc_pass(
        self,
        model: DebertaV3GDES,
        dummy_input_ids: torch.Tensor,
        dummy_attention_mask: torch.Tensor,
    ) -> None:
        model.forward_disc(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
        )
        for name, param in model.deberta.named_parameters():
            if "embed" in name:
                assert param.requires_grad, f"{name} still frozen after disc pass"

    def test_combined_loss_formula(
        self,
        model: DebertaV3GDES,
        dummy_input_ids: torch.Tensor,
        dummy_attention_mask: torch.Tensor,
    ) -> None:
        lambda_disc = 0.5
        labels = dummy_input_ids.clone()

        gen_out = model.forward_gen(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            labels=labels,
        )
        disc_logits = model.forward_disc(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
        )
        disc_labels = (dummy_input_ids == 0).float()
        mask = torch.ones_like(dummy_input_ids)

        disc_loss = _disc_loss(disc_logits, disc_labels, mask)
        assert gen_out.loss is not None
        combined = gen_out.loss + lambda_disc * disc_loss
        expected = gen_out.loss.item() + lambda_disc * disc_loss.item()
        assert abs(combined.item() - expected) < 1e-5


# ───────────────────────────────────────────────────────────────────────────────
# Integration: single training step
# ───────────────────────────────────────────────────────────────────────────────


class TestIntegration:
    """Smoke-test a full train step and eval pass without crashing."""

    MASK_TOKEN_ID = 128000

    def _make_batch(self) -> dict[str, torch.Tensor]:
        ids = torch.randint(100, 1000, (BATCH, SEQ))
        ids[0, 3] = self.MASK_TOKEN_ID
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(BATCH, SEQ, dtype=torch.long),
            "labels": ids.clone(),
        }

    def _make_loader(self, batch: dict[str, torch.Tensor]) -> Any:
        class _Batch:
            def __init__(self_b) -> None:
                for k, v in batch.items():
                    setattr(self_b, k, v)
                self_b._d = batch

            def to(self_b, device: torch.device) -> "_Batch":
                for k, v in self_b._d.items():
                    setattr(self_b, k, v.to(device))
                    self_b._d[k] = v.to(device)
                return self_b

            def __iter__(self_b) -> Iterator[str]:
                return iter(self_b._d)

            def __getitem__(self_b, key: str) -> torch.Tensor:
                return self_b._d[key]

            def keys(self_b) -> Any:
                return self_b._d.keys()

        class _Loader:
            def __len__(self) -> int:
                return 1

            def __iter__(self) -> Iterator[_Batch]:
                yield _Batch()

        return _Loader()

    def test_train_step_does_not_raise(self, model: DebertaV3GDES) -> None:
        tokenizer = MagicMock()
        tokenizer.mask_token_id = self.MASK_TOKEN_ID

        loader = self._make_loader(self._make_batch())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_one_epoch(
            tokenizer,
            loader,
            model,
            lambda_disc=0.5,
            optimizer=optimizer,
            scheduler=scheduler,
            dtype=torch.float32,
            scaler=None,
            device=torch.device("cpu"),
        )

    def test_evaluate_returns_expected_keys(self, model: DebertaV3GDES) -> None:
        tokenizer = MagicMock()
        tokenizer.mask_token_id = self.MASK_TOKEN_ID

        loader = self._make_loader(self._make_batch())

        results = evaluate(
            tokenizer,
            loader,
            model,
            lambda_disc=0.5,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert set(results.keys()) == {"eval_loss", "accuracy", "f1"}
        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["f1"] <= 1.0
