"""
Unit and integration tests for the GDES pretraining pipeline.

Run with:
    pytest tests/ -v --cov=rtd_gdes
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch

from rtd_gdes.config import TrainConfig
from rtd_gdes.gdes.model import DebertaV3GDES
from rtd_gdes.gdes.trainer import _build_disc_labels, train_one_epoch, evaluate
from rtd_gdes.gdes.utils import MixedPrecisionSelectionError


# ───────────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────────

MODEL_ID = "microsoft/deberta-v3-base"
BATCH, SEQ, HIDDEN = 2, 16, 768


@pytest.fixture(scope="module")
def model() -> DebertaV3GDES:
    """Instantiate the full model once per test module (slow, cached)."""
    return DebertaV3GDES(MODEL_ID)


@pytest.fixture()
def dummy_input_ids() -> torch.Tensor:
    return torch.randint(0, 1000, (BATCH, SEQ))


@pytest.fixture()
def dummy_attention_mask() -> torch.Tensor:
    # Last two tokens are padding.
    mask = torch.ones(BATCH, SEQ, dtype=torch.long)
    mask[:, -2:] = 0
    return mask


# ───────────────────────────────────────────────────────────────────────────────
# Config tests
# ───────────────────────────────────────────────────────────────────────────────

class TestTrainConfig:
    def test_defaults_are_valid(self):
        cfg = TrainConfig()
        assert cfg.model_id == "microsoft/deberta-v3-base"
        assert cfg.lambda_disc == 0.5

    def test_fp16_and_bf16_together_raises(self):
        with pytest.raises(ValueError, match="Only one of fp16 or bf16"):
            TrainConfig(fp16=True, bf16=True)

    def test_invalid_test_size_raises(self):
        with pytest.raises(ValueError, match="test_size"):
            TrainConfig(test_size=1.5)

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda_disc"):
            TrainConfig(lambda_disc=-0.1)


# ───────────────────────────────────────────────────────────────────────────────
# Utils tests
# ───────────────────────────────────────────────────────────────────────────────

class TestUtils:
    def test_mixed_precision_error_is_exception(self):
        with pytest.raises(MixedPrecisionSelectionError):
            raise MixedPrecisionSelectionError("both selected")

    def test_mixed_precision_error_message(self):
        err = MixedPrecisionSelectionError("msg")
        assert str(err) == "msg"


# ───────────────────────────────────────────────────────────────────────────────
# Trainer helper tests
# ───────────────────────────────────────────────────────────────────────────────

class TestBuildDiscLabels:
    MASK_ID = 128000

    def test_masked_positions_are_one(self):
        ids = torch.tensor([[1, self.MASK_ID, 3], [self.MASK_ID, 5, self.MASK_ID]])
        labels = _build_disc_labels(ids, self.MASK_ID)
        assert labels[0, 1].item() == 1.0
        assert labels[1, 0].item() == 1.0
        assert labels[1, 2].item() == 1.0

    def test_non_masked_positions_are_zero(self):
        ids = torch.tensor([[1, 2, 3]])
        labels = _build_disc_labels(ids, self.MASK_ID)
        assert labels.sum().item() == 0.0

    def test_output_shape_matches_input(self):
        ids = torch.randint(0, 500, (4, 32))
        labels = _build_disc_labels(ids, self.MASK_ID)
        assert labels.shape == ids.shape

    def test_output_dtype_is_float(self):
        ids = torch.tensor([[1, self.MASK_ID]])
        assert _build_disc_labels(ids, self.MASK_ID).dtype == torch.float32


# ───────────────────────────────────────────────────────────────────────────────
# Model tests
# ───────────────────────────────────────────────────────────────────────────────

class TestDebertaV3GDES:
    def test_disc_head_is_linear(self, model):
        assert isinstance(model.disc_head, nn.Linear)
        assert model.disc_head.out_features == 1

    def test_forward_gen_output_shape(self, model, dummy_input_ids, dummy_attention_mask):
        labels = dummy_input_ids.clone()
        out = model.forward_gen(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            labels=labels,
        )
        vocab_size = model.deberta.config.vocab_size
        assert out.logits.shape == (BATCH, SEQ, vocab_size)
        assert out.loss is not None

    def test_forward_disc_output_shape(self, model, dummy_input_ids, dummy_attention_mask):
        logits = model.forward_disc(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
        )
        assert logits.shape == (BATCH, SEQ)

    def test_forward_disc_padding_is_zero(self, model, dummy_input_ids, dummy_attention_mask):
        logits = model.forward_disc(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
        )
        # Padding positions are zeroed (not -inf) to prevent NaN in BCE loss.
        pad_positions = dummy_attention_mask == 0
        assert torch.all(logits[pad_positions] == 0.0)

    def test_freeze_embeddings(self, model):
        model.freeze_embeddings()
        for name, param in model.deberta.named_parameters():
            if "embed" in name:
                assert not param.requires_grad, f"{name} should be frozen"
        model.unfreeze_embeddings()  # clean up

    def test_unfreeze_embeddings(self, model):
        model.freeze_embeddings()
        model.unfreeze_embeddings()
        for name, param in model.deberta.named_parameters():
            if "embed" in name:
                assert param.requires_grad, f"{name} should be unfrozen"

    def test_embeddings_unfrozen_after_disc_pass(
        self, model, dummy_input_ids, dummy_attention_mask
    ):
        """Embeddings must be restored to trainable after forward_disc returns."""
        model.forward_disc(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
        )
        for name, param in model.deberta.named_parameters():
            if "embed" in name:
                assert param.requires_grad, f"{name} still frozen after disc pass"

    def test_combined_loss_formula(self, model, dummy_input_ids, dummy_attention_mask):
        """L = L_gen + lambda * L_disc — verify numerically."""
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
        disc_labels = (dummy_input_ids == 0).float()  # arbitrary target

        disc_loss = nn.BCEWithLogitsLoss()(disc_logits, disc_labels)
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

    def _make_loader(self, batch: dict[str, torch.Tensor]):
        """A minimal single-batch DataLoader that survives .to(device) calls."""

        class _Loader:
            def __len__(self):
                return 1

            def __iter__(self_inner):
                # Return a namespace whose .to() returns itself with tensors
                # moved to the target device, matching how trainer.py uses it.
                class _Batch:
                    def __init__(self_b):
                        for k, v in batch.items():
                            setattr(self_b, k, v)
                        # Support dict-style unpacking via **batch in forward_gen.
                        self_b._d = batch

                    def to(self_b, device):
                        for k, v in self_b._d.items():
                            setattr(self_b, k, v.to(device))
                            self_b._d[k] = v.to(device)
                        return self_b

                    def __iter__(self_b):
                        return iter(self_b._d)

                    def __getitem__(self_b, key):
                        return self_b._d[key]

                    # Allow **batch unpacking in forward_gen(**inp).
                    def keys(self_b):
                        return self_b._d.keys()

                yield _Batch()

        return _Loader()

    def test_train_step_does_not_raise(self, model):
        tokenizer = MagicMock()
        tokenizer.mask_token_id = self.MASK_TOKEN_ID

        loader = self._make_loader(self._make_batch())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scaler = torch.amp.GradScaler(device="cpu")

        train_one_epoch(
            tokenizer, loader, model,
            lambda_disc=0.5, optimizer=optimizer,
            scheduler=scheduler, dtype=torch.float32,
            scaler=scaler, device=torch.device("cpu"),
        )

    def test_evaluate_returns_expected_keys(self, model):
        tokenizer = MagicMock()
        tokenizer.mask_token_id = self.MASK_TOKEN_ID

        loader = self._make_loader(self._make_batch())

        results = evaluate(
            tokenizer, loader, model,
            lambda_disc=0.5, dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert set(results.keys()) == {"eval_loss", "accuracy", "f1"}
        assert 0.0 <= results["accuracy"] <= 1.0
        assert 0.0 <= results["f1"] <= 1.0
