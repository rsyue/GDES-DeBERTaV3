"""Training and evaluation loops for GDES pretraining."""

from typing import Any

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score  # type: ignore[import-untyped]
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # type: ignore[import-untyped]
from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer

from rtd_gdes.gdes.model import DebertaV3GDES


def _build_disc_labels(input_ids: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """
    Return a float tensor marking which positions held a mask token.

    Args:
        input_ids:      Shape ``(B, T)``.
        mask_token_id:  The tokenizer's mask token id.

    Returns:
        Float tensor of shape ``(B, T)``.
    """
    return (input_ids == mask_token_id).float()


def _disc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute BCEWithLogitsLoss over non-padding positions only.

    Padding positions (attention_mask == 0) are excluded via a per-token
    weight tensor rather than masking logits to -inf, which would produce
    NaN gradients via log(sigmoid(-inf)).

    Args:
        logits:         Shape ``(B, T)``.
        labels:         Float tensor of shape ``(B, T)``.
        attention_mask: Binary mask of shape ``(B, T)``; 0 for padding.

    Returns:
        Scalar loss averaged over non-padding positions.
    """
    weight = attention_mask.float()
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, weight=weight, reduction="sum"
    )
    return loss / weight.sum().clamp(min=1)


def train_one_epoch(
    tokenizer: DebertaV2Tokenizer,
    dataloader: DataLoader[Any],
    model: DebertaV3GDES,
    lambda_disc: float,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dtype: torch.dtype,
    scaler: GradScaler,
    device: torch.device,
) -> None:
    """
    Run one full pass over ``dataloader``, updating ``model`` in place.

    Both forward passes and loss calculation run inside a single autocast
    block. The scaler is always passed in but constructed with
    ``enabled=False`` for BF16/FP32 so its methods are no-ops.

    Args:
        tokenizer:    Used only to obtain ``mask_token_id``.
        dataloader:   Yields masked batches produced by
                      :class:`~transformers.DataCollatorForLanguageModeling`.
        model:        The :class:`DebertaV3GDES` instance to train.
        lambda_disc:  Scaling coefficient for the discriminator loss.
        optimizer:    An already-constructed :class:`~torch.optim.AdamW`.
        scheduler:    Exponential LR scheduler stepped once per epoch.
        dtype:        ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``.
        scaler:       :class:`~torch.cuda.amp.GradScaler` — enabled only for
                      FP16 (5-bit exponent, prone to gradient underflow).
                      BF16 shares FP32's 8-bit exponent range and does not
                      require scaling. A no-op for BF16 and FP32.
        device:       Target device.
    """
    model.train()
    progress = tqdm(dataloader, desc="train", leave=False)

    for batch in progress:
        batch = batch.to(device)
        disc_labels = _build_disc_labels(batch.input_ids, tokenizer.mask_token_id)

        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            gen_out = model.forward_gen(**batch)
            gen_loss: torch.Tensor = gen_out.loss  # type: ignore[assignment]
            filled_ids: torch.Tensor = gen_out.logits.argmax(dim=-1)  # type: ignore[union-attr]

            disc_logits = model.forward_disc(
                input_ids=filled_ids,
                attention_mask=batch.attention_mask,
            )
            disc_loss = _disc_loss(disc_logits, disc_labels, batch.attention_mask)
            loss = gen_loss + lambda_disc * disc_loss

        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()


@torch.no_grad()
def evaluate(
    tokenizer: DebertaV2Tokenizer,
    dataloader: DataLoader[Any],
    model: DebertaV3GDES,
    lambda_disc: float,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate the model on ``dataloader`` and return aggregate metrics.

    Args:
        tokenizer:    Used only to obtain ``mask_token_id``.
        dataloader:   Yields masked batches.
        model:        The :class:`DebertaV3GDES` instance to evaluate.
        lambda_disc:  Scaling coefficient for the discriminator loss.
        dtype:        Autocast dtype.
        device:       Target device.

    Returns:
        A dict with keys ``eval_loss``, ``accuracy``, and ``f1``.
    """
    model.eval()
    progress = tqdm(dataloader, desc="eval", leave=False)

    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0

    for batch in progress:
        batch = batch.to(device)
        disc_labels = _build_disc_labels(batch.input_ids, tokenizer.mask_token_id)

        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            gen_out = model.forward_gen(**batch)
            filled_ids: torch.Tensor = gen_out.logits.argmax(dim=-1)  # type: ignore[union-attr]

            disc_logits = model.forward_disc(
                input_ids=filled_ids,
                attention_mask=batch.attention_mask,
            )
            disc_loss = _disc_loss(disc_logits, disc_labels, batch.attention_mask)

        total_loss += disc_loss.item()

        valid = batch.attention_mask.bool().view(-1)
        preds = (torch.sigmoid(disc_logits).view(-1)[valid] > 0.5).int().cpu().tolist()
        labels = disc_labels.view(-1)[valid].int().cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

    results: dict[str, Any] = {
        "eval_loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    print(results)
    return results
