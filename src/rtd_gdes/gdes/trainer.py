"""Training and evaluation loops for GDES pretraining."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 — used via nn.functional in _disc_loss
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer

from rtd_gdes.gdes.model import DebertaV3GDES


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
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, weight=weight, reduction="sum"
    )
    # Normalise by the number of non-padding tokens to get a true mean.
    return loss / weight.sum().clamp(min=1)


_disc_loss_fn = nn.BCEWithLogitsLoss()


def _build_disc_labels(input_ids: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """
    Return a float tensor marking which positions held a mask token.

    A position is labelled 1.0 (replaced) if it contained ``[MASK]`` in the
    original masked input, and 0.0 (original) otherwise.

    Args:
        input_ids: Shape ``(B, T)``.
        mask_token_id: The tokenizer's mask token id.

    Returns:
        Float tensor of shape ``(B, T)``.
    """
    return (input_ids == mask_token_id).float()


def train_one_epoch(
    tokenizer: DebertaV2Tokenizer,
    dataloader: DataLoader,
    model: DebertaV3GDES,
    lambda_disc: float,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dtype: torch.dtype,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> None:
    """
    Run one full pass over ``dataloader``, updating ``model`` in place.

    The combined loss is:

    .. math::
        \\mathcal{L} = \\mathcal{L}_{\\text{gen}} + \\lambda \\cdot \\mathcal{L}_{\\text{disc}}

    Embedding parameters are frozen during the discriminator forward pass
    (GDES disentanglement) and restored before the next generator step.

    Args:
        tokenizer: Used only to obtain ``mask_token_id``.
        dataloader: Yields masked batches produced by
            :class:`~transformers.DataCollatorForLanguageModeling`.
        model: The :class:`DebertaV3GDES` instance to train.
        lambda_disc: Scaling coefficient for the discriminator loss.
        optimizer: An already-constructed :class:`~torch.optim.AdamW`.
        scheduler: Exponential LR scheduler stepped once per epoch.
        dtype: ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``.
        scaler: :class:`~torch.amp.GradScaler` for mixed-precision training.
        device: Target device.
    """
    model.train()
    progress = tqdm(dataloader, desc="train", leave=False)

    for batch in progress:
        batch = batch.to(device)
        disc_labels = _build_disc_labels(batch.input_ids, tokenizer.mask_token_id)

        # ---- Generator pass ----------------------------------------
        with torch.autocast(device_type=device.type, dtype=dtype):
            gen_out = model.forward_gen(**batch)
            gen_loss: torch.Tensor = gen_out.loss
            # Greedy token predictions from the generator fill masked slots.
            filled_ids: torch.Tensor = gen_out.logits.argmax(dim=-1)

        # ---- Discriminator pass ------------------------------------
        # Embeddings are frozen inside forward_disc (GDES step).
        with torch.autocast(device_type=device.type, dtype=dtype):
            disc_logits = model.forward_disc(
                input_ids=filled_ids,
                attention_mask=batch.attention_mask,
            )
            disc_loss: torch.Tensor = _disc_loss(disc_logits, disc_labels, batch.attention_mask)

        loss = gen_loss + lambda_disc * disc_loss

        # scaler.scale() / scaler.step() are no-ops when the scaler is
        # disabled (BF16 or FP32), so this branch handles all three dtypes.
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()


@torch.no_grad()
def evaluate(
    tokenizer: DebertaV2Tokenizer,
    dataloader: DataLoader,
    model: DebertaV3GDES,
    lambda_disc: float,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate the model on ``dataloader`` and return aggregate metrics.

    Args:
        tokenizer: Used only to obtain ``mask_token_id``.
        dataloader: Yields masked batches.
        model: The :class:`DebertaV3GDES` instance to evaluate.
        lambda_disc: Scaling coefficient for the discriminator loss (reported
            in the returned metrics but not used for backprop here).
        dtype: Autocast dtype.
        device: Target device.

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

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype == torch.float16):
            gen_out = model.forward_gen(**batch)
            filled_ids = gen_out.logits.argmax(dim=-1)

            disc_logits = model.forward_disc(
                input_ids=filled_ids,
                attention_mask=batch.attention_mask,
            )
            disc_loss = _disc_loss(disc_logits, disc_labels, batch.attention_mask)

        total_loss += disc_loss.item()

        # Exclude padding from metrics.
        valid = batch.attention_mask.bool().view(-1)
        preds = (torch.sigmoid(disc_logits).view(-1)[valid] > 0.5).int().cpu().tolist()
        labels = disc_labels.view(-1)[valid].int().cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

    results = {
        "eval_loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    print(results)
    return results
