"""Data loading and collation for the GDES pretraining pipeline."""

from typing import Any

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer

from rtd_gdes.config import TrainConfig


def get_dataloaders_and_tokenizer(
    cfg: TrainConfig,
) -> tuple[DataLoader[Any], DataLoader[Any], DebertaV2Tokenizer]:
    """
    Build train and eval DataLoaders along with the matching tokenizer.

    Args:
        cfg: A populated :class:`TrainConfig` instance.

    Returns:
        A three-tuple of ``(train_dataloader, eval_dataloader, tokenizer)``.
    """
    tokenizer: DebertaV2Tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.model_id)

    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    dataset = dataset.train_test_split(test_size=cfg.test_size)

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        tokenized: dict[str, Any] = tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding=True,
        )
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text", "label"])
    tokenized.set_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, return_tensors="pt")

    train_loader: DataLoader[Any] = DataLoader(
        tokenized["train"],
        batch_size=cfg.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=True,
    )
    eval_loader: DataLoader[Any] = DataLoader(
        tokenized["test"],
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        pin_memory=True,
    )

    return train_loader, eval_loader, tokenizer
