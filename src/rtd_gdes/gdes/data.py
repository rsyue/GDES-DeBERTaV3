"""Data loading and collation for the GDES pretraining pipeline."""

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, DebertaV2Tokenizer

from rtd_gdes.config import TrainConfig


def get_dataloaders_and_tokenizer(
    cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader, DebertaV2Tokenizer]:
    """
    Build train and eval DataLoaders along with the matching tokenizer.

    The dataset name and split, tokenisation options, and worker count are all
    driven by ``cfg`` so that this function is fully configurable and testable
    without touching global state.

    Args:
        cfg: A populated :class:`TrainConfig` instance.

    Returns:
        A three-tuple of ``(train_dataloader, eval_dataloader, tokenizer)``.
    """
    # DeBERTa-v3 does not ship a fast tokeniser, so we fall back to v2.
    tokenizer: DebertaV2Tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.model_id)

    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    dataset = dataset.train_test_split(test_size=cfg.test_size)

    def _tokenize(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding=True,
        )
        # Labels are a copy of input_ids; the MLM collator will overwrite
        # masked positions with -100 so they are ignored by the loss.
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text", "label"])
    tokenized.set_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, return_tensors="pt")

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=cfg.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        tokenized["test"],
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, eval_loader, tokenizer
