"""
Entry point for GDES RTD pretraining.

Usage
-----
python -m rtd_gdes.train --help
"""

import argparse
import os

import torch

from rtd_gdes.config import TrainConfig
from rtd_gdes.gdes.data import get_dataloaders_and_tokenizer
from rtd_gdes.gdes.model import DebertaV3GDES
from rtd_gdes.gdes.trainer import evaluate, train_one_epoch
from rtd_gdes.gdes.utils import MixedPrecisionSelectionError


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ELECTRA-style RTD pretraining with GDES on DeBERTaV3."
    )
    p.add_argument("-m", "--model", type=str, help="HuggingFace model id")
    p.add_argument("-ld", "--lambda_disc", type=float, help="Discriminator loss weight")
    p.add_argument("-bs", "--batch_size", type=int, help="Batch size")
    p.add_argument("-ep", "--epochs", type=int, help="Number of epochs")
    p.add_argument("-lr", "--learning_rate", type=float, help="AdamW learning rate")
    p.add_argument("-wd", "--weight_decay", type=float, help="AdamW weight decay")
    p.add_argument("-g", "--gamma", type=float, help="ExponentialLR gamma")
    p.add_argument("-nw", "--num_workers", type=int, help="DataLoader worker count")
    p.add_argument("--dataset", type=str, help="HuggingFace dataset name (default: imdb)")
    p.add_argument("--fp16", action="store_true", default=False, help="Enable FP16")
    p.add_argument("--bf16", action="store_true", default=False, help="Enable BF16")
    p.add_argument(
        "-c", "--compile", action="store_true", default=False,
        help="torch.compile with max-autotune"
    )
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> TrainConfig:
    """Merge CLI overrides onto the default :class:`TrainConfig`."""
    overrides: dict = {k: v for k, v in vars(args).items() if v is not None}

    # Rename CLI keys that differ from dataclass field names.
    if "model" in overrides:
        overrides["model_id"] = overrides.pop("model")
    if "dataset" in overrides:
        overrides["dataset_name"] = overrides.pop("dataset")
    if "compile" in overrides:
        overrides["compile_model"] = overrides.pop("compile")

    return TrainConfig(**overrides)


def main() -> None:
    args = _parse_args()

    if args.fp16 and args.bf16:
        raise MixedPrecisionSelectionError("Select only fp16 or bf16, not both.")

    cfg = _build_config(args)

    # ------------------------------------------------------------------ #
    # Device & dtype                                                       #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if cfg.fp16:
        dtype = torch.float16
    elif cfg.bf16:
        dtype = torch.bfloat16

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    train_loader, eval_loader, tokenizer = get_dataloaders_and_tokenizer(cfg)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = DebertaV3GDES(cfg.model_id).to(device)

    if cfg.compile_model:
        print("Compiling model with max-autotune …")
        torch._dynamo.reset()
        model.deberta = torch.compile(model.deberta, fullgraph=True, mode="max-autotune")
        torch.cuda.synchronize()
        print("Model compiled.")

    # ------------------------------------------------------------------ #
    # Optimiser & scheduler                                                #
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
    # GradScaler is only valid for FP16 — BF16 and FP32 do not require
    # loss scaling and will error if a scaler is used with them.
    scaler = torch.amp.GradScaler(device=str(device)) if cfg.fp16 else None

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    for epoch in range(1, cfg.epochs + 1):
        print(f"Epoch {epoch}/{cfg.epochs} {'─' * 48}")
        train_one_epoch(
            tokenizer, train_loader, model, cfg.lambda_disc,
            optimizer, scheduler, dtype, scaler, device,
        )
        evaluate(tokenizer, eval_loader, model, cfg.lambda_disc, dtype, device)
        print()

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    save_name = cfg.model_id.replace("-", "_").split("/")[-1] + "_gdes"
    model.deberta.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)
    print(f"Model and tokenizer saved to '{save_name}/'")


if __name__ == "__main__":
    main()
