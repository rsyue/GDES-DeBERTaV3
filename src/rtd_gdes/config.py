"""
Centralised training configuration dataclass.
All CLI defaults and hyperparameter values live here so that
train.py, trainer.py, and tests can import a single source of truth.
"""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Model
    model_id: str = "microsoft/deberta-v3-base"

    # Dataset
    dataset_name: str = "imdb"
    dataset_split: str = "unsupervised"
    test_size: float = 0.1
    max_length: int = 512
    prefetch_factor: int = 2

    # Training
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gamma: float = 0.9
    lambda_disc: float = 0.5
    max_norm: float = 1.0

    # Precision
    fp16: bool = False
    bf16: bool = False

    # Misc
    compile_model: bool = False
    num_workers: int = field(default=4)  # safe default; override via CLI

    def __post_init__(self) -> None:
        if self.fp16 and self.bf16:
            raise ValueError("Only one of fp16 or bf16 may be set, not both.")
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}.")
        if self.lambda_disc < 0.0:
            raise ValueError(f"lambda_disc must be non-negative, got {self.lambda_disc}.")
