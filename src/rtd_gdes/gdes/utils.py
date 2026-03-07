"""Utility helpers for the GDES training pipeline."""


class MixedPrecisionSelectionError(Exception):
    """Raised when both fp16 and bf16 are requested simultaneously."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
