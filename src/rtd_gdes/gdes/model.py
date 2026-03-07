"""
GDES model definition.

The generator and discriminator share the DeBERTa backbone and its
embedding table. During the discriminator forward pass, embedding
gradients are frozen so the two objectives do not interfere — this is
the core of Gradient-Disentangled Embedding Sharing (GDES).
"""

import torch
import torch.nn as nn
from transformers import DebertaV2ForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput


class DebertaV3GDES(nn.Module):
    """
    DeBERTaV3 module supporting both the generator (MLM) and discriminator
    (replaced-token detection) forward passes needed for ELECTRA-style RTD.

    The single shared backbone (``self.deberta``) is used for both passes.
    A lightweight binary classification head (``self.disc_head``) maps the
    backbone's hidden states to per-token replaced/original logits.

    Attributes:
        deberta: The underlying :class:`DebertaV2ForMaskedLM` backbone.
        disc_head: Linear layer projecting hidden states → 1 logit per token.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.deberta = DebertaV2ForMaskedLM.from_pretrained(model_id)
        hidden_size: int = self.deberta.config.hidden_size
        self.disc_head = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    # Embedding gradient helpers
    # ------------------------------------------------------------------

    def _set_embedding_grad(self, requires_grad: bool) -> None:
        """Toggle ``requires_grad`` for all embedding parameters."""
        for name, param in self.deberta.named_parameters():
            if "embed" in name:
                param.requires_grad = requires_grad

    def freeze_embeddings(self) -> None:
        """Freeze embedding parameters (used during the discriminator pass)."""
        self._set_embedding_grad(False)

    def unfreeze_embeddings(self) -> None:
        """Restore embedding gradients (used after the discriminator pass)."""
        self._set_embedding_grad(True)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_gen(self, **inputs) -> MaskedLMOutput:
        """
        Generator (MLM) forward pass.

        Args:
            **inputs: Keyword arguments forwarded directly to the DeBERTa
                backbone (``input_ids``, ``attention_mask``, ``labels``, …).

        Returns:
            A :class:`~transformers.modeling_outputs.MaskedLMOutput` containing
            at minimum ``.loss`` and ``.logits``.
        """
        return self.deberta(**inputs)

    def forward_disc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminator (RTD) forward pass.

        Embeddings are frozen before running the backbone so that the
        discriminator loss does not update the shared embedding table —
        this is the GDES disentanglement step.

        Args:
            input_ids: Token ids of the generator-filled sequence,
                shape ``(B, T)``.
            attention_mask: Binary mask, shape ``(B, T)``;
                0 for padding positions.

        Returns:
            Per-token logits of shape ``(B, T)`` — positive values indicate
            a replaced token, negative values indicate an original token.
            Padding positions carry a logit of ``-inf`` so they are excluded
            from the loss automatically.
        """
        self.freeze_embeddings()

        hidden_states: torch.Tensor = self.deberta.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # (B, T, H)

        self.unfreeze_embeddings()

        logits = self.disc_head(hidden_states).squeeze(-1)  # (B, T)

        # Mask out padding so BCEWithLogitsLoss ignores those positions.
        logits = logits.masked_fill(attention_mask == 0, float("-inf"))

        return logits
