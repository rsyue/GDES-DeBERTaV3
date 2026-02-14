# Model module

# Model constructor for the generator/discriminator GDES model

import torch.nn as nn
import torch

from transformers import DebertaV2ForMaskedLM

class DebertaV3GDES(nn.Module):
    """
    DeBERTaV3 GDES training classifier class.
    This ``Module`` lets the user run a generative
    pass as well as a discriminative pass.

    1. Generative pass: Generate predictions for
       each masked token
    2. Discriminative pass: For each token, predict
       whether it was replaced or part of the original
       text
    
    """
    
    def __init__(self, model_id):
        super(DebertaV3GDES, self).__init__()
        # Idem for tokenizer, V3 not available
        self.deberta = DebertaV2ForMaskedLM.from_pretrained(model_id)

    def forward_gen(self, **inputs):
        """
        Generative forward pass

        Params:

        :inputs: Kwargs as input for the model forward pass
                 e.g. input_ids, attention_mask

        Returns:
            :logits: A tensor of computed logits
        """
        logits = self.deberta(**inputs)
        return logits

    def forward_disc(self, gen_out, attention_mask):
        """
        Discriminator forward pass

        Params:
        
            :gen_out: The output (filled masks) from the generator
            :attention_mask: The attention mask for ignoring
                             padded tokens

        Returns:

            :logits_ignore_pad: The logits with pad tokens ignored
        """
        float_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf'))
        logits_ignore_pad = torch.where(float_mask == float('-inf'), gen_out, float_mask)
        return logits_ignore_pad
