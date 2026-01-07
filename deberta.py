# A little script to run ELECTRA-style training
# with GDES using native PyTorch on a
# HuggingFace Transformers model

# @author: Richard Yue

import copy

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", help="The model to train using RTD with GDES", type=str)
parser.add_argument("-ld", "--lambda_disc", help="The lambda coefficient for the discriminator model", type=float)
parser.add_argument("-bs", "--batch_size", help="The batch size for training and validation", type=int)
parser.add_argument("-ep", "--epochs", help="Number of training epochs", type=int)
parser.add_argument("-lr", "--learning_rate", help="Learning rate for training", type=float)
parser.add_argument("-wd", "--weight_decay", help="Weight decay regularization for the Adam optimizer", type=float)
parser.add_argument("-g", "--gamma", help="Gamma value for exponential lr scheduler", type=float)
parser.add_argument("--fp16", action=argparse.BooleanOptionalAction)
parser.add_argument("--bf16", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

class MixedPrecisionSelectionError(Exception):
    def __init__(self, msg):
        self.msg = msg
        print("Cannot select multiple mixed precision settings at once")

if args.fp16 and args.bf16:
    raise MixedPrecisionSelectionError("Select only fp16 or bf16")

print(f"Args passed:\n\n{args}")

from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch

from sklearn.metrics import accuracy_score, f1_score

from tqdm.auto import tqdm

model = args.model if args.model else "microsoft/deberta-v3-base"
lambda_disc = float(args.lambda_disc) if args.lambda_disc else 0.5
batch_size = int(args.batch_size) if args.batch_size else 8
epochs = int(args.epochs) if args.epochs else 5
learning_rate = float(args.learning_rate) if args.learning_rate else 2e-5
weight_decay = float(args.weight_decay) if args.weight_decay else 0.01
gamma = float(args.gamma) if args.gamma else 0.9

if args.fp16 or args.bf16:
    dtype = torch.float16 if args.fp16 else torch.bfloat16

# Set model id
model_id = model

# Load a fast tokenizer, note that V3 is not available so we use V2
tokenizer = DebertaV2Tokenizer.from_pretrained(model_id, is_fast=True)

# Grab the IMDB dataset for unsupervised training
# and siphon off 10% of data for tok class eval
dataset = load_dataset("imdb", split="unsupervised")
dataset = dataset.train_test_split(test_size=0.1)
print(dataset)

# Tokenize function, truncate at 512, pad, and copy input ids as
# ground truth for masked text
def tokenize(batch):
    tokenized = tokenizer(batch["text"], truncation=True, max_length=512, padding=True)
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

# Batch map tokenization and remove non-numerical columns
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text", "label"])
tokenized_dataset.set_format("torch")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, return_tensors="pt")

# Create dataloaders with the mlm collator
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, collate_fn=data_collator, shuffle=True)
eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator)

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
    
    def __init__(self):
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

# Set device and send model instantiation to it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DebertaV3GDES().to(device)

# Standard loss with BCE with logits for optimized discriminator processing
loss_fn = nn.CrossEntropyLoss()
disc_loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# For fp16/bf16 mixed precision
scaler = torch.amp.GradScaler(device=str(device))

def train(dataloader, model, loss_fn, optimizer):
    """
    One full training loop for the model

    Params:
        dataloader: the dataloader for training
        model: the model to train
        loss_fn: the loss_fn to use in training
        optimizer: the optimizer to use for training
    
    """
    model.train()
    num_batches = len(dataloader)
    progress_bar = tqdm(total=num_batches)
    for inp in dataloader:
        inp = inp.to(device)
        # Generate discriminator labels from the input_ids
        disc_labels = (inp.input_ids == tokenizer.mask_token_id).float().squeeze()
        with torch.autocast(device_type=str(device), dtype=dtype):
            gen_outputs = model.forward_gen(**inp)
            gen_loss, gen_logits = gen_outputs.loss, gen_outputs.logits

            # Get predicted masks for use with discriminator
            masks_filled = gen_logits.argmax(2).float()

        # Save in dict with attention_mask to properly ignore padded tokens
        disc_inp = {"gen_out": masks_filled, "attention_mask": inp.attention_mask}

        # Freeze embeddings so they are not modified by the discriminator
        for name, param in model.named_parameters():
            if "embed" in name:
                param.requires_grad = False

        with torch.autocast(device_type=str(device), dtype=dtype):
            disc_outputs = model.forward_disc(**disc_inp)
            disc_loss = disc_loss_fn(disc_outputs, disc_labels)

        # Compute loss with lambda coefficient for scaling
        loss = gen_loss + (lambda_disc * disc_loss)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Unfreeze params for the generator
        for name, param in model.named_parameters():
            if "embed" in name:
                param.requires_grad = True

        progress_bar.update(1)
        progress_bar.set_description(f"Loss: {loss}")

    scheduler.step()

def eval(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    progress_bar = tqdm(total=num_batches)
    preds = []
    labels = []
    eval_loss = 0.0
    # Dispense with gradient computation during eval
    with torch.no_grad():
        for inp in dataloader:
            inp = inp.to(device)
            disc_labels = (inp.input_ids == tokenizer.mask_token_id).float().squeeze()
            labels += disc_labels.int().squeeze().tolist()[0]
            with torch.autocast(device_type=str(device), dtype=dtype):
                gen_outputs = model.forward_gen(**inp)
                gen_loss, gen_logits = gen_outputs.loss, gen_outputs.logits

                masks_filled = copy.deepcopy(gen_logits).argmax(2).float()

            disc_inp = {"gen_out": masks_filled, "attention_mask": inp.attention_mask}

            with torch.autocast(device_type=str(device), dtype=dtype):
                disc_outputs = model.forward_disc(**disc_inp)
                disc_loss = loss_fn(disc_outputs, disc_labels)

            eval_loss += disc_loss

            preds += (F.sigmoid(disc_outputs) > 0.5).int().squeeze().tolist()[0]

            progress_bar.update(1)
            
    results = {
        "eval_loss": eval_loss.item() / num_batches,
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }
    print(results)

def main():
    # Full training loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n--------------------------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        eval(eval_dataloader, model, loss_fn)
        print()
    print("Done!")
    PATH = "out/"
    torch.save(model.state_dict(), PATH)
    tokenizer.save_pretrained("out/")
    print("Model and tokenizer saved")

if __name__ == "__main__":
    main()
