# A little script to run ELECTRA-style training
# with GDES using native PyTorch on a
# HuggingFace Transformers model

# @author: Richard Yue

import copy
import argparse

from gdes.utils import MixedPrecisionSelectionError

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", help="The model to train using RTD with GDES", type=str)
parser.add_argument("-ld", "--lambda_disc", help="The lambda coefficient for the discriminator model", type=float)
parser.add_argument("-bs", "--batch_size", help="The batch size for training and validation", type=int)
parser.add_argument("-ep", "--epochs", help="Number of training epochs", type=int)
parser.add_argument("-lr", "--learning_rate", help="Learning rate for training", type=float)
parser.add_argument("-wd", "--weight_decay", help="Weight decay regularization for the Adam optimizer", type=float)
parser.add_argument("-g", "--gamma", help="Gamma value for exponential lr scheduler", type=float)
parser.add_argument("-c", "--compile", help="Runs torch.compile with max-autotune if active", action=argparse.BooleanOptionalAction)
parser.add_argument("--fp16", action=argparse.BooleanOptionalAction)
parser.add_argument("--bf16", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

if args.fp16 and args.bf16:
    raise MixedPrecisionSelectionError("Select only fp16 or bf16")

print(f"Args passed:\n\n{args}")

from transformers import DebertaV2ForMaskedLM, DebertaV2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch

from safetensors.torch import save_file

from sklearn.metrics import accuracy_score, f1_score

from tqdm.auto import tqdm

from gdes.model import DebertaV3GDES
from gdes.data import get_dataloaders_and_tokenizer
from gdes.trainer import train, eval

model = args.model if args.model else "microsoft/deberta-v3-base"
batch_size = int(args.batch_size) if args.batch_size else 8
lambda_disc = float(args.lambda_disc) if args.lambda_disc else 0.5
epochs = int(args.epochs) if args.epochs else 5
learning_rate = float(args.learning_rate) if args.learning_rate else 2e-5
weight_decay = float(args.weight_decay) if args.weight_decay else 0.01
gamma = float(args.gamma) if args.gamma else 0.9

dtype = torch.float32
if args.fp16 or args.bf16:
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    lambda_disc = torch.tensor(lambda_disc).to(dtype).item()

# Set model id
model_id = model

# Get dataloaders and tokenizer
# TODO add test size as an argparse arg
train_dataloader, eval_dataloader, tokenizer = get_dataloaders_and_tokenizer(model_id, batch_size)

# Set device and send model instantiation to it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DebertaV3GDES(model_id).to(device)
if args.compile:
    print("Compile activated. Compiling model with max-autotune...")
    torch._dynamo.reset()
    model.deberta = torch.compile(model.deberta, fullgraph=True, mode='max-autotune')
    torch.cuda.synchronize()
    print("Model compiled!")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

# For fp16/bf16 mixed precision
scaler = torch.amp.GradScaler(device=str(device))

def main():
    # Full training loop
    for t in range(epochs):

        print(f"Epoch {t+1}\n--------------------------------------------------")

        train(
            tokenizer,
            train_dataloader,
            model,
            lambda_disc,
            optimizer,
            scheduler,
            dtype,
            scaler,
            device,
        )
        
        eval(
            tokenizer,
            eval_dataloader,
            model,
            lambda_disc,
            dtype,
            scaler,
            device
        )
        
        print()
        
    print("Done!")
    save_friendly_name = model_id.replace("-", "_").split("/")[1] + "_gdes"
    model.deberta.save_pretrained(save_friendly_name)
    tokenizer.save_pretrained(save_friendly_name)
    print(f"Model and tokenizer saved under '{save_friendly_name}'")

if __name__ == "__main__":
    main()
