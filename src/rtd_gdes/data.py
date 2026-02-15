# Data loading and collating
# classes, methods, and functions

from transformers import DebertaV2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

def get_dataloaders_and_tokenizer(model_id, batch_size, test_size=0.1):
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

    # Determine num cpus for workers
    num_workers = os.cpu_count()
    
    # Create dataloaders with the mlm collator
    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, collate_fn=data_collator, shuffle=True, num_workers=num_workers, pin_memory=True)
    eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator, num_workers=num_workers, pin_memory=True)

    return train_dataloader, eval_dataloader, tokenizer
