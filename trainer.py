# Training and eval functions for the trainer

import copy

import torch.nn.functional as F
import torch.nn as nn
import torch

from sklearn.metrics import accuracy_score, f1_score

from tqdm.auto import tqdm

disc_loss_fn = nn.BCEWithLogitsLoss()

def train(tokenizer, dataloader, model, lambda_disc, optimizer, scheduler, dtype, scaler, device):
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

def eval(tokenizer, dataloader, model, lambda_disc, dtype, scaler, device):
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
