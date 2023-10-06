from model import Seq2SeqTransformer
from train_process import train_one_epoch, val_one_epoch
from dataset_loader import create_dataloaders
import torch.nn as nn
import torch
import pandas as pd
import os
import logging
import pathlib
script_path = pathlib.Path(__file__).parent.resolve()
import argparse

def load_model(model_name: str, require_weights: bool, default_model):
    path_to_weights = os.path.join(script_path, f"../../models//{model_name}.pth")
    model = default_model
    
    return model, path_to_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("epochs", type=int)
    parser.add_argument("--weights", action='store_true')
    
    args = parser.parse_args()
    epochs = args.epochs
    
    # Loading dataset
    path_to_dataset = os.path.join(script_path, f"../../data/interim/{args.dataset}")
    train_loader, val_loader, vocab_size = create_dataloaders(pd.read_csv(path_to_dataset))

    # Loading model
    default_model = Seq2SeqTransformer(
        3,
        3,
        512,
        8,
        vocab_size,
        vocab_size,
        )
    model, model_weights_save_path = load_model(args.model_name, args.weights, default_model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_loss = 1e9
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, epoch, loss_fn, optimizer, 'cpu')
        val_loss = val_one_epoch(model, val_loader, epoch, loss_fn, loss_fn, optimizer, 'cpu')
        if train_loss < best_loss:
            best_loss = train_loss
            logging.info("New best loss. Checkpoint is saved!")
            torch.save(model.state_dict(), model_weights_save_path)
        print(f"Epoch {epoch} train_loss:{train_loss}, val_loss:{val_loss}")