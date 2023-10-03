from tokenizers.basic_tokenizer import BasicTokenizer
from tokenizers.custom_torch_tokenizer import CustomTorchTokenizer
from models.model import DetoxificationModel
import torch
import pandas as pd
import os
import logging
import pathlib
script_path = pathlib.Path(__file__).parent.resolve()
import argparse


class ToxicTextDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BasicTokenizer):
        self.tokenizer = tokenizer
        
        self.toxic_texts =  df["toxic"].tolist()
        self.detoxified_texts = df["detoxified"].tolist()


        full_text = self.toxic_texts + self.detoxified_texts
        self.tokenizer.create_vocab(full_text)
        self.tokenized_labels = [self.tokenizer.tokenize(text) for text in self.toxic_texts]
        self.tokenized_targets = [self.tokenizer.tokenize(text) for text in self.detoxified_texts]
        
        

    def __getitem__(self, index: int) -> tuple[list, list]:
        return self.tokenized_labels[index], self.tokenized_targets[index]
    
    def __len__(self):
        return len(self.toxic_texts)


def create_dataloaders(df: pd.DataFrame):
    pass

def load_model(model_name: str, require_weights: bool, default_variant):
    path_to_model = os.path.join(script_path, f"../../models/{model_name}.pt")
    path_to_weights = os.path.join(script_path, f"../../models//{model_name}.pth")

    if not os.path.exists(path_to_model):
        logging.warn(f"Model {model_name} is not enough. Training current one from scratch!")
        model = default_variant()
        model_scripted = torch.jit.script(model)
        model_scripted.save(path_to_model)
    else:
        model = torch.jit.load(path_to_model)
        if require_weights:
            if not os.path.exists(path_to_weights):
                raise Exception("Model is loaded but weights not found!")
            model.load_state_dict(torch.load(path_to_weights))
    
    return model, path_to_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("epochs", type=int)
    #parser.add_argument("loss", choices=list(loss_functions.keys()))
    parser.add_argument("--weights", action='store_true')
    parser.add_argument("--GAN_model", type=str)
    
    args = parser.parse_args()
    epochs = args.epochs
    

    # Loading model
    model, model_weights_save_path = load_model(args.model_name, args.weights, DetoxificationModel)
    

    train_loader, val_loader = create_dataloaders()
    optimizer = torch.optim.Adam(model.parameters())

    best_loss = 1e9
    for epoch in range(epochs):
        pass
        # train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        # val_loss = val_one_epoch(model, val_loader, loss_fn)
        # if train_loss < best_loss:
        #     best_loss = train_loss
        #     logging.info("New best loss. Checkpoint is saved!")
        #     torch.save(model.state_dict(), model_weights_save_path)
        # print(f"Epoch {epoch} train_loss:{train_loss}, val_loss:{val_loss}")