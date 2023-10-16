import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils.constants import PAD_IDX


class Seq2SeqTrainer():
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str) -> None:
        
        self.model = model
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.optmizer = optim.Adam(model.parameters(), 3e-4)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.device = device

    def train_one_epoch(self):
        self.model.train()
        progress = tqdm(self.train_loader)
        for batch in progress:
            input, target = batch
            input, target = input.to(self.device), target.to(self.device)
            output = self.model(input, target)
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            self.optmizer.zero_grad()
            
            loss = self.loss_fn(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optmizer.step()
            progress.set_postfix({"loss":loss.item()})

    def val_one_epoch(self):
        self.model.eval()
        progress = tqdm(self.val_loader)
        with torch.no_grad():
            for batch in progress:
                input, target = batch
                input, target = input.to(self.device), target.to(self.device)

                output = self.model(input, target)
                output = output.reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)
                
                loss = self.loss_fn(output, target)
                progress.set_postfix({"loss":loss.item()})

    def train(self, epochs: int, save_path: str):
        self.train_one_epoch()
        self.val_one_epoch()
        print("SAVE MODEL")
        torch.save(self.model.state_dict(), save_path)