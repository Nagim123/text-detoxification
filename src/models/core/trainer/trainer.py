import torch
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils.constants import PAD_IDX


class Seq2SeqTrainer():
    """
    Class for training Seq2Seq models.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str) -> None:
        """
        Create Seq2Seq trainer by providing model, dataloaders and device to train on.

        Parameters:
            model (Module): Torch model for training.
            train_loader (DataLoader): Training set dataloader.
            val_loader (DataLoader): Validation set dataloader.
            device (str): Device to train on.
        """
        # Store model and dataloaders
        self.model = model
        self.val_loader = val_loader
        self.train_loader = train_loader
        # Create Adam optimizer
        self.optmizer = optim.Adam(model.parameters(), 3e-4)
        # For vocabulary token prediction it is good to use Cross Entropy Loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        # Store device
        self.device = device


    def train_one_epoch(self) -> float:
        """
        Train model for one epoch.
        """
        
        # Set model to train mode
        self.model.train()
        # Keep track of progress
        progress = tqdm(self.train_loader)
        # Mean loss
        mean_loss = 0.0
        for batch in progress:
            # Get inputs and targets from batch and move them into specified device
            input, target = batch
            input, target = input.to(self.device), target.to(self.device)
            # Feed forward model
            output = self.model(input, target)

            # NOTE: Transformer's models output the sequence with length 1 less than target.
            diff = target.shape[0]-output.shape[0]
            # Reshape output to be [BATCH * SEQ_LEN, VOCAB_SIZE]
            output = output.reshape(-1, output.shape[2])
            
            # Cut first element in case of transformer training.
            target = target[diff:]
            # Reshape target to be [BATCH * SEQ_LEN]
            target = target.reshape(-1)
            
            # Clear grads
            self.optmizer.zero_grad()
            
            # Calculate loss and do backpropagation.
            loss = self.loss_fn(output, target)
            loss.backward()

            # Clip gradients to avoid exploding gradients problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            # Optimize weights
            self.optmizer.step()
            # Show loss information
            mean_loss += loss.item()
            progress.set_postfix({"loss":loss.item()})    
        return mean_loss/len(self.train_loader)
    
    def val_one_epoch(self) -> float:
        """
        Validate model for one epoch.
        """

        # Set model to evaluation mode
        self.model.eval()
        # Keep track of progress
        progress = tqdm(self.val_loader)
        # Mean loss
        mean_loss = 0.0
        with torch.no_grad():
            for batch in progress:
                # Get inputs and targets from batch and move them into specified device
                input, target = batch
                input, target = input.to(self.device), target.to(self.device)
                
                # Feed forward model
                output = self.model(input, target)
                
                # NOTE: Transformer's models output the sequence with length 1 less than target.
                diff = target.shape[0]-output.shape[0]
                # Reshape output to be [BATCH * SEQ_LEN, VOCAB_SIZE]
                output = output.reshape(-1, output.shape[2])
                
                # Cut first element in case of transformer training.
                target = target[diff:]
                # Reshape target to be [BATCH * SEQ_LEN]
                target = target.reshape(-1)
                
                # Calculate loss
                loss = self.loss_fn(output, target)
                mean_loss += loss.item()
                progress.set_postfix({"loss":loss.item()})

        return mean_loss/len(self.val_loader)
    
    def train(self, epochs: int, save_path: str) -> None:
        """
        Do training of model for specified number of epochs.

        Parameters:
            epochs (int): How much epochs to train.
            save_path (str): Path where to save model.
        """
        best_loss = 1e9
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            train_loss = self.train_one_epoch()
            val_loss = self.val_one_epoch()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(save_path, "best_weights.pt"))
            print(f"Train loss:{train_loss} Validation loss:{val_loss}")
            torch.save(self.model.state_dict(), os.path.join(save_path, "last_weights.pt"))