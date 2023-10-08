from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch

class DetoxificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.25)
        self.hidden2tag = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim//2, vocab_size),
        )
        
    def forward(self, text):

        # text shape= [sent len, batch size]
        embeds = self.word_embeddings(text)
        lstm_out, _ = self.lstm(embeds)
        predictions = self.hidden2tag(lstm_out)
        # predictions shape = [sent len, batch size, output dim]
        return predictions#F.log_softmax(predictions,dim=1)
    

def train_one_epoch(model, train_loader, optmizer, loss_fn, device):
    model.train()
    progress = tqdm(train_loader)
    for batch in progress:
        input, target = batch
        input, target = input.to(device), target.to(device)
        output = model(input).to(device)
        output = output.reshape(-1, output.shape[2])
        target = target.reshape(-1)
        optmizer.zero_grad()
        
        loss = loss_fn(output, target)
        loss.backward()

        optmizer.step()
        progress.set_postfix({"loss":loss.item()})

def val_one_epoch(model, val_loader, loss_fn, device):
    model.eval()
    progress = tqdm(val_loader)
    with torch.no_grad():
        for batch in progress:
            input, target = batch
            input, target = input.to(device), target.to(device)

            output = model(input).to(device)
            output = output.reshape(-1, output.shape[2])
            target = target.reshape(-1)
            
            loss = loss_fn(output, target)
            progress.set_postfix({"loss":loss.item()})