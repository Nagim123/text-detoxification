from tqdm import tqdm
import torch.nn as nn
import torch
import random

class DetoxificationModel(nn.Module):
    def __init__(self,  embedding_dim, hidden_dim, vocab_size):
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

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

def predict(model, vocab, input_data, tokenizer, max_sentence_size, bos_idx, eos_idx, device):
    model.eval()
    tokenized_data = tokenizer.tokenize(input_data)
    tokenized_data = torch.tensor([bos_idx] + vocab(tokenized_data) + [eos_idx]).unsqueeze(0).permute((1, 0))
    with torch.no_grad():
        pred = model(tokenized_data).to(device)
        _, token_ids = torch.max(pred, axis=2)
    return token_ids.view(-1)