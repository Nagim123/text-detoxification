from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import random

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p) -> None:
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        _, (hidden, cell) = self.rnn(embedding)

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p) -> None:
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

class DetoxificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, device):
        super(DetoxificationModel, self).__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, 2, 0.5)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, vocab_size, 2, 0.5)
        
    def forward(self, source, target , teacher_force_ratio = 0.5):
        batch_size = source.shape[1]
        target_len = source.shape[0]
        target_vocab_size = self.vocab_size

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        hidden, cell = self.encoder(source)
        x = target[0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
        
        return outputs

def train_one_epoch(model, train_loader, optmizer, loss_fn, device):
    model.train()
    progress = tqdm(train_loader)
    for batch in progress:
        input, target = batch
        input, target = input.to(device), target.to(device)
        output = model(input, target).to(device)
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

            output = model(input, target).to(device)
            output = output.reshape(-1, output.shape[2])
            target = target.reshape(-1)
            
            loss = loss_fn(output, target)
            progress.set_postfix({"loss":loss.item()})

def predict(model, vocab, input_data, tokenizer, max_sentence_size, bos_idx, eos_idx, device):
    model.eval()
    tokenized_data = tokenizer.tokenize(input_data)
    tokenized_data = torch.tensor([bos_idx] + vocab(tokenized_data) + [eos_idx]).unsqueeze(0).permute((1, 0))
    result = []
    with torch.no_grad():
        pred = model(tokenized_data).to(device)
        _, token_id = torch.max(pred, axis=2)
        token_id.view(-1)
        print(token_id)
        for token in token_id:
            result.append(token.item())
    cleared_result = []
    for i in range(min(max_sentence_size,len(result))):
        if result[i] == eos_idx:
            break
        cleared_result.append(result[i])
    return torch.tensor(cleared_result)