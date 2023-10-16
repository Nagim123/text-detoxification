from ..utils.constants import VOCAB_SIZE
import torch.nn as nn
import torch

class DetoxificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_dim = 300
        hidden_dim = 1024
        vocab_size = VOCAB_SIZE
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.25)
        self.hidden2tag = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim//2, vocab_size),
        )
        
    def forward(self, text, _):
        embeds = self.word_embeddings(text)
        lstm_out, _ = self.lstm(embeds)
        predictions = self.hidden2tag(lstm_out)
        return predictions

def predict(model, vocab, input_data, tokenizer, max_sentence_size, bos_idx, eos_idx, device):
    model.eval()
    tokenized_data = tokenizer.tokenize(input_data)
    tokenized_data = torch.tensor([bos_idx] + vocab(tokenized_data) + [eos_idx]).unsqueeze(0).permute((1, 0))
    with torch.no_grad():
        pred = model(tokenized_data).to(device)
        _, token_ids = torch.max(pred, axis=2)
    return token_ids.view(-1)