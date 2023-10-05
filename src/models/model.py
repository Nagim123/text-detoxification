import torch.nn as nn
import torch.nn.functional as F

class DetoxificationModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        embedding_dim = 2048
        hidden_dim = 1024
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.25, batch_first=True)
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