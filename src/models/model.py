import torch.nn as nn
import torch.nn.functional as F

class DetoxificationModel(nn.Module):
    def __init__(self,  embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=4, bidirectional=True, dropout=0.25)
        self.hidden2tag = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, tagset_size),
        )
        
    def forward(self, text):

        # text shape= [sent len, batch size]
        embeds = self.word_embeddings(text)
        lstm_out, _ = self.lstm(embeds)
        predictions = self.hidden2tag(lstm_out)
        # predictions shape = [sent len, batch size, output dim]
        return predictions#F.log_softmax(predictions,dim=1)