import torch.nn as nn
import torch
import random
from ..utils.constants import VOCAB_SIZE

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
    def __init__(self, device):
        super(DetoxificationModel, self).__init__()
        self.vocab_size = VOCAB_SIZE
        embedding_dim = 300
        hidden_dim = 1024
        self.device = device
        self.encoder = Encoder(self.vocab_size, embedding_dim, hidden_dim, 2, 0.5)
        self.decoder = Decoder(self.vocab_size, embedding_dim, hidden_dim, self.vocab_size, 2, 0.5)
        
    def forward(self, source, target , teacher_force_ratio = 0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
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