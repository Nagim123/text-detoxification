from ..utils.constants import PAD_IDX, MAX_SENTENCE_SIZE, VOCAB_SIZE
import torch
from torch import nn


class DetoxificationModel(nn.Module):
    def __init__(self, device):
        super(DetoxificationModel, self).__init__()
        embedding_size = 512
        dropout = 0.1
        vocab_size = VOCAB_SIZE

        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(MAX_SENTENCE_SIZE, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(embedding_size, 8, 3, 3, 4, dropout, batch_first=False)
        self.fc_out = nn.Linear(embedding_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = PAD_IDX
    
    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask
    
    def forward(self, src, trg):
        if self.training:
            trg = trg[:-1]

        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N)
            .to(self.device)
        )
        trg_positions = (
            torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N)
            .to(self.device)
        )
        embed_src = self.dropout(
            (self.word_embedding(src) + self.position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.word_embedding(trg) + self.position_embedding(trg_positions))
        )
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len, self.device)
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            tgt_mask = trg_mask
        )
        out = self.fc_out(out)
        return out