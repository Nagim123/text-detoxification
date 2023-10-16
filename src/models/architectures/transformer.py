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



    
def predict(model, vocab, input_data, tokenizer, max_sentence_size, bos_idx, eos_idx, device):
    model.eval()
    tokenized_data = tokenizer.tokenize(input_data)
    tokenized_data = torch.tensor([bos_idx] + vocab(tokenized_data) + [eos_idx]).unsqueeze(0).permute((1, 0))
    y_input = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_sentence_size):
            pred = model(tokenized_data, y_input).to(device)
            _, token_id = torch.max(pred, axis=2)
            next_token = token_id.view(-1)[-1].item()
            if next_token == eos_idx:
                break
            next_tensor = torch.tensor([[next_token]])
            y_input = torch.cat((y_input, next_tensor), dim=0)
    return y_input.view(-1)