import torch
from torch import nn
from tqdm import tqdm

class DetoxificationModel(nn.Module):
    def __init__(self, embedding_size, vocab_size, dropout, max_len, pad_idx, device):
        super(DetoxificationModel, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(embedding_size, 8, 3, 3, 4, dropout, batch_first=False)
        self.fc_out = nn.Linear(embedding_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = pad_idx
    
    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask
    
    def forward(self, src, trg):
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


def train_one_epoch(model, train_loader, optmizer, loss_fn, device):
    model.train()
    progress = tqdm(train_loader)
    for batch in progress:
        input, target = batch
        input, target = input.to(device), target.to(device)
        output = model(input, target[:-1])
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
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

            output = model(input, target[:-1])
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            
            loss = loss_fn(output, target)
            progress.set_postfix({"loss":loss.item()})
    
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