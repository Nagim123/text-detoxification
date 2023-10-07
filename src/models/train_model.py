import torch
import pathlib
import os
import argparse
from torch.utils.data import random_split, Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
from torch import nn
from torch import optim

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
DATASET_PATH = os.path.join(SCRIPT_PATH, "../../data/interim/dataset.pt")
MODEL_WEIGHTS_PATH = os.path.join(SCRIPT_PATH, "../../models/weights.pt")
MAX_SENTENCE_SIZE = 100
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

class DetoxificationModel(nn.Module):
    def __init__(self, embedding_size, vocab_size, dropout, max_len, device):
        super(DetoxificationModel, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(embedding_size, 8, 3, 3, 4, dropout, batch_first=False)
        self.fc_out = nn.Linear(embedding_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = PAD_IDX
    
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

class ToxicTextDataset(Dataset):
    def __init__(self, toxic_texts: list[str], detoxified_texts: list[str]):
        self.toxic_texts = toxic_texts
        self.detoxified_texts = detoxified_texts
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.vocab = build_vocab_from_iterator(toxic_texts + detoxified_texts, special_first=self.special_symbols, max_tokens=10000, min_freq=2)
        self.vocab.set_default_index(0)

    def __getitem__(self, index: int) -> tuple[list, list]:
        return self.vocab(self.toxic_texts[index]), self.vocab(self.detoxified_texts[index])
    
    def __len__(self):
        return len(self.toxic_texts)

def train_one_epoch(model, train_loader, optmizer, loss_fn):
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

def val_one_epoch(model, val_loader, loss_fn):
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
    

def collate_batch(batch: list):
    max_size = MAX_SENTENCE_SIZE
    _toxic_batch, _detoxic_batch = [], []
    for _toxic, _detoxic in batch:
        if len(_toxic) + 2 > max_size:
            _toxic = BOS_IDX + _toxic[:max_size-2] + EOS_IDX 
        else:
            _toxic = BOS_IDX + _toxic + [PAD_IDX]*(max_size - len(_toxic) - 2) + EOS_IDX

        if len(_detoxic) + 2 > max_size:
            _toxic = BOS_IDX + _detoxic[:max_size-2] + EOS_IDX 
        else:
            _detoxic = BOS_IDX + _detoxic + [PAD_IDX]*(max_size - len(_detoxic) - 2) + EOS_IDX
        _toxic_batch.append(torch.tensor(_toxic))
        _detoxic_batch.append(torch.tensor(_detoxic))
    
    _toxic_batch = torch.stack(_toxic_batch, dim=0)
    _detoxic_batch = torch.stack(_detoxic_batch, dim=0)
    return _toxic_batch.T, _detoxic_batch.T

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_data = torch.load(DATASET_PATH)
    toxic_text_dataset = ToxicTextDataset(text_data["toxic"], text_data["detoxified"])
    train_dataset, val_dataset = random_split(toxic_text_dataset, [0.7, 0.3])
    vocab_size = len(toxic_text_dataset.vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = DetoxificationModel(512, vocab_size, 0.1, MAX_SENTENCE_SIZE, device).to(device)
    optmizer = optim.Adam(model.parameters(), 3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(args.epochs):
        train_one_epoch(model, train_dataloader, optmizer, loss_fn)
        val_one_epoch(model, val_dataloader, loss_fn)
        print("SAVE MODEL")
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)