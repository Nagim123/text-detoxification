import torch
import pathlib
import os
import argparse

import transformer_model
import lstm_model
import simple_lstm_model

from torch.utils.data import random_split, Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator

from torch import nn
from torch import optim

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
DATASET_PATH = os.path.join(SCRIPT_PATH, "../../data/interim/dataset.pt")
MODEL_WEIGHTS_PATH = os.path.join(SCRIPT_PATH, "../../models/weights.pt")
MAX_SENTENCE_SIZE = 100
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


class ToxicTextDataset(Dataset):
    def __init__(self, toxic_texts: list[str], detoxified_texts: list[str]):
        self.toxic_texts = toxic_texts
        self.detoxified_texts = detoxified_texts
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.vocab = build_vocab_from_iterator(toxic_texts + detoxified_texts, specials=self.special_symbols, max_tokens=10000, min_freq=2)
        self.vocab.set_default_index(0)

    def __getitem__(self, index: int) -> tuple[list, list]:
        return self.vocab(self.toxic_texts[index]), self.vocab(self.detoxified_texts[index])
    
    def __len__(self):
        return len(self.toxic_texts)
    
def collate_batch(batch: list):
    max_size = MAX_SENTENCE_SIZE
    _toxic_batch, _detoxic_batch = [], []
    for _toxic, _detoxic in batch:
        if len(_toxic) + 2 > max_size:
            _toxic = [BOS_IDX] + _toxic[:max_size-2] + [EOS_IDX] 
        else:
            _toxic = [BOS_IDX] + _toxic + [EOS_IDX] + [PAD_IDX]*(max_size - len(_toxic) - 2)

        if len(_detoxic) + 2 > max_size:
            _detoxic = [BOS_IDX] + _detoxic[:max_size-2] + [EOS_IDX] 
        else:
            _detoxic = [BOS_IDX] + _detoxic + [EOS_IDX] + [PAD_IDX]*(max_size - len(_detoxic) - 2)
        _toxic_batch.append(torch.tensor(_toxic))
        _detoxic_batch.append(torch.tensor(_detoxic))
    
    _toxic_batch = torch.stack(_toxic_batch, dim=0)
    _detoxic_batch = torch.stack(_detoxic_batch, dim=0)
    return _toxic_batch.T, _detoxic_batch.T

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_data = torch.load(DATASET_PATH)
    toxic_text_dataset = ToxicTextDataset(text_data["toxic"], text_data["detoxified"])
    train_dataset, val_dataset = random_split(toxic_text_dataset, [0.7, 0.3])
    vocab_size = len(toxic_text_dataset.vocab)

    available_models = {
        "transformer": {
            "model": transformer_model.DetoxificationModel(512, vocab_size, 0.1, MAX_SENTENCE_SIZE, PAD_IDX, device).to(device),
            "train": transformer_model.train_one_epoch,
            "validate": transformer_model.val_one_epoch,
        },
        "LSTM": {
            "model": lstm_model.DetoxificationModel(vocab_size, 300, 1024, device).to(device),
            "train": lstm_model.train_one_epoch,
            "validate": lstm_model.val_one_epoch,
        },
        "simple_LSTM": {
            "model": simple_lstm_model.DetoxificationModel(300, 1024, vocab_size).to(device),
            "train": simple_lstm_model.train_one_epoch,
            "validate": simple_lstm_model.val_one_epoch,
        }
    }

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_type", choices=list(available_models.keys()))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)


    model = available_models[args.model_type]["model"]
    train_one_epoch = available_models[args.model_type]["train"]
    val_one_epoch = available_models[args.model_type]["validate"]
    optmizer = optim.Adam(model.parameters(), 3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(args.epochs):
        train_one_epoch(model, train_dataloader, optmizer, loss_fn, device)
        val_one_epoch(model, val_dataloader, loss_fn, device)
        print("SAVE MODEL")
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)