from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import torch

MAX_SENTENCE_SIZE = 255
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

class ToxicTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.vocab_size = int(df['vocab_size'][0])
        
        self.toxic_texts = df.apply(lambda x: eval(x['input']), axis=1)
        self.detoxified_texts = df.apply(lambda x: eval(x['label']), axis=1)

    def __getitem__(self, index: int) -> tuple[list, list]:
        return self.toxic_texts[index], self.detoxified_texts[index]
    
    def __len__(self):
        return len(self.toxic_texts)

def collate_batch_train(batch: list):
    max_size = MAX_SENTENCE_SIZE
    _toxic_batch, _detoxic_batch = [], []
    for _toxic, _detoxic in batch:
        if len(_toxic) > max_size:
            _toxic = _toxic[:max_size]
        else:
            _toxic += [PAD_IDX]*(max_size - len(_toxic))
        if len(_detoxic) > max_size:
            _detoxic = _detoxic[:max_size]
        else:
            _detoxic += [PAD_IDX]*(max_size - len(_detoxic))
        _toxic_batch.append(torch.tensor(_toxic))
        _detoxic_batch.append(torch.tensor(_detoxic))
    
    _toxic_batch = torch.stack(_toxic_batch, dim=0)
    _detoxic_batch = torch.stack(_detoxic_batch, dim=0)
    return _toxic_batch, _detoxic_batch

def collate_batch_test(batch: list):
    max_size = MAX_SENTENCE_SIZE
    _toxic_batch = []
    for _toxic in batch:
        if len(_toxic) > max_size:
            _toxic = _toxic[:max_size]
        else:
            _toxic += [PAD_IDX]*(max_size - len(_toxic))
        _toxic_batch.append(torch.tensor(_toxic))
    
    _toxic_batch = torch.stack(_toxic_batch, dim=0)
    return _toxic_batch

def create_dataloaders(df: pd.DataFrame, batch_size=1) -> tuple[DataLoader, DataLoader]:
    dataset = ToxicTextDataset(df)
    train_dataset, val_dataset = random_split(dataset, [0.7, 0.3])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_batch_train)
    return train_loader, val_loader, dataset.vocab_size
