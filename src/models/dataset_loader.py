from torch.utils.data import DataLoader, Dataset, random_split

class ToxicTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BasicTokenizer):
        self.tokenizer = tokenizer
        
        self.toxic_texts =  df["toxic"].tolist()
        self.detoxified_texts = df["detoxified"].tolist()
        

    def __getitem__(self, index: int) -> tuple[list, list]:
        return self.tokenized_labels[index], self.tokenized_targets[index]
    
    def __len__(self):
        return len(self.toxic_texts)


def create_dataloaders(df: pd.DataFrame, batch_size=32) -> tuple[DataLoader, DataLoader]:
    dataset = ToxicTextDataset(df, CustomTorchTokenizer())
    train_dataset, val_dataset = random_split(dataset, [0.7, 0.3])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, dataset.vocab_size