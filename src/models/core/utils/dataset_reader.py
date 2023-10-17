import torch
import os
from torch.utils.data import random_split, Dataset, DataLoader
from .constants import MAX_SENTENCE_SIZE, BOS_IDX, EOS_IDX, PAD_IDX

class ToxicTextDataset(Dataset):
    """
    Class that represent dataset with toxic and detoxified texts.
    """
    def __init__(self, toxic_texts: list[str], detoxified_texts: list[str]):
        """
        Initialize dataset by providing toxic texts list and detoxified version of this list.

        Parameters:
            toxic_texts (list[str]): List of toxic texts.
            detoxified_texts (list[str]): List of detoxified texts.
        """
        # Store texts
        self.toxic_texts = toxic_texts
        self.detoxified_texts = detoxified_texts
        
    def __getitem__(self, index: int) -> tuple[list, list]:
        return self.toxic_texts[index], self.detoxified_texts[index]
    
    def __len__(self):
        return len(self.toxic_texts)

def __pad_sequence(sequence: list[int], max_size: int) -> list[int]:
    """
    Pad or truncate sequence based on maximum sequence size.

    Parameters:
        sequence (list[int]): Sequence to process.

    Returns:
        list[int]: Processed sequence. 
    """
    if len(sequence) + 2 > max_size:
        return [BOS_IDX] + sequence[:max_size-2] + [EOS_IDX] 
    return [BOS_IDX] + sequence + [EOS_IDX] + [PAD_IDX]*(max_size - len(sequence) - 2)

def __collate_batch(batch: list) -> tuple[torch.tensor, torch.tensor]:
    """
    Perform padding and necessary permutations for Seq2Seq models in dataloader's batches.

    Parameters:
        batch (list): Batch from dataloader.

    Returns:
        list: Updated batch.
    """
    toxic_batch, detoxic_batch = [], []
    for _toxic, _detoxic in batch:
        # Perfrorm padding or truncuation on toxic and detoxified sequences from batch.
        toxic_batch.append(torch.tensor(__pad_sequence(_toxic, MAX_SENTENCE_SIZE)))
        detoxic_batch.append(torch.tensor(__pad_sequence(_detoxic, MAX_SENTENCE_SIZE)))
    
    # Union all sequences to batch.
    toxic_batch = torch.stack(toxic_batch, dim=0)
    detoxic_batch = torch.stack(detoxic_batch, dim=0)

    # Transpose batches to have shape [SEQ_LEN, BATCH]
    return toxic_batch.T, detoxic_batch.T

def create_dataloaders_from_dataset_file(filepath: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """
    Create dataloaders from .pt file created with 'make_dataset.py' script.

    Parameters:
        filepath (str): Path to dataset.pt file.
        batch_size (int): Size of a batch.

    Returns:
        DataLoader: Training dataloader.
        DataLoader: Validation dataloader.
    """
    if not os.path.exists(filepath):
        raise Exception(f"Cannot find dataset file {filepath}.\n Did you forget to create dataset using 'make_dataset.py'?")

    # Read text from saved file
    text_data = torch.load(filepath)
    # Create toxic text dataset
    toxic_text_dataset = ToxicTextDataset(text_data["toxic"], text_data["detoxified"])
    # Perform random splitting for train and validation datasets.
    train_dataset, val_dataset = random_split(toxic_text_dataset, [0.7, 0.3])
    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=__collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=__collate_batch)

    return train_dataloader, val_dataloader