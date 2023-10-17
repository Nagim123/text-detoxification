import os
import torch
from ..utils.constants import PREPROCESS_SCRIPT_PATH, INTERIM_PATH, BOS_IDX, EOS_IDX

class TextManager():
    """
    Class to process raw text into tokenized and encoded ones.
    """

    def __init__(self, toxic_data_filepath: str, detoxified_data_filepath: str = None, device: str = "cpu") -> None:
        """
        Create toxic text manager.

        Parameters:
            toxic_data_filepath (str): Path to file with toxic texts separated by new lines.
            detoxified_data_filepath (str): Path to file with detoxified texts separated by new lines (Optional for comparison).
            device (str): Device to load data on.
        """

        # Read toxic texts
        with open(toxic_data_filepath, "r") as read_file:
            self.raw_toxic = read_file.read().split('\n')
        
        # Has command line arguments if detoxified texts are provided.
        compare_arg = ""
        # Read detoxified texts
        if not detoxified_data_filepath is None:
            compare_arg = "--translated_text_file " + detoxified_data_filepath
            with open(detoxified_data_filepath, "r") as read_file:
                self.clean_detoxified = read_file.read().split('\n')

        # Call external script to prepare text for prediction
        os.system(f"python {PREPROCESS_SCRIPT_PATH} {toxic_data_filepath} temp.pt {compare_arg}")
        
        # Get processed data from script resulting file and delete it
        processed_data_path = os.path.join(INTERIM_PATH, "temp.pt")
        data_file = torch.load(processed_data_path, map_location=device)
        os.remove(processed_data_path)

        # Get toxic and detoxified texts
        self.tokenized_toxic, self.tokenized_detoxified = data_file["toxic"], data_file["detoxified"]
        
        # Read vocabulary of dataset
        self.vocab = torch.load(os.path.join(INTERIM_PATH, "vocab.pt"), map_location=device)

    def text2tensor(self, tokenized_text: list[str]) -> torch.tensor:
        """
        Convert list of tokens to tensor and put special symbols.

        Parameters:
            tokenized_text (list[str]): Sequence of tokens.

        Returns:
            torch.tensor: Encoded by vocabulary tensor.
        """
        
        return torch.tensor([BOS_IDX] + self.vocab(tokenized_text) + [EOS_IDX])

    def tensor2text(self, encoded_tensor: str) -> list[str]:
        """
        Convert tensor to list of tokens.

        Parameters:
            encoded_tensor (torch.tensor): Tensor.

        Returns:
            list[str]: Sequence of tokens.
        """
        
        # Get index to string dictionary from vocabulary
        idx2word = self.vocab.get_itos()
        # Use it on tensor
        return [idx2word[idx] for idx in encoded_tensor.tolist()]
    
    def get_raw(self, index: int) -> str:
        """
        Get raw toxic text.

        Parameters:
            index (int): Index of text.

        Returns:
            str: Raw toxic text.
        """
        return self.raw_toxic[index]
    
    def get_tokenized(self, index: int) -> list[str]:
        """
        Get tokenized toxic text.

        Parameters:
            index (int): Index of text.

        Returns:
            list[str]: Tokenized toxic text.
        """
        return self.tokenized_toxic[index]
    
    def get_encoded(self, index: int) -> list[int]:
        """
        Get vocabulary encoded toxic text.

        Parameters:
            index (int): Index of text.

        Returns:
            list[int]: Encoded toxic text.
        """
        # Use vocabulary for encoding
        return self.text2tensor(self.tokenized_toxic[index])

    def __len__(self) -> int:
        return len(self.raw_toxic)