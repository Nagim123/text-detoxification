import torch
import spacy
import argparse
import os
from tqdm import tqdm
from constants import OUTPUT_DIR_PATH, VOCAB_SIZE, SPECIAL_SYMBOLS
from torchtext.vocab import build_vocab_from_iterator

class TextPreprocessor():
    """
    Class to tokenize and create vocabulary over text.
    """
    def __init__(self, toxic_data_filepath: str, detoxified_data_filepath: str = None, do_logging: bool = False, vocab_save_path: str = None) -> None:
        """
        Create text processor.
        
        Parameters:
            toxic_data_filepath (str): File with toxic sentences separated by new line.
            detoxified_data_filepath (str): File with detoxified sentences separated by new line.
            do_logging (bool): Do logging in console.
            vocab_save_path (str): If it is not None, then vocabulary will be created and used for encoding.
        """
        
        # Set toxic and detoxified texts as empty lists in the beginning
        self.toxic_texts, self.detoxified_texts = [], []

        # Store arguments
        self.do_logging = do_logging
        self.vocab_save_path = vocab_save_path
        
        # Read toxic data
        with open(toxic_data_filepath, "r", encoding="UTF-8") as input_file:
            self.toxic_texts = input_file.read().lower().split('\n')
        
        # Read detoxified data if provided
        if not detoxified_data_filepath is None:
            with open(detoxified_data_filepath, "r", encoding="UTF-8") as input_file:
                self.detoxified_texts = input_file.read().lower().split('\n')
        
        # Tokenize texts
        self.toxic_texts = self.__tokenize_texts(self.toxic_texts)
        self.detoxified_texts = self.__tokenize_texts(self.detoxified_texts)

        # Create vocabulary if needed
        if not self.vocab_save_path is None:
            self.vocab = build_vocab_from_iterator(self.toxic_texts + self.detoxified_texts, min_freq=2, specials=SPECIAL_SYMBOLS, max_tokens=VOCAB_SIZE)
            self.vocab.set_default_index(0)
            
            # Encode texts using vocabulary
            self.toxic_texts = self.__apply_vocabulary(self.toxic_texts)
            self.detoxified_texts = self.__apply_vocabulary(self.detoxified_texts)

    def __tokenize_texts(self, texts: list[str]) -> list[list[str]]:
        """
        Tokenize list of texts
        
        Parameters:
            texts (list[str]): List of texts to tokenize.

        Returns:
            list[list[str]]: List of list of tokens for each text.
        """
        # Load spacy tokenizer
        spacy_eng = spacy.load("en_core_web_sm")
        
        # Set up progress tracking if logging is on
        if self.do_logging:
            texts = tqdm(texts)
        
        # Tokenize each text using spacy tokenizer
        tokenized_texts = []
        for text in texts:
            tokenized_texts.append([tok.text for tok in spacy_eng.tokenizer(text)])
        return tokenized_texts
    
    def __apply_vocabulary(self, texts: list[list[int]]) -> list[list[int]]:
        """
        Encode tokens of list of texts to numbers using vocabulary.
        
        Parameters:
            texts (list[int]): List of list of tokens for each text.

        Returns:
            list[list[int]]: List of number sequences.
        """
        # Set up progress tracking if logging is on
        if self.do_logging:
            texts = tqdm(texts)
        
        # Use vocabulary to convert tokens to numbers
        return [self.vocab(text) for text in texts]

    def save_to_pt(self, filepath: str) -> None:
        """
        Save preprocessed texts as dataset.
        
        Parameters:
            filepath (str): Save path of dataset.
        """

        # Assign data to a dictionary
        data_to_save = {
            "toxic": self.toxic_texts,
            "detoxified": self.detoxified_texts
        }

        # If we encode texts using vocabulary we need to save it for future usage.
        if not self.vocab_save_path is None:
            vocab_save_path = os.path.join(OUTPUT_DIR_PATH, self.vocab_save_path)
            torch.save(self.vocab, vocab_save_path)
        # Save data
        torch.save(data_to_save, filepath)

if __name__ == "__main__":
    # Read command line arguments.
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("text_file", type=str)
    parser.add_argument("output_name", type=str)
    parser.add_argument("--translated_text_file", type=str)
    parser.add_argument("--vocab_encode")
    parser.add_argument("--logging", action="store_true")
    args = parser.parse_args()

    # Preprocess texts from input files
    text_preprocessor = TextPreprocessor(args.text_file, args.translated_text_file, args.logging, args.vocab_encode)
    # Save the result
    text_preprocessor.save_to_pt(os.path.join(OUTPUT_DIR_PATH, args.output_name))