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
        self.toxic_texts, self.detoxified_texts = [], []
        self.do_logging = do_logging
        self.vocab_save_path = vocab_save_path

        with open(toxic_data_filepath, "r", encoding="UTF-8") as input_file:
            self.toxic_texts = input_file.read().split('\n')

        if not detoxified_data_filepath is None:
            with open(detoxified_data_filepath, "r", encoding="UTF-8") as input_file:
                self.detoxified_texts = input_file.read().split('\n')
        
        self.toxic_texts = self.__tokenize_texts(self.toxic_texts)
        self.detoxified_texts = self.__tokenize_texts(self.detoxified_texts)

        if not self.vocab_save_path is None:
            self.vocab = build_vocab_from_iterator(self.toxic_texts + self.detoxified_texts, min_freq=2, specials=SPECIAL_SYMBOLS, max_tokens=VOCAB_SIZE)
            self.vocab.set_default_index(0)
            
            self.toxic_texts = self.__apply_vocabulary(self.toxic_texts)
            self.detoxified_texts = self.__apply_vocabulary(self.detoxified_texts)

    def __tokenize_texts(self, texts: list[str]) -> list[list[str]]:
        spacy_eng = spacy.load("en_core_web_sm")
        tokenized_texts = []
        if self.do_logging:
            texts = tqdm(texts)
        for text in texts:
            tokenized_texts.append([tok.text for tok in spacy_eng.tokenizer(text)])
        return tokenized_texts
    
    def __apply_vocabulary(self, texts: list[str]) -> list[list[int]]:
        if self.do_logging:
            texts = tqdm(texts)
        return [self.vocab(text) for text in texts]

    def save_to_pt(self, filepath: str) -> None:
        data_to_save = {
            "toxic": self.toxic_texts,
            "detoxified": self.detoxified_texts
        }
        if not self.vocab_save_path is None:
            vocab_save_path = os.path.join(OUTPUT_DIR_PATH, self.vocab_save_path)
            torch.save(self.vocab, vocab_save_path)
        
        torch.save(data_to_save, filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("text_file", type=str)
    parser.add_argument("output_name", type=str)
    parser.add_argument("--tranlated_text_file", type=str)
    parser.add_argument("--vocab_encode")
    parser.add_argument("--logging", action="store_true")
    args = parser.parse_args()

    text_preprocessor = TextPreprocessor(args.text_file, args.tranlated_text_file, args.logging, args.vocab_encode)
    text_preprocessor.save_to_pt(os.path.join(OUTPUT_DIR_PATH, args.output_name))