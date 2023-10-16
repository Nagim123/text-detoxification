import torch
import spacy
import argparse
import os
from tqdm import tqdm
from constants import OUTPUT_DIR_PATH

class TextPreprocessor():
    def __init__(self, toxic_data_filepath: str, detoxified_data_filepath: str = None, do_logging: bool = False) -> None:
        self.toxic_texts = None
        self.detoxified_texts = None
        self.do_logging = do_logging

        with open(toxic_data_filepath, "r", encoding="UTF-8") as input_file:
            self.toxic_texts = input_file.read().split('\n')
        
        if not detoxified_data_filepath is None:
            with open(detoxified_data_filepath, "r", encoding="UTF-8") as input_file:
                self.detoxified_texts = input_file.read().split('\n')

    def __tokenize_texts(self, texts: list[str]) -> list[list[str]]:
        spacy_eng = spacy.load("en_core_web_sm")
        tokenized_texts = []
        if self.do_logging:
            texts = tqdm(texts)
        for text in texts:
            tokenized_texts.append([tok.text for tok in spacy_eng.tokenizer(text)])
        return tokenized_texts

    def save_to_pt(self, filepath: str) -> None:
        data_to_save = {
            "toxic": self.__tokenize_texts(self.toxic_texts)
        }
        if not self.detoxified_texts is None:
            data_to_save["detoxified"] = self.__tokenize_texts(self.detoxified_texts)
        
        torch.save(data_to_save, filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("text_file", type=str)
    parser.add_argument("output_name", type=str)
    parser.add_argument("--tranlated_text_file", type=str)
    parser.add_argument("--logging", action="store_true")
    args = parser.parse_args()

    text_preprocessor = TextPreprocessor(args.text_file, args.tranlated_text_file, args.logging)
    text_preprocessor.save_to_pt(os.path.join(OUTPUT_DIR_PATH, args.output_name))

