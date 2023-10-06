from text_tokenizers.basic_tokenizer import BasicTokenizer
from transformers import RobertaTokenizerFast
import os
import pathlib
script_path = pathlib.Path(__file__).parent.resolve()

class BPETokenizer(BasicTokenizer):

    def create_vocab(self, full_text: list[str]) -> None:
        self.vocab = RobertaTokenizerFast(os.path.join(script_path, "BPETokenizer"))

    
    def tokenize(self, text: str) -> list:
        if self.vocab is None:
            raise Exception("Call create_vocab() first!")
        return self.vocab(text)