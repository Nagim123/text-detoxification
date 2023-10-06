from text_tokenizers.basic_tokenizer import BasicTokenizer
from transformers import RobertaTokenizerFast
import os
import pathlib
script_path = pathlib.Path(__file__).parent.resolve()

class BPETokenizer(BasicTokenizer):

    def create_vocab(self, full_text: list[str]) -> None:
        vocab_path = os.path.join(script_path, "BPETokenizer/vocab.json")
        merge_path = os.path.join(script_path, "BPETokenizer/merges.txt")
        self.vocab = RobertaTokenizerFast(vocab_path, merge_path)

    
    def tokenize(self, text: str) -> list:
        if self.vocab is None:
            raise Exception("Call create_vocab() first!")
        return self.vocab(text)['input_ids']