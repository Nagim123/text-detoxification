from torchtext.vocab import build_vocab_from_iterator
from tokenizers.basic_tokenizer import BasicTokenizer
import nltk
nltk.download('punkt')

class CustomTorchTokenizer(BasicTokenizer):

    def create_vocab(self, full_text: list[str]) -> None:
        full_tokenized_text = [nltk.tokenize.word_tokenize(text) for text in full_text]
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.vocab = build_vocab_from_iterator(full_tokenized_text, specials=special_symbols, min_freq=2)
        self.vocab.set_default_index(0)

    
    def tokenize(self, text: str) -> list:
        if self.vocab is None:
            raise Exception("Call create_vocab() first!")
        text = nltk.tokenize.word_tokenize(text)
        return self.vocab(text)