class BasicTokenizer():
    def create_vocab(self, full_text: list[str]):
        pass

    def tokenize(self, text: str):
        raise Exception("Not implemented")
    
    def __len__(self):
        return len(self.vocab)