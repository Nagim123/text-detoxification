import torch
import pathlib
import os
import spacy
from torchtext.vocab import build_vocab_from_iterator
from torch import nn

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
INPUT_FILE_PATH = os.path.join(SCRIPT_PATH, "../../data/raw/test.txt")
MODEL_WEIGHTS_PATH = os.path.join(SCRIPT_PATH, "../../models/weights.pt")
DATASET_PATH = os.path.join(SCRIPT_PATH, "../../data/interim/dataset.pt")
MAX_SENTENCE_SIZE = 100
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

class DetoxificationModel(nn.Module):
    def __init__(self, embedding_size, vocab_size, dropout, max_len, device):
        super(DetoxificationModel, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(embedding_size, 8, 3, 3, 4, dropout, batch_first=False)
        self.fc_out = nn.Linear(embedding_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = PAD_IDX
    
    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        return src_mask
    
    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N)
            .to(self.device)
        )
        trg_positions = (
            torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N)
            .to(self.device)
        )
        embed_src = self.dropout(
            (self.word_embedding(src) + self.position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.word_embedding(trg) + self.position_embedding(trg_positions))
        )
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len, self.device)
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            tgt_mask = trg_mask
        )
        out = self.fc_out(out)
        return out

class SpacyTokenizer:
    def __init__(self) -> None:
        self.spacy_eng = spacy.load("en_core_web_sm")
    def tokenize(self, input_data: str) -> list[str]:
        return [tok.text for tok in self.spacy_eng.tokenizer(input_data.lower())]

def tensor2text(input_tensor, vocab):
    idx2word = vocab.get_itos()
    tokens_list = input_tensor.tolist()
    return [idx2word[i] for i in tokens_list]

def model_predict(model, vocab, input_data, tokenizer, device):
    model.eval()
    tokenized_data = tokenizer.tokenize(input_data)
    tokenized_data = torch.tensor(vocab(tokenized_data)).unsqueeze(0).permute((1, 0))
    y_input = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(MAX_SENTENCE_SIZE):
            pred = model(tokenized_data, y_input).to(device)
            _, token_id = torch.max(pred, axis=2)
            next_token = token_id.view(-1)[-1].item()
            if next_token == EOS_IDX:
                break
            next_tensor = torch.tensor([[next_token]])
            y_input = torch.cat((y_input, next_tensor), dim=0)
    return tensor2text(y_input.view(-1))

if __name__ == "__main__":

    with open(INPUT_FILE_PATH, "r") as input_file:
        input_data = input_file.read()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_data = torch.load(DATASET_PATH)
    
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    vocab = build_vocab_from_iterator(text_data["toxic"] + text_data["detoxified"], special_first=special_symbols, max_tokens=10000, min_freq=2)
    vocab.set_default_index(0)

    model = DetoxificationModel(512, len(vocab), 0.1, MAX_SENTENCE_SIZE, device).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    
    print(" ".join(model_predict(model, vocab, input_data, SpacyTokenizer())))