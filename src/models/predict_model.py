import torch
import argparse
import pathlib
import os
import spacy
import transformer_model
import lstm_model
from torchtext.vocab import build_vocab_from_iterator

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
INPUT_FILE_PATH = os.path.join(SCRIPT_PATH, "../../data/raw/test.txt")
MODEL_WEIGHTS_PATH = os.path.join(SCRIPT_PATH, "../../models")
DATASET_PATH = os.path.join(SCRIPT_PATH, "../../data/interim/dataset.pt")
MAX_SENTENCE_SIZE = 100
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

class SpacyTokenizer:
    def __init__(self) -> None:
        self.spacy_eng = spacy.load("en_core_web_sm")
    def tokenize(self, input_data: str) -> list[str]:
        return [tok.text for tok in self.spacy_eng.tokenizer(input_data.lower())]

def tensor2text(input_tensor, vocab):
    idx2word = vocab.get_itos()
    tokens_list = input_tensor.tolist()
    return [idx2word[i] for i in tokens_list]

if __name__ == "__main__":

    with open(INPUT_FILE_PATH, "r") as input_file:
        input_data = input_file.read().split('\n')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_data = torch.load(DATASET_PATH)

    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    vocab = build_vocab_from_iterator(text_data["toxic"] + text_data["detoxified"], specials=special_symbols, max_tokens=10000, min_freq=2)
    vocab.set_default_index(0)
    
    available_models = {
        "transformer": {
            "model": transformer_model.DetoxificationModel(512, len(vocab), 0.1, MAX_SENTENCE_SIZE, PAD_IDX, device).to(device),
            "predict": transformer_model.predict,
        },
        "LSTM": {
            "model": lstm_model.DetoxificationModel(len(vocab), 300, 1024, device).to(device),
            "predict": lstm_model.predict,
        }
    }
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_type", choices=list(available_models.keys()))
    parser.add_argument("--weights", type=str, default="weights.pt")
    args = parser.parse_args()

    model = available_models[args.model_type]["model"]
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_PATH, args.weights), map_location=device))
    model_predict = available_models[args.model_type]["predict"]

    result = []
    for data in input_data:
        result.append(tensor2text(model_predict(model, vocab, data, SpacyTokenizer(), MAX_SENTENCE_SIZE, BOS_IDX, EOS_IDX, device), vocab))
        result[-1] = " ".join(result[-1][1:])
    print(result)