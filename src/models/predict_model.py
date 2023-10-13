import torch
import argparse
import pathlib
import os
import spacy
import transformer_model
import lstm_model
from torchtext.vocab import build_vocab_from_iterator
from torchmetrics import TranslationEditRate
from torchmetrics.text.rouge import ROUGEScore

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
EXTERNAL_PATH = os.path.join(SCRIPT_PATH, "../../data/external")
INTERIM_PATH = os.path.join(SCRIPT_PATH, "../../data/interim")
MODEL_WEIGHTS_PATH = os.path.join(SCRIPT_PATH, "../../models")
DATASET_PATH = os.path.join(INTERIM_PATH, "dataset.pt")
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
    parser.add_argument("file_path", type=str)
    parser.add_argument("--weights", type=str, default="weights.pt")
    parser.add_argument("--compare", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    with open(os.path.join(EXTERNAL_PATH, args.file_path), "r") as input_file:
        input_data = input_file.read().split('\n')
    if args.compare:
        with open(os.path.join(EXTERNAL_PATH, args.file_path), "r") as compare_file:
            compare_data = compare_file.read().split('\n')
        if len(input_data) != len(compare_data):
            raise Exception("The number of lines in input and compare files must be equal!")

    model = available_models[args.model_type]["model"]
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_PATH, args.weights), map_location=device))
    model_predict = available_models[args.model_type]["predict"]

    s_tokenizer = SpacyTokenizer()
    ter = TranslationEditRate()
    rogue = ROUGEScore(rouge_keys=("rouge1"))

    result = []
    ter_scores = []
    rog_score = []
    for i in range(len(input_data)):
        result.append(tensor2text(model_predict(model, vocab, input_data[i], s_tokenizer, MAX_SENTENCE_SIZE, BOS_IDX, EOS_IDX, device), vocab))    
        result[-1] = " ".join(result[-1][1:])
        if args.compare:
            ter_scores.append(ter(result[-1][1:], [compare_data[i]]).item())
            rog_score.append(rogue(result[-1][1:], compare_data[i])["rouge1_fmeasure"].item())
    if args.out_dir:
        with open(os.path.join(EXTERNAL_PATH, args.out_dir), "w") as write_file:
            for i in range(len(result)):
                if args.compare:
                    write_file.write(f"{result[i]}${ter_scores[i]}${rog_score[i]}\n")
                else:
                    write_file.write(result[i] + '\n')
    print(result)
    print(ter_scores)