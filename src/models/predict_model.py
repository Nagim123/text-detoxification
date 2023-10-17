import torch
import argparse
import os

from torch import nn
from torchtext.vocab import Vocab
from core.utils.constants import MODEL_WEIGHTS_PATH, PREPROCESS_SCRIPT_PATH, INTERIM_PATH, BOS_IDX, EOS_IDX, MAX_SENTENCE_SIZE
from core.architectures import lstm, ae_lstm, transformer

class TextProcessor():
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def text2tensor(self, tokenized_text: list[str]) -> torch.tensor:
        return torch.tensor([BOS_IDX] + self.vocab(tokenized_text) + [EOS_IDX])

    def tensor2text(self, encoded_tensor: str) -> list[str]:
        idx2word = self.vocab.get_itos()
        return [idx2word[idx] for idx in encoded_tensor.tolist()]

def model_predict(model: nn.Module, tensor_input: list[int]):
    model.eval()
    tensor_input = tensor_input.unsqueeze(1)
    y_input = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(MAX_SENTENCE_SIZE):
            pred = model(tensor_input, y_input).to(device)
            _, token_id = torch.max(pred, axis=2)
            next_token = token_id.view(-1)[-1].item()
            if next_token == EOS_IDX:
                break
            next_tensor = torch.tensor([[next_token]])
            y_input = torch.cat((y_input, next_tensor), dim=0)
    return y_input.view(-1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available_models = {
        "lstm": lstm.DetoxificationModel(),
        "ae_lstm": ae_lstm.DetoxificationModel(device),
        "transformer": transformer.DetoxificationModel(device)
    }
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_type", choices=list(available_models.keys()))
    parser.add_argument("weights", type=str)
    parser.add_argument("file_path", type=str)
    parser.add_argument("--compare", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    compare_arg = "" if args.compare is None else "--translated_text_file " + args.compare
    os.system(f"python {PREPROCESS_SCRIPT_PATH} {args.file_path} temp.pt {compare_arg}")
    processed_data_path = os.path.join(INTERIM_PATH, "temp.pt")
    data_file = torch.load(processed_data_path, map_location=device)
    vocab = torch.load(os.path.join(INTERIM_PATH, "vocab.pt"), map_location=device)
    
    toxic_texts, detoxified_texts = data_file["toxic"], data_file["detoxified"]
    os.remove(processed_data_path)
    text_processor = TextProcessor(vocab)

    model = available_models[args.model_type]
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_PATH, args.weights), map_location=device))

    result = []
    for i in range(len(toxic_texts)):
        encoded_text = text_processor.text2tensor(toxic_texts[i])
        output = model_predict(model, encoded_text)
        result.append(text_processor.tensor2text(output))
        result[-1] = " ".join(result[-1][1:])
        
    print(result)