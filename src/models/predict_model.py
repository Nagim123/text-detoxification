from model import DetoxificationModel
from dataset_loader import create_dataloaders
import torch
import os
import logging
import pathlib
script_path = pathlib.Path(__file__).parent.resolve()
import argparse
from text_tokenizers.bpe_tokenizer import BPETokenizer


def load_model(model_name: str, require_weights: bool, default_model):
    path_to_model = os.path.join(script_path, f"../../models/{model_name}.pt")
    path_to_weights = os.path.join(script_path, f"../../models//{model_name}.pth")

    if not os.path.exists(path_to_model):
        logging.warn(f"Model {model_name} is not enough. Training current one from scratch!")
        model = default_model
        model_scripted = torch.jit.script(model)
        model_scripted.save(path_to_model)
    else:
        model = torch.jit.load(path_to_model)
        if require_weights:
            if not os.path.exists(path_to_weights):
                raise Exception("Model is loaded but weights not found!")
            model.load_state_dict(torch.load(path_to_weights,map_location=torch.device('cpu')))
    
    return model, path_to_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_name", type=str)
    parser.add_argument("input_file", type=str)
    
    args = parser.parse_args()

    tokenizer = BPETokenizer()
    tokenizer.create_vocab("")
    # Loading model
    model, model_weights_save_path = load_model(args.model_name, True, DetoxificationModel(len(tokenizer)))
    
    text_file = open(os.path.join(script_path, f"../../data/raw/{args.input_file}"), "r")
    text_input = text_file.read()
    text_file.close()
    

    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(tokenizer.tokenize(text_input))
        output = model(input_tensor)
        _, output = torch.max(output, axis=1)
        print(output.shape)
        #print(output.tolist())
        print(tokenizer.vocab.decode(output.tolist()))