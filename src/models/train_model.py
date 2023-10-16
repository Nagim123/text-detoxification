import torch
import argparse

from core.utils.dataset_reader import create_dataloaders_from_dataset_file
from core.utils.constants import DATASET_PATH, MODEL_WEIGHTS_PATH
from core.trainer.trainer import Seq2SeqTrainer
from core.architectures import lstm, ae_lstm, transformer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    available_models = {
        "lstm": lstm.DetoxificationModel(),
        "ae_lstm": ae_lstm.DetoxificationModel(device),
        "transformer": transformer.DetoxificationModel(device)
    }
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("model_type", choices=list(available_models.keys()))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train_dataloader, val_dataloader = create_dataloaders_from_dataset_file(DATASET_PATH, args.batch_size)
    
    model = available_models[args.model_type].to(device)
    trainer = Seq2SeqTrainer(model, train_dataloader, val_dataloader, device)
    trainer.train(args.epochs, MODEL_WEIGHTS_PATH)