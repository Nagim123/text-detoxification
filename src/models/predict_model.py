import torch
import argparse
import os

from torch import nn
from torchtext.vocab import Vocab
from core.utils.constants import MODEL_WEIGHTS_PATH, PREPROCESS_SCRIPT_PATH, INTERIM_PATH, BOS_IDX, EOS_IDX, MAX_SENTENCE_SIZE
from core.architectures import lstm, ae_lstm, transformer

class ToxicTextManager():
    """
    Class to process raw text into tokenized and encoded ones.
    """

    def __init__(self, toxic_data_filepath: str, detoxified_data_filepath: str = None) -> None:
        """
        Create toxic text manager.

        Parameters:
            toxic_data_filepath (str): Path to file with toxic texts separated by new lines.
            detoxified_data_filepath (str): Path to file with detoxified texts separated by new lines (Optional for comparison).
        """

        # Read toxic texts
        with open(toxic_data_filepath, "r") as read_file:
            self.clean_toxic = read_file.read().split('\n')
        
        # Has command line arguments if detoxified texts are provided.
        compare_arg = ""
        # Read detoxified texts
        if not detoxified_data_filepath is None:
            compare_arg = "--translated_text_file " + detoxified_data_filepath
            with open(detoxified_data_filepath, "r") as read_file:
                self.clean_detoxified = read_file.read().split('\n')

        # Call external script to prepare text for prediction
        os.system(f"python {PREPROCESS_SCRIPT_PATH} {args.file_path} temp.pt {compare_arg}")
        
        # Get processed data from script resulting file and delete it
        processed_data_path = os.path.join(INTERIM_PATH, "temp.pt")
        data_file = torch.load(processed_data_path, map_location=device)
        os.remove(processed_data_path)

        # Get toxic and detoxified texts
        self.tokenized_toxic, self.tokenized_detoxified = data_file["toxic"], data_file["detoxified"]
        
        # Read vocabulary of dataset
        self.vocab = torch.load(os.path.join(INTERIM_PATH, "vocab.pt"), map_location=device)

    def text2tensor(self, tokenized_text: list[str]) -> torch.tensor:
        """
        Convert list of tokens to tensor and put special symbols.

        Parameters:
            tokenized_text (list[str]): Sequence of tokens.

        Returns:
            torch.tensor: Encoded by vocabulary tensor.
        """
        
        return torch.tensor([BOS_IDX] + self.vocab(tokenized_text) + [EOS_IDX])

    def tensor2text(self, encoded_tensor: str) -> list[str]:
        """
        Convert tensor to list of tokens.

        Parameters:
            encoded_tensor (torch.tensor): Tensor.

        Returns:
            list[str]: Sequence of tokens.
        """
        
        # Get index to string dictionary from vocabulary
        idx2word = self.vocab.get_itos()
        # Use it on tensor
        return [idx2word[idx] for idx in encoded_tensor.tolist()]
    
    def get_clean(self, index):
        return self.clean_toxic[index]
    
    def get_tokenized(self, index):
        return self.tokenized_toxic[index]
    
    def get_encoded(self, index):
        return self.text2tensor(self.tokenized_toxic[index])

    def __len__(self):
        return len(self.clean_toxic)

def model_predict(model: nn.Module, tensor_input: torch.tensor) -> torch.tensor:
    """
    Do prediction for any pytorch Seq2Seq model.

    Parameters:
        model (nn.Module): Model to do prediction.
        tensor_input (torch.tensor): Input to model (sequence of numbers tensor of shape)

    Returns:
        torch.tensor: Model output result.
    """

    # Set model to evaluation mode
    model.eval()
    # Make tensor shape be [SEQ_LEN, 1]
    tensor_input = tensor_input.unsqueeze(1)
    # Create additional tensor that will be used as target but with model outputs instead of real data.
    y_input = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
    # Can model do inference in one shot (not token by token)
    one_shot = True
    
    with torch.no_grad():
        # Until maximum sentence length is not reached
        for _ in range(MAX_SENTENCE_SIZE):
            # Get predictions
            pred = model(tensor_input, y_input).to(device)
            # Get tokens with maximum probability
            _, token_id = torch.max(pred, axis=2)
            # Get last token
            next_token = token_id.view(-1)[-1].item()
            if next_token == EOS_IDX:
                # If end of sequence then stop and return response
                return token_id.view(-1) if one_shot else y_input.view(-1)
            # Increase artificial target tensor by model output
            next_tensor = torch.tensor([[next_token]])
            y_input = torch.cat((y_input, next_tensor), dim=0)
            # If end of sequence was not produce in first try, then model is token by token.
            one_shot = False
    # Return response if maximum sentence length exceeded
    return y_input.view(-1)

if __name__ == "__main__":
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of available models
    available_models = {
        "lstm": lstm.DetoxificationModel(),
        "ae_lstm": ae_lstm.DetoxificationModel(device),
        "transformer": transformer.DetoxificationModel(device),
        "T5": None,
    }
    
    # Read command line arguments
    parser = argparse.ArgumentParser(description="Do prediction for any model that are supported")
    parser.add_argument("model_type", choices=list(available_models.keys()))
    parser.add_argument("weights", type=str)
    parser.add_argument("file_path", type=str)
    parser.add_argument("--compare", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    # Create preprocessor with dataset vocabulary
    toxic_text_manager = ToxicTextManager(args.file_path, args.compare)

    # Load model with weights
    model = available_models[args.model_type]
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_PATH, args.weights), map_location=device))

    # Run prediction for each sentence in input
    result = []
    for i in range(len(toxic_text_manager)):
        # Get encoded text
        encoded_text = toxic_text_manager.get_encoded(i)
        # Predict detoxified text
        output = model_predict(model, encoded_text)
        # Add decoded result to result list
        result.append(toxic_text_manager.tensor2text(output))
        # Detokenize text
        result[-1] = " ".join(result[-1][1:])
        
    print(result)