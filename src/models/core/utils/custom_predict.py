import torch
from torch import nn
from .text_manager import TextManager
from ..utils.constants import BOS_IDX, EOS_IDX, MAX_SENTENCE_SIZE

def model_predict(model: nn.Module, tensor_input: torch.tensor, device: str) -> torch.tensor:
    """
    Do prediction for any pytorch Seq2Seq model.

    Parameters:
        model (nn.Module): Model to do prediction.
        tensor_input (torch.tensor): Input to model (sequence of numbers tensor of shape)
        device (str): Device where prediction will be executed.

    Returns:
        torch.tensor: Model output result.
    """

    # Set model to evaluation mode
    model.eval()
    # Make tensor shape be [SEQ_LEN, 1]
    tensor_input = tensor_input.unsqueeze(1).to(device)
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
            next_tensor = torch.tensor([[next_token]], device=device)
            y_input = torch.cat((y_input, next_tensor), dim=0)
            # If end of sequence was not produce in first try, then model is token by token.
            one_shot = False
    # Return response if maximum sentence length exceeded
    return y_input.view(-1).detach().cpu()

def predict_using_custom_models(model: nn.Module, toxic_text_manager: TextManager, device: str) -> list[str]:
    """
    Predict detoxified texts using models implemented by me.

    Parameters:
        model (nn.Module): Model to do prediction.
        toxic_text_manager (TextManager): Input data container.
        device (str): Device where prediction will be executed.

    Returns:
        list[str]: Detoxified texts.
    """
    # Run prediction for each sentence in input
    result = []
    for i in range(len(toxic_text_manager)):
        # Get encoded text
        encoded_text = toxic_text_manager.get_encoded(i)
        # Predict detoxified text
        output = model_predict(model, encoded_text, device)
        # Add decoded result to result list
        result.append(toxic_text_manager.tensor2text(output))
        # Detokenize text
        result[-1] = " ".join(result[-1][1:])
    return result