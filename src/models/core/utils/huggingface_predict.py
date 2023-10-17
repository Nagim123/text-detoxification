from ..architectures.t5_paranmt import DetoxificationModel
from .text_manager import TextManager

def predict_using_external_models(model: DetoxificationModel, toxic_text_manager: TextManager) -> list[str]:
    """
    Predict detoxified texts using models from hugging face.

    Parameters:
        model (DetoxificationModel): Model to do prediction.
        toxic_text_manager (TextManager): Input data container.
        
    Returns:
        list[str]: Detoxified texts.
    """
    
    result = []
    for i in range(len(toxic_text_manager)):
        # Feed forward each text through model
        result.append(model.forward(toxic_text_manager.get_raw(i)))
    return result

