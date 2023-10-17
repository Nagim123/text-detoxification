from ..architectures.t5_paranmt import DetoxificationModel
from .text_manager import TextManager

def predict_using_external_models(model: DetoxificationModel, toxic_text_manager: TextManager) -> list[str]:
    result = []
    for i in range(len(toxic_text_manager)):
        result.append(model.forward(toxic_text_manager.get_raw(i)))
    return result

