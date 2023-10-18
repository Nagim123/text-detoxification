from .text_manager import TextManager
from ..metrics import rouges_metric, ter_metric, bleu_metric

def generate_metric_evalulation(predictions: list[str], text_manager: TextManager):

    report_data = []
    for i in range(len(text_manager)):
        target, original = text_manager.get_raw_detoxified(i), text_manager.get_raw(i)
        prediction = predictions[i]

        evaluation_info = {
            "generated": prediction,
            "original": original,
            "true_translated": target,
            "BLEU": bleu_metric.calculate_bleu(target, prediction),
            "TER score": ter_metric.calculate_ter(target, prediction),
            "ROUGES": rouges_metric.calculate_rogues(target, prediction)
        }

        report_data.append(evaluation_info)
    return report_data
