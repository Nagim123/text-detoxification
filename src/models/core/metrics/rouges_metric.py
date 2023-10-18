from torchmetrics.text.rouge import ROUGEScore

def calculate_rogues(original: str, translated: str):
    rogue = ROUGEScore(rouge_keys=("rouge1", "rouge2"))
    info = rogue(translated, original)
    for key in info:
        info[key] = info[key].item()
    return info

