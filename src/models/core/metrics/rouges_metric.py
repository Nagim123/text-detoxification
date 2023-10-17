from torchmetrics.text.rouge import ROUGEScore

def calculate_rogues(original: str, translated: str):
    rogue = ROUGEScore(rouge_keys=("rouge1"))