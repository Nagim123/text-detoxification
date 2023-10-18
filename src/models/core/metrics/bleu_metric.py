from torchmetrics.text import BLEUScore

def calculate_bleu(original: str, translated: str):
    bleu = BLEUScore(smooth=True)
    return bleu([translated], [[original]]).item()