from torchmetrics import TranslationEditRate

def calculate_ter(original: str, translated: str):
    ter = TranslationEditRate()
    return ter(translated, [original]).item()