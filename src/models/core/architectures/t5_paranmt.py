from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class DetoxificationModel():
    def __init__(self) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained()
        self.tokenizer = AutoTokenizer.from_pretrained()
    
    def forward(self, text: str) -> str:
        tokenized_text = self.tokenizer(text)["input_ids"]
        outputs = self.model.generate(tokenized_text)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def predict_for_batch(self, texts: list[str]) -> list[str]:
        result = []
        for text in texts:
            result.append(self.forward(text))
        return result