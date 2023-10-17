from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class DetoxificationModel():
    def load_model(self, model_path: str) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def forward(self, text: str) -> str:
        tokenized_text = self.tokenizer(text, return_tensors="pt")["input_ids"]
        outputs = self.model.generate(tokenized_text)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
