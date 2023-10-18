from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class DetoxificationModel():
    def load_model(self) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained("s-nlp/t5-paranmt-detox")
        self.tokenizer = AutoTokenizer.from_pretrained("s-nlp/t5-paranmt-detox")
    
    def forward(self, text: str) -> str:
        tokenized_text = self.tokenizer(text, return_tensors="pt")["input_ids"]
        outputs = self.model.generate(tokenized_text)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
