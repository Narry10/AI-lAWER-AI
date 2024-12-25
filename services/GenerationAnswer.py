import re
import torch
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast


def format_answer(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'(\s+(\d+|[a-zA-Z])[\.\):])', r'\n\1', text)

    text = text.strip()

    return text


class GenerationAnswer:
    def __init__(self, model_name="Captone2C2SE13/legal-support"):
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

        self.pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

    def get_answer(self, question):
        answer = self.pipe(question)[0]['generated_text']
        return format_answer(answer)


if __name__ == "__main__":
    from datetime import datetime
    qa_model = GenerationAnswer()
    question = ""
    start_time = datetime.now()
    answer = qa_model.get_answer(question)
    end_time = datetime.now()
    print(answer)
    duration = end_time - start_time
    print("\nTime execution:", duration)
