# scripts/infer.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys

def infer(question, biased_answer):
    model_path = "../models/flan-t5-debias"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_text = f"<Question:> {question} <Answer:> {biased_answer}"
    enc = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**enc, max_length=100)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n✅ Unbiased Answer: {decoded}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python infer.py '<question>' '<biased_answer>'")
        exit(1)

    question = sys.argv[1]
    biased_answer = sys.argv[2]

    infer(question, biased_answer)
