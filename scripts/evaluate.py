# scripts/evaluate.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from train import DebiasDataset
from tqdm import tqdm
import numpy as np

def evaluate():
    model_path = "../models/flan-t5-debias"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    dataset = DebiasDataset("../data/test.json", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.item())

    avg_loss = np.mean(losses)
    perplexity = torch.exp(torch.tensor(avg_loss))

    print(f"✅ Evaluation complete. Val Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    evaluate()