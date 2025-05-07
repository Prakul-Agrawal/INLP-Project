# scripts/train.py
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm


class DebiasDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_enc = self.tokenizer(item['input'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        target_enc = self.tokenizer(item['target'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': input_enc['input_ids'].squeeze(),
            'attention_mask': input_enc['attention_mask'].squeeze(),
            'labels': target_enc['input_ids'].squeeze()
        }


def train():
    model_name = "google/flan-t5-small" # change to "google/flan-t5-base" for larger model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)

    # Dataset
    train_dataset = DebiasDataset("../data/train.json", tokenizer)
    test_dataset = DebiasDataset("../data/test.json", tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    num_epochs = 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(test_loader)
        perplexity = torch.exp(torch.tensor(avg_val_loss))

        print(f"\nEpoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}")

        # Save best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("../models", exist_ok=True)
            model.save_pretrained("../models/flan-t5-debias")
            tokenizer.save_pretrained("../models/flan-t5-debias")
            print("✅ Best model saved.")


if __name__ == "__main__":
    train()