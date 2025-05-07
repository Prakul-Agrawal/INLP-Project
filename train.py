import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from config import config
from data_processing import create_prompt
import json
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Enable TF32 for better performance (add these right after imports)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"Enabled TF32: matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}")

def load_datasets():
    """Load processed datasets"""
    with open(config.processed_data_path, 'r') as f:
        data = json.load(f)

    train_data = data["train"]
    test_data = data["test"]

    # Convert to HF dataset format
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    return train_dataset, test_dataset


def tokenize_function(examples):
    """Tokenize the input data"""
    inputs = [create_prompt(q, ba) for q, ba in zip(examples["question"], examples["biased_answer"])]
    targets = examples["unbiased_answer"]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=config.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=config.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    # Set labels (we want to predict the unbiased answer)
    model_inputs["labels"] = labels
    return model_inputs


def setup_training():
    """Setup model without flash attention"""
    global tokenizer

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=False,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.local_model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model WITHOUT flash attention
    model = AutoModelForCausalLM.from_pretrained(
        config.local_model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
    )

    # Prepare for PEFT/LoRA
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.use_gradient_checkpointing
    )

    # More efficient LoRA config
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules  # Only specified modules
    )
    model = get_peft_model(model, peft_config)

    # Enable gradient checkpointing if needed
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Enabled gradient checkpointing with reentrant=False")

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load and tokenize datasets
    train_dataset, test_dataset = load_datasets()

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=test_dataset.column_names
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        weight_decay=0.01,
        warmup_steps=config.warmup_steps,
        dataloader_pin_memory=False,  # Saves ~0.5GB memory
        dataloader_num_workers=2,  # Optimal for most systems
        dataloader_prefetch_factor=2,  # Helps with throughput
        fp16=config.fp16,
        bf16=config.bf16,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )

    return trainer


def main():
    # Setup training
    trainer = setup_training()

    # Start training
    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(config.output_dir)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()