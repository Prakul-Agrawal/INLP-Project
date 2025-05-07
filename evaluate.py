import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
from config import config
import json
from tqdm import tqdm
import numpy as np
from data_processing import create_prompt

def calculate_perplexity(model, tokenizer, texts):
    """Calculate perplexity for a list of texts"""
    perplexities = []
    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
    return np.mean(perplexities)

def load_model():
    """Properly load the model with all parameters"""
    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization (if used during training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load model with explicit device map
    model = AutoModelForCausalLM.from_pretrained(
        config.output_dir,
        quantization_config=bnb_config if config.use_4bit else None,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Ensure all parameters are properly loaded
    for name, param in model.named_parameters():
        if param.is_meta:
            # Get the original parameter to copy metadata
            original_param = next(
                p for n, p in model.named_parameters() 
                if n == name.replace(".weight", "").replace(".bias", "")
            )
            # Create new tensor with correct specs
            param.data = torch.empty_like(
                original_param,
                device=model.device,
                dtype=original_param.dtype
            )
    
    return model, tokenizer

def generate_answer(model, tokenizer, prompt):
    """Manual generation without pipeline"""
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=config.max_seq_length
    ).to(model.device)
    
    generation_config = GenerationConfig(
        max_new_tokens=256,  # Reduced from max_seq_length for safety
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model():
    """Evaluate the trained model on test set"""
    # Load model and tokenizer
    model, tokenizer = load_model()
    print("Model and tokenizer loaded successfully")
    
    # Load test data
    with open(config.processed_data_path, 'r') as f:
        data = json.load(f)
    test_data = data["test"]
    print(f"Loaded {len(test_data)} test samples")
    
    # Generate unbiased answers
    generated_answers = []
    true_answers = []
    
    for i, item in enumerate(tqdm(test_data, desc="Generating answers")):
        prompt = create_prompt(item["question"], item["biased_answer"])
        
        try:
            generated_text = generate_answer(model, tokenizer, prompt)
            unbiased_answer = generated_text.split("<Unbiased Answer:>")[-1].strip()
            
            generated_answers.append(unbiased_answer)
            true_answers.append(item["unbiased_answer"])
            
            # Print first few samples as we go
            if i < 3:
                print(f"\nSample {i+1}:")
                print(f"Question: {item['question']}")
                print(f"Biased Answer: {item['biased_answer']}")
                print(f"Generated Answer: {unbiased_answer}")
                print(f"True Answer: {item['unbiased_answer']}")
                
        except Exception as e:
            print(f"\nError processing sample {i}: {str(e)}")
            continue
    
    # Calculate perplexity if we have successful generations
    if generated_answers:
        print("\nCalculating perplexity metrics...")
        gen_perplexity = calculate_perplexity(model, tokenizer, generated_answers)
        true_perplexity = calculate_perplexity(model, tokenizer, true_answers)
        
        print(f"\nEvaluation Results:")
        print(f"Generated answers perplexity: {gen_perplexity:.2f}")
        print(f"True unbiased answers perplexity: {true_perplexity:.2f}")
    else:
        print("\nNo successful generations to evaluate")

if __name__ == "__main__":
    evaluate_model()
