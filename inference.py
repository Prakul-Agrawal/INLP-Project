import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
import argparse
from config import config
from data_processing import create_prompt

# Global variables for loaded model
_loaded_model = None
_loaded_tokenizer = None

def load_model():
    """Load the model once and cache it globally"""
    global _loaded_model, _loaded_tokenizer
    
    if _loaded_model is None or _loaded_tokenizer is None:
        print("Loading model and tokenizer...")
        
        # Configure tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ) if config.use_4bit else None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.output_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Ensure all parameters are properly loaded
        for name, param in model.named_parameters():
            if param.is_meta:
                original_param = next(
                    p for n, p in model.named_parameters() 
                    if n == name.replace(".weight", "").replace(".bias", "")
                )
                param.data = torch.empty_like(
                    original_param,
                    device=model.device,
                    dtype=original_param.dtype
                )
        
        _loaded_model = model
        _loaded_tokenizer = tokenizer
        print("Model loaded successfully")
    
    return _loaded_model, _loaded_tokenizer

def generate_answer(question: str, biased_answer: str) -> str:
    """Generate answer using the pre-loaded model"""
    model, tokenizer = load_model()
    
    try:
        prompt = create_prompt(question, biased_answer)
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=config.max_seq_length
        ).to(model.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output.split("<Unbiased Answer:>")[-1].strip()
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Generate unbiased answers from biased Q&A pairs")
    parser.add_argument("--question", type=str, required=True, help="The question to answer")
    parser.add_argument("--biased_answer", type=str, required=True, help="The biased answer to correct")
    
    args = parser.parse_args()
    
    print("\nGenerating unbiased answer...")
    unbiased_answer = generate_answer(args.question, args.biased_answer)
    
    print("\n=== Results ===")
    print(f"Question: {args.question}")
    print(f"Biased Answer: {args.biased_answer}")
    print(f"\nUnbiased Answer: {unbiased_answer}")

if __name__ == "__main__":
    main()
