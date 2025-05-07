import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
from config import config
from data_processing import create_prompt

# Global variables to keep model in memory
_original_model = None
_original_tokenizer = None

def load_original_model():
    """Load the original model once and reuse it"""
    global _original_model, _original_tokenizer
    
    if _original_model is None:
        print("Loading original Mistral-7B model...")
        
        tokenizer = AutoTokenizer.from_pretrained("models/mistral-7b")
        tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True  # Critical for multi-turn
        )
        
        # Custom device map for better memory management
        device_map = {
            "model": 0,  # Main GPU
            "lm_head": 0,
            "embed_tokens": 0,
            "layers.0": 0,
            "layers.1": 0,
            # ... add more layers if needed ...
            "norm": "cpu",  # Offload normalization to CPU
            "final_layer_norm": "cpu"
        }
        
        _original_model = AutoModelForCausalLM.from_pretrained(
            "models/mistral-7b",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=device_map,
            offload_folder="offload",
            low_cpu_mem_usage=True
        )
        
        _original_tokenizer = tokenizer
        print("Original model loaded successfully")
    
    return _original_model, _original_tokenizer

def generate_response(question, biased_answer):
    """Generate response using cached original model"""
    model, tokenizer = load_original_model()
    
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
        return f"Generation Error: {str(e)}"

def clear_cuda_cache():
    """Helper to free GPU memory"""
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def interactive_test():
    """Interactive comparison with memory management"""
    from inference import generate_answer as generate_finetuned
    
    print("\n=== Model Comparison Tool ===")
    print("Type 'clear' to free memory or 'quit' to exit")
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            if question.lower() == 'quit':
                break
            if question.lower() == 'clear':
                clear_cuda_cache()
                continue
                
            biased_answer = input("Biased answer: ").strip()
            
            print("\nGenerating responses...")
            
            # Original model
            print("\n=== ORIGINAL ===")
            print(generate_response(question, biased_answer))
            
            # Finetuned model
            print("\n=== FINETUNED ===")
            print(generate_finetuned(question, biased_answer))
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    interactive_test()
