from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    # Data paths
    raw_data_path: str = "data/data.json"
    processed_data_path: str = "data/processed_data.json"

    # Model paths
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    local_model_dir: str = "models/mistral-7b"
    output_dir: str = "models/finetuned_mistral"

    # Training hyperparameters
    # Updated for 10GB VRAM
    batch_size: int = 2  # Reduced from 4
    gradient_accumulation_steps: int = 8  # Increased to compensate
    use_4bit: bool = True  # New - enable 4-bit quantization
    bnb_4bit_quant_type: str = "nf4"  # New - 4-bit quantization type
    bnb_4bit_compute_dtype: str = "float16"  # New - computation dtype
    use_flash_attention: bool = False  # New - enable flash attention
    use_gradient_checkpointing: bool = True  # New - enable gradient checkpointing

    # Updated LoRA config
    lora_rank: int = 4  # Reduced from 8
    lora_alpha: int = 8  # Reduced from 16
    lora_target_modules: List[str] = ("q_proj", "v_proj")
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    max_seq_length: int = 512
    warmup_steps: int = 100
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500

    # Hardware
    fp16: bool = True
    bf16: bool = False
    use_gpu: bool = True
    multi_gpu: bool = False

    # Evaluation
    test_size: float = 0.2
    random_seed: int = 42


config = Config()