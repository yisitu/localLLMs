import torch
import sys
from transformers import pipeline, BitsAndBytesConfig

# 1. Configure 4-bit Quantization (Reduces VRAM usage & memory bandwidth bottleneck)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "meta-llama/Llama-3.1-8B"

# 2. Initialize Pipeline with Optimizations
pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "quantization_config": quant_config,
        "attn_implementation": "flash_attention_2"  # <--- Massive speedup for Llama 3
    },
    device_map="auto"
)

# 3. Handle Input
if len(sys.argv) < 2:
    prompt = "Hey how are you doing today?"
else:
    prompt = " ".join(sys.argv[1:])

# 4. Generate
# Setting pad_token_id is best practice for Llama 3 to avoid warnings
output = pipeline(
    prompt,
    max_new_tokens=128,
    pad_token_id=pipeline.tokenizer.eos_token_id
)

print(output[0]['generated_text'])
