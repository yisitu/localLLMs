import torch
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2")

model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

if len(sys.argv) < 2:
    prompt = "Write a haiku about GPUs"
else:
    prompt = " ".join(sys.argv[1:])

messages = [
    {"role": "system", "content": "/think"},
    {"role": "user", "content": prompt},
]

tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    tokenized_chat,
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id
)
output = tokenizer.decode(outputs[0])
print(output)