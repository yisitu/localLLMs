from vllm import LLM, SamplingParams
import sys

# 1. Configure the model with Tensor Parallelism
# tensor_parallel_size=8 ensures all 8 GPUs work on the single input simultaneously.
llm = LLM(
    model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    tensor_parallel_size=8,  # <--- This is the specific fix for your hardware setup
    trust_remote_code=True,
    dtype="bfloat16",
    # gpu_memory_utilization=0.9, # Adjust if you hit OOM, though unlikely with 9B on 8 GPUs
)

if len(sys.argv) < 2:
    prompt = "Write a haiku about GPUs"
else:
    prompt = " ".join(sys.argv[1:])

# 2. Define Sampling Parameters
# Corresponds to your max_new_tokens=512
sampling_params = SamplingParams(
    max_tokens=512,
    temperature=0.0, # Greedy decoding (deterministic)
)

# 3. Apply Template (Manual or via vLLM's chat method if supported, manual is safer for control)
# vLLM expects raw prompt strings usually, so we format the string first.
messages = [
    {"role": "system", "content": "/think"},
    {"role": "user", "content": prompt},
]

# We use the tokenizer to format the string, but pass the *string* to vLLM
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Generate
# This runs the C++ optimized runtime. No Python loop overhead.
outputs = llm.generate([prompt_text], sampling_params)

# 5. Output
generated_text = outputs[0].outputs[0].text
print(generated_text)
