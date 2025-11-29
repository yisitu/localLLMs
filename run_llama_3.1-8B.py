import transformers
import torch
import sys

#from huggingface_hub import login
#login(token='paste token here')

model_id = "meta-llama/Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

if len(sys.argv) < 2:
    prompt = "Hey how are you doing today?"
else:
    prompt = " ".join(sys.argv[1:])

output = pipeline(prompt)
print(output)
