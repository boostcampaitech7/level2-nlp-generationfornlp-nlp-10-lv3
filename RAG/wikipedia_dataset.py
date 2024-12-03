import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_api_key = os.getenv('HF_API_KEY')
wandb_api_key = os.getenv('WANDB_API_KEY')

from datasets import load_dataset



# 데이터셋 로드
dataset = load_dataset("Cohere/wikipedia-22-12-ko-embeddings", split="train")

# Embeddings
print(dataset["title"],dataset["text"])