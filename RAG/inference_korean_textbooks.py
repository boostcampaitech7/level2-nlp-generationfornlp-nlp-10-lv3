import os 
import argparse

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

from data_loader.datasets import BaseDataset
from models.base_model import BaseModel
from utils.utils import load_config, set_seed

from dotenv import load_dotenv
from huggingface_hub import login
import wandb

load_dotenv()
hf_api_key = os.getenv('HF_API_KEY')
wandb_api_key = os.getenv('WANDB_API_KEY')

login(hf_api_key)
wandb.login(key=wandb_api_key)

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        help="")

    args = parser.parse_args() 
    print("Config Path :",args.config_path) # Check Config path
    
    configs = load_config(args.config_path)
    set_seed(configs.seed) 

    tokenizer = AutoTokenizer.from_pretrained(
        configs.test_model_path_or_name,
        trust_remote_code = True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        configs.test_model_path_or_name,
        trust_remote_code = True,
        device_map = 'auto',
        torch_dtype=torch.float16,
    )

    rag_model = SentenceTransformer(configs.rag_model_path_or_name, device='cuda')

    test_data = pd.read_csv(os.path.join(configs.data_dir, "test.csv"))# .iloc[:10,:]
    test_dataset = BaseDataset(test_data, tokenizer, configs, 
                               False, True, rag_model)
    
    model = BaseModel(configs, tokenizer, model)

    outputs = model.inference_pipeline(test_dataset)

    os.makedirs("./saved/outputs", exist_ok=True)
    pd.DataFrame(outputs).to_csv(os.path.join("./saved/outputs", configs.output_file), index=False)


if __name__=="__main__":
    main()
