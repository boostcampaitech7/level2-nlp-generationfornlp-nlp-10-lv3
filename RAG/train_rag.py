# 표준 라이브러리
import argparse
import os
import random

# 외부 라이브러리
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 로컬 모듈
from data_loader.datasets import BaseDataset
from data_loader.rag_datasets import RAGDataset
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
    
    model = AutoModelForCausalLM.from_pretrained(
        configs.train_model_path_or_name,
        trust_remote_code = True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Train Model Name :", configs.train_model_path_or_name) # Check Configs model name

    tokenizer = AutoTokenizer.from_pretrained(
        configs.train_model_path_or_name,
        trust_remote_code=True,
    )

    if configs.chat_template is not None :
        tokenizer.chat_template = configs.chat_template

    train_data = pd.read_csv(os.path.join(configs.data_dir, configs.train_path))
    eval_data = pd.read_csv(os.path.join(configs.data_dir, configs.val_path))

    train_dataset = RAGDataset(train_data, tokenizer, configs)
    eval_dataset = RAGDataset(eval_data, tokenizer, configs)

    model = BaseModel(configs, tokenizer, model)

    wandb.init(project=configs.project, 
               name=configs.sub_project,
               )
    model.train(train_dataset, eval_dataset)

    # val_outputs = model.eval(eval_dataset)
    # val_outputs.to_csv(os.path.join(configs.output_dir, configs.~~~), index = False)


if __name__ == "__main__":
    main()




