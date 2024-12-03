# 표준 라이브러리
import argparse
import os
import random

# 외부 라이브러리
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel

# 로컬 모듈
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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        max_seq_length = 2048,
        dtype = torch.float16,
        load_in_4bit = True,
        device_map="auto",
        trust_remote_code=True,   
    )
    

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )        

    if configs.chat_template is not None :
        tokenizer.chat_template = configs.chat_template


    train_data = pd.read_csv(os.path.join(configs.data_dir, configs.train_path))
    eval_data = pd.read_csv(os.path.join(configs.data_dir, configs.val_path))

    train_dataset = BaseDataset(train_data, tokenizer, configs)
    eval_dataset = BaseDataset(eval_data, tokenizer, configs)

    model = BaseModel(configs, tokenizer, model)

    wandb.init(project=configs.project, 
               name=configs.sub_project,
               )
    model.train(train_dataset, eval_dataset)

    # val_outputs = model.eval(eval_dataset)
    # val_outputs.to_csv(os.path.join(configs.output_dir, configs.~~~), index = False)


if __name__ == "__main__":
    main()




