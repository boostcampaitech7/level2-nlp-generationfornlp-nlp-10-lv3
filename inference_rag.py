# 표준 라이브러리
import argparse
import os

# 외부 라이브러리
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
# 로컬 모듈
from data_loader.datasets import BaseDataset
from data_loader.rag_datasets import RAGDataset
from models.base_model import BaseModel
from utils.utils import load_config, set_seed

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_api_key = os.getenv('HF_API_KEY')
wandb_api_key = os.getenv('WANDB_API_KEY')

login(hf_api_key)

def post_processing(outputs): # dict
    post_outputs = []
    for output in outputs:
        output["target"] = output["target"].split()[-1]
        post_outputs.append(output)

    return post_outputs

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        help="")

    args = parser.parse_args() 
    print("Config Path :",args.config_path) # Check Config path
    
    configs = load_config(args.config_path)
    set_seed(configs.seed) 

    test_model_path_or_name = os.path.join("./saved/models", configs.test_model_path_or_name)
    print("Inference Model Name :", test_model_path_or_name) # Check Configs model name

    model = AutoPeftModelForCausalLM.from_pretrained(
        test_model_path_or_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     configs.test_model_path_or_name,
    #     trust_remote_code=True,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        test_model_path_or_name,
        trust_remote_code=True,
    )

    if configs.chat_template is not None :
        tokenizer.chat_template = configs.chat_template

    test_data = pd.read_csv(os.path.join("./data", 'test.csv'))

    test_dataset = RAGDataset(test_data, tokenizer, configs, False)

    model = BaseModel(configs, tokenizer, model)

    outputs, decoder_output = model.inference(test_dataset)
    # decoder_output = model.inference_generate(test_dataset)
    # post_output = post_processing(decoder_output)

    os.makedirs("./saved/outputs", exist_ok=True)
    pd.DataFrame(outputs).to_csv(os.path.join("./saved/outputs", configs.output_file), index=False)
    pd.DataFrame(decoder_output).to_csv(os.path.join("./saved/outputs", "exaone_output.csv"), index=False)


if __name__ == "__main__":
    main()