# 표준 라이브러리
import argparse
import os

# 외부 라이브러리
import pandas as pd
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch
# 로컬 모듈
from data_loader.datasets import BaseDataset
from models.base_model import BaseModel
from utils.utils import load_config, set_seed
from unsloth import FastLanguageModel


def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        help="")

    args = parser.parse_args() 
    print("Config Path :",args.config_path) # Check Config path
    
    configs = load_config(args.config_path)
    set_seed(configs.seed) 

    # test_model_path_or_name = os.path.join("./saved/models", configs.test_model_path_or_name)
    # print("Inference Model Name :", configs.test_model_path_or_name) # Check Configs model name

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        max_seq_length = 8192,
        dtype = torch.float16,
        load_in_4bit = True,
        device_map="auto",
        trust_remote_code=True,   
    )
    

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["k_proj","v_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 42,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )        


    if configs.chat_template is not None :
        tokenizer.chat_template = configs.chat_template

    test_data = pd.read_csv(os.path.join(configs.data_dir, 'test.csv'))

    test_dataset = BaseDataset(test_data, tokenizer, configs, False)
    
    model = BaseModel(configs, tokenizer, model=model)

    # outputs, decoder_output = model.inference(test_dataset)
    generate_output = model.inference_generate(test_dataset)

    os.makedirs("./saved/outputs", exist_ok=True)
    # pd.DataFrame(outputs).to_csv(os.path.join("./saved/outputs", configs.output_file), index=False)
    pd.DataFrame(generate_output).to_csv(os.path.join("./saved/outputs", "qwen32b_generation_output.csv"), index=False)


if __name__ == "__main__":
    main()