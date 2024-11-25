# 표준 라이브러리
import os
import argparse

# 외부 라이브러리
from datasets import load_from_disk
from dotenv import load_dotenv
from huggingface_hub import login
import wandb
from transformers import AutoTokenizer

# 로컬 모듈
from utils.utils import load_config, set_seed
from data_loader.datasets import FineTuningDataset
from models.fine_tuning_model import FineTuningModel


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

    configs = load_config(args.config_path)

    set_seed(configs.seed)

    wandb.init(project=configs.project, 
               name=configs.sub_project,
               )

    tokenizer = AutoTokenizer.from_pretrained(
        configs.ft_model_path_or_name,
        trust_remote_code = True,
    )

    dataset = load_from_disk(configs.data_path)
    train_dataset = FineTuningDataset(dataset['train'], tokenizer, configs)
    val_dataset = FineTuningDataset(dataset['validation'], tokenizer, configs, True)
    test_dataset = FineTuningDataset(dataset['test'], tokenizer, configs, True)

    model = FineTuningModel(configs, tokenizer)

    if configs.do_train:
        model.train(train_dataset, val_dataset)
    else:
        outputs, metrics = model.inference(test_dataset) 

        outputs.to_csv(configs.output_path, index = False)
        wandb.log(metrics)

    wandb.finish()


if __name__ == "__main__":
    main()