# 표준 라이브러리
import argparse
import os

# 외부 라이브러리
import pandas as pd
from transformers import AutoTokenizer

# 로컬 모듈
from data_loader.datasets import BaseDataset
from models.base_model import BaseModel
from utils.utils import load_config, set_seed


def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        help="")

    args = parser.parse_args() 

    configs = load_config(args.config_path)

    set_seed(configs.seed) 

    test_model_path_or_name = os.path.join("./saved/models", configs.test_model_path_or_name)
    tokenizer = AutoTokenizer.from_pretrained(
        test_model_path_or_name,
        trust_remote_code=True,
    )

    if configs.chat_template is not None :
        tokenizer.chat_template = configs.chat_template

    test_data = pd.read_csv(os.path.join(configs.data_dir, 'test.csv'))

    test_dataset = BaseDataset(test_data, tokenizer, configs, False)
    
    model = BaseModel(configs, tokenizer)

    outputs = model.inference(test_dataset)

    os.makedirs("./saved/outputs", exist_ok=True)
    pd.DataFrame(outputs).to_csv(os.path.join("./saved/outputs", configs.output_file), index=False)


if __name__ == "__main__":
    main()