import os
import re
import yaml
import argparse
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from box import Box

from data_loader.datasets import ReasoningDataset


def main(arg):
    BASE_DIR = os.getcwd()
    CONFIG_DIR = "Reasoning"
    ## set seed 빠졌다
    ## utils에 load_config 있으므로 반영
    with open(os.path.join(BASE_DIR, CONFIG_DIR), 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
    configs = Box(configs)

    MODEL_DIR = os.path.join("saved", "models")
    model_path = os.path.join(BASE_DIR, MODEL_DIR, arg.checkpoint_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)

    DATA_DIR = "../../data"
    FILE_NAME = "output.csv" if arg.inference else "reasoning_valid.csv"
    eval_data = pd.read_csv(os.path.join(BASE_DIR, DATA_DIR, FILE_NAME))
    eval_dataset = ReasoningDataset(
        data=eval_data,
        tokenizer=tokenizer,
        configs=configs,
        do_train=~arg.inference
    )

    outputs = []
    cnt = 0
    for row in tqdm(eval_dataset):
        answer = model.generate(
            torch.tensor(row["input_ids"], device="cuda").unsqueeze(0),
            max_new_tokens=configs.max_new_tokens
        )
        answer = tokenizer.decode(answer[0].tolist())

        ## answer extraction in prompt
        answer = answer.split("[|assistant|]").pop()
        answer = re.findall(r"정답\s*:\s*[1-5]", answer)
        if answer:
            answer = answer[0][-1]
        else:
            cnt += 1
            answer = 3 ## 정답 못 찾으면 3번으로 찍기
        outputs.append(int(answer))
    
    print(f"There are {cnt}-rows that model doesn't know the answer")

    if arg.inference:
        eval_data["answer"] = outputs
        eval_data.to_csv(os.path.join(BASE_DIR, DATA_DIR, FILE_NAME), index=False)
        print("Submission Updated!")
    else:
        eval_data["pred"] = outputs
        print(f"This model has an accuracy of {sum(eval_data["pred"] == eval_data["answer"])/eval_data.shape[0]:.4f}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-i",
        "--inference",
        default=False,
        type=bool,
        help="to inference or not (default: False)",
    )
    args.add_argument(
        "-c",
        "--checkpoint_path",
        default=None,
        type=str,
        help="model checkpoint path (default: None)",
    )

    arg = args.parse_args()
    main(arg)