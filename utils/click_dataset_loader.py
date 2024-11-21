import os
import re
import argparse

import pandas as pd
from datasets import load_dataset

from utils.utils import load_config

def main(args):
    configs = load_config(args.config_path)

    data_id = "EunsuKim/CLIcK"
    dataset = load_dataset(data_id)
    dataset = pd.DataFrame(dataset["train"])

    # formatting dataset
    records = []
    for _, row in dataset.iterrows():
        pattern = r'다음은 .+에 대한 문제이다\.\n'  ## remove label instruction text
        row["question"] = re.sub(pattern, "", row["question"])

        id = row["id"]
        paragraph = row["paragraph"] if len(row["paragraph"])!=0 else row["question"]
        question = row["question"]
        choices = row["choices"]
        answer = choices.index(row["answer"])+1  ## answer: str to int

        records.append({
            'id': id,
            'paragraph': paragraph,
            'question': question,
            'choices': choices,
            'answer': answer,
        })
    df = pd.DataFrame(records)

    df.to_csv(os.path.join(configs.data_dir, "click_dataset.csv"), index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="file path for conig.yaml"
    )
    args = parser.parse_args()

    main(args)
