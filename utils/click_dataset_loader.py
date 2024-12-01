import os
import re
import argparse

import pandas as pd
from datasets import load_dataset

from utils import load_config, split_questions


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

    # clean
    ## delete data
    _id = ["KHB_66_22", "CSAT_korean_17_4"]
    idx = df["id"].apply(lambda x: x in _id)
    df.drop(index=df[idx].index, inplace=True)

    ## remove duplicated texdt
    idx = (df["id"] == "Kedu_tradition_12")
    df.loc[idx, "question"] = "한국의 전통 식문화 중 반상차림에 관한 설명으로 옳지 않은 것은?"
    df.loc[idx, "paragraph"] = "한국의 전통 식문화 중 반상차림에 관한 설명으로 옳지 않은 것은?"

    # split
    split_types = [
        {
            "search_patterns": [r"^다음.+\?.+", r"^다음.+\?\n"],
            "pattern": r"^다음.+\?"
        },
        {
            "search_patterns": [r"^다음.+고르십시오\..+", r"^다음.+고르십시오\.\n"],
            "pattern": r"^다음.+고르십시오\."
        },
        {
            "search_patterns": [r"^다음.+답하십시오\..+", r"^다음.+답하십시오\.\n"],
            "pattern": r"^다음.+답하십시오\."
        },
        {
            "search_patterns": [r".+것은\?.+", r".+것은\?\n"],
            "pattern": r".+것은\?"
        },
    ]
    df = split_questions(df, split_types)

    # group the data by id type / group = ["train", "augment", "remove"]
    ## ID tags for each training, augmentation
    tags_train = ["Kedu_politics_1", "CSAT", "KHB", "TK"] ## + PSE(only paragraph != question)
    tags_augment = ["KIIP_economy", "KIIP_geography", "KIIP_tradition",
                   "KIIP_law", "Kedu_economy", "Kedu_tradition", "PSAT"] ## + Kedu, Kedu_society(only paragraph == question)

    ## index extraction
    idx_train_1 = df["id"].apply(lambda x: any([x.startswith(tag) for tag in tags_train]))
    idx_train_2 = (
        df["id"].apply(
            lambda x: any([x.startswith(tag) for tag in ["Kedu_society", "Kedu_history", "PSE"]])
        )
    ) & (df["question"] == df["paragraph"]) ## PSE, Kedu_society, Kedu_history(only paragraph != question)
    idx_train_3 = (
        df["id"].apply(
            lambda x: bool(re.match(r"^Kedu_[0-9]+", x))
        )
    ) & (df["question"] != df["paragraph"]) ## Kedu(only paragraph == question)
    idx_train = idx_train_1 | idx_train_2 | idx_train_3

    idx_augment_1 = df["id"].apply(lambda x: any([x.startswith(tag) for tag in tags_augment]))
    idx_augment_2 = (
        df["id"].apply(
            lambda x: any([x.startswith(tag) for tag in ["Kedu_society", "Kedu_history", "PSE"]])
        )
    ) & (df["question"] == df["paragraph"]) ## PSE, Kedu_society, Kedu_history(only paragraph != question)
    idx_augment_3 = (
        df["id"].apply(
            lambda x: bool(re.match(r"^Kedu_[0-9]+", x))
        )
    ) & (df["question"] == df["paragraph"]) ## Kedu(only paragraph == question)
    idx_augment = idx_augment_1 | idx_augment_2 | idx_augment_3

    ## slicing
    click_for_train = df.loc[idx_train]
    click_for_augment = df.loc[idx_augment]
    cnt = {"For Training": click_for_train.shape[0], "For Augmentation": click_for_augment.shape[0]}
    cnt["Removed"] = df.shape[0] - sum(cnt.values())
    cnt["Total"] = df.shape[0]
    results = pd.DataFrame.from_dict(cnt, orient='index', columns=["count"])
    print("\n**Results for grouping by id type**")
    print(results)
    
    ## saving
    click_for_train.to_csv(os.path.join(configs.data_dir, "click_for_train.csv"), index=False)
    click_for_augment.to_csv(os.path.join(configs.data_dir, "click_for_augment.csv"), index=False)
    print(f"\nClick Datasets(for training + for augmentation) successfully saved at {configs.data_dir}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="file path for conig.yaml"
    )
    args = parser.parse_args()

    main(args)
