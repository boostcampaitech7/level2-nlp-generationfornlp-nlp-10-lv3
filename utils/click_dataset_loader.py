import os
import re
import argparse

import pandas as pd
from datasets import load_dataset

from utils.utils import load_config

def srch_ptrn(pattern, text):  ## search for data that needs to be split using a search pattern
    return True if re.match(pattern, text) else False

def split_question(
        question,
        paragraph,
        pattern
):
    if question == paragraph:  ## for case the paragraph is empty
        return re.findall(pattern, question)[0], re.sub(pattern, "", question).strip()
    else:  ## not empty
        return re.findall(pattern, question)[0], paragraph + "\n" + re.sub(pattern, "", question).strip()

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
    ## type1. 다음 ~~ ? + paragraph
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
    ]

    num_of_cases = len(split_types)
    cnt = {f"Type_{i}": 0 for i in range(1, num_of_cases+1)}
    questions = []
    paragraphs = []
    for _, row in df.iterrows():
        for idx, split_type in enumerate(split_types):
            if any([row.id.startswith(x) for x in ["KITP", "PSAT"]]): ## exclude simple question type
                continue
            if any([srch_ptrn(search_pattern, row.question) for search_pattern in split_type["search_patterns"]]):
                question, paragraph = split_question(
                    question=row.question,
                    paragraph=row.paragraph,
                    pattern=split_type["pattern"]
                )
                questions.append(question)
                paragraphs.append(paragraph)
                cnt[f"Type_{idx+1}"] += 1

                break
        else:
            questions.append(row.question)
            paragraphs.append(row.paragraph)

    df["question"] = questions
    df["paragraph"] = paragraphs

    cnt["unsplitted"] = df.shape[0] - sum(cnt.values())
    cnt["total"] = df.shape[0]
    summary = pd.DataFrame.from_dict(cnt, orient='index', columns=["count"])
    print("**Summary of text split results**")
    print(summary)
    
    df.to_csv(os.path.join(configs.data_dir, "click_dataset.csv"), index=False)
    print(f"\nClick Dataset successfully saved at {configs.data_dir}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="file path for conig.yaml"
    )
    args = parser.parse_args()

    main(args)
