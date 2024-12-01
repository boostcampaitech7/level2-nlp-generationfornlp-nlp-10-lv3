# 표준 라이브러리
import random
import yaml
import re

# 외부 라이브러리
import numpy as np
import pandas as pd
from box import Box
import torch


def load_config(config_file):
    # Load Config.yaml
    with open(config_file) as file:
        config = yaml.safe_load(file) # Dictionary
        config = Box(config) # . 

    return config


# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def extract_answer(text):
    if "정답:" in text:
        # "정답:" 뒤의 텍스트 추출
        answer_part = text.split("정답:")[1].strip()
        # 개행 문자나 "설명:" 등으로 끝나는 부분까지 제거
        answer = answer_part.split("\n")[0]
        return answer
    return ""


def split_question(
        question,
        paragraph,
        pattern
):
    if question == paragraph:  ## for case the paragraph is empty
        return re.findall(pattern, question)[0], re.sub(pattern, "", question).strip()
    else:  ## not empty
        return re.findall(pattern, question)[0], paragraph + "\n" + re.sub(pattern, "", question).strip()

def split_questions(df, split_types):
    num_of_cases = len(split_types)
    cnt = {f"Type_{i}": 0 for i in range(1, num_of_cases+1)}

    questions = []
    paragraphs = []
    for _, row in df.iterrows():
        for idx, split_type in enumerate(split_types):
            if any([row.id.startswith(x) for x in ["KIIP", "PSAT"]]): ## exclude simple question type
                continue
            if any([bool(re.match(search_pattern, row.question)) for search_pattern in split_type["search_patterns"]]):
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

    ## logging
    cnt["unsplitted"] = df.shape[0] - sum(cnt.values())
    cnt["total"] = df.shape[0]
    results = pd.DataFrame.from_dict(cnt, orient='index', columns=["count"])
    print("**Results for text split**")
    print(results)

    return df


def tag_indexing(data_id, tags):
    return any([data_id.startswith(tag) for tag in tags])


def tag_indexing_df(df, tags):
    return df["id"].apply(tag_indexing, tags=tags)