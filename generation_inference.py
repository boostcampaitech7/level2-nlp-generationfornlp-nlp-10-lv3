import os
import re
import argparse
from collections import Counter

import yaml
import pandas as pd
from ast import literal_eval
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.utils import set_seed

def main(arg):
    ## parameters
    SEED = arg.seed
    MODEL_ID = arg.model_id
    DATA_VER = arg.data_ver
    K = arg.k

    ## random seeding
    set_seed(SEED)

    ## data loading
    BASE_DIR = os.getcwd()
    BASE_TO_DATA = os.path.join("..", "..", "data")
    DATA_DIR = os.path.join(BASE_DIR, BASE_TO_DATA, DATA_VER)
    df = pd.read_csv(os.path.join(DATA_DIR, "train_v0.1.6.csv"))

    records = []
    for _, row in df.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems['question'],
            'choices': problems['choices'],
            'answer': problems.get('answer', None),
            "question_plus": problems.get('question_plus', None),
        }
        # Include 'question_plus' if it exists
        if 'question_plus' in problems:
            record['question_plus'] = problems['question_plus']
        records.append(record)
    
    df = pd.DataFrame(records)
    df['question_plus'] = df['question_plus'].fillna('')
    df['full_question'] = df.apply(lambda x: x['question'] + ' ' + x['question_plus'] if x['question_plus'] else x['question'], axis=1)
    
    ## prompt template loading
    PROMPT_DIR = os.path.join(BASE_DIR, "configs", "chat_template.yaml")
    with open(PROMPT_DIR, 'r') as f:
        template = yaml.load(f, Loader=yaml.SafeLoader)
    template = [tup for message in template for tup in message.items()]

    ## model loading
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

    ## pipline constructing
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        device=0,
        batch_size=1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    llm = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_messages(template)
    chain = prompt | llm | StrOutputParser()

    ## inference
    outputs = []
    rep_out = []
    for _, row in tqdm(df.iterrows()):
        for _ in range(K):
            answer = chain.invoke(
                {
                    "paragraph": row.paragraph,
                    "question": row.question,
                    "choices":  row.choices,
                }
            )

            ## answer extraction in prompt
            answer = answer.split("[|assistant|]").pop()
            answer = re.findall(r"정답\s*:\s*[1-5]", answer)
            if answer:
                answer = answer[0][-1]
            else:
                answer = 3 ## 정답 못 찾으면 3번으로 찍기
            rep_out.append(answer)

        majority = Counter(rep_out)
        majority = list(sorted(majority.item(), key=lambda x: x[1]))
        majority = majority.pop()[0]
        outputs.append(majority)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-s",
        "--seed",
        default=42,
        type=int,
        help="setting random seed (default: 456)",
    )
    args.add_argument(
        "-m",
        "--model_id",
        default=None,
        type=str,
        help="hugging face model id (default: None)",
    )
    args.add_argument(
        "-v",
        "--data_ver",
        default=1.0,
        type=str,
        help="dataset version (default: 1.0)",
    )
    args.add_argument(
        "-k",
        "--k",
        default=5,
        type=int,
        help="the number of repeatition for self-consistency prompting (default: 5)",
    )

    arg = args.parse_args()
    main(arg)