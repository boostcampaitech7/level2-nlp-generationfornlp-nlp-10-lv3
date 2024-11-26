import os
import re
import argparse
from collections import Counter

import pandas as pd
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
    DATA_DIR = os.path.join(BASE_DIR, "../../data", DATA_VER)
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    
    ## prompt template loading
    PROMPT_DIR = os.path.join(BASE_DIR, "chat_template.yaml")
    with open(PROMPT_DIR, 'r') as f:
        template = f.read()
    template = [(message["role"], message["content"]) for message in template["messages"]]

    ## model loading
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    ## pipline constructing
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device=0,
        batch_size=1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    llm = ChatHuggingFace(llm=llm)

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm.bind(stop=[r"\n"]) | StrOutputParser()

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
            answer = answer.split("\n").pop()
            rep_out.append(answer) ## parsing하고 넣어야 함

        majority = Counter(rep_out)
        majority = list(sorted(majority.item(), key=lambda x: x[1]))
        majority = majority.pop()[0]
        outputs.append(majority)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-s",
        "--seed",
        default=456,
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