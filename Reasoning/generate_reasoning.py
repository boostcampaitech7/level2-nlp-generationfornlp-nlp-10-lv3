import os
import yaml
import json
import time
import argparse
from dotenv import load_dotenv

from box import Box
from ast import literal_eval
import pandas as pd
from sklearn.model_selection import train_test_split
from openai import OpenAI


def refactor_data(dataset):
    records = []
    for _, row in dataset.iterrows():
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

    return df


def main(arg):

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, arg.data_dir)
    df = pd.read_csv(DATA_DIR)
    df = refactor_data(df)

    PROMPT_DIR = os.path.join(BASE_DIR, "Reasoning", "prompts.yaml")
    with open(PROMPT_DIR, 'r') as f:
        prompts = yaml.load(f, Loader=yaml.SafeLoader)

    sys_prompts = prompts["sys_prompts"]
    usr_prompt_template = prompts["usr_prompt_template"]
    usr_prompts = [
        usr_prompt_template.format(
            paragraph=row.paragraph,
            question=row.question,
            choices=row.choices
        ) for _, row in df.iterrows()
    ]

    results = []
    for i, sys_prompt in enumerate(sys_prompts):
        requests = []
        for id, usr_prompt in enumerate(usr_prompts):
            request = {
                'custom_id': f"{id}",
                'method': 'POST',
                'url': "/v1/chat/completions",
                'body': {
                    'model': 'gpt-4o-mini',
                    'messages': [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt}
                    ],
                    ## hard coding 수정 해야함
                    'max_tokens': 500,
                    'temperature': 0,
                    'top_p': 1,
                    'frequency_penalty': 0,
                    'presence_penalty': 0,
                    'stop': None,
                    'logprobs': True,
                    'top_logprobs': 10,
                    'n': 1,
                }
            }
            requests.append(request)

        with open("requests.jsonl", "w") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")
        
        client = OpenAI(api_key=api_key)
        batch_input_file = client.files.create(
            file=open("requests.jsonl", "rb"),
            purpose="batch"
        )
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f"\nNow using {i}-th system prompt for generating reasoning...")
        while True:
            batches = client.batches.list().data[0]
            status = batches.status
            if status == "validating":
                print("Validating the inputs")
                while status == "validating":
                    time.spleep(5)
                    status = client.batches.list().data[0].status
            elif status == "in_progress":
                print(f"{batches.request_counts.completed}/{batches.request_counts.total}")
                while status == "in_progress":
                    time.spleep(60)
                    status = client.batches.list().data[0].status
            elif status == "finalizing":
                print("Finalizing the process")
                while status == "in_progress":
                    time.spleep(30)
                    status = client.batches.list().data[0].status
            elif status == "completed":
                print(f"{batches.request_counts.completed}/{batches.request_counts.total}")
                print("Request is successfully done")
                break
            elif status == "failed":
                print("There are some problems in API calling")
                break
            elif status == "expired":
                print("The process is expired, because it is not finished in 24hours")
                break
            else:
                print("The process is canceled")
                break

        result_file_id = client.batches.retrieve(batch_job.id).output_file_id
        result = client.files.content(result_file_id).content.decode('utf-8')
        result = [json.loads(line)["response"]["body"]["choices"][0]["message"]["content"]
                for line in result.split('\n') if line]
        
        df_res = df.copy()
        df_res["reason"] = result
        df["sys"] = sys_prompt
        results.append(df_res)
        print(f"{i}-th system prompt's generation is done.\n")

    df_final = pd.concat(results)
    train, valid = train_test_split(df_final, test_size=0.1, random_state=arg.seed)
    train.to_csv(os.path.join(DATA_DIR,"reasoning_train.csv"), index=False)
    valid.to_csv(os.path.join(DATA_DIR,"reasoning_valid.csv"), index=False)
    print("All datasets are saved into train and valid")

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
        "-d",
        "--data_dir",
        default="../../data/v0",
        type=str,
        help="data directory path (default: ../../data/v0)",
    )

    arg = args.parse_args()
    main(arg)