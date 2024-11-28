# 표준 라이브러리
from ast import literal_eval
from copy import deepcopy

# 외부 라이브러리
import pandas as pd
import torch
from datasets import Dataset


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data,
                 tokenizer, configs, do_train = True):
        self.tokenizer = tokenizer
        self.max_length = configs.max_length
        self.configs = configs
        self.do_train = do_train
        self.dataset = self.format_data(data)
        self.tokens = self.tokenize(self.dataset)
        # if self.max_length is not None:
        #     self.tokens = self.tokens.filter(lambda x: len(x["input_ids"]) <= self.max_length)
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        if self.do_train:
            return self.tokens[idx]
        else:
            return self.dataset[idx]

    def tokenize(self, dataset):
        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example["messages"])):
                output_texts.append(
                    self.tokenizer.apply_chat_template(
                        example["messages"][i],
                        tokenize=False,
                    )
                )
            return output_texts

        def _tokenize(element):
            outputs = self.tokenizer(
                formatting_prompts_func(element),
                truncation=False,
                padding=False,
                return_overflowing_tokens=False,
                return_length=False,
            )
            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
            }

        tokenized_dataset = dataset.map(
            _tokenize,
            remove_columns=list(dataset.features),
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )

        return tokenized_dataset

    def format_data(self, dataset):
        def refactor_data(dataset) : # pandas Dataframe 
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

            return df # pandas Dataframe 
        
        dataset = refactor_data(dataset)
        processed_dataset = []

        system_prompt = "지문을 읽고 질문의 답을 구하세요."

        for i, row in dataset.iterrows():
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])

            if not self.do_train :
                len_choices = len(row['choices'])

            # <보기>가 있을 때
            if row["question_plus"]:
                user_message = self.configs.PROMPT_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = self.configs.PROMPT_NO_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    choices=choices_string,
                )
            
            # chat message 형식으로 변환
            if self.do_train:
                processed_dataset.append(
                    {
                        "id": row["id"],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": f"{row['answer']}"}
                        ],
                        "label": row["answer"],
                    }
                )
            else:
                processed_dataset.append(
                    {
                        "id": row["id"],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "len_choices": len_choices,
                    }
                )
        processed_dataset = pd.DataFrame(processed_dataset)
        # inference시에 해당부분 주석
        processed_dataset['label'] = processed_dataset['label'].astype(int)
        
        return Dataset.from_pandas(processed_dataset)


class FineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, configs, is_eval=False):
        self.tokenizer = tokenizer
        self.configs = configs
        self.is_eval = is_eval

        # 데이터 전처리를 한 번만 수행
        self.tokenized_data = data.map(self.preprocess_function, batched=True)

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def preprocess_function(self, examples):
        if self.is_eval:
            # 평가 데이터 전처리
            prompt = "\n질문에 알맞은 선택지를 골라 정답만 출력하세요. \n정답:"
            inputs = self.tokenizer(
                ["질문: "+t+"\n선택지: "+c+prompt for t, c in zip(examples["question"], examples['choice'])], # 
                truncation=True,
                padding="max_length",
                max_length=self.configs.max_length,
            )
            # 타겟 데이터 전처리
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples["answer"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.configs.max_length,
                )

            # 패딩 토큰을 -100으로 변경
            labels["input_ids"] = [
                [-100 if token == self.tokenizer.pad_token_id else token for token in label]
                for label in labels["input_ids"]
            ]
            inputs["labels"] = labels["input_ids"]

        else:
            # 학습 데이터 전처리
            inputs = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.configs.max_length,
            )

            # labels 생성 및 한 칸씩 이동
            inputs["labels"] = deepcopy(inputs["input_ids"])
            for i in range(len(inputs["labels"])):
                inputs["labels"][i] = inputs["labels"][i][1:] + [self.tokenizer.pad_token_id]

            # 패딩 토큰을 -100으로 변경
            inputs["labels"] = [
                [-100 if token == self.tokenizer.pad_token_id else token for token in label]
                for label in inputs["labels"]
            ]

        return inputs