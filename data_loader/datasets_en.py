# 표준 라이브러리
from ast import literal_eval

# 외부 라이브러리
import pandas as pd
import torch
from datasets import Dataset
import deepl # 번역을 위함

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

        system_prompt = "chose an answer number from reading paragraph and question"

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
            
            #EN으로 번역
            auth_key = "0bf96e91-bb7b-4661-99ce-25e3c5ff7e83:fx"
            translator = deepl.Translator(auth_key)
            user_message_english = translator.translate_text(user_message, source_lang="KO", target_lang="EN-US")
            
            # chat message 형식으로 변환
            if self.do_train:
                processed_dataset.append(
                    {
                        "id": row["id"],
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message_english.text},
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
                            {"role": "user", "content": user_message_english.text},
                        ],
                        "len_choices": len_choices,
                    }
                )
        return Dataset.from_pandas(pd.DataFrame(processed_dataset)) 
