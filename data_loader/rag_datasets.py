# 표준 라이브러리
from ast import literal_eval
from tqdm import tqdm
# 외부 라이브러리
import pandas as pd
import torch
from datasets import Dataset
from RAG.retrieval import Retrieval

class RAGDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, configs,
                 do_train = True):
        self.tokenizer = tokenizer
        self.max_length = configs.max_length
        self.configs = configs
        self.do_train = do_train
        # self.dataset = self.format_data(data)
        # self.tokens = self.tokenize(self.dataset)
        # if self.max_length is not None:
        #     self.tokens = self.tokens.filter(lambda x: len(x["input_ids"]) <= self.max_length)
        
        self.model_name = "dragonkue/BGE-m3-ko"
        self.collection_name = "wiki_collection"
        self.db_name = "../milvus/milvus_rag2.db"

        self.retrieval = Retrieval(self.model_name,
                             self.collection_name,
                             self.db_name)

        self.dataset = self.hint_generate(data)
        self.formatted_dataset = self.format_data(self.dataset)
        self.tokens = self.tokenize(self.formatted_dataset)
        

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        if self.do_train:
            return self.tokens[idx]
        else:
            return self.formatted_dataset[idx]

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

    def refactor_data(self, dataset) : # pandas Dataframe 
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

    def hint_generate(self, dataset):
        
        # 이 포맷이 과연 적절한가?
        QUERYFORMAT_NOPLUS="""
        지문:{paragraph}
        질문:{question}
        선택지:{choices}
        """
        QUERYFORMAT_PLUS="""
        지문:{paragraph}
        질문:{question}
        보기:{question_plus}
        선택지:{choices}
        """
    
        df = self.refactor_data(dataset)
        hints = []
        for i in tqdm(range(len(df))):
            data = df.iloc[i]

            paragraph = data["paragraph"]
            question = data["question"]
            choices = data["choices"]
            question_plus = data["question_plus"]
            if data["question_plus"]:
                question_plus = data["question_plus"]
            
            if question_plus:
                query = QUERYFORMAT_NOPLUS.format(
                    paragraph=paragraph,
                    question=question,
                    choices=choices,
                )
            else:
                query = QUERYFORMAT_PLUS.format(
                    paragraph=paragraph,
                    question=question,
                    question_plus=question_plus,
                    choices=choices,
                )

            hint = self.retrieval.search(query)
            for result, score in hint:
                hints.append(result.page_content) # Hint Paragraph

        df["hint"] = hints
        df.to_csv("./RAG/hints_df.csv", encoding="utf-8-sig")
        del self.retrieval
        return df

    def format_data(self, dataset):
        
        # dataset = self.refactor_data(dataset)
        processed_dataset = []

        system_prompt = self.configs.PROMPT_SYSTEM_MESSAGE

        for i, row in dataset.iterrows():
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])

            if not self.do_train :
                len_choices = len(row['choices'])

            # <보기>가 있을 때
            if row["question_plus"]:
                user_message = self.configs.PROMPT_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    hint=row["hint"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = self.configs.PROMPT_NO_QUESTION_PLUS.format(
                    paragraph=row["paragraph"],
                    hint=row["hint"],
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
        return Dataset.from_pandas(pd.DataFrame(processed_dataset)) 
