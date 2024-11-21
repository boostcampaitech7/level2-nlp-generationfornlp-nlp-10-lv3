

from datasets import load_from_disk
import torch
from copy import deepcopy

class FineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, configs, is_eval=False):
        self.tokenizer = tokenizer
        self.configs = configs
        self.is_eval = is_eval
        self.tokenized_data = data.map(self.preprocess_function, batched=True)

    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def preprocess_function(self, examples):
        if self.is_eval:
            inputs = self.tokenizer("다음 문서를 요약하세요:\n"+ examples['text'],
                                    truncation=True,
                                    padding='max_length')
            labels = self.tokenizer(examples['summary'],
                                    truncation=True,
                                    padding='max_length')['input_ids']
            # 패딩 토큰은 -100으로 설정하여 손실 계산에서 제외
            labels = [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
            inputs['labels'] = labels
        else:

            inputs = self.tokenizer(examples['text'], 
                                    truncation=True,
                                    padding='max_length',
                                    )
            
            # 깊은 복사를 통해 labels를 input_ids와 분리
            inputs['labels'] = deepcopy(inputs['input_ids'])

            # 각 샘플에 대해 한 칸씩 밀기
            for i in range(len(inputs['labels'])):
                # 한 칸씩 밀기
                inputs['labels'][i][:-1] = inputs['labels'][i][1:]
                # 마지막 토큰을 패딩 토큰으로 설정
                inputs['labels'][i][-1] = self.tokenizer.pad_token_id

            # 패딩 토큰은 -100으로 설정하여 손실 계산에서 제외
            inputs['labels'] = [
                [-100 if token == self.tokenizer.pad_token_id else token for token in label]
                for label in inputs['labels']
            ]

        return inputs
    

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
class FineTuningModel:
    def __init__(self, configs, tokenizer):
        self.configs = configs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.tokenizer = tokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            configs.ft_model_path_or_name,
            torch_dtype = torch.float16,
            trust_remote_code=True,
        )
        
        self.lora_config = LoraConfig(
            r = configs.lora_rank,
            lora__alpha = configs.lora_alpha,
            target_modules = configs.lora_target_modules,
            lora_dropout = configs.lora_dropout,
            bias = configs.lora_bias,
            task_type = configs.lora_task_type,
        )
        
        self.model = get_peft_model(self.model, self.lora_config)

        self.data_collator = DataCollatorWithPadding(tokenizer, padding='longest')
        

        self.training_args = TrainingArguments(
            output_dir = f"../saved/fine_tuning/{configs.ft_model_path_or_name}",
            # eval_strategy = "steps",
            # save_strategy = "steps",
            # eval_steps=configs.steps,
            # save_steps=configs.steps,
            eval_strategy = "epoch",
            save_strategy = "epoch",
            save_total_limit = configs.save_total_limit,
            load_best_model_at_end=True,
            save_only_model = True,
            per_device_train_batch_size = configs.per_device_train_batch_size,
            gradient_accumulation_steps= configs.gradient_accumulation_steps,
            num_train_epochs = configs.num_train_epochs,
            lr_scheduler_type = configs.lr_scheduler_type,
            learning_rate = configs.learning_rate,
            weight_decay = configs.weight_decay,
            logging_steps = configs.steps,
            report_to = "wandb"
        )

    def train(self, train_dataset, eval_dataset=None):
        self.trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = train_dataset,
            eval_dataset=eval_dataset,
            data_collator = self.data_collator,
            tokenizer = self.tokenizer,
            compute_metrics = lambda eval_results: ft_compute_metrics(eval_results, self.tokenizer),
        )

        self.trainer.train()
    
    def inference(self, test_dataset):
        results = []
        for data in test_dataset:
            inputs = self.tokenizer(data['text'],
                                    return_tensors='pt',
                                    truncation=True,
                                    ).to(self.device)
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length = 256,
                num_beams = 5,
            )
            generated_text = self.tokenizer.decode(outputs[0],
                                                   skip_special_tokens=True
                                                   )
            results.append({
                "inputs": data['text'],
                "origin": data['summary'],
                "generated": generated_text
            })
        
        return results


from datasts import load_metric

def ft_compute_metrics(eval_preds, tokenizer):
    rouge_metric = load_metric("rouge")

    logits, labels = eval_preds

    # 생성된 텍스트와 레이블을 디코딩
    predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE 계산
    rouge_result = rouge_metric.compute(predictions=predictions, references=references)
    
    return {
        "rouge1": rouge_result["rouge1"].mid.fmeasure,
        "rouge2": rouge_result["rouge2"].mid.fmeasure,
        "rougeL": rouge_result["rougeL"].mid.fmeasure,
    }



from utils.utils import load_config, set_seed
import argparse

from dotenv import load_dotenv
from huggingface_hub import login
import wandb

load_dotenv()
hf_api_key = os.getenv('HF_API_KEY')
wandb_api_key = os.getenv('WANDB_API_KEY')

login(hf_api_key)
wandb.login(key=wandb_api_key)

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        help="")

    args = parser.parse_args() 

    configs = load_config(args.config_path)

    set_seed(configs.seed)

    wandb.init(project=configs.project, 
               name=configs.sub_project,
               )

    tokenizer = AutoTokenizer.from_pretrained(
        configs.ft_model_path_or_name,
        trust_remote_code = True,
    )

    dataset = load_from_disk(configs.data_path)
    train_dataset = FineTuningDataset(dataset['train'], tokenizer, configs)
    val_dataset = FineTuningDataset(dataset['validation'], tokenizer, configs, True)
    test_dataset = FineTuningDataset(dataset['test'], tokenizer, configs, True)

    model = FineTuningModel(configs, tokenizer)

    if configs.do_train:
        model.train(train_dataset, val_dataset)

    outputs = model.inference(test_dataset)

    outputs.to_csv(configs.output_path, index = False)


if __name__ == "__main__":
    main()


#!/bin/bash

# python ./train.py --config_path "./configs/.yaml"


