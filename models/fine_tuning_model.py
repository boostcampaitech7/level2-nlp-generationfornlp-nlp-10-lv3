# 표준 라이브러리
import os
import torch
from tqdm import tqdm

# 외부 라이브러리
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score

# 로컬 모듈
from utils.metrics import ft_preprocess_logits_for_metrics, compute_qa_metrics
from utils.utils import extract_answer

class FineTuningModel:
    def __init__(self, configs, tokenizer):
        self.configs = configs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.tokenizer = tokenizer
        
        self.lora_config = LoraConfig(
            r = configs.lora_rank,
            lora_alpha = configs.lora_alpha,
            target_modules = configs.lora_target_modules,
            lora_dropout = configs.lora_dropout,
            bias = configs.lora_bias,
            task_type = configs.lora_task_type,
        )
        
        if configs.do_train :
            self.model = AutoModelForCausalLM.from_pretrained(
                configs.ft_model_path_or_name,
                torch_dtype = torch.float16,
                trust_remote_code=True,
            )
            self.model = get_peft_model(self.model, self.lora_config)
        else :
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                configs.ft_model_path_or_name,
                torch_dtype = torch.float16,
                trust_remote_code=True,
            )
        self.model.to(self.device)

        output_dir = os.path.join(configs.output_dir, configs.ft_model_path_or_name)
        self.training_args = TrainingArguments(
            output_dir = output_dir,
            # eval_strategy = "steps",
            # save_strategy = "steps",
            # eval_steps=configs.steps,
            # save_steps=configs.steps,
            save_strategy = "epoch",
            eval_strategy = "epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            # eval_strategy = "no",
            # load_best_model_at_end=False,
            save_total_limit = configs.save_total_limit,
            save_only_model = True,
            per_device_train_batch_size = configs.batch_size,
            per_device_eval_batch_size = configs.batch_size,
            gradient_accumulation_steps= configs.gradient_accumulation_steps,
            num_train_epochs = configs.num_train_epochs,
            lr_scheduler_type = configs.lr_scheduler_type,
            learning_rate = float(configs.learning_rate),
            weight_decay = configs.weight_decay,
            logging_steps = configs.steps,
            report_to = "wandb",
            # fp16=True,
            eval_accumulation_steps=10
        )

    def train(self, train_dataset, eval_dataset=None):
        self.trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = train_dataset,
            eval_dataset=eval_dataset,
            tokenizer = self.tokenizer,
            compute_metrics = lambda eval_results: compute_qa_metrics(eval_results, self.tokenizer),
            preprocess_logits_for_metrics=lambda logits, labels: ft_preprocess_logits_for_metrics(logits, labels, self.tokenizer),
        )

        self.trainer.train()
    
    def inference(self, test_dataset):
        question = []
        choice = []
        predictions = []
        references = []

        for data in tqdm(test_dataset):
            input_ids = torch.tensor(data['input_ids'], device=self.device).unsqueeze(0)
            attention_mask = torch.tensor(data['attention_mask'], device=self.device).unsqueeze(0)
                
            outputs = self.model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = 50,
                num_beams = 5,
            )
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            generated_text  = extract_answer(generated_text)

            # 원본 정답 가져오기
            true_answer = data['answer']
            # 정답 저장
            predictions.append(generated_text)
            references.append(true_answer)

            question.append(data['question'])
            choice.append(data['choice'])    

        accuracy = accuracy_score(references, predictions)
        f1 = f1_score(references, predictions, average="macro")
        metrics = {
            "accuracy" : accuracy,
            "f1" : f1,
        }
        results = pd.DataFrame({'question' : question,
                                'choice' : choice,
                                'origin_answer': references,
                                'generated_answer' : predictions})
        return results, metrics