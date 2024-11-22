# 표준 라이브러리
import torch
from tqdm import tqdm

# 외부 라이브러리
import pandas as pd
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer,
    DataCollatorWithPadding, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# 로컬 모듈
from utils.metrics import ft_compute_metrics, single_sample_perplexity_evaluate


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
            lora_alpha = configs.lora_alpha,
            target_modules = configs.lora_target_modules,
            lora_dropout = configs.lora_dropout,
            bias = configs.lora_bias,
            task_type = configs.lora_task_type,
        )
        
        if configs.use_lora:
            self.model = get_peft_model(self.model, self.lora_config)
        
        self.model.to(self.device)

        # self.data_collator = DataCollatorWithPadding(tokenizer, 
        #                                              padding=True,  # 패딩 활성화
        #                                             )
        # self.data_collator = DataCollatorForSeq2Seq(
        #     tokenizer=self.tokenizer,
        #     model=self.model,
        #     padding=True
        # )
        
        self.training_args = TrainingArguments(
            output_dir = f"../saved/fine_tuning/{configs.ft_model_path_or_name}",
            # eval_strategy = "steps",
            # save_strategy = "steps",
            # eval_steps=configs.steps,
            # save_steps=configs.steps,
            save_strategy = "epoch",
            # eval_strategy = "epoch",
            # load_best_model_at_end=True,
            eval_strategy = "no",
            load_best_model_at_end=False,
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
            fp16=True,
            eval_accumulation_steps=10
        )

    def train(self, train_dataset, eval_dataset=None):
        self.trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset = train_dataset,
            eval_dataset=eval_dataset,
            # data_collator = self.data_collator,
            tokenizer = self.tokenizer,
            # compute_metrics = lambda eval_results: ft_compute_metrics(eval_results, self.tokenizer),
            compute_metrics = single_sample_perplexity_evaluate,
        )

        self.trainer.train()
    
    def inference(self, test_dataset):
        text = []
        origin_summary = []
        generated_summary = []
        for data in tqdm(test_dataset):
            inputs = self.tokenizer(data['text'],
                                    return_tensors='pt',
                                    truncation=True,
                                    ).to(self.device)
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens = 256,
                num_beams = 5,
            )
            generated_text = self.tokenizer.decode(outputs[0],
                                                   skip_special_tokens=True
                                                   )
            text.append(data['text'])
            origin_summary.append(data['summary'])
            generated_summary.append(generated_text)        
       
        pd

        return pd.DataFrame({'text' : text,
                             'origin': origin_summary,
                             'generated' : generated_summary})