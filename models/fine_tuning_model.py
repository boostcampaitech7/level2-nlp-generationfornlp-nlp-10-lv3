# 표준 라이브러리
import torch

# 외부 라이브러리
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model

# 로컬 모듈
from utils.metrics import ft_compute_metrics


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
        
        if configs.use_lora:
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