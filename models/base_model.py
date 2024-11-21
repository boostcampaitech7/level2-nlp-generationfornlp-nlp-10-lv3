# 표준 라이브러리
import os
import numpy as np

# 외부 라이브러리
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig

# 로컬 모듈
from utils.metrics import preprocess_logits_for_metrics, compute_metrics


class BaseModel:
    def __init__(self, configs, tokenizer) :
        self.configs = configs

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.special_tokens_map
        tokenizer.padding_side = configs.padding_side
        
        self.model = AutoModelForCausalLM.from_pretrained(
            configs.train_model_path_or_name,
            torch_dtype = torch.float16,
            trust_remote_code = True,
            device_map="auto",
            load_in_4bit=False,
        ).to(self.device)

        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template=configs.response_template,
            tokenizer=tokenizer
        )

        self.sft_config = SFTConfig(
            do_train=True,
            do_eval=True,
            lr_scheduler_type="cosine", # 바꾸고 싶으면 요청.
            max_seq_length=configs.max_length,
            output_dir=os.path.join("./saved/models", configs.train_model_path_or_name),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            # gradient_checkpointing=True, # 연산속도 느려짐.
            num_train_epochs=self.configs.num_train_epochs,
            learning_rate=self.configs.learning_rate,
            weight_decay=self.configs.weight_decay,
            logging_steps=self.configs.logging_steps,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=self.configs.save_total_limit,
            save_only_model=True,
            report_to="wandb",
            # fp16=False,
            # bf16=False
        )

        self.peft_config = LoraConfig(
            r=self.configs.rank,
            lora_alpha=self.configs.lora_alpha,
            lora_dropout=self.configs.lora_dropout,
            target_modules=self.configs.target_modules,
            bias=self.configs.bias,
            task_type=self.configs.task_type,
        )
        
    def train(self, train_dataset, eval_dataset):
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=lambda eval_result: compute_metrics(eval_result, self.tokenizer),
            preprocess_logits_for_metrics=lambda logits, labels: preprocess_logits_for_metrics(logits, labels, self.tokenizer),
            peft_config=self.peft_config,
            args=self.sft_config,
        )

        self.trainer.train()

    def eval(self, eval_dataset):
        pass 

        return # 데이터 프레임 (판다스)

    def inference(self, test_dataset):
        infer_results = []

        pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
       
        self.model.eval()
        with torch.inference_mode():
            for idx in tqdm(range(len(test_dataset))):
                _id = test_dataset[idx]['id']
                messages = test_dataset[idx]["messages"]
                len_choices = test_dataset[idx]["len_choices"]

                outputs = self.model(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenizer=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(self.device)
                )

                logits = outputs.logits[:, -1].flatten().cpu()

                target_logit_list = [logits[self.tokenizer.vocab[str(i + 1)]] for i in range(len_choices)]

                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(target_logit_list, dtype=torch.float32)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

                predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
                infer_results.append({"id": _id, "answer": predict_value})
            
        return infer_results