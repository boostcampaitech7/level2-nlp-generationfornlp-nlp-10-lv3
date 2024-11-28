# 표준 라이브러리
import os
import numpy as np
import pandas as pd

# 외부 라이브러리
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, pipeline
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig, AutoPeftModelForCausalLM

# 로컬 모듈
from utils.metrics import preprocess_logits_for_metrics, compute_metrics
from utils.utils import extract_answer


class BaseModel:
    def __init__(self, configs, tokenizer, model) :
        self.configs = configs
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.model = model
        self.tokenizer = tokenizer
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = configs.padding_side

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
            gradient_checkpointing=True, # 연산속도 느려짐. # VRAM 줄이는 용도
            gradient_accumulation_steps=4,
            num_train_epochs=self.configs.num_train_epochs,
            learning_rate=self.configs.learning_rate,
            weight_decay=self.configs.weight_decay,
            logging_steps=self.configs.logging_steps,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=self.configs.save_total_limit,
            save_only_model=True,
            report_to="wandb",
            fp16=True, # Mix Precision
            bf16=False
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
    
    def inference_pipeline(self, test_dataset):
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # torch_dtype=torch.float16,
            # device="cuda",
            do_sample=True,
            top_k=10,
            max_new_tokens=1,
            return_full_text = False,   
        )
        id_list = []
        generate_text_list = []
        for idx in tqdm(range(len(test_dataset)), desc="Generating answer"):
            _id = test_dataset[idx]['id']

            input_texts = self.tokenizer.apply_chat_template(
                test_dataset[idx]['messages'],
                tokenize=False,
            )

            output = pipe(input_texts)
            output_text = output[0]['generated_text']

            id_list.append(_id)
            generate_text_list.append(extract_answer(output_text))
        
        return {
            'id': id_list,
            'answer': generate_text_list,
        }



    def inference_generate(self, test_dataset): # 현재 작동 X
        # test_datsaet은 Tokenized가 된 데이터셋이 아님
        generated_infer_results = []
        
        self.model.eval()
        
        with torch.inference_mode():
            for idx in tqdm(range(len(test_dataset))):
                _id = test_dataset[idx]['id']
                messages = test_dataset[idx]["messages"]
                len_choices = test_dataset[idx]["len_choices"]

                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self.device)
                
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=2,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                generate_text = self.tokenizer.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
                )[0]
                print(generate_text)
                generate_text.strip()
                generated_infer_results.append({"id":idx, "target":generate_text})
        
        return generated_infer_results

    def inference_vllm(self, test_dataset):
        infer_results = []
        pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
        # self.model.eval()
        # test_dataset : Not Tokenized Dataset
        # apply_chat_template
        
        with torch.inference_mode():
            for idx in tqdm(range(len(test_dataset))):
                # Tokenizing
                _id = test_dataset[idx]['id']
                messages = test_dataset[idx]["messages"]
                len_choices = test_dataset[idx]["len_choices"]

                tokenized = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self.device)

                print(self.tokenizer.decode(tokenized[0], skip_special_tokens=False))
                
                        

    def inference(self, test_dataset):
        infer_results = []
        decoded_results = []

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
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(self.device)
                )
                # logits에서 생성된 토큰 ID를 추출
                generated_ids = torch.argmax(outputs.logits, dim=-1)

                # 디코드
                decoded_sent = self.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=False
                )
                # print(f"Decoded sentence for ID {_id}: {decoded_sent}")

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
                decoded_results.append({"id": _id, "decoded": decoded_sent})

        return infer_results, decoded_results