import os
import yaml

from box import Box
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig

from data_loader.datasets import ReasoningDataset
from utils.utils import load_config


def main():
    # loading configuration
    BASE_DIR = os.getcwd()
    CONFIG_DIR = os.path.join(BASE_DIR, "Reasoning", "prompts.yaml")
    configs = load_config(CONFIG_DIR)

    # loading model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        configs.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.max_new_tokens = configs.max_new_tokens ## setting model max new token

    # loading dataset
    train_data = pd.read_csv(os.path.join(configs.data_dir, configs.train_path))
    eval_data = pd.read_csv(os.path.join(configs.data_dir, configs.val_path))

    # setting response template
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=configs.response_template,
        tokenizer=tokenizer
    )

    # training setting
    sft_config = SFTConfig(
        do_train=True,
        do_eval=True,
        lr_scheduler_type="cosine",
        output_dir=os.path.join("./saved/models", configs.model_id),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True, # 연산속도 느려짐. # VRAM 줄이는 용도
        gradient_accumulation_steps=4,
        max_seq_length=configs.max_seq_length,
        num_train_epochs=configs.num_train_epochs,
        learning_rate=configs.learning_rate,
        weight_decay=configs.weight_decay,
        logging_steps=configs.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=configs.save_total_limit,
        save_only_model=True,
        report_to="wandb",
        fp16=True, # Mix Precision
        bf16=False
    )
    # LoRA setting
    peft_config = LoraConfig(
        r=configs.rank,
        lora_alpha=configs.lora_alpha,
        lora_dropout=configs.lora_dropout,
        target_modules=configs.target_modules,
        bias=configs.bias,
        task_type=configs.task_type,
    )

    # calling custon dataset class
    train_dataset = ReasoningDataset(
        data=train_data,
        tokenizer=tokenizer,
        configs=configs,
        do_train=True
    )
    eval_dataset = ReasoningDataset(
        data=eval_data,
        tokenizer=tokenizer,
        configs=configs,
        do_train=True
    )

    # training
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        peft_config=peft_config,
        args=sft_config,
    )
    trainer.train()

if __name__ == "__main__":
    main()