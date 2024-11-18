
import torch 
import numpy as np
import random

from utils.utils import format_data, load_config

import sys



import torch
import transformers
from ast import literal_eval
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
import json
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM, LoraConfig


# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

set_seed(42) # magic number :)


train한걸로 실제로 돌려봤을 때 결과 
test는 ㄴㄴ

val로 결과 파악 
train data 나 validataion 에 대한 prediction


Reason:
Answer: 
-> val결과 파악용 


def main() :
    config_path = sys.argv[1] # -> argpaser로 고치기 
    config = load_config(config_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path_or_name,
        trust_remote_code=True,
    )

    # 템플릿 읽어서 데이터셋에 전달 


    loader 

    model.train(train_loader, val_loader)

    return 0

if __name__ == "__main__":
    main()




