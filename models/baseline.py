
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

pd.set_option('display.max_columns', None)

class Baseline:
    def __init__(self, config) :
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path_or_name,
            torch_dtype = torch.bfloat16,
            trust_remote_code = True
        )
        self.tokenizer = AutoTokenizer
        """
        토크나이저를 어디다가 넣어야하는가? 
        일단 프롬프트를 이용해서 토큰화를 진행하니까...
        근데 만약 랭체인을 이용하면??? 

        
        아니면 데이터셋 필요 없어 보이는데,
        아싸리 데이터셋 자체를 들고와서 모델 내부에서 알아서 사용하라고 할까?
        그럼 데이터로더도 알아서 불러오라고 시키고 ㅇㅅㅇ 
        """
    


