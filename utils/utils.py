# 표준 라이브러리
import random
import yaml

# 외부 라이브러리
import numpy as np
from box import Box
import torch


def load_config(config_file):
    # Load Config.yaml
    with open(config_file) as file:
        config = yaml.safe_load(file) # Dictionary
        config = Box(config) # . 

    return config


# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def extract_answer(text):
    try:
        answer = int(text)
    except:
        answer = "0" # random.randint(1, 5)
    return answer