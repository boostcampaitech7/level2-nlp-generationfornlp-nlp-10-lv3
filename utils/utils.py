from ast import literal_eval
import pandas as pd 
import yaml
from datasets import Dataset

# torch Dataset 






def load_config(config_path) :
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config 