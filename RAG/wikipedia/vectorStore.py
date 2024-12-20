from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from datasets import load_dataset

import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from milvus_database import MilvusDatabase
import logging

load_dotenv()
hf_api_key = os.getenv('HF_API_KEY')

def embedding_text(model, datasets):

    batch_size = 128
    
    embeddings_list = []

    # 데이터를 batch_size로 나눕니다.
    for i in tqdm(range(0, len(datasets), batch_size), desc="Embedding batches"):
        batch_texts = datasets['text'].iloc[i:i + batch_size].tolist()
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)  # 각 배치에 대해 embedding 수행
        embeddings_list.extend(batch_embeddings)  # 결과를 리스트에 추가
        # print(batch_embeddings.dtype)
        torch.cuda.empty_cache()

    datasets["emb"] = embeddings_list
    print(datasets.shape)
    return datasets

def wiki_dataset(dataset_name): 

    print("Load Dataset :",dataset_name)
    dataset = load_dataset(dataset_name, split="train")

    # id, title, text만 가져감
    df = dataset.to_pandas()
    df = df[["id","title","text"]]
    df = df[df['text'].str.len() < 2024] # 2024자 이상인경우 삭제
    print("Dataset Shape :",df.shape)
    print(df.head())

    return df

# Data를 Embedding하고 milvus로 전달하는 과정 수행
if __name__=="__main__":

    model_name = "dragonkue/BGE-m3-ko" # max_seq_length = 8192
    dataset_name = "Cohere/wikipedia-22-12-ko-embeddings" # wiki Data
    db_path = "../../milvus/milvus_rag2.db"
    collection_name = "wiki_collection"
    
    # Load Model, Dataset, Milvus
    model = SentenceTransformer(model_name,
                                model_kwargs={
                                    "torch_dtype":torch.float16
                                }) # embedding model
    datasets = wiki_dataset(dataset_name)

    # Embedding
    datasets = embedding_text(model, datasets)
    
    # Insert Data to milvus
    database = MilvusDatabase(db_path, collection_name, 1024) # load database
    database.drop_collection() # 기존에 존재하면 삭제
    database.set_collection()
    database.insert_data(collection_name, datasets)
    print(database.describe())
