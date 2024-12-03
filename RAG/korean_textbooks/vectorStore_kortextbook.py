import argparse
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from tqdm import tqdm
from milvus_database import MilvusDatabase
import numpy as np
import pandas as pd 

def embedding_text_save_db(model, datasets, batch_size,
                   database, collection_name, start_idx=0):

    # 데이터를 batch_size로 나눕니다.
    for i in tqdm(range(start_idx, len(datasets), batch_size), desc="Embedding batches"):
        batch_texts = datasets[i:i+batch_size]['text']
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)  # 각 배치에 대해 embedding 수행

        data = pd.DataFrame({
            'id' : datasets[i:i+batch_size]['id'],
            'text': batch_texts,
            'emb' : batch_embeddings.tolist()
        })
            
        database.insert_data(collection_name, data)
        print(f"{i // batch_size}th batch saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        help="Path to store data")
    parser.add_argument("--model_name", type=str, 
                        default="dragonkue/bge-m3-ko",
                        help="")
    parser.add_argument("--db_path", type=str,
                        help="")
    parser.add_argument("--collection_name", type=str,
                        default="korean_textbooks",
                        help="")
    parser.add_argument("--batch_size", type=int,
                        default=128,
                        help="")
    parser.add_argument("--drop_collection", type=bool,
                        default=True,
                        help="")
    args = parser.parse_args() 
    
    model_config = AutoConfig.from_pretrained(args.model_name)
    model = SentenceTransformer(args.model_name, device="cuda")
    embed_dim = model_config.hidden_size

    korean_textbook = load_from_disk(args.data_path)

    database = MilvusDatabase(args.db_path, 
                              args.collection_name, 
                              embed_dim,
                              False)
    if args.drop_collection:
        database.drop_collection()
        database.set_collection()
    
    korean_textbook = embedding_text_save_db(model, korean_textbook, args.batch_size,
                                             database, args.collection_name)    
    # database.insert_data(args.collection_name, korean_textbook)

if __name__=="__main__":
    main()