import argparse
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig
from tqdm import tqdm
from milvus_database import MilvusDatabase
import numpy as np


def embedding_text(model, datasets, batch_size):
    embeddings_list = []

    # 데이터를 batch_size로 나눕니다.
    for i in tqdm(range(0, len(datasets), batch_size), desc="Embedding batches"):
        batch_texts = datasets['text'].iloc[i:i + batch_size].tolist()
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)  # 각 배치에 대해 embedding 수행
        embeddings_list.extend(batch_embeddings)  # 결과를 리스트에 추가

    datasets["emb"] = embeddings_list
    return datasets


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
                        default=512,
                        help="")
    parser.add_argument("--drop_collection", type=bool,
                        default=True,
                        help="")
    args = parser.parse_args() 
    
    model_config = AutoConfig.from_pretrained(args.model_name)
    model = SentenceTransformer(args.model_name, device="cuda")
    embed_dim = model_config.hidden_size

    korean_textbook = load_from_disk(args.data_path).select(range(100))
    korean_textbook = korean_textbook.to_pandas()
    korean_textbook['id'] = list(range(len(korean_textbook)))

    database = MilvusDatabase(args.db_path, 
                              args.collection_name, 
                              embed_dim)
    if args.drop_collection:
        database.drop_collection()
        database.set_collection()
    
    korean_textbook = embedding_text(model, korean_textbook, args.batch_size)    
    database.insert_data(args.collection_name, korean_textbook)

if __name__=="__main__":
    main()