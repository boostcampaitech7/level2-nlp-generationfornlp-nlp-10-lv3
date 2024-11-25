import argparse
from datasets import load_from_disk
from retrievals import AutoModelForEmbedding
from transformers import AutoConfig
from tqdm import tqdm
from milvus_database import MilvusDatabase


def get_embedding(model, dataset, db, collection_name, 
                  batch_size, start_idx=0, save_interval=100):
    
    # Too much data, so saved for a specific step.
    save_step = save_interval * batch_size
    ids_list = []
    texts_list = []
    vectors_list = []
    for i in tqdm(range(start_idx, len(dataset), batch_size), desc="Calculating embeddings"):
        batch_texts = dataset[i:i+batch_size]['text']
        batch_embeddings = model.encode(batch_texts).tolist() # numpy.ndarray -> List[List]
        ids = list(range(i, i+batch_size))

        ids_list.extend(ids)
        texts_list.extend(batch_texts)
        vectors_list.extend(batch_embeddings)

        if (i % save_step == 0) & (i > 0):
            saved_data = {
                "id" : ids_list,
                "vector" : texts_list,
                "text" : vectors_list,
            }
            db.milvus_client.insert(
                collection_name=collection_name,
                data=saved_data,
            )
            ids_list = []
            texts_list = []
            vectors_list = []
            print(f"{i // save_step}th step saved.")
    
    if ids_list: # Check if there is remaining data
        saved_data = {
            "id" : ids_list,
            "vector" : texts_list,
            "text" : vectors_list,
        }
        db.milvus_client.insert(
            collection_name=collection_name,
            data=saved_data,
        )
        print("Final step saved.")


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
                        default=4,
                        help="")
    parser.add_argument("--drop_collection", type=bool,
                        default=True,
                        help="")
    args = parser.parse_args() 
    
    model_config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForEmbedding.from_pretrained(
        args.model_name,
        pooling_method="mean",
    )
    embed_dim = model_config.hidden_size

    korean_textbook = load_from_disk(args.data_path)

    database = MilvusDatabase(args.db_path, 
                              args.collection_name, 
                              embed_dim)
    if args.drop_collection:
        database.drop_collection()
        database.set_collection()
    
    get_embedding(model, korean_textbook, database, 
                  args.collection_name, args.batch_size)


if __name__=="__main__":
    main()