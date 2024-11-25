from pymilvus import MilvusClient
from tqdm import tqdm

# Vector DB 구현
class MilvusDatabase:
    def __init__(self, uri, collection_name, embedding_dim):
        self.milvus_client = MilvusClient(uri=uri) # milvus init
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

    def set_collection(self):
        # Create Collection
        if not self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedding_dim, # embedding dimension
                metric_type="COSINE", #innuer product
                consistency_level="Strong"
            )

    def insert_data(self, collection_name, datasets):
        # Insert Data to Collection
        
        for i in tqdm(range(len(datasets)), desc="Insert Data to Milvus DB"):
            row = datasets.iloc[i]

            data = {
                "id": row["id"],
                "text": row["text"],
                "vector": row["emb"],
            }
            
            self.milvus_client.insert(collection_name=collection_name,
                                  data=data)
        
        print("Done. Insert Data")
  
    def drop_collection(self):
        # Drop Collection
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)
