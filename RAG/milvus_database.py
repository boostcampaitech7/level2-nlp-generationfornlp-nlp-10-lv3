from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from tqdm import tqdm
import numpy as np 
# Vector DB 구현

class MilvusDatabase:
    def __init__(self, uri, collection_name, embedding_dim):
        self.milvus_client = MilvusClient(uri=uri) # milvus init
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

    def set_collection(self):
        # Create Collection
        if not self.milvus_client.has_collection(self.collection_name):
            schema = self.create_schema()
            index_param = self.create_index()

            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_param
            )

     # 스키마 생성
    def create_schema(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=6000, description="raw Text"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim, description="vector")
        ]
        schema = CollectionSchema(fields=fields, auto_id=True, description="Rag schema")
        return schema
    
    def create_index(self):
        index_params = self.milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",
            metric_type="IP"
        )
        return index_params    
    

    def insert_data(self, collection_name, datasets):
        # Insert Data to Collection
        
        for i in tqdm(range(len(datasets)), desc="Insert Data to Milvus DB"):
            row = datasets.iloc[i]
            vector = np.array(row["emb"], dtype=np.float32)
            data = {
                "text": row["text"],
                "embedding": vector,
            }
            
            self.milvus_client.insert(collection_name=collection_name,
                                  data=data)
        
        print("Done. Insert Data")
  
    def drop_collection(self):
        # Drop Collection
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

    def describe(self):
        return self.milvus_client.describe_collection(self.collection_name)