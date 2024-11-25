from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

class RAG:
    def __init__(self, model_name, collection_name, db_name):
        self.model_name = model_name
        self.collection_name = collection_name
        self.db_name = db_name

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name   
        )
        
        self.vector_store= Milvus(
            embedding_function=self.embedding_model,
            connection_args={
                "uri":self.db_name
            },
            collection_name=self.collection_name
        )
        
    def describe_milvus(self):
        client = MilvusClient(
            uri="../../milvus/milvus_rag.db"
        )
        print(client.describe_collection("wiki_collection"))

    def search(self, query):
        print(query)
        results = self.vector_store.similarity_search(query, k=5)
        print(results)
        for result in results:
            print(f"Document: {result.page_content}, Score: {result.score}")
        
if __name__=="__main__":
    
    # config
    model_name = "dragonkue/BGE-m3-ko" # max_seq_length = 8192
    collection_name = "wiki_collection"
    db_name = "../../milvus/milvus_rag.db"

    # Retrieval
    retrieval = RAG(model_name, collection_name, db_name)

    query = "5.18 민주화운동이 뭐야?"

    # retrieval.search(query)
    retrieval.describe_milvus()
