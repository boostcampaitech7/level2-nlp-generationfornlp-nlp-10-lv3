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
        
    def describe_milvus(self, db_name):
        client = MilvusClient(
            uri=db_name
        )
        print(client.describe_collection("wiki_collection"))

    def search(self, query):
        print(query)
        results = self.vector_store.similarity_search_with_score(query, k=5)
        for result, score in results:
            print(f"Document: {result.page_content}, Score: {score}")
        
if __name__=="__main__":
    
    # config
    model_name = "dragonkue/BGE-m3-ko" # max_seq_length = 8192
    collection_name = "wiki_collection"
    db_name = "../../milvus/milvus_rag2.db"

    # Retrieval
    retrieval = RAG(model_name, collection_name, db_name)

    query = "5.18 민주화 운동은 어디서 일어났어?"

    retrieval.search(query)