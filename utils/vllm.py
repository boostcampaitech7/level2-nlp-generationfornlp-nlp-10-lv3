"""
vLLM 실행방식을 적용하고자 해당 코드를 작성함
vLLM은 inference 속도를 24배나 빠르게 해준다고함..!

근데 방금 생각난건데, 이거 의미있나??
아 근데 RAG 쓰면 의미 있을 것 같음
hint를 RAG를 통해서 생성해낸다! 라는 느낌이니까
"""
from vllm import LLM

class vLLM:
    # init
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        self.llm = LLM(model)

    # 
    def inference(self, test_dataset):
        
        pass
    
    