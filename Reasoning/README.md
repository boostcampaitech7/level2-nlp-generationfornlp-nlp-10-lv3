# Learning reasoning process of LLM
## Concept
- LLM이 등장한 이후로, 단순히 LLM 자체의 역량을 통해 굉장히 다양한 태스크의 처리가 가능해졌다.
- 하지만, LLM을 사용하기 위해서는 매우 높은 컴퓨팅 리소스 등이 요구된다.
- 따라서, 상대적으로 작은 모델을 이용해 LLM의 성능을 흉내낼 수 있다면, 이러한 비용을 절약하는 데 매우 큰 도움이 될 것이다.
- 본 실험에서는 LLM의 성능을 흉내내기 위해, Mukherjee, Subhabrata, et al.(2023)의 방법을 참고하여 상대적으로 작은 모델 기반으로 실험을 진행했다.
- 해당 방법론은 문제 풀이 task에 대해서 LLM이 정답을 도출하는 과정을 뽑아낸 뒤, 그 과정을 상대적으로 작은 모델로 학습하는 방법이다.

## Method
- 먼저, 기존의 question, paragraph, answer data를 기반으로 문제 풀이에 대한 자세한 풀이를 유도하는 prompt를 만든다.
- 접근할 수 있는 LLM 모델에 해당 prompt를 입력하고 문제 풀이에 대한 reasoning을 도출하여 저장한다.
- Reasoning을 기존의 데이터에 결합한다.
- 이후, 활용하는 데 큰 무리가 없는 상대적으로 작은 모델을 활용하여, 생성한 reasoning을 target으로서 학습한다.
- Reasoning을 생성하기 위한 systemp prompt는 Mukherjee, Subhabrata, et al.(2023)에서 제시한 system prompt 중 4개를 발췌하여 사용했다.
- LLM은 openAI의 GPT-40-mini를 활용했다.
