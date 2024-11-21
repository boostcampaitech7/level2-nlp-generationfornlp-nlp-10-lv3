# 외부 라이브러리
import evaluate
import numpy as np
import torch
from datasts import load_metric

def preprocess_logits_for_metrics(logits, labels, tokenizer):
    logits = logits if not isinstance(logits, tuple) else logits[0]
    logit_idx = [
        tokenizer.vocab["1"],
        tokenizer.vocab["2"],
        tokenizer.vocab["3"],
        tokenizer.vocab["4"],
        tokenizer.vocab["5"]
    ]
    logits = logits[:, -2, logit_idx]  # -2: answer token, -1: eos token
    return logits


def compute_metrics(evaluation_result, tokenizer):
    # metric 로드
    acc_metric = evaluate.load("accuracy")
    
    # 정답 토큰 매핑
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
    
    logits, labels = evaluation_result

    # 토큰화된 레이블 디코딩
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
    labels = list(map(lambda x: int_output_map[x], labels))

    # 소프트맥스 함수를 사용하여 로그트 변환
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    # 정확도 계산
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc


def ft_compute_metrics(eval_preds, tokenizer):
    rouge_metric = load_metric("rouge")

    logits, labels = eval_preds

    # 생성된 텍스트와 레이블을 디코딩
    predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE 계산
    rouge_result = rouge_metric.compute(predictions=predictions, references=references)
    
    return {
        "rouge1": rouge_result["rouge1"].mid.fmeasure,
        "rouge2": rouge_result["rouge2"].mid.fmeasure,
        "rougeL": rouge_result["rougeL"].mid.fmeasure,
    }