# 외부 라이브러리
import evaluate
import numpy as np
import torch
import torch.nn.functional as F

# 로컬 모듈
from utils.utils import extract_answer


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
    # DISTINCT-N 계산 함수
    def distinct_n(predictions, n=2):
        ngrams = set()
        total_ngrams = 0
        for pred in predictions:
            tokens = pred.split()
            total_ngrams += len(tokens) - n + 1
            ngrams.update(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        return len(ngrams) / total_ngrams if total_ngrams > 0 else 0

    # Perplexity 계산 함수
    def calculate_perplexity(logits):
        logits_tensor = torch.tensor(logits).to("cuda")  # GPU로 이동
        probs = F.softmax(logits_tensor, dim=-1)  # logits을 확률로 변환
        log_probs = torch.log(probs + 1e-9)  # log 계산 시 NaN 방지
        perplexity = torch.exp(-torch.mean(log_probs))  # perplexity 계산
        return perplexity.item()

    # 메트릭 초기화
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    ter_metric = evaluate.load("ter")

    # eval_preds에서 logits과 labels 추출
    logits, labels = eval_preds

    # logits에서 가장 높은 확률을 가진 토큰 ID 추출
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits_tensor = torch.tensor(logits) # GPU로 이동
    predicted_ids = torch.argmax(logits_tensor, dim=-1)

    # labels의 -100 값을 tokenizer.pad_token_id로 변경
    labels = [
        [token if token != -100 else tokenizer.pad_token_id for token in label]
        for label in labels
    ]

    # 생성된 텍스트와 레이블을 디코딩
    predictions = tokenizer.batch_decode(predicted_ids.tolist(), skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 메트릭 계산
    rouge_result = rouge_metric.compute(predictions=predictions, references=references)
    bleu_result = bleu_metric.compute(predictions=predictions, references=references)
    meteor_result = meteor_metric.compute(predictions=predictions, references=references)
    ter_result = ter_metric.compute(predictions=predictions, references=references)
    distinct2 = distinct_n(predictions, n=2)
    perplexity = calculate_perplexity(logits)

    # 결과 반환
    return {
        "rouge1": rouge_result["rouge1"].mid.fmeasure,
        "rouge2": rouge_result["rouge2"].mid.fmeasure,
        "rougeL": rouge_result["rougeL"].mid.fmeasure,
        "bleu": bleu_result["bleu"],
        "meteor": meteor_result["meteor"],
        "ter": ter_result["score"],
        "distinct2": distinct2,
        "perplexity": perplexity,
    }

def single_sample_evaluate(eval_preds, tokenizer):
    # DISTINCT-N 계산 함수
    def distinct_n(predictions, n=2):
        ngrams = set()
        total_ngrams = 0
        for pred in predictions:
            tokens = pred.split()
            total_ngrams += len(tokens) - n + 1
            ngrams.update(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        return len(ngrams) / total_ngrams if total_ngrams > 0 else 0

    # Perplexity 계산 함수
    def calculate_perplexity(logits):
        logits_tensor = torch.tensor(logits).to("cuda")  # GPU로 이동
        probs = F.softmax(logits_tensor, dim=-1)  # logits을 확률로 변환
        log_probs = torch.log(probs + 1e-9)  # log 계산 시 NaN 방지
        perplexity = torch.exp(-torch.mean(log_probs))  # perplexity 계산
        return perplexity.item()
    
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    ter_metric = evaluate.load("ter")

    logits, labels = eval_preds
    total_perplexity = 0

    metrics_results = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "bleu": [],
        "meteor": [],
        "ter": [],
        "distinct2": [],
    }

    for logit, label in zip(logits, labels):
        logit_tensor = torch.tensor(logit).unsqueeze(0).to("cuda")
        predicted_ids = torch.argmax(logit_tensor, dim=-1)

        label = [token if token != -100 else tokenizer.pad_token_id for token in label]

        # 생성된 텍스트와 레이블을 디코딩
        prediction = tokenizer.decode(predicted_ids[0].tolist(), skip_special_tokens=True)
        reference = tokenizer.decode(label, skip_special_tokens=True)

        # Perplexity 계산
        perplexity = calculate_perplexity(logit_tensor)
        total_perplexity += perplexity

        # 메트릭 계산
        rouge_result = rouge_metric.compute(predictions=[prediction], references=[reference])
        bleu_result = bleu_metric.compute(predictions=[prediction], references=[reference])
        meteor_result = meteor_metric.compute(predictions=[prediction], references=[reference])
        ter_result = ter_metric.compute(predictions=[prediction], references=[reference])
        distinct2 = distinct_n([prediction], n=2)

        # 결과 저장
        metrics_results["rouge1"].append(rouge_result["rouge1"].mid.fmeasure)
        metrics_results["rouge2"].append(rouge_result["rouge2"].mid.fmeasure)
        metrics_results["rougeL"].append(rouge_result["rougeL"].mid.fmeasure)
        metrics_results["bleu"].append(bleu_result["bleu"])
        metrics_results["meteor"].append(meteor_result["meteor"])
        metrics_results["ter"].append(ter_result["score"])
        metrics_results["distinct2"].append(distinct2)

    # 최종 평균 계산
    num_samples = len(logits)
    metrics_results = {k: np.mean(v) for k, v in metrics_results.items()}
    metrics_results["perplexity"] = total_perplexity / num_samples

    return metrics_results


def single_sample_perplexity_evaluate(eval_preds):
    # Perplexity 계산 함수
    def calculate_perplexity(logits):
        logits_tensor = torch.tensor(logits).to("cuda")  # GPU로 이동
        probs = F.softmax(logits_tensor, dim=-1)  # logits을 확률로 변환
        log_probs = torch.log(probs + 1e-9)  # log 계산 시 NaN 방지
        perplexity = torch.exp(-torch.mean(log_probs))  # perplexity 계산
        return perplexity.item()
    
    logits, _ = eval_preds
    total_perplexity = 0

    for logit in logits:
        logit_tensor = torch.tensor(logit).unsqueeze(0).to("cuda")
        perplexity = calculate_perplexity(logit_tensor)
        total_perplexity += perplexity

    num_samples = len(logits)
    average_perplexity = total_perplexity / num_samples

    return {"perplexity": average_perplexity}

def compute_qa_metrics(eval_preds, tokenizer):
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_preds

    # Convert logits to predictions
    predictions = np.argmax(logits, axis=-1)

    # Remove padding tokens from labels
    labels = [[label for label in label_row if label != -100] for label_row in labels]

    # Match the length of predictions and labels
    flat_predictions = []
    flat_labels = []

    for pred, label in zip(predictions, labels):
        length = min(len(pred), len(label))
        flat_predictions.extend(pred[:length])
        flat_labels.extend(label[:length])

    # Compute metrics
    accuracy_result = accuracy_metric.compute(predictions=flat_predictions, references=flat_labels)
    f1_result = f1_metric.compute(predictions=flat_predictions, references=flat_labels, average="macro")

    return {
        "accuracy": accuracy_result["accuracy"],
        "f1": f1_result["f1"]
    }

def ft_preprocess_logits_for_metrics(logits, labels, tokenizer):
    predictions = torch.argmax(logits, dim=-1) # logits -> predictions: [batch_size, seq_len]

    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 디코딩 (토큰 ID -> 텍스트)

    extracted_answers = [extract_answer(pred) for pred in decoded_predictions]
    # "정답:" 이후 텍스트 추출

    # 추출된 텍스트를 다시 토큰화 -> 로짓과 동일한 텐서 크기로 변환
    tokenized_answers = tokenizer(
        extracted_answers,
        padding="max_length",  # 로짓과 동일한 길이로 패딩
        max_length=logits.size(1),
        truncation=True,
        return_tensors="pt"
    )

    # tokenized_answers["input_ids"]를 3D 텐서로 변환
    # input_ids: [batch_size, seq_len] -> 3D 형태로 복원
    processed_logits = torch.zeros_like(logits)  # 원래 로짓과 동일한 크기의 텐서 생성
    for i, input_id in enumerate(tokenized_answers["input_ids"]):
        processed_logits[i, :len(input_id), input_id] = 1.0  # One-hot encoding 형태로 설정

    return processed_logits
