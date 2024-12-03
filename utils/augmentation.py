import os
import torch
import random
import numpy as np
import pandas as pd
from ast import literal_eval
from dotenv import load_dotenv
from tqdm import tqdm
import re
import ast

# 환경 변수 로드
load_dotenv()

# Pandas 설정
pd.set_option('display.max_columns', None)

# 난수 고정
def set_seed(random_seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if multi-GPU is used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


set_seed(42)

# 데이터셋 로드 및 플래트닝
dataset = pd.read_csv('./data/v0.1.6.csv')


def flatten_dataset(dataset):
    """Flatten nested JSON-like data."""
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'question': problems.get('question', None),
            'choices': problems.get('choices', None),
            'answer': problems.get('answer', None),
            'question_plus': problems.get('question_plus', None),
        }
        records.append(record)
    return pd.DataFrame(records)


result = flatten_dataset(dataset)

# OpenAI 클라이언트 설정
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def check_mismatch(paragraph):

    prompt = f"""당신은 수능 출제 전문가 입니다. 과목은 수능의 국어, 사회 영역(윤리, 정치, 사회, 역사, 경제, 지리) 
        아래 예시를 보고 주어진 paragraph, question와 대한민국 수능 범위를 참고해서 출제할 수 있는 수능문제 형식의 question, choices(5지선다), answer를 생성해주세요
        
        예시:
        paragraph:
        미국 정신 의학회(AmericanPsychiatricAssociation)에서 발행한 "정신 질환 진단 및 통계 편 람(DSM-IV)"은 다음의 모든 항목에 대한 정보를 제공합니다.
        
        question: 
        여기에 해당하지 않는 것은 무엇입니까?
        
        choices: 
        1. 정신 질환 명
        2. 모든 정신 질환의 분류
        3. 모든 정신 질환의 주요 증상
        4. 모든 정신 질환의 원인
        5. 정신 질환 정보 
        
        answer: 
        4

        paragraph:
        국회 정무 위원회 전해철 국회의원(더불어민주당)은 16일 금융 위원회가 운영하는 옴부즈만 제도가 소비자 보호와 관련해서 운영 실적이 저조하다고 밝혔다.2016년 2월 출범한 금융위 옴부즈만은 금융 규제를 상시 점검하고 불합리한 금융 규제를 개선하여 금융 회사의 고충을 경감하고 금융 소비자를 보호하기 위한 취지로 설치되었다. 전해철 의원실이 금융 위원회로부터 제출받은 국정 감사 자료에 따르면,16년 출범 이후 17년 9월 말까지 총 20개월 동안 금융 회사 고충 처리 민원은 65건, 소비자 보호를 위한 제도 개선은 16건으로 나타났다. 이 중 고충 처리 민원 6건과 소비자 보호 제도 개선 7건은 금감원 등 타 기관으로 이첩하여 실제 옴부즈만이 처리한 고충 처리 민원은 59건, 소비자 보호 제도 개선 9건으로 각각 월평균 2.95건,0.45건이다. 현재 옴부즈만 메일, 금융 규제 민원 포털 등 공식적인 채널로는 사건 접수가 매우 저조하다. 총 81건 중 단 1건만이 온라인으로 신청된 건이고, 나머지 80건 중 80 %이상이 금융 회사가 개별 옴부즈만에게 개인적, 비공식적으로 건의하는 민원들이 안건으로 주로 상정되는 상황은 금융 회사 고충 민원 처리에 집중되어 있는 옴부즈만 활용 현황을 반영하고 있는 것으로 보인다. 또한 옴부즈만은 2016년 2월 출범 후 연 4회로 회의를 예정하여 3회 개최하였다가 1주년인 2017년 2월 연 8회로 회의를 확대 개최하겠다고 계획하였으나 2017년 10월 현재까지도 4회 개최하는 데 그치는 등 활발한 운용이 이루어지고 있지 않은 것으로 나타났다. 금융 위원회는 올 2월 옴부즈만 설치 1주년을 맞아 옴 부즈 만 기능을 강화하겠다고 하였으나 동 제도는 별도의 법령이 아니라 국무총리 훈령인「금융 규제 운영 규정」의 위임을 받아 금융 위원회 고시인「금융 위원회 옴부즈만 운영 규칙 제정 안」에 따른 자문 기구에 불과하다는 점도 제도적 맹점이다. 전해철 의원은“옴부즈만 제도의 취지와 목적에 따라 금융 위원회가 옴부즈만을 내실 있게 운영하기 위해서는 소비자 보호 제도 개선 기능을 강화할 필요가 있다”며“제도의 실질화 및 독립성 강화, 홍보 확대 등을 위한 실질적인 방안을 모색해야한다”고 밝혔다.
        
        question: 전해철 국회 의원이 언급한 옴부즈만 제도의 주요 목적은 무엇인가?
        
        choices:
        1. 금융 소비자 보호
        2. 금융 회사의 이익 증대
        3. 금융 규제 완화
        4. 소비자 민원 처리
        5. 금융 시장 활성화
        
        answer: 1

        paragraph:
        당신이 정말 부지런하고 똑똑하다면, 그리고 꿈이 있다면 10 대 대기업에는 가지 마라.”대기업 입사를 위해 학점을 높이고 영어 점수를 만들고 공모전에 여념이 없는‘취준생’에게 이 말은 다소 황당하게 들릴지도 모르겠다. 하지만 연 매출 1 조 원이 넘는 세원 그룹의 김문기 회장은“(대기업에선)임원이 부하 직원의 이름도 모르는데 어떻게 제대로 평가하고 보상할 수 있겠는가”라며“(중소기업)조직에서 일에 집중하며 회사를 성장시키면 회사와 함께 나도 성장한다”고 강조한다. 이것이 또 다른 의미의‘창업’이란 설명이다.《제로 플러스》는 현대자동차 품질 평가 팀에서 9년째 부품 협력사를 진단· 평가하는 업무를 맡고 있는 저자가 들려주는 중견· 중소기업인 이야기다.1000명 이상의 중소기업 창업주와 전문 경영인을 만난 저자는 끈기와 치열함, 신뢰와 오너 십, 상생과 나눔 등 자신만의 경쟁력으로 기업을 성장시킨 기업인들의 경영 노하우와 인생관을 책에 담아냈다. 책에는 총 9명의 기업인이 등장한다. 김문기 회장은“화장실이 깨끗해야 회사가 바로 선다”는‘화장실 경영학’과 자발적 리더십으로 기업을 일궜다 .서중호아 진 산업 사장은 모든 직원에게 스포츠카를 사 줄 수 있는 회사를 만들겠다는 목표로 나눔 경영을 실천해 왔고, 강성진 월드 솔루션 사장은 고아· 전과자· 고교 중퇴라는 스펙을 갖고도 오기와 성실을 무기로 경영 신화를 썼다. 양진 석호 원사 장은 야전 침대에서 신발을 신은 채 잠자며 일등 부품 사를 키워 냈고, 고(故)김인찬 신기 인터 모빌 회장은 움직이는 자가 반드시 이긴다는 정신으로 최고의 자동차 내장재 전문 기업을 일궜다. 직접 발로 뛰며 현장에서 문제를 찾아 해결하는 최광 오 대풍 공업 사장, 과감한 연구 개발 투자와 현장 작업 환경 개선을 단행한 정 순백 위너 콤 주식회사 사장, 대기업에서 뛰쳐나와 부지런함을 무기로 건실한 기업을 일군 행복 경영 추구자 김은호 동진 이공 회장, 자신이 할 수 있는 일을 열심히 하다 보면 반드시 기회가 찾아온다는 진성 현명 진 테크 사장의 이야기도 감동적이다. 저자는 기업을 세우고 성공적으로 경영한 이들로부터 공통된 특징을 찾을 수 있다고 말한다. 긍정적인 마음가짐을 갖고 있으며 스펙보다 경험을 중시한다. 일에 대한 집중과 몰입도가 높고 성공을 이룬 뒤에도 초심을 잃지 않는다. 자신만의 이익을 추구하지 않고 직원에게 돌려주고 나아가 국가 발전을 위해 일한다는 마음으로 노력한다. 이들의 또 한 가지 공통점은 중소기업 예찬론. 대기업보다 중소 기업에서 일할 때 더 성장할 수 있다는 것이다 .서중호 사장은“사장이 끌고 직원들이 졸졸 따라 가면 딱 사장 수준만큼만 결과가 나온다”며“직원들이 모두‘오너’와 같은 마음을 가질 때 회사는 사장의 수준을 뛰어넘는 성과를 낼 수 있다”고 강조한다.
        
        question': '김문기 회장이 강조한 중소기업의 장점은 무엇인가?
        
        choices:
        1. 임원이 부하 직원의 이름을 모른다
        2. 조직에서 일에 집중하며 회사와 함께 성장할 수 있다
        3. 대기업보다 높은 연봉을 제공한다
        4. 직원들이 모두 사장과 같은 마음을 가진다
        5. 스펙보다 학력을 중시한다
        
        answer: 2

        paragraph:
        백금이‘귀금속의 제왕’으로 불리는 금의 자리를 꿰차고 있다.13일 월스트리트저널(WSJ)에 따르면 올 들어 세계 경기 회복과 함께 산업용 백금의 수요가 늘고 있고, 최근에는 목걸이, 팔찌 등 장식용 귀금속 시장에서도 빠른 속도로 금을 대체하고 있다. 특히 전 세계에서 두 번째로 금을 많이 소비하는 인도의 백금 수요가 올해는 지난해보다 35 %나 증가한 21 만 온스에 달할 것으로 마케팅 조사 기관인 길드 인터내셔널은 전망했다.WSJ 는 금 일변도 장식에 싫증 난 인도 소비자들의 백금 선호 현상이 급속도로 확산되고 있다고 전했다. 미국과 일본도 경기 회복 조짐과 함께 백금 수요가 올해 각각 11 %와 12 %증가할 것으로 예상했다. 백금은 디젤 자동차의 촉매 변환 장치와 의료용 기기 등에 필수 소재로 사용된다. 백금 가격도 올 들어 4.3 %오르면서 금과의 격차가 확대되고 있다. 톰슨 로이터에 따르면 이날 기준 백금 가격은 온스당 1429달러로 3개월 전과 비교해 18달러 오른 반면 금값은 온스당 1291달러로 최근 3개월간 11달러 하락했다.
        
        question: 
        올해 인도의 백금 수요가 지난해보다 얼마나 증가할 것으로 전망되었는가?
        
        choices: 
        1. 10 %
        2. 20 %
        3. 30 %
        4. 35 %
        5. 50 %
        
        answer: 4

        paragraph:
        일본 유통업체가 자율 계산 점포를 늘리고 있다. 계산원 부족 문제를 해결하고 신속한 계산으로 고객 대기 시간을 줄이기 위해서다. 일본 편의점 업체 패밀리마트가 30일부터 전철역 안에 있는 점포를 중심으로 고객이 직접 계산하는 자율 계산대를 도입한다고 니혼게이자이신문이 29일 보도했다.2017년까지 전체 점포의 10 %에 해당하는 1500개 점포로 확대할 예정이다. 이 신문에 따르면 패밀리마트는 도쿄 닛 포리 역에 있는 점포 등 5개 점포에 자율 계산대를 우선 도입한다. 소비자는 제품 포장재에 인쇄된 바코드를 계산대에 갖다 댄 뒤 금액을 교통 카드 등 전자 머니 카드로 결제하면 된다. 점포 관리를 위해 직원이 배치된 계산대도 일단 유지하기로 했다.1회 구매 한도는 1 만 엔 미만으로 대상 제품은 식품과 일용품, 잡지· 신문 등이다 .구매허가가 필요한 담배와 주류는 살 수 없다. 편의점 업계 2위인 로손도 2010년 자율 계산 점포를 도입해 현재 50개 점포에서 71개 계산대를 운영하고 있다. 철도 회사 JR 동일본 자회사인 JR 동일본 리테일 넷도 자율 계산 점포를 늘리고 있다. 편의점 외 슈퍼마켓 등 다른 유통 업체로 보급이 확대되고 있다고 니혼게이자이신문은 보도했다. 일본 슈퍼마켓 협회 등 3 대 소매 유통 단체에 따르면 51개 이상 점포를 운영하는 슈퍼마켓 중 60 %가량이 자율 계산대를 도입하고 있다. 유통업체는 자율 계산 점포 도입이 인력 부족이나 아르바이트 임금 상승 등의 문제를 해결하는 하나의 방법이 될 것으로 기대한다고 이 신문은 전했다.
        
        question: 일본의 자율 계산 점포 도입을 주도하고 있는 편의점 업체는 무엇인가?
        
        choices: 
        1. 세븐일레븐
        2. 로손
        3. 패밀리마트
        4. 미니스톱
        5. 이온 몰
        
        answer: 
        3

        paragraph: 
        방하남 고용 노동부 장관(사진)이 취임 후 첫 외부 행사로 5년 만에 일자리를 3.5 배 늘린 셀트리온을 방문했다. 셀트리온은 제약과 생명 공학의 융· 복합으로 만들어진 바이오 의약품 기업이다. 박근혜 대통령의‘창조 경제’를 염두에 둔 행보로 보인다. 방 장관은 12일 인천 송도에 있는 셀트리온을 방문, 이 회사 노사와 간담회를 했다. 셀트리온은 세계 2위 수준의 생산 설비를 갖춘 바이오 의약품 분야 선도 기업으로 지난해 25.2 %의 매출 증가율을 기록했다.2008년 260명이던 직원은 지난해 890명으로 늘었고‘2012일 자리 창출 유공 포상’도 받았다. 이 회사 계약 직 사원의 정규직 전환율은 95 %에 이른다. 간담회에서 서정진 셀트리온 대표는“은행 대출이 아니라 투자를 많이 받을 수 있어야 기업이 부담 없이 사업을 확장해 일자리를 늘릴 수 있다”며 투자 여건 개선을 건의했다. 방 장관은 기업의 건의 사항을 메모한 뒤“앞으로 더 많은 현장을 다니며 건의· 애로 사항을 듣고 개선해 좋은 일자리 창출로 이어지도록하겠다”고 말했다. 방 장관은 오후에는 한국노총을 방문해 문 진국 위원장을 만났다. 방 장관은“일자리 문제 해결을 위해서는 노사정 모두의 협력이 필수적”이라며 고용률 70 %달성을 위한 한국노총의 협력을 당부했다. 문 위원장은“중산층 70 %복원, 고용률 70 %달성이라는 정부 목표는 한국노총도 지향하는 방향”이라며 화답했다.
        
        question: 
        방하남 고용 노동부 장관이 방문한 기업의 이름은 무엇인가?
        
        choices: 
        1. 현대자동차
        2. 삼성전자
        3. LG화학
        4. 한화그룹
        5. 셀 트리온
        
        answer: 1





        paragraph:
        {paragraph}


        양식:
        ### question:
        (paragraph와 대한민국 수능 범위를 참고해서 출제한 수능형 문제)

        ### choices:
        (question에 해당하는 정답 1개와 오답 4개 정답의 위치는 랜덤)
        1. :
        2. :
        3. :
        4. :
        5. :
        
        ### Answer:
        (choices중 question에 해당하는 정답 번호)

        """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 또는 사용 가능한 다른 모델
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0  # 항상 동일한 응답을 위해 설정
    )
    return response.choices[0].message.content.strip()


def parse_output(output):
    """Parse OpenAI response into question, choices, and answer."""
    question_start = output.find("### question:") + len("### question:")
    question_end = output.find("### choices:")
    question = output[question_start:question_end].strip()

    choices_start = output.find("### choices:") + len("### choices:")
    choices_end = output.find("### Answer:")
    choices = output[choices_start:choices_end].strip()

    answer_start = output.find("### Answer:") + len("### Answer:")
    answer = output[answer_start:].strip()

    return question, choices, answer


def generate_ag_data(paragraph):
    """Generate augmented data using check_mismatch."""
    output = check_mismatch(paragraph)
    question, choices, answer = parse_output(output)
    return pd.Series([question, choices, answer])


# Augmented 데이터 생성
tqdm.pandas()
ag_result = result[['id', 'paragraph']].copy()
ag_result[['question', 'choices', 'answer']] = ag_result['paragraph'].progress_apply(generate_ag_data)

# DataFrame 병합
ag_result['problems'] = ag_result.apply(
    lambda row: {
        'question': row['question'],
        'choices': row['choices'],
        'answer': row['answer']
    }, axis=1
)

# 'choices' 파싱
def parse_choices(problem):
    if isinstance(problem, dict) and isinstance(problem.get('choices'), str):
        choices = re.split(r'\d+\.\s', problem['choices'])
        problem['choices'] = [c.strip() for c in choices if c.strip()]
    return problem


ag_result['problems'] = ag_result['problems'].apply(parse_choices)
ag_result = ag_result.drop(columns=['question', 'choices', 'answer'])
concat_result = pd.concat([result, ag_result], ignore_index=True)

# 최종 결과 저장
concat_result.to_csv('./data/augmentation.csv', index=False)
