import sys
import os
sys.path.append("/data/ephemeral/pip_packages") # for config
sys.path.append(os.path.join(os.getcwd(),"../../")) # for config
import streamlit as st
import pandas as pd
from ast import literal_eval
from config import HF_CONFIG

# get Data
data_path = os.path.join(os.getcwd(),"../../data")
datas = pd.read_csv(os.path.join(data_path,"train.csv")) # 이곳에 파일명 변경해서 보기

# 현재 선택된 데이터 출력    
# # get Paragraph, Question, Answer
def getDataFromId(id_index):
    rowById = datas.iloc[id_index]
    problems = literal_eval(rowById['problems'])
    
    paragraph = rowById["paragraph"]
    question = problems["question"]
    choices = problems["choices"]
    answer = problems["answer"]

    return paragraph, question, choices, answer

def main():
    st.set_page_config(layout="wide", page_title="Data 확인 데모")
    
    st.title("Data Checker")

    # ID를 문자열로 변환
    options = datas["id"].astype(str).tolist()  # 문자열 리스트로 변환

    # 세션 초기화
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0  # 초기 인덱스 설정

    if "id_selectbox" not in st.session_state:
        st.session_state.id_selectbox = options[0]  # 첫 번째 값 문자열로 설정

    def prev_data():
        st.session_state.current_index = (st.session_state.current_index - 1) % len(options)
    
    def next_data():
        st.session_state.current_index = (st.session_state.current_index + 1) % len(options)
    
    def update_index_by_selectbox():
        # 선택된 ID로 인덱스 찾기
        selected_id = st.session_state.id_selectbox  # 이미 문자열 상태
        st.session_state.current_index = int(datas[datas['id'] == selected_id].index[0])
    
    current_index = st.session_state.current_index

    # 3개의 컬럼으로 분리
    id_box_col, prev_arrow_button, next_arrow_button = st.columns(3)

    # Selectbox로 데이터 선택
    with id_box_col:
        ids = st.selectbox(
            label="-",
            options=options,
            index=current_index,
            key="id_selectbox",
            on_change=update_index_by_selectbox,
            label_visibility="collapsed"
        )

    st.write(f"선택된 ID : {st.session_state.id_selectbox}")
    
    # 이전 버튼
    with prev_arrow_button:
        st.button("PREV", on_click=prev_data, use_container_width=True)
    
    # 다음 버튼
    with next_arrow_button:
        st.button("NEXT", on_click=next_data, use_container_width=True)
    
    paragraph, question, choices, answer = getDataFromId(st.session_state.current_index)

    paragraph = paragraph.replace("\n", "  ")
    # 전체 레이아웃 구성
    paragraph_area, problem_area = st.columns([3, 1])  # 첫 번째 열(지문)이 더 넓게 설정됨

    # 지문 영역
    with paragraph_area:
        st.subheader("지문")
        st.markdown(f"```{paragraph}```")
        
        st.subheader("질문")
        st.markdown("Q : " + question)
        
    # 질문 및 선지 영역
    with problem_area:
        st.subheader("선택지")
        for i, ch in enumerate(choices):
            st.markdown(f"{i+1}. {ch}")

        st.subheader("답")
        st.markdown("Answer : " + str(answer))

    

if __name__=="__main__":
    main()