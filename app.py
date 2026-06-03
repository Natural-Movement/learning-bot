"""
장원덕 학습 자동화 앱 (Streamlit PWA)
"""
import os
import json
from datetime import datetime, timedelta
from io import BytesIO

import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import google.generativeai as genai
import asyncio
import edge_tts

# ==================== 설정 ====================
st.set_page_config(
    page_title="Learning Bot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 비밀키 로드 (Streamlit Cloud의 Secrets 사용)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SHEET_ID = st.secrets["SHEET_ID"]
GOOGLE_CREDS_JSON = st.secrets["GOOGLE_CREDS_JSON"]
APP_PASSWORD = st.secrets["APP_PASSWORD"]

REVIEW_INTERVALS = [1, 3, 7, 14, 30]
TTS_VOICES = {
    "한국어 여성 - SunHi": "ko-KR-SunHiNeural",
    "한국어 남성 - InJoon": "ko-KR-InJoonNeural",
    "영어 여성 - Jenny": "en-US-JennyNeural",
    "영어 남성 - Guy": "en-US-GuyNeural",
    "일본어 여성 - Nanami": "ja-JP-NanamiNeural",
    "중국어 여성 - Xiaoxiao": "zh-CN-XiaoxiaoNeural"
}

TTS_RATES = {
    "매우 느리게": "-30%",
    "느리게": "-15%",
    "보통": "+0%",
    "빠르게": "+15%",
    "매우 빠르게": "+30%"
}

# Gemini 초기화
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# ==================== 인증 ====================
def check_password():
    """앱 접근 비밀번호 (간단한 보호)"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    st.title("🎓 Learning Bot")
    pw = st.text_input("비밀번호", type="password")
    if st.button("입장"):
        if pw == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다")
    return False

# ==================== Google Sheets ====================
@st.cache_resource
def get_sheet():
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_ID).worksheet('learning_data')

def load_data():
    sheet = get_sheet()
    records = sheet.get_all_records()
    if not records:
        return pd.DataFrame(columns=[
            'id', '영역', '콘텐츠', '원본설명', 'AI가공내용',
            '등록일', '복습횟수', '다음복습일', '상태', '음성파일경로'
        ])
    return pd.DataFrame(records)

# ==================== AI 처리 ====================
def classify_domain(content: str) -> str:
    prompt = f"""다음 학습 내용이 어느 영역에 속하는지 한 단어로만 답하세요.
선택지: 국어, 영어, 컴퓨터, 뇌과학, 명상, 기타
학습 내용: {content}
답변:"""
    try:
        response = model.generate_content(prompt)
        domain = response.text.strip()
        if domain in ['국어', '영어', '컴퓨터', '뇌과학', '명상']:
            return domain
        return '기타'
    except:
        return '기타'

def enrich_content(domain: str, content: str) -> str:
    prompts = {
        '국어': f"""다음 한국어 단어/표현을 학습용 음성 콘텐츠로 만들어주세요.
1) 단어를 또박또박 발음 2) 뜻 설명 3) 한자 풀이 (해당 시) 
4) 비슷한 표현 1-2개 5) 일상 예문 2개 6) 마지막에 단어 한 번 더
음성용 자연스러운 말투. 약 1분 30초. 이모지/마크다운 금지.
단어: {content}""",
        '영어': f"""다음 영어 표현을 학습용 음성 콘텐츠로 만들어주세요.
1) 표현 발음 2) 의미와 뉘앙스 3) 격식도 4) 비슷한 표현과의 차이
5) 대화 예문 2개 (영+한) 6) 마지막에 표현 한 번 더
음성용 자연스러운 한국어 말투. 약 1분 30초. 이모지/마크다운 금지.
표현: {content}""",
        '컴퓨터': f"""다음 컴퓨터/프로그래밍 개념을 학습용 음성으로 만들어주세요.
1) 한 줄 정의 2) 왜 필요한지 3) 실생활 비유 4) 동작 원리 3단계 5) 한 줄 요약
음성용 자연스러운 말투. 약 2분. 이모지/마크다운 금지.
개념: {content}""",
        '뇌과학': f"""다음 뇌과학 개념을 학습용 음성으로 만들어주세요.
1) 핵심 정의 2) 관련 뇌 부위/메커니즘 3) 일상에서의 의미 
4) 비판적 관점 5) 한 줄 정리
음성용 강의 톤. 약 2분. 이모지/마크다운 금지.
내용: {content}""",
        '명상': f"""다음 명상 경험에 대한 피드백 음성을 만들어주세요.
1) 경험의 가치 인정 2) 뇌과학적/심리학적 해석 
3) 다음에 시도할 것 4) 격려
차분하고 따뜻한 톤. 약 1분 30초. 이모지/마크다운 금지.
내용: {content}""",
        '기타': f"""다음 학습 내용을 음성 콘텐츠로 풍부하게 만들어주세요.
핵심 개념, 중요한 이유, 적용 방법, 한 줄 요약.
자연스러운 말투. 약 1분 30초. 이모지/마크다운 금지.
내용: {content}"""
    }
    try:
        response = model.generate_content(prompts.get(domain, prompts['기타']))
        return response.text.strip()
    except Exception as e:
        return f"AI 가공 실패: {e}"

async def make_edge_tts_audio(text: str, voice: str, rate: str) -> bytes:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate
    )
    
    audio_buffer = BytesIO()
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
    
    audio_buffer.seek(0)
    return audio_buffer.read()


def text_to_speech(text: str, voice: str, rate: str) -> bytes:
    return asyncio.run(make_edge_tts_audio(text, voice, rate))

def get_next_review_date(review_count: int) -> str:
    if review_count >= len(REVIEW_INTERVALS):
        return ''
    days = REVIEW_INTERVALS[review_count]
    return (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')

# ==================== 메인 앱 ====================
def main():
    if not check_password():
        return
    
    st.title("🎓 Learning Bot")

    with st.expander("🔊 음성 설정", expanded=True):
    voice_label = st.selectbox(
        "목소리 선택",
        list(TTS_VOICES.keys()),
        index=0
    )
    
    rate_label = st.selectbox(
        "읽기 속도",
        list(TTS_RATES.keys()),
        index=2
    )
    
    selected_voice = TTS_VOICES[voice_label]
    selected_rate = TTS_RATES[rate_label]
    
    # 탭 구조
    tab1, tab2, tab3, tab4 = st.tabs(["📝 입력", "🎧 오늘 복습", "📊 대시보드", "📚 전체"])
    
    # --- 탭1: 입력 ---
    with tab1:
        st.subheader("새 학습 내용 입력")
        
        content = st.text_area(
            "학습 내용",
            placeholder="예: 사필귀정 / as it were / 재귀함수 / 신경가소성 / 오늘 명상 인사이트",
            height=100
        )
        
        col1, col2 = st.columns([1, 1])
        manual_domain = col1.selectbox(
            "영역 (자동 분류 원하면 '자동')",
            ['자동', '국어', '영어', '컴퓨터', '뇌과학', '명상', '기타']
        )
        
        if col2.button("💾 저장 + AI 가공", type="primary", use_container_width=True):
            if not content.strip():
                st.warning("내용을 입력하세요")
            else:
                with st.spinner("AI가 가공 중... (10-20초)"):
                    domain = manual_domain if manual_domain != '자동' else classify_domain(content)
                    enriched = enrich_content(domain, content)
                    
                    sheet = get_sheet()
                    all_data = sheet.get_all_values()
                    new_id = len(all_data)
                    today = datetime.now().strftime('%Y-%m-%d')
                    next_review = get_next_review_date(0)
                    
                    sheet.append_row([
                        new_id, domain, content, '', enriched,
                        today, 0, next_review, '학습중', ''
                    ])
                
                st.success(f"✅ 저장 완료 | 영역: {domain} | 다음 복습: {next_review}")
                
                with st.expander("📖 AI 가공 내용 보기", expanded=True):
                    st.write(enriched)
                
                st.subheader("🔊 음성 미리듣기")
                with st.spinner("음성 생성 중..."):
                    audio = text_to_speech(enriched, selected_voice, selected_rate)
                st.audio(audio, format='audio/mp3')
                st.cache_resource.clear()  # 데이터 새로고침
    
    # --- 탭2: 오늘 복습 ---
    with tab2:
        st.subheader("🎧 오늘의 복습")
        df = load_data()
        
        if df.empty:
            st.info("아직 학습 내용이 없습니다. '입력' 탭에서 시작하세요!")
        else:
            today = datetime.now().strftime('%Y-%m-%d')
            today_df = df[(df['다음복습일'] == today) & (df['상태'] == '학습중')]
            
            if today_df.empty:
                st.success("✨ 오늘 복습할 항목이 없습니다!")
                # 가장 최근 등록 항목 보여주기
                st.subheader("📝 최근 등록 항목")
                recent = df.tail(3)
                for _, row in recent.iterrows():
                    with st.expander(f"[{row['영역']}] {row['콘텐츠']}"):
                        st.write(row['AI가공내용'])
                        if st.button(f"🔊 음성 듣기", key=f"recent_{row['id']}"):
                            audio = text_to_speech(row['AI가공내용'], selected_voice, selected_rate)
                            st.audio(audio, format='audio/mp3')
            else:
                st.write(f"오늘 복습 대상: **{len(today_df)}개**")
                
                # 영역 필터
                domains = ['전체'] + sorted(today_df['영역'].unique().tolist())
                selected = st.selectbox("영역 필터", domains)
                
                filtered = today_df if selected == '전체' else today_df[today_df['영역'] == selected]
                
                for _, row in filtered.iterrows():
                    with st.expander(f"[{row['영역']}] {row['콘텐츠']}"):
                        st.write(row['AI가공내용'])
                        col1, col2 = st.columns(2)
                        if col1.button(f"🔊 음성 듣기", key=f"play_{row['id']}"):
                            audio = text_to_speech(row['AI가공내용'])
                            st.audio(audio, format='audio/mp3')
                        if col2.button(f"✅ 복습 완료", key=f"done_{row['id']}"):
                            sheet = get_sheet()
                            row_idx = int(row['id']) + 1  # 헤더 1행
                            new_count = int(row['복습횟수']) + 1
                            new_next = get_next_review_date(new_count)
                            new_status = '마스터' if not new_next else '학습중'
                            sheet.update_cell(row_idx + 1, 7, new_count)  # 복습횟수
                            sheet.update_cell(row_idx + 1, 8, new_next)   # 다음복습일
                            sheet.update_cell(row_idx + 1, 9, new_status) # 상태
                            st.success("복습 완료 처리됨!")
                            st.cache_resource.clear()
                            st.rerun()
    
    # --- 탭3: 대시보드 ---
    with tab3:
        st.subheader("📊 학습 대시보드")
        df = load_data()
        
        if df.empty:
            st.info("데이터가 쌓이면 차트가 표시됩니다")
        else:
            # 핵심 지표
            col1, col2, col3 = st.columns(3)
            col1.metric("전체", f"{len(df)}개")
            col2.metric("학습 중", f"{(df['상태'] == '학습중').sum()}개")
            col3.metric("마스터", f"{(df['상태'] == '마스터').sum()}개")
            
            # 영역별 차트
            domain_counts = df['영역'].value_counts().reset_index()
            domain_counts.columns = ['영역', '개수']
            fig = px.pie(domain_counts, values='개수', names='영역', 
                         title='영역별 분포', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
            # 일별 등록 추이
            df['등록일_dt'] = pd.to_datetime(df['등록일'])
            daily = df.groupby(df['등록일_dt'].dt.date).size().reset_index(name='개수')
            daily.columns = ['날짜', '개수']
            fig2 = px.bar(daily, x='날짜', y='개수', title='일별 학습 등록량')
            st.plotly_chart(fig2, use_container_width=True)
    
    # --- 탭4: 전체 데이터 ---
    with tab4:
        st.subheader("📚 전체 학습 데이터")
        df = load_data()
        if df.empty:
            st.info("아직 데이터가 없습니다")
        else:
            # 검색
            keyword = st.text_input("🔍 검색", placeholder="콘텐츠 검색")
            if keyword:
                df = df[df['콘텐츠'].str.contains(keyword, case=False, na=False)]
            
            # 영역 필터
            domains = ['전체'] + sorted(df['영역'].unique().tolist())
            selected = st.selectbox("영역", domains, key='all_domain')
            if selected != '전체':
                df = df[df['영역'] == selected]
            
            st.dataframe(
                df[['영역', '콘텐츠', '등록일', '복습횟수', '다음복습일', '상태']],
                use_container_width=True,
                height=400
            )

if __name__ == '__main__':
    main()
