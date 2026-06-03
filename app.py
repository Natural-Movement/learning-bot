"""
Learning Bot
- Streamlit 기반 PWA 학습 앱
- Google Sheets 저장
- Gemini AI 가공
- edge-tts 음성 변환
"""

import json
import asyncio
from datetime import datetime, timedelta
from io import BytesIO

import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import google.generativeai as genai
import edge_tts


# ==================== 기본 설정 ====================

st.set_page_config(
    page_title="Learning Bot",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SHEET_ID = st.secrets["SHEET_ID"]
GOOGLE_CREDS_JSON = st.secrets["GOOGLE_CREDS_JSON"]
APP_PASSWORD = st.secrets["APP_PASSWORD"]

REVIEW_INTERVALS = [1, 3, 7, 14, 30]

REQUIRED_COLUMNS = [
    "id",
    "영역",
    "콘텐츠",
    "원본설명",
    "AI가공내용",
    "등록일",
    "복습횟수",
    "다음복습일",
    "상태",
    "음성파일경로"
]

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


# ==================== Gemini 설정 ====================

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


# ==================== 로그인 ====================

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🎓 Learning Bot")
    st.caption("개인 학습 자동화 시스템")

    pw = st.text_input("비밀번호", type="password")

    if st.button("입장"):
        if pw == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")

    return False


# ==================== Google Sheets 연결 ====================

@st.cache_resource
def get_sheet():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    if isinstance(GOOGLE_CREDS_JSON, str):
        creds_dict = json.loads(GOOGLE_CREDS_JSON)
    else:
        creds_dict = dict(GOOGLE_CREDS_JSON)

    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    spreadsheet = client.open_by_key(SHEET_ID)

    try:
        sheet = spreadsheet.worksheet("learning_data")
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(
            title="learning_data",
            rows=1000,
            cols=len(REQUIRED_COLUMNS)
        )

    first_row = sheet.row_values(1)

    if first_row != REQUIRED_COLUMNS:
        sheet.update("A1:J1", [REQUIRED_COLUMNS])

    return sheet


def load_data():
    sheet = get_sheet()
    records = sheet.get_all_records()

    if not records:
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        df["_row_number"] = []
        return df

    df = pd.DataFrame(records)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[REQUIRED_COLUMNS]
    df["_row_number"] = range(2, len(df) + 2)

    return df


# ==================== AI 처리 ====================

def classify_domain(content: str) -> str:
    prompt = f"""
다음 학습 내용이 어느 영역에 속하는지 한 단어로만 답하세요.

선택지:
국어, 영어, 컴퓨터, 뇌과학, 명상, 기타

학습 내용:
{content}

답변:
"""

    try:
        response = model.generate_content(prompt)
        domain = response.text.strip()

        if domain in ["국어", "영어", "컴퓨터", "뇌과학", "명상", "기타"]:
            return domain

        return "기타"

    except Exception:
        return "기타"


def enrich_content(domain: str, content: str) -> str:
    prompts = {
        "국어": f"""
다음 한국어 단어 또는 표현을 학습용 음성 콘텐츠로 만들어주세요.

구성:
1. 단어를 또박또박 제시
2. 뜻을 쉽게 설명
3. 한자어라면 한자 풀이
4. 비슷한 표현 1~2개
5. 일상 예문 2개
6. 마지막에 단어를 다시 한 번 반복

조건:
- 음성으로 들을 내용입니다.
- 자연스러운 말투로 작성하세요.
- 이모지와 마크다운 기호는 쓰지 마세요.
- 약 1분 30초 분량으로 작성하세요.

단어 또는 표현:
{content}
""",
        "영어": f"""
다음 영어 표현을 학습용 음성 콘텐츠로 만들어주세요.

구성:
1. 영어 표현 제시
2. 뜻과 뉘앙스 설명
3. 격식도 설명: 캐주얼, 중립, 격식 중 어디에 가까운지
4. 비슷한 표현과의 차이
5. 실제 대화 예문 2개: 영어 문장과 한국어 해석
6. 마지막에 표현을 다시 한 번 반복

조건:
- 음성으로 들을 내용입니다.
- 자연스러운 한국어 설명으로 작성하세요.
- 이모지와 마크다운 기호는 쓰지 마세요.
- 약 1분 30초 분량으로 작성하세요.

영어 표현:
{content}
""",
        "컴퓨터": f"""
다음 컴퓨터 또는 프로그래밍 개념을 학습용 음성 콘텐츠로 만들어주세요.

구성:
1. 한 줄 정의
2. 왜 필요한지
3. 실생활 비유
4. 핵심 동작 원리 3단계
5. 마지막 한 줄 요약

조건:
- 음성으로 들을 내용입니다.
- 자연스러운 강의 말투로 작성하세요.
- 이모지와 마크다운 기호는 쓰지 마세요.
- 약 2분 분량으로 작성하세요.

개념:
{content}
""",
        "뇌과학": f"""
다음 뇌과학 개념 또는 문장을 학습용 음성 콘텐츠로 만들어주세요.

구성:
1. 핵심 개념 정의
2. 관련 뇌 부위 또는 작동 메커니즘
3. 일상생활에서의 의미
4. 비판적 관점 또는 보충 설명
5. 마지막 한 줄 정리

조건:
- 음성으로 들을 내용입니다.
- 자연스러운 강의 말투로 작성하세요.
- 이모지와 마크다운 기호는 쓰지 마세요.
- 약 2분 분량으로 작성하세요.

내용:
{content}
""",
        "명상": f"""
다음 명상 경험 또는 인사이트에 대해 학습용 피드백 음성 콘텐츠를 만들어주세요.

구성:
1. 경험의 의미 인정
2. 심리학적 또는 뇌과학적 해석
3. 다음 명상에서 시도해볼 점
4. 짧은 격려

조건:
- 음성으로 들을 내용입니다.
- 차분하고 따뜻한 말투로 작성하세요.
- 이모지와 마크다운 기호는 쓰지 마세요.
- 약 1분 30초 분량으로 작성하세요.

내용:
{content}
""",
        "기타": f"""
다음 학습 내용을 음성 콘텐츠로 풍부하게 만들어주세요.

구성:
1. 핵심 개념 설명
2. 중요한 이유
3. 일상이나 학습에 적용하는 방법
4. 마지막 한 줄 요약

조건:
- 음성으로 들을 내용입니다.
- 자연스러운 말투로 작성하세요.
- 이모지와 마크다운 기호는 쓰지 마세요.
- 약 1분 30초 분량으로 작성하세요.

내용:
{content}
"""
    }

    prompt = prompts.get(domain, prompts["기타"])

    try:
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"AI 가공 실패: {e}"


# ==================== TTS 처리 ====================

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
    loop = asyncio.new_event_loop()

    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            make_edge_tts_audio(text, voice, rate)
        )

    finally:
        loop.close()


# ==================== 복습일 계산 ====================

def get_next_review_date(review_count: int) -> str:
    if review_count >= len(REVIEW_INTERVALS):
        return ""

    days = REVIEW_INTERVALS[review_count]
    next_date = datetime.now() + timedelta(days=days)

    return next_date.strftime("%Y-%m-%d")


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

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 입력", "🎧 오늘 복습", "📊 대시보드", "📚 전체"]
    )

    # ==================== 입력 탭 ====================

    with tab1:
        st.subheader("새 학습 내용 입력")

        content = st.text_area(
            "학습 내용",
            placeholder="예: 사필귀정 / as it were / 재귀함수 / 신경가소성 / 오늘 명상 인사이트",
            height=120
        )

        col1, col2 = st.columns([1, 1])

        manual_domain = col1.selectbox(
            "영역",
            ["자동", "국어", "영어", "컴퓨터", "뇌과학", "명상", "기타"]
        )

        save_button = col2.button(
            "💾 저장 + AI 가공",
            type="primary",
            use_container_width=True
        )

        if save_button:
            if not content.strip():
                st.warning("학습 내용을 입력하세요.")

            else:
                with st.spinner("AI가 내용을 가공 중입니다. 잠시만 기다려주세요."):
                    domain = manual_domain if manual_domain != "자동" else classify_domain(content)
                    enriched = enrich_content(domain, content)

                    sheet = get_sheet()
                    all_values = sheet.get_all_values()
                    new_id = len(all_values)

                    today = datetime.now().strftime("%Y-%m-%d")
                    next_review = get_next_review_date(0)

                    new_row = [
                        new_id,
                        domain,
                        content,
                        "",
                        enriched,
                        today,
                        0,
                        next_review,
                        "학습중",
                        ""
                    ]

                    sheet.append_row(new_row)

                st.success(f"저장 완료 | 영역: {domain} | 다음 복습: {next_review}")

                with st.expander("📖 AI 가공 내용 보기", expanded=True):
                    st.write(enriched)

                st.subheader("🔊 음성 미리듣기")

                with st.spinner("음성을 생성 중입니다."):
                    try:
                        audio = text_to_speech(
                            enriched,
                            selected_voice,
                            selected_rate
                        )
                        st.audio(audio, format="audio/mp3")

                    except Exception as e:
                        st.error(f"음성 생성 실패: {e}")

    # ==================== 오늘 복습 탭 ====================

    with tab2:
        st.subheader("🎧 오늘의 복습")

        df = load_data()

        if df.empty:
            st.info("아직 학습 내용이 없습니다. 입력 탭에서 시작하세요.")

        else:
            today = datetime.now().strftime("%Y-%m-%d")

            today_df = df[
                (df["다음복습일"].astype(str) == today)
                & (df["상태"].astype(str) == "학습중")
            ]

            if today_df.empty:
                st.success("오늘 복습할 항목이 없습니다.")

                st.subheader("최근 등록 항목")

                recent_df = df.tail(5)

                for _, row in recent_df.iterrows():
                    title = f"[{row['영역']}] {row['콘텐츠']}"

                    with st.expander(title):
                        st.write(row["AI가공내용"])

                        if st.button("🔊 음성 듣기", key=f"recent_play_{row['_row_number']}"):
                            try:
                                audio = text_to_speech(
                                    str(row["AI가공내용"]),
                                    selected_voice,
                                    selected_rate
                                )
                                st.audio(audio, format="audio/mp3")

                            except Exception as e:
                                st.error(f"음성 생성 실패: {e}")

            else:
                st.write(f"오늘 복습 대상: **{len(today_df)}개**")

                domain_options = ["전체"] + sorted(today_df["영역"].dropna().unique().tolist())

                selected_domain = st.selectbox(
                    "영역 필터",
                    domain_options
                )

                if selected_domain == "전체":
                    filtered_df = today_df
                else:
                    filtered_df = today_df[today_df["영역"] == selected_domain]

                for _, row in filtered_df.iterrows():
                    title = f"[{row['영역']}] {row['콘텐츠']}"

                    with st.expander(title):
                        st.write(row["AI가공내용"])

                        col1, col2 = st.columns(2)

                        if col1.button("🔊 음성 듣기", key=f"today_play_{row['_row_number']}"):
                            try:
                                audio = text_to_speech(
                                    str(row["AI가공내용"]),
                                    selected_voice,
                                    selected_rate
                                )
                                st.audio(audio, format="audio/mp3")

                            except Exception as e:
                                st.error(f"음성 생성 실패: {e}")

                        if col2.button("✅ 복습 완료", key=f"today_done_{row['_row_number']}"):
                            sheet = get_sheet()
                            sheet_row = int(row["_row_number"])

                            current_count = int(row["복습횟수"]) if str(row["복습횟수"]).isdigit() else 0
                            new_count = current_count + 1
                            new_next = get_next_review_date(new_count)
                            new_status = "마스터" if not new_next else "학습중"

                            sheet.update_cell(sheet_row, 7, new_count)
                            sheet.update_cell(sheet_row, 8, new_next)
                            sheet.update_cell(sheet_row, 9, new_status)

                            st.success("복습 완료 처리되었습니다.")
                            st.rerun()

    # ==================== 대시보드 탭 ====================

    with tab3:
        st.subheader("📊 학습 대시보드")

        df = load_data()

        if df.empty:
            st.info("데이터가 쌓이면 대시보드가 표시됩니다.")

        else:
            col1, col2, col3 = st.columns(3)

            total_count = len(df)
            learning_count = (df["상태"].astype(str) == "학습중").sum()
            master_count = (df["상태"].astype(str) == "마스터").sum()

            col1.metric("전체", f"{total_count}개")
            col2.metric("학습 중", f"{learning_count}개")
            col3.metric("마스터", f"{master_count}개")

            st.divider()

            domain_counts = df["영역"].value_counts().reset_index()
            domain_counts.columns = ["영역", "개수"]

            fig = px.pie(
                domain_counts,
                values="개수",
                names="영역",
                title="영역별 분포",
                hole=0.4
            )

            st.plotly_chart(fig, use_container_width=True)

            df["등록일_dt"] = pd.to_datetime(df["등록일"], errors="coerce")
            daily_df = (
                df.dropna(subset=["등록일_dt"])
                .groupby(df["등록일_dt"].dt.date)
                .size()
                .reset_index(name="개수")
            )

            if not daily_df.empty:
                daily_df.columns = ["날짜", "개수"]

                fig2 = px.bar(
                    daily_df,
                    x="날짜",
                    y="개수",
                    title="일별 학습 등록량"
                )

                st.plotly_chart(fig2, use_container_width=True)

    # ==================== 전체 데이터 탭 ====================

    with tab4:
        st.subheader("📚 전체 학습 데이터")

        df = load_data()

        if df.empty:
            st.info("아직 데이터가 없습니다.")

        else:
            keyword = st.text_input(
                "검색",
                placeholder="검색어를 입력하세요."
            )

            filtered_df = df.copy()

            if keyword:
                filtered_df = filtered_df[
                    filtered_df["콘텐츠"].astype(str).str.contains(
                        keyword,
                        case=False,
                        na=False
                    )
                ]

            domain_options = ["전체"] + sorted(filtered_df["영역"].dropna().unique().tolist())

            selected_domain = st.selectbox(
                "영역 선택",
                domain_options,
                key="all_domain_select"
            )

            if selected_domain != "전체":
                filtered_df = filtered_df[filtered_df["영역"] == selected_domain]

            st.dataframe(
                filtered_df[
                    ["영역", "콘텐츠", "등록일", "복습횟수", "다음복습일", "상태"]
                ],
                use_container_width=True,
                height=400
            )


if __name__ == "__main__":
    main()
