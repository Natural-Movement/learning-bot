"""
Learning Bot
- Streamlit based personal learning app
- Google Sheets storage
- Gemini AI enrichment
- edge-tts voice preview
"""

import asyncio
import json
from datetime import datetime, timedelta
from io import BytesIO

import edge_tts
import google.generativeai as genai
import gspread
import pandas as pd
import plotly.express as px
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials


st.set_page_config(
    page_title="Learning Bot",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
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
    "음성파일경로",
]

DOMAINS = ["국어", "영어", "컴퓨터", "뇌과학", "명상", "기타"]

TTS_VOICES = {
    "한국어 여성 - SunHi": "ko-KR-SunHiNeural",
    "한국어 남성 - InJoon": "ko-KR-InJoonNeural",
}

TTS_RATES = {
    "매우 느리게": "-30%",
    "느리게": "-15%",
    "보통": "+0%",
    "빠르게": "+15%",
    "매우 빠르게": "+30%",
}


def apply_theme(theme: str) -> None:
    if theme == "다크 모드":
        st.markdown(
            """
            <style>
            .stApp {
                background: #111827;
                color: #f9fafb;
            }
            [data-testid="stHeader"],
            [data-testid="stSidebar"] {
                background: #111827;
            }
            [data-testid="stExpander"],
            div[data-testid="stMetric"],
            div[data-testid="stDataFrame"] {
                background: #1f2937;
                border-color: #374151;
                color: #f9fafb;
            }
            .stTextInput input,
            .stTextArea textarea,
            .stSelectbox div[data-baseweb="select"] > div {
                background: #1f2937;
                color: #f9fafb;
                border-color: #4b5563;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                background: #1f2937;
                border-radius: 8px 8px 0 0;
                color: #e5e7eb;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background: #ffffff;
                color: #111827;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


with st.sidebar:
    selected_theme = st.radio("화면 모드", ["라이트 모드", "다크 모드"], horizontal=True)
apply_theme(selected_theme)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


def check_password() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("📚 Learning Bot")
    st.caption("개인 학습 자동화 시스템")

    password = st.text_input("비밀번호", type="password")

    if st.button("입장"):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        st.error("비밀번호가 틀렸습니다.")

    return False


@st.cache_resource
def get_sheet():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    creds_dict = json.loads(GOOGLE_CREDS_JSON) if isinstance(GOOGLE_CREDS_JSON, str) else dict(GOOGLE_CREDS_JSON)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SHEET_ID)

    try:
        sheet = spreadsheet.worksheet("learning_data")
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title="learning_data", rows=1000, cols=len(REQUIRED_COLUMNS))

    if sheet.row_values(1) != REQUIRED_COLUMNS:
        sheet.update("A1:J1", [REQUIRED_COLUMNS])

    return sheet


def load_data() -> pd.DataFrame:
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


def classify_domain(content: str) -> str:
    prompt = f"""
다음 학습 내용이 어느 영역에 속하는지 아래 선택지 중 하나만 답하세요.

선택지: {", ".join(DOMAINS)}

학습 내용:
{content}

답:
"""
    try:
        domain = model.generate_content(prompt).text.strip()
        return domain if domain in DOMAINS else "기타"
    except Exception:
        return "기타"


def enrich_content(domain: str, content: str) -> str:
    prompt = f"""
다음 학습 내용을 음성으로 듣기 좋은 한국어 학습 콘텐츠로 만들어 주세요.

영역: {domain}
학습 내용: {content}

조건:
- 자연스러운 말투로 작성하세요.
- 이모지와 마크다운 기호는 쓰지 마세요.
- 핵심 설명, 쉬운 예시, 마지막 요약을 포함하세요.
- 약 1분 30초에서 2분 분량으로 작성하세요.
"""
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as exc:
        return f"AI 가공 실패: {exc}"


async def make_edge_tts_audio(text: str, voice: str, rate: str) -> bytes:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
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
        return loop.run_until_complete(make_edge_tts_audio(text, voice, rate))
    finally:
        loop.close()


def get_next_review_date(review_count: int) -> str:
    if review_count >= len(REVIEW_INTERVALS):
        return ""
    next_date = datetime.now() + timedelta(days=REVIEW_INTERVALS[review_count])
    return next_date.strftime("%Y-%m-%d")


def play_audio_button(label: str, key: str, text: str, voice: str, rate: str) -> None:
    if st.button(label, key=key):
        try:
            audio = text_to_speech(text, voice, rate)
            st.audio(audio, format="audio/mp3")
        except Exception as exc:
            st.error(f"음성 생성 실패: {exc}")


def main() -> None:
    if not check_password():
        return

    st.title("📚 Learning Bot")

    with st.expander("🔊 음성 설정", expanded=True):
        voice_label = st.selectbox("목소리 선택", list(TTS_VOICES.keys()), index=0)
        rate_label = st.selectbox("읽기 속도", list(TTS_RATES.keys()), index=2)
        selected_voice = TTS_VOICES[voice_label]
        selected_rate = TTS_RATES[rate_label]

    tab1, tab2, tab3, tab4 = st.tabs(["입력", "오늘 복습", "대시보드", "전체"])

    with tab1:
        st.subheader("학습 내용 입력")
        content = st.text_area("학습 내용", placeholder="예: as it were / 함수 / 명상 인사이트", height=120)
        col1, col2 = st.columns([1, 1])
        manual_domain = col1.selectbox("영역", ["자동"] + DOMAINS)
        save_button = col2.button("저장 + AI 가공", type="primary", use_container_width=True)

        if save_button:
            if not content.strip():
                st.warning("학습 내용을 입력하세요.")
            else:
                with st.spinner("AI가 내용을 가공하고 있습니다."):
                    domain = manual_domain if manual_domain != "자동" else classify_domain(content)
                    enriched = enrich_content(domain, content)
                    sheet = get_sheet()
                    new_id = len(sheet.get_all_values())
                    today = datetime.now().strftime("%Y-%m-%d")
                    next_review = get_next_review_date(0)
                    sheet.append_row([new_id, domain, content, "", enriched, today, 0, next_review, "학습중", ""])

                st.success(f"저장 완료 | 영역: {domain} | 다음 복습: {next_review}")
                with st.expander("AI 가공 내용 보기", expanded=True):
                    st.write(enriched)
                with st.spinner("음성을 생성하고 있습니다."):
                    st.audio(text_to_speech(enriched, selected_voice, selected_rate), format="audio/mp3")

    with tab2:
        st.subheader("오늘의 복습")
        df = load_data()

        if df.empty:
            st.info("아직 학습 내용이 없습니다. 입력 탭에서 시작하세요.")
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            today_df = df[(df["다음복습일"].astype(str) == today) & (df["상태"].astype(str) == "학습중")]

            if today_df.empty:
                st.success("오늘 복습할 항목이 없습니다.")
                recent_df = df.tail(5)
                st.subheader("최근 등록 항목")
                for _, row in recent_df.iterrows():
                    with st.expander(f"[{row['영역']}] {row['콘텐츠']}"):
                        st.write(row["AI가공내용"])
                        play_audio_button("음성 듣기", f"recent_play_{row['_row_number']}", str(row["AI가공내용"]), selected_voice, selected_rate)
            else:
                st.write(f"오늘 복습 항목: **{len(today_df)}개**")
                domain_options = ["전체"] + sorted(today_df["영역"].dropna().unique().tolist())
                selected_domain = st.selectbox("영역 필터", domain_options)
                filtered_df = today_df if selected_domain == "전체" else today_df[today_df["영역"] == selected_domain]

                for _, row in filtered_df.iterrows():
                    with st.expander(f"[{row['영역']}] {row['콘텐츠']}"):
                        st.write(row["AI가공내용"])
                        col1, col2 = st.columns(2)
                        with col1:
                            play_audio_button("음성 듣기", f"today_play_{row['_row_number']}", str(row["AI가공내용"]), selected_voice, selected_rate)
                        if col2.button("복습 완료", key=f"today_done_{row['_row_number']}"):
                            sheet = get_sheet()
                            sheet_row = int(row["_row_number"])
                            current_count = int(row["복습횟수"]) if str(row["복습횟수"]).isdigit() else 0
                            new_count = current_count + 1
                            new_next = get_next_review_date(new_count)
                            new_status = "마스터" if not new_next else "학습중"
                            sheet.update_cell(sheet_row, 7, new_count)
                            sheet.update_cell(sheet_row, 8, new_next)
                            sheet.update_cell(sheet_row, 9, new_status)
                            st.success("복습 완료 처리했습니다.")
                            st.rerun()

    with tab3:
        st.subheader("학습 대시보드")
        df = load_data()

        if df.empty:
            st.info("데이터가 없으면 대시보드가 표시되지 않습니다.")
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
            st.plotly_chart(px.pie(domain_counts, values="개수", names="영역", title="영역별 분포", hole=0.4), use_container_width=True)

            df["등록일_dt"] = pd.to_datetime(df["등록일"], errors="coerce")
            daily_df = df.dropna(subset=["등록일_dt"]).groupby(df["등록일_dt"].dt.date).size().reset_index(name="개수")
            if not daily_df.empty:
                daily_df.columns = ["날짜", "개수"]
                st.plotly_chart(px.bar(daily_df, x="날짜", y="개수", title="일별 학습 등록"), use_container_width=True)

    with tab4:
        st.subheader("전체 학습 데이터")
        df = load_data()

        if df.empty:
            st.info("아직 데이터가 없습니다.")
        else:
            keyword = st.text_input("검색", placeholder="검색어를 입력하세요.")
            filtered_df = df.copy()
            if keyword:
                filtered_df = filtered_df[filtered_df["콘텐츠"].astype(str).str.contains(keyword, case=False, na=False)]

            domain_options = ["전체"] + sorted(filtered_df["영역"].dropna().unique().tolist())
            selected_domain = st.selectbox("영역 선택", domain_options, key="all_domain_select")
            if selected_domain != "전체":
                filtered_df = filtered_df[filtered_df["영역"] == selected_domain]

            st.dataframe(
                filtered_df[["영역", "콘텐츠", "등록일", "복습횟수", "다음복습일", "상태"]],
                use_container_width=True,
                height=400,
            )


if __name__ == "__main__":
    main()
