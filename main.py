"""
장원덕 학습 자동화 봇
- 텔레그램으로 학습 내용 입력
- Gemini AI가 가공
- gTTS로 음성 변환
- Google Drive에 저장
- 매일 아침 복습 음성 자동 전송
"""

import os
import json
import logging
from datetime import datetime, timedelta
from io import BytesIO

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import google.generativeai as genai
from gtts import gTTS
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# 로깅 설정
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 환경변수 로드
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
MY_TELEGRAM_ID = int(os.environ['MY_TELEGRAM_ID'])
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
SHEET_ID = os.environ['SHEET_ID']
GOOGLE_CREDS_JSON = os.environ['GOOGLE_CREDS_JSON']

# 간격 반복 스케줄 (일 단위)
REVIEW_INTERVALS = [1, 3, 7, 14, 30]

# Gemini 초기화
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Google Sheets 초기화
def get_sheet():
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_ID).worksheet('learning_data')


# 영역 자동 분류 (간단 버전)
def classify_domain(content: str) -> str:
    prompt = f"""
다음 학습 내용이 어느 영역에 속하는지 한 단어로만 답하세요.
선택지: 국어, 영어, 컴퓨터, 뇌과학, 명상, 기타

학습 내용: {content}

답변 (한 단어):
"""
    try:
        response = model.generate_content(prompt)
        domain = response.text.strip()
        if domain in ['국어', '영어', '컴퓨터', '뇌과학', '명상']:
            return domain
        return '기타'
    except Exception as e:
        logger.error(f"분류 실패: {e}")
        return '기타'


# AI 가공: 학습 콘텐츠를 풍부하게 만들기
def enrich_content(domain: str, content: str) -> str:
    prompts = {
        '국어': f"""
다음 한국어 단어/표현을 학습용 음성 콘텐츠로 만들어주세요.
형식:
1. 단어를 또박또박 발음
2. 뜻 설명 (1-2문장)
3. 한자어라면 한자 풀이
4. 비슷한 표현 1-2개
5. 일상 예문 2개
6. 마지막에 단어 한 번 더 반복

음성으로 들을 거니까 자연스러운 말투로, 약 1분 30초 분량으로.
이모지나 마크다운 기호 사용 금지. 순수 텍스트만.

단어: {content}
""",
        '영어': f"""
다음 영어 표현을 학습용 음성 콘텐츠로 만들어주세요.
형식:
1. 표현 발음 (영어 그대로)
2. 의미와 뉘앙스 설명 (한국어)
3. 격식도 레벨 (캐주얼/중립/격식)
4. 비슷한 표현과의 차이
5. 실제 대화 예문 2개 (영어 + 한국어 해석)
6. 마지막에 표현 한 번 더

음성용 자연스러운 한국어 말투. 약 1분 30초 분량.
이모지나 마크다운 금지.

표현: {content}
""",
        '컴퓨터': f"""
다음 컴퓨터/프로그래밍 개념을 학습용 음성으로 만들어주세요.
형식:
1. 개념 한 줄 정의
2. 왜 필요한지 (실무적 이유)
3. 실생활 비유로 설명
4. 핵심 동작 원리 (3단계)
5. 마지막에 한 줄 요약

음성용 자연스러운 말투. 약 2분 분량.
이모지나 마크다운 금지.

개념: {content}
""",
        '뇌과학': f"""
다음 뇌과학 개념/문장을 학습용 음성으로 만들어주세요.
형식:
1. 핵심 개념 정의
2. 관련 뇌 부위/메커니즘
3. 일상에서의 의미
4. 비판적 관점 또는 보충 의견
5. 마지막에 한 줄로 정리

음성용 자연스러운 강의 톤. 약 2분 분량.
이모지나 마크다운 금지.

내용: {content}
""",
        '명상': f"""
다음 명상 경험/인사이트에 대한 피드백 음성을 만들어주세요.
형식:
1. 이 경험의 가치 인정
2. 뇌과학적/심리학적 해석
3. 다음 명상에서 시도해볼 것
4. 격려 한마디

차분하고 따뜻한 톤. 약 1분 30초.
이모지나 마크다운 금지.

내용: {content}
""",
        '기타': f"""
다음 학습 내용을 음성 콘텐츠로 풍부하게 만들어주세요.
- 핵심 개념 설명
- 중요한 이유
- 일상에 적용하는 방법
- 한 줄 요약

자연스러운 말투. 약 1분 30초.
이모지나 마크다운 금지.

내용: {content}
"""
    }
    
    prompt = prompts.get(domain, prompts['기타'])
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"AI 가공 실패: {e}")
        return content


# 다음 복습일 계산
def get_next_review_date(review_count: int) -> str:
    if review_count >= len(REVIEW_INTERVALS):
        return ''  # 마스터 완료
    days = REVIEW_INTERVALS[review_count]
    next_date = datetime.now() + timedelta(days=days)
    return next_date.strftime('%Y-%m-%d')


# 텔레그램 핸들러: /start
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MY_TELEGRAM_ID:
        await update.message.reply_text("권한이 없습니다.")
        return
    
    msg = """🎓 장원덕 학습봇입니다.

학습하고 싶은 내용을 그냥 메시지로 보내주세요.
- 단어, 표현, 개념, 명상 인사이트 무엇이든 OK
- AI가 영역을 자동 분류하고 풍부한 학습 콘텐츠로 가공해서 저장합니다.
- 매일 아침 7시에 오늘의 복습 음성을 보내드려요.

명령어:
/list - 오늘 복습 대상 보기
/stats - 전체 통계
/help - 도움말"""
    await update.message.reply_text(msg)


# 텔레그램 핸들러: 일반 메시지 (학습 내용 입력)
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MY_TELEGRAM_ID:
        return
    
    content = update.message.text.strip()
    if not content:
        return
    
    await update.message.reply_text("⏳ AI가 가공 중입니다... (10-20초)")
    
    try:
        # 1. 영역 자동 분류
        domain = classify_domain(content)
        
        # 2. AI 가공
        enriched = enrich_content(domain, content)
        
        # 3. Google Sheets에 저장
        sheet = get_sheet()
        all_data = sheet.get_all_values()
        new_id = len(all_data)  # 헤더 제외 다음 행 번호
        
        today = datetime.now().strftime('%Y-%m-%d')
        next_review = get_next_review_date(0)
        
        new_row = [
            new_id,                  # id
            domain,                  # 영역
            content,                 # 콘텐츠
            '',                      # 원본설명 (지금은 비움)
            enriched,                # AI가공내용
            today,                   # 등록일
            0,                       # 복습횟수
            next_review,             # 다음복습일
            '학습중',                # 상태
            ''                       # 음성파일경로 (나중에 채움)
        ]
        sheet.append_row(new_row)
        
        # 4. 즉시 음성 미리보기 생성
        tts = gTTS(text=enriched, lang='ko', slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # 5. 텔레그램으로 응답
        reply = f"""✅ 저장 완료
영역: {domain}
다음 복습: {next_review}

[AI 가공 미리보기]
{enriched[:300]}{'...' if len(enriched) > 300 else ''}"""
        await update.message.reply_text(reply)
        await update.message.reply_voice(voice=audio_buffer)
        
    except Exception as e:
        logger.error(f"처리 실패: {e}")
        await update.message.reply_text(f"❌ 오류: {str(e)[:200]}")


# 텔레그램 핸들러: /list (오늘 복습 대상)
async def list_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MY_TELEGRAM_ID:
        return
    
    try:
        sheet = get_sheet()
        all_data = sheet.get_all_records()
        today = datetime.now().strftime('%Y-%m-%d')
        
        today_items = [r for r in all_data if r.get('다음복습일') == today and r.get('상태') == '학습중']
        
        if not today_items:
            await update.message.reply_text("오늘 복습 대상이 없습니다. 😊")
            return
        
        msg = f"📚 오늘의 복습 대상 ({len(today_items)}개)\n\n"
        for i, item in enumerate(today_items, 1):
            msg += f"{i}. [{item['영역']}] {item['콘텐츠']}\n"
        
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"오류: {str(e)[:200]}")


# 텔레그램 핸들러: /stats
async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != MY_TELEGRAM_ID:
        return
    
    try:
        sheet = get_sheet()
        all_data = sheet.get_all_records()
        
        total = len(all_data)
        learning = sum(1 for r in all_data if r.get('상태') == '학습중')
        mastered = sum(1 for r in all_data if r.get('상태') == '마스터')
        
        domains = {}
        for r in all_data:
            d = r.get('영역', '기타')
            domains[d] = domains.get(d, 0) + 1
        
        msg = f"""📊 학습 통계

전체: {total}개
학습 중: {learning}개
마스터 완료: {mastered}개

영역별:
"""
        for d, c in sorted(domains.items(), key=lambda x: -x[1]):
            msg += f"  · {d}: {c}개\n"
        
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"오류: {str(e)[:200]}")


# 메인 실행
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler('start', start_handler))
    application.add_handler(CommandHandler('help', start_handler))
    application.add_handler(CommandHandler('list', list_handler))
    application.add_handler(CommandHandler('stats', stats_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    logger.info("봇 시작!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
