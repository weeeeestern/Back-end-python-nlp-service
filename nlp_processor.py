# nlp_processor.py
import pickle
import re
import pymysql
from konlpy.tag import Okt # 형태소 분석기 (예시)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import config # 설정 파일 임포트

# --- 전역 변수 (애플리케이션 시작 시 한 번만 로드) ---
okt = None # Okt 형태소 분석기 인스턴스
tfidf_vectorizer = None # 학습된 TF-IDF Vectorizer 모델
stopwords_set = set() # '불용어' 집합 (메모리 로드)

def should_exclude_token(token: str) -> bool:
    # 쉼표 포함된 숫자 → "31,231" 등
    if ',' in token and token.replace(',', '').isdigit():
        return True
    # 순수 숫자 (정수)만 포함된 토큰 → "2024", "518"
    if token.isdigit():
        return True
    return False

# --- 모델 및 데이터 로드 함수 (FastAPI startup 이벤트에서 호출) ---
def load_models_and_data():
    global okt, tfidf_vectorizer, stopwords_set

    try:
        # 1. 형태소 분석기 초기화
        okt = Okt()
        print("Okt 형태소 분석기 초기화 완료.")

        # 2. TF-IDF Vectorizer 모델 로드
        with open(config.TFIDF_MODEL_PATH, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        print(f"TF-IDF Vectorizer 로드 완료: {config.TFIDF_MODEL_PATH}")

        # 3. '불용어'를 데이터베이스에서 로드
        conn = pymysql.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            db=config.DB_NAME,
            charset='utf8mb4'
        )
        cursor = conn.cursor()

        # 'stopwords' 테이블에서 'word' 컬럼의 모든 단어 로드
        cursor.execute("SELECT word FROM stopwords") # 해당 테이블과 컬럼이 DB에 존재해야 합니다.
        stopwords_set = {row[0] for row in cursor.fetchall()}
        print(f"DB에서 불용어 로드 완료. 개수: {len(stopwords_set)}")

        cursor.close()
        conn.close()
        print("DB 리소스 로드 완료.")

    except FileNotFoundError as e:
        print(f"오류: 필수 모델/데이터 파일이 없습니다: {e.filename}. 'data/' 디렉토리에 파일이 있는지 확인하세요.")
        raise RuntimeError(f"필수 파일 누락: {e.filename}") from e
    except pymysql.Error as e:
        print(f"DB 연결 또는 쿼리 오류 발생: {e}. DB 인증 정보와 테이블 이름을 확인하세요.")
        raise RuntimeError(f"DB에서 NLP 리소스 로드 실패: {e}") from e
    except Exception as e:
        print(f"NLP 리소스 로드 중 예상치 못한 오류 발생: {e}")
        raise RuntimeError(f"NLP 리소스 로드 실패: {e}") from e

# --- 텍스트 전처리 공통 함수 ---
def preprocess_text_for_nlp(text: str) -> str:
    """텍스트에서 특수문자를 제거하고 공백을 정규화합니다."""
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 형태소 분석 함수 ---
def analyze_morphs(text: str) -> list[tuple[str, str]]:
    """입력 텍스트를 형태소 분석하고 품사를 태깅합니다."""
    if okt is None:
        raise RuntimeError("형태소 분석기 (Okt)가 초기화되지 않았습니다. load_models_and_data()를 먼저 호출하세요.")

    processed_text = preprocess_text_for_nlp(text)
    if not processed_text:
        return []

    return okt.pos(processed_text, norm=True, stem=True)

# --- TF-IDF 벡터화 로직 ---
def vectorize_tfidf(text: str) -> list[float]:
    """입력 텍스트를 TF-IDF 벡터로 변환합니다."""
    if tfidf_vectorizer is None:
        raise RuntimeError("TF-IDF Vectorizer가 로드되지 않았습니다. load_models_and_data()를 먼저 호출하세요.")

    # 텍스트를 형태소 분석 후 명사만 추출하고 불용어 제거 후 공백으로 조인
    tokens_with_pos = analyze_morphs(text)
    # 명사만 추출하고, 불용어 목록에 없는 단어만 필터링합니다.
    processed_tokens = [
        word for word, pos in tokens_with_pos
        if pos.startswith('N') and word not in stopwords_set and len(word) > 1 and not should_exclude_token(word) # 명사, 불용어, 한 글자 이상, 단순 숫자 아님
    ]

    print(f"[키워드 추출 결과] {processed_tokens}")

    text_for_vectorizer = ' '.join(processed_tokens)

    if not text_for_vectorizer: # 처리할 유효한 토큰이 없으면 빈 벡터 반환
        return []

    # 학습된 TfidfVectorizer를 사용하여 벡터화 수행
    vector = tfidf_vectorizer.transform([text_for_vectorizer])

    # 희소 행렬을 밀집 행렬(dense array)로 변환하고 파이썬 리스트로 반환
    return vector.toarray()[0].tolist()
