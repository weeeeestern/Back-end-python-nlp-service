# config.py
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# --- TF-IDF 모델 파일 경로 ---
TFIDF_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'tfidf_vectorizer.pkl')

# --- 데이터베이스 연결 설정 (MySQL) ---
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 3306))
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'your_password') # 개발 환경 기본값, 실제로는 .env에서 로드
DB_NAME = os.getenv('DB_NAME', 'your_database') # 개발 환경 기본값, 실제로는 .env에서 로드
