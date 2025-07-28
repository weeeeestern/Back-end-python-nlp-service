# tokenizer_utils.py
from konlpy.tag import Okt
import re
import os
import pymysql

okt = Okt()

def load_stopwords_from_db():
    stopwords = set()
    try:
        conn = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', 'your_password'),
            db=os.getenv('DB_NAME', 'your_database'),
            charset='utf8mb4'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT word FROM stopwords")
        stopwords = {row[0] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"불용어 로드 실패: {e}")
    return stopwords

stopwords_set = load_stopwords_from_db()

def dummy_tokenizer(text):
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = okt.pos(text, norm=True, stem=True)
    return [
        word for word, pos in tokens
        if pos.startswith('N') and word not in stopwords_set and len(word) > 1
    ]
