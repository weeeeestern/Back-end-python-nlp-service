# scripts/insert_stopwords_to_db.py
import csv
import pymysql
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# DB 설정
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 3306))
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'snack')

# CSV 경로
DATA_SOURCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_source')
STOPWORDS_CSV = os.path.join(DATA_SOURCE_DIR, 'stopwords.csv')

def init_database_and_table():
    """DB 및 테이블이 없으면 자동 생성"""
    print(f"DB '{DB_NAME}' 및 테이블 'stopwords' 초기화 시도 중...")

    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        charset='utf8mb4',
        autocommit=True  # CREATE DATABASE는 autocommit 필요
    )

    try:
        with conn.cursor() as cursor:
            # 1. DB 생성
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;")
            cursor.execute(f"USE {DB_NAME};")

            # 2. 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stopwords (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    word VARCHAR(255) NOT NULL UNIQUE
                );
            """)
        print("✅ DB 및 테이블이 준비되었습니다.")
    finally:
        conn.close()

def insert_data_to_db(filepath: str, table_name: str, column_name: str):
    """CSV 파일을 읽어 DB 테이블에 데이터를 삽입"""
    print(f"테이블 '{table_name}'에 '{filepath}' 파일의 데이터 삽입 시도 중...")
    if not os.path.exists(filepath):
        print(f"오류: 파일이 존재하지 않습니다: '{filepath}'. '{table_name}' 삽입을 건너뜁니다.")
        return

    conn = None
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME,
            charset='utf8mb4'
        )
        cur = conn.cursor()

        insert_query = f"INSERT IGNORE INTO {table_name} ({column_name}) VALUES (%s)"

        with open(filepath, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    value = row[column_name]
                    if value:
                        cur.execute(insert_query, (value,))
                except KeyError:
                    print(f"경고: CSV 파일에 컬럼 '{column_name}'이 없습니다. 행을 건너뜁니다: {row}")
                except Exception as e:
                    print(f"오류: 행 '{row}' 삽입 실패: {e}")

        conn.commit()
        print(f"'{filepath}' 파일의 데이터가 테이블 '{table_name}'에 성공적으로 삽입되었습니다.")

    except pymysql.Error as e:
        print(f"❌ DB 작업 실패: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    print("--- 불용어 삽입 스크립트 실행 ---")
    os.makedirs(DATA_SOURCE_DIR, exist_ok=True)

    # ✅ DB 및 테이블 초기화
    init_database_and_table()

    # ✅ 데이터 삽입
    insert_data_to_db(STOPWORDS_CSV, 'stopwords', 'word')

    print("--- 불용어 삽입 스크립트 완료 ---")
