import os
import subprocess

# 불용어 DB 삽입
print("불용어 DB에 삽입 중...")
subprocess.run(["python", "scripts/insert_stopwords_to_db.py"], check=True)

# TF-IDF 모델 학습
print("TF-IDF 모델 학습 중...")
subprocess.run(["python", "scripts/train_tfidf_model.py"], check=True)

# 서버 실행
print("FastAPI 서버 실행 중...")
subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"])
