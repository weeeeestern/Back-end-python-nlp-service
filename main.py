# main.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel # 데이터 모델 정의
import uvicorn # FastAPI 서버 실행 유틸리티

import nlp_processor # 핵심 NLP 로직 모듈 임포트

# --- FastAPI 애플리케이션 인스턴스 생성 ---
app = FastAPI(
    title="기사 NLP 마이크로서비스",
    description="기사 분석 및 추천을 위한 한국어 자연어 처리 마이크로서비스 (TF-IDF 벡터화)",
    version="1.0.0",
    docs_url="/docs",     # Swagger UI URL (http://localhost:5000/docs)
    redoc_url="/redoc"    # ReDoc UI URL (http://localhost:5000/redoc)
)

# --- 애플리케이션 시작 시 실행되는 이벤트 핸들러 ---
@app.on_event("startup")
async def startup_event():
    print("NLP 모델 및 데이터 로딩 시작...")
    try:
        nlp_processor.load_models_and_data()
        print("NLP 모델 및 데이터 로드 성공.")
    except Exception as e:
        print(f"시작 시 NLP 리소스 로드 실패: {e}")
        # 실제 운영 환경에서는 애플리케이션이 정상적으로 시작되지 않도록 처리할 수 있습니다.
        # 여기서는 단순히 로깅하고, API 호출 시 에러를 발생시킵니다.
        # raise RuntimeError("서버 시작 실패: NLP 리소스 로드 오류.") from e


# --- Pydantic 모델 정의: API 요청/응답 데이터 구조 및 유효성 검사 ---

# TF-IDF 벡터화 요청을 위한 데이터 모델
class TextVectorRequest(BaseModel):
    text: str # 벡터화할 텍스트 (기사 본문 또는 검색어)

# TF-IDF 벡터화 응답을 위한 데이터 모델
class VectorResponse(BaseModel):
    vector: list[float] # TF-IDF 벡터 (float 값의 리스트)

# --- API 엔드포인트: TF-IDF 벡터화 ---
@app.post("/vectorize-tfidf", response_model=VectorResponse, status_code=status.HTTP_200_OK, summary="텍스트 TF-IDF 벡터화")
async def vectorize_tfidf_api(request_data: TextVectorRequest):
    """
    제공된 텍스트(기사 본문 또는 검색어)를 TF-IDF 벡터로 변환합니다.
    - **text**: 벡터화할 텍스트 (필수)
    - 반환: TF-IDF 벡터 (float 값의 리스트)
    """
    text = request_data.text
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'text' 필드는 필수입니다.")

    try:
        vector = nlp_processor.vectorize_tfidf(text)
        return {"vector": vector}
    except RuntimeError as e: # nlp_processor에서 발생시킨 초기화 오류
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"NLP 서비스 준비 안됨: {e}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"텍스트 벡터화 중 오류 발생: {e}")


# --- FastAPI 애플리케이션 실행 명령어 ---
# 이 파일을 직접 실행할 때 (예: python main.py) Uvicorn 서버를 시작합니다.
# 개발 중에는 --reload 옵션을 사용하여 코드 변경 시 자동 재시작하게 할 수 있습니다.
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
