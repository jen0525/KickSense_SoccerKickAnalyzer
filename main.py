# main.py

import os
import uuid
import logging
import tempfile
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# src 폴더의 모듈들을 가져옵니다.
from src.analyzer import run_full_analysis
from src.firebase_client import get_firebase_client, download_video_from_firebase
from src.utils import convert_numpy_types

# FastAPI 앱 생성
app = FastAPI(title="Soccer Kick Analysis API")
logger = logging.getLogger("uvicorn")


from typing import Literal, Optional
from pydantic import BaseModel, Field, validator


class AnalysisRequest(BaseModel):
    video_url: str
    # 허용값: auto, right, left
    preferredFoot: Literal["auto", "right", "left"] = "auto"
    job_id: Optional[str] = None
    user_id: Optional[str] = Field(None, alias="userId")

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True

    @validator("preferredFoot", pre=True, always=True)
    def default_preferred_foot(cls, v):
        # null 또는 빈 문자열인 경우 auto 로 대체
        if v is None:
            return "auto"
        return v


@app.post("/analyze/", tags=["Analysis"])
async def create_analysis_from_url(request: AnalysisRequest):
    """
    Firebase Storage 경로를 입력받아 동영상을 다운로드하고,
    킥 분석을 수행한 뒤, 모든 결과물을 다시 Firebase에 저장하고
    최종 결과 데이터(URL 포함)를 반환합니다.
    """
    # 🔍 받은 요청 데이터 전체 로그 출력 (디버깅용)
    logger.info("🔍🔍🔍 === AI 서버가 받은 요청 데이터 === 🔍🔍🔍")
    logger.info(f"   • 요청 본문 파싱 결과: {request.dict()}")
    logger.info("🔍🔍🔍 ======================================== 🔍🔍🔍")

    # Flutter에서 받은 job_id를 사용하고, 없으면 새로 생성합니다.
    job_id = request.job_id or f"api-fastapi-{uuid.uuid4().hex[:8]}"

    logger.info(
        f"[{job_id}] Received new analysis request for url: {request.video_url}"
    )

    try:
        firebase_client = get_firebase_client()
        if not firebase_client.db:
            raise ConnectionError("Firebase client could not be initialized.")
    except Exception as e:
        logger.error(f"[{job_id}] Firebase connection error: {e}")
        raise HTTPException(status_code=500, detail="Could not connect to Firebase.")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}.mp4")

        # 1. Firebase Storage에서 동영상 다운로드
        download_success = download_video_from_firebase(request.video_url, video_path)
        if not download_success:
            firebase_client.update_job_status(
                job_id, "failed", "Could not download video from Firebase."
            )
            raise HTTPException(
                status_code=404,
                detail="Could not download video from the provided URL.",
            )

        try:
            # 2. 핵심 분석 함수 실행
            final_data, video_out_path, backswing_img_path, impact_img_path = (
                run_full_analysis(
                    video_path=video_path,
                    kicking_foot_preference=request.preferredFoot,
                    job_id=job_id,
                )
            )

            if not final_data:
                raise ValueError("Analysis process returned no data.")

            # 3. 분석 결과물 업로드
            save_success = firebase_client.save_full_analysis_package(
                job_id=job_id,
                analysis_data=final_data,
                result_video_path=video_out_path,
                backswing_image_path=backswing_img_path,
                impact_image_path=impact_img_path,
            )
            if not save_success:
                raise Exception("Failed to save analysis package to Firebase.")

            # 4. 반환할 데이터 준비
            serializable_data = convert_numpy_types(final_data)

            # 최종 응답에 user_id와 job_id를 추가
            response_payload = {
                "user_id": request.user_id,
                "job_id": job_id,
                **serializable_data,
            }

            logger.info(
                f"[{job_id}] Analysis and Firebase upload successful. Returning data directly."
            )
            return JSONResponse(content=response_payload)

        except Exception as e:
            logger.error(f"[{job_id}] Analysis pipeline failed: {e}", exc_info=True)
            firebase_client.update_job_status(job_id, "failed", str(e))
            raise HTTPException(
                status_code=500, detail=f"An error occurred during analysis: {e}"
            )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
