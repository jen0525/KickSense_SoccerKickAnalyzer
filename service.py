# service.py
"""
BentoML 기반 축구 킥 분석 API 서비스 (개선 적용 버전)

- Pydantic을 사용한 명확한 API 요청/응답 모델
- 주발 미설정 시 경고 출력 및 응답 전달
- 작업 상태 메시지 개선
- 안정적인 임시 파일 처리 로직
"""

import os
import shutil
import tempfile
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field, HttpUrl
from typing import Literal, Optional

# 내부 모듈 임포트
from src import analyzer, config
from src.utils import setup_logger, download_file
from src.firebase_client import get_firebase_client

# 로거 및 Firebase 클라이언트 설정
logger = setup_logger(__name__)
firebase_client = get_firebase_client()

# --- Pydantic 모델 정의 ---


class AnalysisRequest(BaseModel):
    """API 요청을 위한 입력 모델"""

    jobID: str = Field(..., description="Flutter 앱에서 생성한 고유 작업 ID")
    videoURL: HttpUrl = Field(..., description="분석할 영상의 Firebase Storage URL")
    kickingFoot: Literal["left", "right", "auto"] = Field(
        default="auto", description="분석할 주발 설정"
    )


class AnalysisResponse(BaseModel):
    """API 응답을 위한 기본 모델"""

    jobID: str
    status: str
    message: str
    warning: Optional[str] = None  # 경고 메시지 필드 추가


# --- BentoML 서비스 정의 ---


@bentoml.service(name="kick_analysis_service", traffic={"timeout": 600})
class KickAnalysisService:

    def __init__(self):
        """서비스 초기화"""
        config.ensure_directories()
        if not firebase_client.db:
            logger.error(
                "Firestore 클라이언트 초기화 실패. 서비스가 정상적으로 동작하지 않을 수 있습니다."
            )
        logger.info(
            f"'{config.PROJECT_NAME}' 서비스가 성공적으로 초기화되었습니다 (v{config.VERSION})."
        )

    @bentoml.api(
        input=JSON(pydantic_model=AnalysisRequest),
        output=JSON(pydantic_model=AnalysisResponse),
    )
    async def analyze_kick(self, request: AnalysisRequest) -> AnalysisResponse:
        """축구 킥 영상 분석 API의 메인 엔드포인트"""
        job_id = request.jobID
        logger.info(f"[{job_id}] 분석 요청 수신. 주발 설정: {request.kickingFoot}")

        # 요청마다 격리된 임시 작업 폴더를 생성합니다.
        temp_dir = tempfile.mkdtemp(prefix=f"{job_id}_")
        local_video_path = os.path.join(temp_dir, "input.mp4")

        warning_msg = None
        if request.kickingFoot == "auto":
            warning_msg = "⚠️ 주발 설정이 '자동(auto)'으로 되어 있습니다. 직접 선택하면 더 정확한 분석 결과를 받을 수 있습니다."
            logger.warning(f"[{job_id}] {warning_msg}")
            firebase_client.update_job_status(job_id, "processing", warning_msg)

        try:
            # 1. 영상 다운로드
            firebase_client.update_job_status(
                job_id, "processing", "영상을 서버로 다운로드하고 있습니다."
            )
            if not download_file(str(request.videoURL), local_video_path):
                raise ValueError("영상 다운로드에 실패했습니다. URL을 확인해주세요.")
            logger.info(f"[{job_id}] 영상 다운로드 완료: {local_video_path}")

            # 2. 핵심 분석 실행
            firebase_client.update_job_status(
                job_id,
                "processing",
                "킥 동작을 정밀 분석 중입니다. 잠시만 기다려주세요...",
            )
            analysis_data, result_video_path = analyzer.run_full_analysis(
                video_path=local_video_path, kicking_foot=request.kickingFoot
            )

            # 3. 결과 업로드
            firebase_client.update_job_status(
                job_id, "processing", "분석 결과를 안전하게 저장하고 있습니다."
            )
            success = firebase_client.save_results_to_firebase(
                job_id=job_id,
                analysis_data=analysis_data,
                result_video_path=result_video_path,
            )

            if not success:
                raise RuntimeError("분석 결과를 Firebase에 저장하는 데 실패했습니다.")
            logger.info(f"[{job_id}] 모든 처리 과정이 성공적으로 완료되었습니다.")

            return AnalysisResponse(
                jobID=job_id,
                status="success",
                message="분석이 성공적으로 완료되었습니다.",
                warning=warning_msg,
            )

        except Exception as e:
            error_message = f"작업 처리 중 오류 발생: {e}"
            logger.exception(f"[{job_id}] {error_message}")
            firebase_client.update_job_status(job_id, "failed", str(e))
            return AnalysisResponse(
                jobID=job_id, status="error", message=str(e), warning=warning_msg
            )
        finally:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(
                        f"[{job_id}] 임시 디렉토리 '{temp_dir}'를 정리했습니다."
                    )
                except Exception as e:
                    logger.error(f"[{job_id}] 임시 디렉토리 정리에 실패했습니다: {e}")
