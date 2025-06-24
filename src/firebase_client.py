# src/firebase_client.py

import os
import logging
import json
from datetime import datetime, timedelta
import requests
from typing import Dict, Any, Optional
import subprocess

# .env 파일에서 환경 변수를 로드 (Optional)
from dotenv import load_dotenv

load_dotenv()

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, storage
except ImportError:
    raise RuntimeError(
        "firebase-admin 라이브러리가 설치되어 있어야 합니다. 'pip install firebase-admin'으로 설치하세요."
    )

from . import config
from .utils import convert_numpy_types, setup_logger

logger = setup_logger(__name__)


class FirebaseClient:
    """
    Firebase와 통신하는 모든 기능을 캡슐화한 클라이언트 클래스
    """

    def __init__(self):
        self.db = None
        self.bucket = None
        self._initialize_firebase()

    def _initialize_firebase(self):
        """
        Firebase Admin SDK를 안정적으로 초기화합니다.
        서버 재시작 시에도 중복 초기화 오류가 발생하지 않도록 처리합니다.
        """
        try:
            # 앱이 이미 초기화되었는지 확인
            if not firebase_admin._apps:
                # 1. 환경 변수에서 인증서 경로를 우선적으로 찾음
                cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH") or os.getenv(
                    "GOOGLE_APPLICATION_CREDENTIALS"
                )

                if cred_path and os.path.exists(cred_path):
                    # 파일 경로가 있으면 해당 파일로 인증
                    cred = credentials.Certificate(cred_path)
                    logger.info(f"서비스 계정 파일로 Firebase 인증: {cred_path}")
                else:
                    # 파일 경로가 없으면 GCP의 기본 인증 정보 사용 (Cloud Run 등 환경에서 유용)
                    cred = credentials.ApplicationDefault()
                    logger.info("Application Default Credentials로 Firebase 인증")

                firebase_admin.initialize_app(
                    cred, {"storageBucket": config.FIREBASE_STORAGE_BUCKET}
                )

            self.db = firestore.client()
            self.bucket = storage.bucket()
            logger.info("Firebase 클라이언트가 성공적으로 초기화되었습니다.")

        except Exception as e:
            logger.error(f"Firebase 초기화 실패: {e}", exc_info=True)
            raise e

    def download_video(self, storage_path: str, local_path: str) -> bool:
        """
        Firebase Storage 경로 또는 전체 URL을 받아 동영상을 다운로드합니다.
        """
        # 1. HTTP URL 형태면 requests로 직접 다운로드
        if storage_path.startswith("http"):
            try:
                logger.info(f"전체 URL로 감지하여 영상 다운로드 시작: {storage_path}")
                resp = requests.get(storage_path, stream=True, timeout=60)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"URL 다운로드 완료: {local_path}")
                return True
            except Exception as e:
                logger.error(f"URL을 통한 다운로드 실패: {e}")
                return False

        # 2. Storage 객체 경로일 경우 Admin SDK로 다운로드
        try:
            logger.info(f"Storage 경로로 감지하여 영상 다운로드 시작: {storage_path}")
            blob = self.bucket.blob(storage_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            logger.info(f"Storage 경로 다운로드 완료: {local_path}")
            return True
        except Exception as e:
            logger.error(f"Storage 경로를 통한 다운로드 실패 ('{storage_path}'): {e}")
            return False

    def _transcode_for_upload(self, src_path: str) -> str:
        """
        FFmpeg로 H.264/AAC + faststart 처리한 파일을 만들고
        그 경로를 반환합니다.
        """
        # 파일명과 확장자를 분리하여 더 안전하게 새로운 파일명 생성
        base, _ = os.path.splitext(src_path)
        dst = f"{base}_fixed.mp4"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                src_path,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-profile:v",
                "baseline",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                dst,
            ],
            check=True,
            capture_output=True,
            text=True,
        )  # capture_output, text 추가 시 ffmpeg 로그도 확인 가능
        return dst

    def upload_image_file(self, local_path: str, remote_path: str) -> Optional[str]:
        """
        로컬 이미지 파일을 Firebase Storage에 업로드하고 공개 URL을 반환합니다.
        (이미지이므로 트랜스코딩 없음)
        """
        if not self.bucket:
            logger.error("Storage 클라이언트가 초기화되지 않았습니다.")
            return None
        if not os.path.exists(local_path):
            logger.error(f"업로드할 로컬 파일이 없습니다: {local_path}")
            return None

        try:
            blob = self.bucket.blob(remote_path)

            # 파일 확장자에 맞는 올바른 content_type 설정
            content_type = (
                "image/jpeg" if local_path.lower().endswith(".jpg") else "image/png"
            )

            blob.upload_from_filename(local_path, content_type=content_type)
            blob.make_public()

            logger.info(
                f"이미지 업로드 성공: {local_path} -> gs://{self.bucket.name}/{remote_path}"
            )
            return blob.public_url

        except Exception as e:
            logger.error(f"이미지 파일 업로드 실패 {local_path}: {e}", exc_info=True)
            return None

    def upload_video_file(self, local_path: str, remote_path: str) -> Optional[str]:
        """
        로컬 '비디오' 파일을 스트리밍용으로 트랜스코드 후
        Firebase Storage에 업로드하고 공개 URL을 반환합니다.
        """
        if not self.bucket:
            logger.error("Storage 클라이언트가 초기화되지 않았습니다.")
            return None
        if not os.path.exists(local_path):
            logger.error(f"업로드할 로컬 파일이 없습니다: {local_path}")
            return None

        fixed_path = None  # finally 블록에서 참조할 수 있도록 외부에 선언
        try:
            # 1) 스트리밍용으로 트랜스코드
            logger.info(f"스트리밍용 트랜스코딩 시작: {local_path}")
            fixed_path = self._transcode_for_upload(local_path)
            logger.info(f"스트리밍용 트랜스코딩 완료: {fixed_path}")

            # 2) 업로드
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(fixed_path, content_type="video/mp4")
            blob.make_public()

            logger.info(
                f"파일 업로드 성공: {fixed_path} -> gs://{self.bucket.name}/{remote_path}"
            )
            return blob.public_url

        except Exception as e:
            logger.error(f"파일 업로드 실패 {local_path}: {e}", exc_info=True)
            return None
        finally:
            # 3) 업로드 성공/실패와 관계없이 임시 파일 삭제
            if fixed_path and os.path.exists(fixed_path):
                try:
                    os.remove(fixed_path)
                    logger.info(f"임시 파일 삭제 완료: {fixed_path}")
                except OSError as e:
                    logger.error(f"임시 파일 삭제 실패 {fixed_path}: {e}")

    def save_full_analysis_package(
        self,
        job_id: str,
        analysis_data: dict,
        result_video_path: str,
        backswing_image_path: str,
        impact_image_path: str,
    ) -> bool:
        """
        모든 분석 결과물(영상, 캡쳐 이미지, JSON 데이터)을 Firebase에 저장합니다.
        """
        if not self.db:
            logger.error("Firestore 클라이언트가 초기화되지 않았습니다.")
            return False

        try:
            # 1. 각 파일 타입에 맞는 업로드 함수 호출
            video_url = self.upload_video_file(  # 비디오용 함수 호출
                result_video_path, f"analysis_results/{job_id}/result.mp4"
            )
            backswing_url = self.upload_image_file(  # 이미지용 함수 호출
                backswing_image_path, f"analysis_results/{job_id}/backswing.jpg"
            )
            impact_url = self.upload_image_file(  # 이미지용 함수 호출
                impact_image_path, f"analysis_results/{job_id}/impact.jpg"
            )

            if not all([video_url, backswing_url, impact_url]):
                logger.error(
                    f"[{job_id}] 하나 이상의 파일 업로드에 실패하여 결과 저장을 중단합니다."
                )
                self.update_job_status(
                    job_id, "failed", "File upload to Storage failed."
                )
                return False

            # 2. 데이터를 UTF-8 JSON 문자열로 변환했다가 다시 파싱하여 인코딩을 보장합니다.
            temp_data = convert_numpy_types(analysis_data)
            json_string = json.dumps(temp_data, ensure_ascii=False)
            serializable_data = json.loads(json_string)

            # 3. Firestore에 저장할 최종 데이터 구조화
            result_doc = {
                "jobId": job_id,
                "status": "completed",
                "completedAt": datetime.now(),
                "resultVideoURL": video_url,
                "backswingImageURL": backswing_url,
                "impactImageURL": impact_url,
                "analysisData": serializable_data,  # 깨끗하게 인코딩된 데이터 사용
                "metadata": {"version": config.VERSION},
            }

            # 4. Firestore에 문서 저장
            doc_ref = self.db.collection("analysis_results").document(job_id)
            doc_ref.set(result_doc, merge=True)

            logger.info(f"[{job_id}] 분석 결과 패키지 전체 저장 완료.")
            return True
        except Exception as e:
            logger.error(f"분석 결과 패키지 저장 실패 {job_id}: {e}", exc_info=True)
            self.update_job_status(job_id, "failed", str(e))
            return False

    def get_analysis_result(self, job_id: str) -> Optional[dict]:
        if not self.db:
            return None
        try:
            doc_ref = self.db.collection("analysis_results").document(job_id)
            doc = doc_ref.get()
            return doc.to_dict() if doc.exists else None
        except Exception:
            return None

    def update_job_status(self, job_id: str, status: str, message: str = None) -> bool:
        if not self.db:
            return False
        try:
            update_data = {"status": status, "lastUpdated": datetime.now()}
            if message:
                update_data["message"] = message
            doc_ref = self.db.collection("analysis_results").document(job_id)
            doc_ref.update(update_data)
            return True
        except Exception:
            return False


# --- 싱글톤 패턴으로 클라이언트 인스턴스 관리 ---
_firebase_client_instance = None


def get_firebase_client() -> FirebaseClient:
    """
    FirebaseClient의 싱글톤 인스턴스를 반환합니다.
    """
    global _firebase_client_instance
    if _firebase_client_instance is None:
        _firebase_client_instance = FirebaseClient()
    return _firebase_client_instance


# --- 편의 함수 (다른 모듈에서 간단히 호출하기 위함) ---
def download_video_from_firebase(firebase_path: str, local_path: str) -> bool:
    """
    [Helper] Firebase Storage에서 동영상 다운로드를 실행합니다.
    """
    client = get_firebase_client()
    return client.download_video(firebase_path, local_path)
