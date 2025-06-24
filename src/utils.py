"""
utils.py - 공용 유틸리티 함수 모듈

프로젝트 전반에서 사용되는 공통 유틸리티 함수들
"""

import os
import cv2
import uuid
import hashlib
import requests
import numpy as np
from typing import Tuple, Optional, List, Union, Dict
from datetime import datetime
import logging
from sklearn.metrics import r2_score


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """로거 설정 및 반환"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def generate_unique_id(prefix: str = "") -> str:
    """고유 ID 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 형식: YYYYMMDD_HHMMSS
    unique_part = str(uuid.uuid4())[:8]
    return (
        f"{prefix}{timestamp}_{unique_part}" if prefix else f"{timestamp}_{unique_part}"
    )


def create_timestamp() -> str:
    """현재 시간의 타임스탬프 생성"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 형식: YYYY-MM-DD HH:MM:SS


def calculate_file_hash(file_path: str) -> str:
    """파일의 MD5 해시값 계산"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"파일 해시 계산 오류: {e}")
        return ""


def download_file(url: str, output_path: str, timeout: int = 30) -> bool:
    """URL에서 파일 다운로드"""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        logging.error(f"파일 다운로드 실패 {url}: {e}")
        return False


def validate_video_file(file_path: str) -> Tuple[bool, str]:
    """영상 파일 유효성 검사"""
    if not os.path.exists(file_path):
        return False, "파일이 존재하지 않습니다."

    # 파일 확장자 체크
    ext = os.path.splitext(file_path)[1].lower()
    supported_formats = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

    if ext not in supported_formats:
        return (
            False,
            f"지원하지 않는 파일 형식입니다. 지원 형식: {', '.join(supported_formats)}",
        )

    # OpenCV로 영상 열기 테스트
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False, "영상 파일을 열 수 없습니다."

    # 기본 영상 정보 확인
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    if frame_count == 0:
        return False, "영상에 프레임이 없습니다."

    if fps <= 0:
        return False, "유효하지 않은 FPS입니다."

    return True, "유효한 영상 파일입니다."


def get_video_info(file_path: str) -> Optional[dict]:
    """영상 파일 정보 추출"""
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        return None

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        "file_size": os.path.getsize(file_path),
    }

    cap.release()
    return info


def resize_video_if_needed(
    input_path: str, output_path: str, max_width: int = 1920, max_height: int = 1080
) -> bool:
    """필요시 영상 크기 조정"""
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return False

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 리사이즈가 필요한지 확인
    if original_width <= max_width and original_height <= max_height:
        cap.release()
        return True  # 리사이즈 불필요

    # 비율 유지하며 크기 계산
    ratio = min(max_width / original_width, max_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # 출력 영상 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (new_width, new_height))
            out.write(resized_frame)

        return True
    except Exception as e:
        logging.error(f"영상 리사이즈 오류: {e}")
        return False
    finally:
        cap.release()
        out.release()


def extract_frames(
    video_path: str, output_dir: str, frame_skip: int = 1, max_frames: int = None
) -> List[str]:
    """영상에서 프레임 추출"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    os.makedirs(output_dir, exist_ok=True)

    frame_paths = []
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame_skip 간격으로 프레임 저장
        if frame_count % frame_skip == 0:
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1

            # 최대 프레임 수 제한
            if max_frames and saved_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    return frame_paths


def cleanup_temp_files(temp_dir: str, max_age_hours: int = 24):
    """임시 파일 정리"""
    if not os.path.exists(temp_dir):
        return

    current_time = datetime.now().timestamp()
    max_age_seconds = max_age_hours * 3600

    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)

        if os.path.isfile(file_path):
            file_age = current_time - os.path.getctime(file_path)

            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    logging.info(f"임시 파일 삭제: {file_path}")
                except Exception as e:
                    logging.error(f"임시 파일 삭제 실패 {file_path}: {e}")


def convert_numpy_types(obj):
    """NumPy 타입을 JSON 직렬화 가능한 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def format_duration(seconds: float) -> str:
    """초를 읽기 쉬운 시간 형식으로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}분 {remaining_seconds:.1f}초"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}시간 {minutes}분 {remaining_seconds:.1f}초"


def format_file_size(bytes_size: int) -> str:
    """바이트를 읽기 쉬운 크기 형식으로 변환"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """안전한 나눗셈 (0으로 나누기 방지)"""
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """값을 지정된 범위로 제한"""
    return max(min_val, min(value, max_val))


# 예외 출력 도우미
def print_traceback():
    """예외 발생 시 traceback 전체 출력"""
    print("예외 발생:")
    import traceback

    traceback.print_exc()


# 시계열 데이터 보간
from scipy.interpolate import interp1d


def interpolate_array(
    x: np.ndarray, y: np.ndarray, num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """배열을 주어진 밀도로 보간"""
    f = interp1d(x, y, kind="linear")
    x_new = np.linspace(x[0], x[-1], num_points)
    y_new = f(x_new)
    return x_new, y_new


# 거리 스무딩 처리
from scipy.ndimage import gaussian_filter1d


def smooth_distance_array(distances: List[float], sigma: float = 1.0) -> List[float]:
    """거리 배열을 Gaussian 필터로 부드럽게 만듦"""
    return gaussian_filter1d(distances, sigma=sigma).tolist()


def create_timestamp() -> str:
    """현재 시간의 타임스탬프 생성"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_valid_url(url: str) -> bool:
    """URL 유효성 검사"""
    try:
        from urllib.parse import urlparse

        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


# ===============================================================
# (노트북에서 가져온 분석용 헬퍼 함수들)
# ===============================================================

import pandas as pd
import mediapipe as mp
from . import config  # config.py의 변수를 사용하기 위해 임포트

# MediaPipe Pose 전역 변수 (get_specific_landmark_position 등에서 사용)
mp_pose = mp.solutions.pose


# ⚽️ 영상에서 측정한 속도를 실제 단위(km/h)로 바꾸되, FPS 차이도 고려해서 정규화해주는 함수
def convert_speed_to_kmh_normalized(px_per_frame, pixel_scale, current_effective_fps):
    """
    측정된 px/frame 속도를 기준 FPS로 정규화하고, 물리적 단위(km/h)로 변환합니다.
    이것이 바로 '소프트웨어적 FPS 정규화'의 핵심 구현입니다.

    Args:
        px_per_frame (float): 현재 유효 FPS에서 측정된 프레임당 픽셀 이동량
        pixel_scale (float): cm/pixel 변환 계수
        current_effective_fps (float): 분석에 실제 사용된 유효 FPS (원본FPS / 샘플링 간격)

    Returns:
        float: km/h 단위의 최종 보정된 속도
    """
    if current_effective_fps <= 1e-6 or pd.isna(px_per_frame):
        return 0.0

    # 1. 기준 FPS(REFERENCE_FPS)로 속도를 정규화
    # 공식: 측정값 * (실제 분석 FPS / 기준 FPS) -> 모든 영상의 속도를 기준 FPS 영상에서 측정한 것처럼 환산
    normalized_px_per_reference_frame = px_per_frame * (
        current_effective_fps / config.REFERENCE_FPS
    )

    # 2. 정규화된 속도(px/기준프레임)를 cm/s로 변환
    # 이제 속도는 기준 FPS(REFERENCE_FPS)에 대한 값이므로, 계산 시에도 기준 FPS를 사용
    speed_cm_per_sec = (
        normalized_px_per_reference_frame * pixel_scale * config.REFERENCE_FPS
    )

    # 3. cm/s를 km/h로 변환 (CM_S_TO_KM_H = 0.036)
    speed_km_per_hour = speed_cm_per_sec * config.CM_S_TO_KM_H

    return speed_km_per_hour


# ⚽️ 공의 픽셀 반지름과 실제 지름(cm)을 바탕으로,
# 현재 영상의 픽셀-센티미터 환산 계수 (cm/pixel)를 동적으로 추정하는 함수 (기초)
def calculate_dynamic_pixel_to_cm_scale(ball_radius_px, real_ball_diameter_cm_param):
    if ball_radius_px and ball_radius_px > 1e-3 and real_ball_diameter_cm_param > 0:
        detected_ball_diameter_px = ball_radius_px * 2.0
        if detected_ball_diameter_px > 1e-3:
            return real_ball_diameter_cm_param / detected_ball_diameter_px
    return None


# ⚽️ 비정상적인 픽셀-센티미터 스케일 값을 감지하고 기본값으로 대체
#  경고를 발생시켜 사용자에게 결과 왜곡 가능성을 알리는 함수 (보완)
def get_safe_pixel_to_cm_scale(scale_val: float) -> float:
    """비정상적인 픽셀-센티미터 스케일 값을 감지하고 기본값으로 대체합니다."""
    if scale_val is None or pd.isna(scale_val) or scale_val <= 1e-6:
        # warnings.warn(...) 대신 로거 사용을 권장
        setup_logger(__name__).warning(
            f"잘못된 scale_val={scale_val} 감지. 기본값으로 대체합니다."
        )
        return config.DEFAULT_PIXEL_TO_CM_SCALE
    return scale_val


def get_confidence_level(r2_x, r2_y):
    r2_min = min(r2_x, r2_y)
    if r2_min >= 0.9:
        return "high"
    elif r2_min >= 0.6:
        return "medium"
    else:
        return "low"


# ⚽️ 벡터가 가리키는 방향을 기준으로 공 표면 어디에 임팩트가 있었는지 시각적 위치 분류를 해주는 함수
# analyzer.py를 구현할 때, 이 ball.py가 반환해주는 '예측된' 중심점 정보를 어떻게 잘 활용하여 get_ball_contact_region을 호출할지 고민)
def get_ball_contact_region(vec_x: float, vec_y: float) -> str:
    """벡터 방향을 기반으로 공의 접촉 부위를 추정합니다."""
    angle = np.degrees(np.arctan2(vec_y, vec_x))
    if -22.5 <= angle < 22.5:
        return "Right"
    if 22.5 <= angle < 67.5:
        return "Top-Right"
    if 67.5 <= angle < 112.5:
        return "Top"
    if 112.5 <= angle < 157.5:
        return "Top-Left"
    if 157.5 <= angle or angle < -157.5:
        return "Left"
    if -157.5 <= angle < -112.5:
        return "Bottom-Left"
    if -112.5 <= angle < -67.5:
        return "Bottom"
    if -67.5 <= angle < -22.5:
        return "Bottom-Right"
    return "Center"


# ⚽️ 두 개의 사각형(바운딩 박스) box1, box2 사이의 IoU (교집합/합집합 비율)를 계산하는 함수
# 의의: 단일 공을 계속 추적하기 위함
def calculate_iou(box1: List, box2: List) -> float:
    """두 바운딩 박스 사이의 IoU를 계산합니다."""
    if not (box1 and box2 and len(box1) == 4 and len(box2) == 4):
        return 0.0
    x1i, y1i = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2i, y2i = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2i - x1i) * max(0, y2i - y1i)
    if inter_area == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area < 1e-6:
        return 0.0
    return inter_area / union_area


# ⚽️ MediaPipe Pose의 결과에서 특정 관절(lm_enum)의 프레임 내 좌표(x, y)를 추출한다.
# 조건: 해당 관절의 감지 신뢰도가 충분히 높을 때만 반환한다.
def get_specific_landmark_position(
    landmarks, lm_enum, frame_shape: tuple
) -> Optional[tuple]:
    """MediaPipe Pose 결과에서 특정 관절의 (x, y) 좌표를 추출합니다."""
    if not landmarks or not landmarks.landmark:
        return None
    lm = landmarks.landmark[lm_enum.value]
    if lm.visibility < config.MP_POSE_MIN_DETECTION_CONFIDENCE:
        return None
    return int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])


# ⚽️ 지정된 관절(lm_enum)의 (x, y, z) 좌표를 반환
# visibility와 z값의 유효성을 함께 검사하며, 조건을 만족하지 않으면 None을 반환
def get_landmark_with_z(landmarks, lm_enum, frame_shape, z_valid_range=(-1.0, 1.0)):
    if not landmarks or not landmarks.landmark:
        return None
    if not (
        isinstance(lm_enum, mp_pose.PoseLandmark)
        and 0 <= lm_enum.value < len(landmarks.landmark)
    ):
        return None
    lm = landmarks.landmark[lm_enum.value]
    if lm.visibility < config.MP_POSE_MIN_DETECTION_CONFIDENCE:
        return None
    if not (z_valid_range[0] <= lm.z <= z_valid_range[1]):
        return None
    return (int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0]), lm.z)


def calculate_angle(p1: tuple, p2: tuple, p3: tuple) -> float:
    """세 점 사이의 각도를 계산합니다. (calculate_angle_for_df에서 이름 변경)"""
    if p1 is None or p2 is None or p3 is None:
        return np.nan
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return np.nan
    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_z_score(user_value, mean, std):
    if pd.isna(user_value) or pd.isna(mean) or pd.isna(std):
        return float("inf")
    if std == 0:
        return 0 if user_value == mean else float("inf")
    return abs((user_value - mean) / std)


def z_score_to_similarity(z_score):
    if z_score == float("inf"):
        return 0
    return np.exp(-0.5 * z_score**2)


# ⚽️ 목적: 좌표가 None인 경우에 쓸만한 추정값을 보간으로 채워넣기 위함 + fallback 로직
def interpolate_point_data_with_quality_and_fallback(
    df, point_column_name, target_idx, window_radius=5, fallback_strategy=True
):
    start_idx = max(0, int(target_idx) - window_radius)
    end_idx = min(len(df) - 1, int(target_idx) + window_radius)
    window_df = df.loc[start_idx:end_idx].copy()

    x_coords = window_df.apply(
        lambda row: (
            row[point_column_name][0]
            if isinstance(row[point_column_name], tuple)
            else np.nan
        ),
        axis=1,
    )
    y_coords = window_df.apply(
        lambda row: (
            row[point_column_name][1]
            if isinstance(row[point_column_name], tuple)
            else np.nan
        ),
        axis=1,
    )

    interp_data_x = pd.DataFrame(
        {"idx": window_df["original_idx"], "coord": x_coords}
    ).dropna()
    interp_data_y = pd.DataFrame(
        {"idx": window_df["original_idx"], "coord": y_coords}
    ).dropna()

    if len(interp_data_x) < 2 or len(interp_data_y) < 2:
        if config.DEBUG_MODE:
            print(f"[DEBUG] 보간 불가: 데이터 수 부족 (idx={target_idx})")
        return {"interpolated_point": None, "quality": None}

    try:
        kind = "cubic" if len(interp_data_x) >= 4 else "linear"

        interp_func_x = interp1d(
            interp_data_x["idx"],
            interp_data_x["coord"],
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_func_y = interp1d(
            interp_data_y["idx"],
            interp_data_y["coord"],
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
        )

        new_x = float(interp_func_x(target_idx))
        new_y = float(interp_func_y(target_idx))

        # R² 계산
        r2_x = r2_score(interp_data_x["coord"], interp_func_x(interp_data_x["idx"]))
        r2_y = r2_score(interp_data_y["coord"], interp_func_y(interp_data_y["idx"]))
        confidence = get_confidence_level(r2_x, r2_y)

        if config.DEBUG_MODE:
            print(
                f"[DEBUG] idx={target_idx}, 보간=({new_x:.1f}, {new_y:.1f}), R²=({r2_x:.2f}, {r2_y:.2f}), 신뢰도={confidence}"
            )

        # 신뢰도 낮고 fallback 전략 사용
        if confidence == "low" and fallback_strategy:
            fallback = None
            for offset in range(1, window_radius + 1):
                for neighbor in [target_idx - offset, target_idx + offset]:
                    if 0 <= neighbor < len(df):
                        neighbor_val = df.iloc[neighbor][point_column_name]
                        if isinstance(neighbor_val, tuple):
                            fallback = neighbor_val
                            break
                if fallback is not None:
                    break
            if config.DEBUG_MODE:
                print(f"[DEBUG] → Fallback 사용됨: {fallback}")
            return {
                "interpolated_point": fallback,
                "quality": {"r2_x": r2_x, "r2_y": r2_y, "level": confidence},
                "fallback_used": True,
            }

        return {
            "interpolated_point": (new_x, new_y),
            "quality": {"r2_x": r2_x, "r2_y": r2_y, "level": confidence},
            "fallback_used": False,
        }

    except Exception as e:
        if config.DEBUG_MODE:
            print(f"[DEBUG] 예외 발생: {e}")
        return {"interpolated_point": None, "quality": None, "error": str(e)}


# def convert_speed_to_kmh_normalized(...): 순수한 단위 변환 함수. ✅
# def calculate_dynamic_pixel_to_cm_scale(...): 픽셀-cm 스케일 계산 함수. ✅
# def get_safe_pixel_to_cm_scale(...): 스케일 값 안전성 확인 함수. ✅
# def get_ball_contact_region(...): 벡터 기반의 접촉 부위 계산 함수. ✅
# def calculate_iou(...): 두 사각형의 IoU 계산 함수. ✅
# def get_specific_landmark_position(...): MediaPipe 랜드마크 좌표 추출 헬퍼. ✅
# def get_landmark_with_z(...): z값을 포함한 랜드마크 추출 헬퍼. ✅
# def calculate_angle_for_df(...): 세 점의 각도를 계산하는 순수 수학 함수. (이름은 calculate_angle로 변경해도 좋습니다.) ✅
# def calculate_z_score(...): 통계 Z-점수 계산 함수.✅
# def z_score_to_similarity(...): Z-점수를 유사도로 변환하는 함수. ✅
# def interpolate_point_data_with_quality_and_fallback(...): 좌표 보간 함수. ✅


# ⚽️ MediaPipe 관절 정보를 기반으로 발의 방향, 길이, 안/바깥쪽 위치 등을 정밀 추정하는 함수
def enhance_foot_model(
    pose_landmarks, side: str, frame_shape: tuple
) -> Optional[Dict[str, tuple]]:
    """
    (완성된 버전)
    MediaPipe 랜드마크를 기반으로 발의 주요 부위(발등, 인사이드, 토)의 2D 좌표를 추정합니다.
    """
    if not pose_landmarks:
        return None

    # 1. 필요한 랜드마크의 enum 값을 결정합니다.
    ankle_enum = (
        mp_pose.PoseLandmark.LEFT_ANKLE
        if side == "left"
        else mp_pose.PoseLandmark.RIGHT_ANKLE
    )
    heel_enum = (
        mp_pose.PoseLandmark.LEFT_HEEL
        if side == "left"
        else mp_pose.PoseLandmark.RIGHT_HEEL
    )
    toe_enum = (
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        if side == "left"
        else mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    )

    # 2. 각 랜드마크의 2D 좌표를 가져옵니다.
    ankle_pos = get_specific_landmark_position(pose_landmarks, ankle_enum, frame_shape)
    heel_pos = get_specific_landmark_position(pose_landmarks, heel_enum, frame_shape)
    toe_pos = get_specific_landmark_position(pose_landmarks, toe_enum, frame_shape)

    # 3. 필수 랜드마크(발목, 발끝)가 없으면 계산을 중단합니다.
    if not ankle_pos or not toe_pos:
        return None

    # 4. 발의 방향과 길이를 나타내는 벡터를 계산합니다.
    foot_vec = np.array(toe_pos) - np.array(ankle_pos)
    foot_len = np.linalg.norm(foot_vec)
    if foot_len < 1e-6:  # 길이가 너무 짧으면 계산 불가
        return None

    foot_unit_vec = foot_vec / foot_len

    # 5. 주요 부위의 위치를 추정합니다.
    # - 발등(instep): 발목과 발끝의 중간 지점에서 약간 발끝 쪽으로 이동
    instep_pos = (
        np.array(ankle_pos) + foot_unit_vec * foot_len * config.INSTEP_FORWARD_SCALE
    )

    # - 인사이드(inside): 발의 방향 벡터에 수직인 벡터를 이용해 계산
    #   (왼발/오른발에 따라 방향이 다름)
    if side == "left":
        perp_vec = np.array(
            [-foot_unit_vec[1], foot_unit_vec[0]]
        )  # 시계 방향 90도 회전
    else:  # right
        perp_vec = np.array(
            [foot_unit_vec[1], -foot_unit_vec[0]]
        )  # 반시계 방향 90도 회전

    #   인사이드 위치: 발등 위치에서 발 너비의 절반만큼 안쪽으로 이동
    inside_pos = instep_pos + perp_vec * foot_len * config.FOOT_WIDTH_SCALE * 0.5

    # 6. 계산된 좌표들을 딕셔너리 형태로 반환합니다.
    foot_model = {
        "ankle": tuple(ankle_pos),
        "toe": tuple(toe_pos),
        "heel": tuple(heel_pos) if heel_pos else None,
        "instep": tuple(instep_pos.astype(int)),
        "inside": tuple(inside_pos.astype(int)),
    }

    return foot_model


# ⚽️ 공의 위치 좌표 리스트를 바탕으로, 스무딩을 적용하여 평균/최대/초기 속도(km/h)를 계산하여 딕셔너리로 반환
# def calculate_ball_speed(
#     positions: pd.Series, fps: float, scale_cm_per_px: float
# ) -> Dict[str, float]:
#     # 위치 데이터가 3개 미만이면 계산 불가
#     if len(positions) < 3:
#         return {"avg_speed_kmh": 0.0, "max_speed_kmh": 0.0, "initial_speed_kmh": 0.0}

#     # 1. 위치 좌표(x, y)를 각각 스무딩
#     x_coords = pd.Series([p[0] for p in positions.values])
#     y_coords = pd.Series([p[1] for p in positions.values])
#     x_smoothed = x_coords.rolling(window=3, center=True, min_periods=1).mean()
#     y_smoothed = y_coords.rolling(window=3, center=True, min_periods=1).mean()
#     smooth_positions = list(zip(x_smoothed, y_smoothed))

#     # 2. 스무딩된 좌표로 프레임별 속도(km/h) 계산
#     speeds_kmh = []
#     for i in range(1, len(smooth_positions)):
#         # 픽셀 이동 거리 계산
#         d_px = np.linalg.norm(
#             np.array(smooth_positions[i]) - np.array(smooth_positions[i - 1])
#         )
#         # cm/s 로 변환
#         cm_s = (d_px * scale_cm_per_px) * fps
#         # km/h 로 변환
#         km_h = cm_s * config.CM_S_TO_KM_H
#         speeds_kmh.append(km_h)

#     # 3. 최종 결과 정리하여 반환
#     if not speeds_kmh:
#         return {"avg_speed_kmh": 0.0, "max_speed_kmh": 0.0, "initial_speed_kmh": 0.0}

#     return {
#         "avg_speed_kmh": float(np.mean(speeds_kmh)),
#         "max_speed_kmh": float(np.max(speeds_kmh)),
#         "initial_speed_kmh": float(speeds_kmh[0]),
#     }

# utils.py 파일에 아래 함수를 추가하거나 교체하세요.

# utils.py 파일

import numpy as np
import pandas as pd
from typing import Dict, Any, Union


def calculate_ball_speed(
    positions: Union[pd.Series, pd.DataFrame], fps: float, scale_cm_per_px: float
) -> Dict[str, Any]:
    """
    (최종 보정 버전)
    공의 위치 데이터(좌표의 시리즈)를 바탕으로 공의 최고 속도와 평균 속도를 km/h 단위로 계산합니다.
    프레임 인덱스를 사용하여 프레임 누락 시에도 정확한 시간을 계산합니다.
    """
    if positions.empty or len(positions) < 2:
        return {"max_speed_kmh": 0.0, "avg_speed_kmh": 0.0, "speeds_kmh": []}

    # Series가 아니면 변환 (일관성 유지)
    if not isinstance(positions, pd.Series):
        positions = pd.Series(positions)

    # 위치 좌표와 프레임 인덱스를 추출
    coords = positions.dropna()
    frame_indices = coords.index.to_numpy()
    coord_values = np.array(coords.to_list())

    # --- 프레임 간격(delta_frames)을 고려한 속도 계산 ---
    speeds_kmh = []
    for i in range(1, len(coord_values)):
        # 이전 프레임과 현재 프레임 정보
        prev_pos = coord_values[i - 1]
        curr_pos = coord_values[i]
        prev_frame_idx = frame_indices[i - 1]
        curr_frame_idx = frame_indices[i]

        # 픽셀 이동량 (유클리드 거리)
        dist_px = np.linalg.norm(curr_pos - prev_pos)

        # 실제 경과 시간 계산 (프레임 간격 / FPS)
        delta_frames = curr_frame_idx - prev_frame_idx
        if delta_frames == 0:
            continue  # 같은 프레임이면 계산 스킵
        time_s = delta_frames / fps

        # cm 단위 거리
        dist_cm = dist_px * scale_cm_per_px

        # cm/s 속도
        speed_cm_per_s = dist_cm / time_s

        # km/h 속도로 변환 (1 cm/s = 0.036 km/h)
        speed_kmh = speed_cm_per_s * 0.036
        speeds_kmh.append(speed_kmh)

    if not speeds_kmh:
        return {"max_speed_kmh": 0.0, "avg_speed_kmh": 0.0, "speeds_kmh": []}

    # 결과 정리
    max_speed_kmh = np.max(speeds_kmh)
    avg_speed_kmh = np.mean(speeds_kmh)

    return {
        "max_speed_kmh": float(max_speed_kmh),
        "avg_speed_kmh": float(avg_speed_kmh),
        "speeds_kmh": speeds_kmh,
    }
