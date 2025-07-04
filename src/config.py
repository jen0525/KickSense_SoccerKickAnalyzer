# src/config.py
"""
축구 킥 분석 AI 프로젝트의 모든 설정을 관리하는 파일입니다.
모델 경로, 하이퍼파라미터, API 설정 등을 중앙에서 관리합니다.
"""
import cv2
import os
import time
import warnings
from pathlib import Path

# --- 기본 프로젝트 정보 ---
PROJECT_NAME = "soccer-kick-analyzer"
VERSION = "0.1.0"  # 프로젝트 버전

# --- 파일 및 디렉토리 경로 ---
# 이 파일(config.py)의 위치를 기준으로 프로젝트 루트 디렉토리를 찾습니다.
# src/config.py -> ../ -> 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent

# 주요 디렉토리 정의
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
LOGS_DIR = BASE_DIR / "logs"
# 모델 파일 경로
YOLO_MODEL_NAME = "ball_yolom.pt"
YOLO_MODEL_PATH = MODELS_DIR / YOLO_MODEL_NAME

# 선수 데이터 파일 경로
PLAYER_STATS_PATH = BASE_DIR / "data" / "pro_player_stats.json"


def ensure_directories():
    """프로젝트에 필요한 주요 디렉토리가 존재하는지 확인하고 없으면 생성합니다."""
    dirs_to_create = [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"필요한 디렉토리 확인/생성 완료: {', '.join(map(str, dirs_to_create))}")


def create_timestamp() -> str:
    """현재 시간을 기준으로 타임스탬프 문자열을 생성합니다."""
    return time.strftime("%Y%m%d-%H%M%S")


# --- BentoML 서비스 설정 ---
BENTOML_SERVICE_NAME = "kick_analysis_service"
BENTOML_API_TIMEOUT = 600  # API 타임아웃 (초)


# --- 로깅 설정 ---
LOG_LEVEL = "DEBUG"  # "DEBUG", "INFO", "WARNING", "ERROR"


# --- 분석 하이퍼파라미터 ---

# 물리적 상수
REAL_BALL_DIAMETER_CM = 22.0
DEFAULT_PIXEL_TO_CM_SCALE = 0.85  # 공 크기 추정 실패 시 사용할 기본 스케일
REFERENCE_FPS = 30.0  # 속도 정규화를 위한 기준 FPS
CM_S_TO_KM_H = 0.036  # cm/s -> km/h 변환 계수

# 모델 신뢰도 임계값
YOLO_CONF_THRESHOLD = 0.4  # original = 0.15
YOLO_IOU_THRESHOLD = 0.45
MP_POSE_MIN_DETECTION_CONFIDENCE = 0.3
MP_POSE_MIN_TRACKING_CONFIDENCE = 0.3

# 공 추적 및 선택 관련
BALL_SELECTION_ALPHA_FOOT_DIST = 0.008  # 발-공 거리 기반 공 선택 가중치
BALL_TRACKING_MAX_DIST_FROM_PREV = (
    200  # 이전 프레임으로부터 공이 이동할 수 있는 최대 픽셀 거리
)
BETA_PROXIMITY_SCALE = 1.3  # 기존 1.1에서 1.3으로 수정
INITIAL_FRAMES_FOR_BALL_SIZE_EST = 30  # 초기 공 크기 측정을 위한 프레임 수
BALL_SIZE_EST_MIN_CONFIDENCE = 0.4  # 공 크기 측정 시 필요한 최소 YOLO 신뢰도

# 임팩트 및 백스윙 감지 관련
IOU_CONTACT_THRESHOLD = 0.01  # 기존 0.15 또는 0.05에서 0.01로 수정
BACKSWING_PEAK_PROMINENCE = 4  # 백스윙 정점(peak) 감지 민감도
BACKSWING_MIN_PEAK_DISTANCE_FRAMES = 8  # 감지될 백스윙 정점 간의 최소 프레임 간격
MINIMUM_DISTANCE_WINDOW = (
    2  # 현재 프레임의 발-공 거리가 최소값인지 확인할 좌우 프레임 반경
)
ABSOLUTE_DISTANCE_THRESHOLD = (
    12.0  # 임팩트 후보가 되기 위한 발-공 간 최대 절대 거리 (픽셀)
)
# --- 공 추적 고급 설정 ---
BALL_TRACKING_RADIUS_WEIGHT = 1.0
BALL_TRACKING_CONF_WEIGHT = -20.0
BALL_TRACKING_MAX_RADIUS_DIFF_RATIO = 0.7
MAX_CONSECUTIVE_PREDICTIONS = 2
PREDICTED_BALL_CONFIDENCE_DECAY = 0.6

# --- 발-공 임팩트 조건 ---
USE_KICKING_FOOT_KINEMATICS_CONDITION = False
KICK_FOOT_APPROACH_VEL_THRESHOLD = 0.05
MIN_KICKING_FOOT_SPEED_AT_IMPACT = 2.0
MAX_KICKING_FOOT_ACCEL_AT_IMPACT = 15.0

# --- 양발 간 거리 조건 ---
USE_INTER_FOOT_DISTANCE_CHECK = True
MAX_INTER_FOOT_DISTANCE_AT_IMPACT = 400

# --- 발 속도 점수화 관련 ---
FOOT_SPEED_SCORE_WEIGHT = -0.8
FOOT_SPEED_PEAK_BONUS = -10.0
FOOT_SPEED_PEAK_WINDOW = 1

# --- 발-공 최소 거리 조건 관련 ---
MINIMUM_DISTANCE_WINDOW = 2
ABSOLUTE_DISTANCE_THRESHOLD = 80.0  # original = 30.0

# --- 마스크 관련 설정 (segmentation 기반 추적 시 사용) ---
USE_BALL_SEGMENTATION = False
MASK_ASSOCIATION_IOU_THRESHOLD = 0.3

# --- shrinking 공 보정 관련 ---
USE_BALL_SHRINK_DETECTION = False
BALL_RADIUS_SHRINK_THRESHOLD_PX = 1.0
BALL_SHRINK_RECOVERY_FACTOR = 0.5

# --- 공 가림 처리 설정 ---
OCCLUSION_IOU_THRESHOLD = 0.1
OCCLUSION_MIN_NEW_BALL_CONF = 0.25

# --- 공 속도 최대 탐색 프레임 수 ---
MAX_FRAMES_FOR_MAX_BALL_SPEED_SEARCH = 30

# 프레임 보간 관련
USE_FRAME_INTERPOLATION = True
INTERPOLATION_WINDOW_RADIUS = 7  # 보간에 사용할 주변 프레임 반경
INTERPOLATION_DENSITY_FACTOR = 50  # 보간 시 얼마나 촘촘하게 계산할지 결정하는 요소

# 발 모델링 관련
USE_ENHANCED_FOOT_MODEL = True
FOOT_WIDTH_SCALE = 0.4  # 발 길이를 기준으로 발 너비를 추정할 때 사용되는 비율
INSTEP_FORWARD_SCALE = 0.7  # 발등 위치 추정을 위한 스케일
SIDE_FOOT_FORWARD_SCALE = 0.6  # 인사이드/아웃사이드 위치 추정을 위한 스케일

# 디딤발 안정성 계산 관련
CALCULATE_SUPPORTING_FOOT_STABILITY = True
SUPPORTING_FOOT_STABILITY_WINDOW_SIZE = (
    5  # 임팩트 전, 디딤발의 안정성을 계산할 프레임 수
)

# 선수 각도 비교 관련
USE_PLAYER_ANGLE_COMPARISON = True

# 최대 스윙속도 관련
CORRECTION_FACTOR = 1.8

# --- 기능 활성화 플래그 ---
# True/False로 특정 분석 기능의 사용 여부를 제어합니다.
DEBUG_MODE = True  # 디버그 모드 (상세 로그 및 시각화 출력)
USE_DISTANCE_SMOOTHING = True  # 발-공 거리 데이터 스무딩 사용 여부
USE_PREDICTIVE_TRACKING = True  # 칼만 필터 기반 공 위치 예측 사용 여부
USE_OCCLUSION_HANDLING = True  # 공 가림 현상 처리 로직 사용 여부


# --- 시각화 관련 설정 ---
# 분석 영상에 표시될 마스크 및 박스의 색상 (R, G, B, Alpha)
SEGMENTATION_VIS_COLOR = (255, 150, 150, 150)
FOOT_VIS_COLOR = (150, 255, 150, 120)

# --- 칼만 필터 하이퍼파라미터 ---
KF_PROCESS_NOISE = 0.1  # 프로세스 노이즈 (모델의 불확실성)
KF_MEASUREMENT_NOISE = 10.0  # 측정 노이즈 (YOLO 탐지의 불확실성)
KF_KICK_NOISE_MULTIPLIER = 10.0  # 킥 감지 시 프로세스 노이즈를 몇 배로 늘릴지
GRAVITY_PIXELS_PER_SECOND_SQUARED = (
    9.8 * 50
)  # 픽셀 단위 중력 가속도 (화면 해상도에 따라 튜닝 필요)

# 어이쿠 이렇게 빠뜨리시면 어떡해여
IMPACT_AV_CHANGE_BONUS = 20  # 접근 속도가 (+)로 바뀌는 프레임에 부여하는 점수 보너스
KICK_TO_SUPPORT_SPEED_RATIO_THRESHOLD = (
    1.2  # 차는 발이 디딤발보다 최소 2배는 빨라야 유효한 킥으로 간주
)
FOOT_BOX_RADIUS = 50  # 40 -> 50
KNEE_ANGLE_THRESHOLD = 100
MIN_IMPACT_SPEED_PX_FR = 1.5
Z_VALID_RANGE = (-1.0, 1.0)
USE_HEEL_CORRECTION = True
USE_DIFFERENTIAL_FOOT_SPEED_CHECK = True
USE_BACKSWING_APPROACH_PATTERN = True
SAMPLING_RATE_TARGET_FRAMES = 30
SMOOTHING_WINDOW_SIZE = 5
BACKSWING_SEARCH_WINDOW = 5

# --- 임팩트 후보 필터링 임계값 ---
IMPACT_DISTANCE_MAX = 150  # 픽셀(px), 이 거리 이내 프레임만 1차 후보
BETA_PROXIMITY_SCALE = 1.2  # 공 반지름 대비 근접 체크 배율
FOOT_BOX_RADIUS = 20  # 발 박스 생성 반경(px)
IOU_CONTACT_THRESHOLD = 0.1  # IoU 최소 임계값(0~1)

# --- 동역학 점수 정규화 ---
IMPACT_DYNAMICS_MAX_SPEED = 30  # px/frame 단위, 속도차 정규화용 최대값

# --- 스코어 가중치 ---
# IMPACT_WEIGHT_DISTANCE = 1.0  # 거리 점수 가중치
# IMPACT_WEIGHT_PROXIMITY = 1.2  # 근접성 점수 가중치
# IMPACT_WEIGHT_IOU = 2.0  # IoU 점수 가중치
# IMPACT_WEIGHT_DYNAMICS = 1.5  # 속도 차 점수 가중치
# IMPACT_WEIGHT_LOCAL_MIN_BONUS = 0.2  # 로컬-미니멈 보너스

# --- 수정 후 (defalut)---
IMPACT_WEIGHT_DISTANCE = 0.10  # 거리 점수 비중을 낮춤
IMPACT_WEIGHT_PROXIMITY = 0.10  # 근접도 점수 비중을 낮춤
IMPACT_WEIGHT_IOU = 0.15
IMPACT_WEIGHT_DYNAMICS = 0.25  # 발 움직임 점수 비중을 높임
IMPACT_WEIGHT_LOCAL_MIN_BONUS = 0.05
IMPACT_WEIGHT_POST_SPEED_BONUS = 0.35  # 임팩트 후 공 속도 보너스를 대폭 높임


# --- 로컬-미니멈 윈도우 크기(프레임) ---
MINIMUM_DISTANCE_WINDOW = 3  # ±1프레임(총 3), 윈도우 내 최소 거리 여부 확인

# --- 최종 후보 개수 제한 ---
IMPACT_CANDIDATE_MAX = 5  # 상위 N개 프레임만 최종 후보로

# --- 임팩트 후보 사후 검증 가중치 ---
IMPACT_WEIGHT_POST_SPEED_BONUS = 5.0  # 임팩트 후 공 속도에 대한 보너스 가중치

# MANUAL_ROTATE_CODE = cv2.ROTATE_90_COUNTERCLOCKWISE  # for messi (edge-case)
MANUAL_ROTATE_CODE = None

##### 영상 속 공이 2개인 경우 #####
# --- 공 선택(Ball Selection) 가중치 ---
BALL_SELECTION_WEIGHT_PREDICTION = 1.0  # 칼만 필터 예측 위치와의 거리 가중치
BALL_SELECTION_WEIGHT_SIZE = 1.5  # 이전 공과의 크기 유사도 가중치
BALL_SELECTION_WEIGHT_PLAYER = 10.0  # 선수 발과의 거리 가중치 (가장 중요하게 설정)
# --- 공 선택(Ball Selection) 정규화 기준값 ---
BALL_SELECT_MAX_PREDICTION_DIST = (
    200.0  # 예측 위치에서 이 거리(px) 이상 벗어나면 점수 1점
)
BALL_SELECT_MAX_SIZE_DIFF = 20.0  # 반지름 차이가 이 값(px) 이상 나면 점수 1점
BALL_SELECT_MAX_FOOT_DIST = 1000.0  # 발에서 이 거리(px) 이상 떨어져 있으면 점수 1점
# [추가] 킥 직후 사용할 가중치
BALL_SELECTION_WEIGHT_PREDICTION_POST_KICK = 0.1  # 예측의 중요도를 대폭 낮춤
BALL_SELECTION_WEIGHT_SIZE_POST_KICK = 1.0
BALL_SELECTION_WEIGHT_PLAYER_POST_KICK = 15.0  # 선수의 중요도를 대폭 높임
# Firebase Admin SDK 서비스 계정 키 파일(.json) 경로


FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")


LOGO_IMAGE_PATH = BASE_DIR / "assets" / "logo.png"
