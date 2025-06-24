# src/visualizer.py
"""
분석 결과를 프레임 위에 시각화하는 모든 함수를 포함하는 모듈입니다.
- 스켈레톤, 공, 발 폴리곤 등 그리기
- 프레임별 분석 정보 텍스트 오버레이
- 최종 점수 카드 생성
- 공 궤적, 오류 경고 시각화, 분석 영상 저장 기능 추가
"""
import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
from .feedback_generator import FEEDBACK_MESSAGES

# 내부 모듈 임포트
from src import config
from src.utils import setup_logger
from typing import Dict, Any, Optional, List

# src/visualizer.py 상단에 추가
import os  # os 모듈 추가


# --- [추가] 워터마크 로고 로드 ---
LOGO_IMG = None
LOGO_RESIZED = None
try:
    if config.LOGO_IMAGE_PATH and os.path.exists(config.LOGO_IMAGE_PATH):
        LOGO_IMG = cv2.imread(str(config.LOGO_IMAGE_PATH), cv2.IMREAD_UNCHANGED)
        if LOGO_IMG is None:
            logger.warning(f"로고 파일을 읽을 수 없습니다: {config.LOGO_IMAGE_PATH}")
    else:
        logger.warning("설정된 로고 이미지 경로를 찾을 수 없습니다.")
except Exception as e:
    logger.error(f"워터마크 로고 이미지 로드 실패: {e}")


# --- [추가] 워터마크 그리는 함수 ---
def draw_watermark(image: np.ndarray):
    """
    프레임의 우측 하단에 워터마크 로고를 그립니다.
    """
    global LOGO_RESIZED
    if LOGO_IMG is None:
        return

    # 로고 크기를 프레임 높이의 10%로 조절 (최초 한 번만 실행)
    if LOGO_RESIZED is None:
        frame_h = image.shape[0]
        logo_h = int(frame_h * 0.1)
        ratio = logo_h / LOGO_IMG.shape[0]
        logo_w = int(LOGO_IMG.shape[1] * ratio)
        LOGO_RESIZED = cv2.resize(LOGO_IMG, (logo_w, logo_h))

    # 로고를 넣을 위치 계산 (우측 하단, 약간의 여백)
    h, w, _ = image.shape
    logo_h, logo_w, _ = LOGO_RESIZED.shape
    margin = 20
    roi_x1, roi_y1 = w - logo_w - margin, h - logo_h - margin
    roi_x2, roi_y2 = w - margin, h - margin

    # 로고에 알파 채널(투명도)이 있을 경우 처리
    if LOGO_RESIZED.shape[2] == 4:
        alpha_channel = LOGO_RESIZED[:, :, 3] / 255.0
        overlay_colors = LOGO_RESIZED[:, :, :3]

        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

        # ROI와 로고를 알파 값에 따라 합성
        composite = roi * (1 - alpha_mask) + overlay_colors * alpha_mask
        image[roi_y1:roi_y2, roi_x1:roi_x2] = composite
    else:  # 알파 채널이 없을 경우 그냥 덮어쓰기
        image[roi_y1:roi_y2, roi_x1:roi_x2] = LOGO_RESIZED[:, :, :3]


logger = setup_logger(__name__)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# 하체 랜드마크 인덱스(힙 → 무릎 → 발목 → 발끝)
LOWER_LANDMARKS = {
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_HEEL,
}
# POSE_CONNECTIONS 중 하체 연결만 필터
LOWER_CONNECTIONS = {
    conn
    for conn in mp_pose.POSE_CONNECTIONS
    if conn[0] in LOWER_LANDMARKS and conn[1] in LOWER_LANDMARKS
}


def draw_pose_landmarks(image: np.ndarray, landmarks):
    if not landmarks:
        return

    # 1) 상체 연결(기본 회색) 그리기
    UPPER_CONNECTIONS = mp_pose.POSE_CONNECTIONS - LOWER_CONNECTIONS
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        UPPER_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 0, 0), thickness=3, circle_radius=3
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 0, 0), thickness=3, circle_radius=2
        ),
    )

    # 2) 하체 연결(원하는 색) 그리기
    mp_drawing.draw_landmarks(
        image,
        landmarks,
        LOWER_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 165, 255), thickness=4, circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 165, 255), thickness=4, circle_radius=2
        ),
    )


# --- Private Helper Functions ---


def _draw_text_with_alpha_bg(
    img: np.ndarray,
    text: str,
    pos: tuple,
    font_scale: float = 0.7,
    font_thickness: int = 2,
    text_color: tuple = (255, 255, 255),
    bg_color: tuple = (0, 0, 0),
    alpha: float = 0.5,
):
    x, y = pos
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    bg_rect_x1, bg_rect_y1 = max(0, x), max(0, y - text_h - baseline)
    bg_rect_x2, bg_rect_y2 = min(img.shape[1], x + text_w), min(
        img.shape[0], y + baseline
    )
    if bg_rect_x1 < bg_rect_x2 and bg_rect_y1 < bg_rect_y2:
        sub_img = img[bg_rect_y1:bg_rect_y2, bg_rect_x1:bg_rect_x2]
        bg = np.full(sub_img.shape, bg_color, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 1 - alpha, bg, alpha, 1.0)
        img[bg_rect_y1:bg_rect_y2, bg_rect_x1:bg_rect_x2] = res
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )


# in visualizer.py


# in visualizer.py


def _draw_analysis_lines(image: np.ndarray, pose_landmarks, kicking_foot: str):
    """
    (최종 수정 버전)
    분석에 필요한 기준선(수직선, 수평선)을 프레임 위에 그립니다.
    - 각 랜드마크의 visibility를 개별적으로 확인합니다.
    - 모든 선의 길이를 동적으로 동일하게 맞춥니다.
    """
    if not pose_landmarks or kicking_foot == "N/A":
        return

    h, w, _ = image.shape
    lm = pose_landmarks.landmark

    # 디딤발과 차는 발 랜드마크 정의
    supporting_foot_ankle_lm = lm[
        (
            mp_pose.PoseLandmark.LEFT_ANKLE
            if kicking_foot == "right"
            else mp_pose.PoseLandmark.RIGHT_ANKLE
        )
    ]
    supporting_foot_hip_lm = lm[
        (
            mp_pose.PoseLandmark.LEFT_HIP
            if kicking_foot == "right"
            else mp_pose.PoseLandmark.RIGHT_HIP
        )
    ]
    kicking_foot_hip_lm = lm[
        (
            mp_pose.PoseLandmark.RIGHT_HIP
            if kicking_foot == "right"
            else mp_pose.PoseLandmark.LEFT_HIP
        )
    ]

    # 1. 디딤발 기준 수직선 그리기 (하늘색)
    if supporting_foot_ankle_lm.visibility >= config.MP_POSE_MIN_DETECTION_CONFIDENCE:
        s_ankle_x = int(supporting_foot_ankle_lm.x * w)
        cv2.line(
            image,
            (s_ankle_x, int(h * 0.3)),
            (s_ankle_x, int(h * 0.9)),
            (255, 255, 0),
            2,
        )

    # 2. 디딤발 힙 기준 수평선 그리기 (주황색)
    if supporting_foot_hip_lm.visibility >= config.MP_POSE_MIN_DETECTION_CONFIDENCE:
        s_hip_x = int(supporting_foot_hip_lm.x * w)
        s_hip_y = int(supporting_foot_hip_lm.y * h)

        # --- ▼▼▼ 핵심 수정 ▼▼▼ ---
        # 수직선의 총 길이(h * 0.6)의 절반을 계산하여 길이를 맞춤
        line_half_length = int(h * 0.3)

        # 엉덩이 좌표를 중심으로 좌우로 line_half_length 만큼 선을 그림
        cv2.line(
            image,
            (s_hip_x - line_half_length, s_hip_y),
            (s_hip_x + line_half_length, s_hip_y),
            (0, 165, 255),
            2,
        )
        # --- ▲▲▲ 핵심 수정 ▲▲▲ ---

    # 3. 차는 발 힙 기준 수직선 그리기 (노란색)
    if kicking_foot_hip_lm.visibility >= config.MP_POSE_MIN_DETECTION_CONFIDENCE:
        k_hip_x = int(kicking_foot_hip_lm.x * w)
        cv2.line(
            image, (k_hip_x, int(h * 0.3)), (k_hip_x, int(h * 0.9)), (0, 255, 255), 2
        )


# --- 폰트 경로 설정 ---
KOREAN_FONT_PATH = "assets/fonts/DoHyeon-Regular.ttf"  # 2단계에서 준비한 폰트 경로


def _draw_korean_text_with_pil(
    img: np.ndarray,
    text: str,
    pos: tuple,
    font_size: int = 20,
    text_color: tuple = (255, 255, 255),
):
    """Pillow를 사용하여 OpenCV 이미지(Numpy 배열) 위에 한글을 그립니다."""
    # OpenCV 이미지를 Pillow 이미지로 변환
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(KOREAN_FONT_PATH, font_size)
    except IOError:
        font = ImageFont.load_default()
        print(f"폰트를 찾을 수 없습니다: {KOREAN_FONT_PATH}. 기본 폰트로 대체합니다.")

    draw.text(pos, text, font=font, fill=text_color)

    # 다시 OpenCV 이미지로 변환
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_foot_polygon(
    pose_landmarks, side: str, frame_shape: tuple
) -> Optional[np.ndarray]:
    if not pose_landmarks:
        return None
    lm_indices = {
        "left": [
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.LEFT_HEEL,
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        ],
        "right": [
            mp_pose.PoseLandmark.RIGHT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_HEEL,
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
        ],
    }
    points = []
    for lm_index in lm_indices.get(side, []):
        lm = pose_landmarks.landmark[lm_index]
        if lm.visibility > config.MP_POSE_MIN_DETECTION_CONFIDENCE:
            points.append((int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])))
    if len(points) < 3:
        return None
    try:
        return cv2.convexHull(np.array(points, dtype=np.int32))
    except Exception:
        return None


def draw_segmentation_mask(image: np.ndarray, mask_points: np.ndarray, color: tuple):
    if mask_points is None or len(mask_points) == 0:
        return
    overlay = image.copy()
    alpha = color[3] / 255.0 if len(color) == 4 else 0.5
    cv2.fillPoly(overlay, [mask_points.astype(np.int32)], color[:3])
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


# src/visualizer.py 에 아래 함수를 추가합니다.


# in visualizer.py


def _draw_gauge_bar(
    draw: ImageDraw.ImageDraw,
    pos: tuple,
    score: float,
    max_score: float,
    bar_width: int,
    bar_height: int,
    font: ImageFont.ImageFont,
):
    """(Helper) Pillow를 사용하여 게이지 바와 점수 텍스트를 그립니다."""
    x, y = int(pos[0]), int(pos[1])

    # 바 배경
    draw.rectangle(
        [(x, y), (x + bar_width, y + bar_height)],
        fill=(50, 50, 50),
        outline=(80, 80, 80),
    )

    # 채워지는 바
    fill_width = int((score / max_score) * bar_width)
    if score / max_score < 0.5:
        fill_color = (220, 48, 32)
    elif score / max_score < 0.8:
        fill_color = (230, 196, 22)
    else:
        fill_color = (76, 175, 80)

    if fill_width > 0:
        draw.rectangle([(x, y), (x + fill_width, y + bar_height)], fill=fill_color)

    # 바 위의 점수 텍스트
    score_text = f"{score:.1f}"
    # Pillow 10.0.0 이후 textbbox 사용, 이전 버전은 textsize
    try:
        bbox = draw.textbbox((0, 0), score_text, font=font)
        score_text_w = bbox[2] - bbox[0]
        score_text_h = bbox[3] - bbox[1]
    except AttributeError:
        score_text_w, score_text_h = draw.textsize(score_text, font=font)

    draw.text(
        (x + (bar_width - score_text_w) / 2, y + (bar_height - score_text_h) / 2 - 2),
        score_text,
        font=font,
        fill=(255, 255, 255),
    )


def draw_ball(image: np.ndarray, ball_info: Dict[str, Any]):
    """
    공의 Bounding Box 정보를 바탕으로, 중심점과 외곽선을 원 형태로 시각화합니다.
    """
    if not ball_info:
        return

    box = ball_info.get("box")
    if not box or len(box) != 4:
        return

    # Bounding Box에서 원의 중심과 반지름 계산
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    radius = int(((x2 - x1) + (y2 - y1)) / 4)

    # 상태에 따라 색상 결정
    is_predicted = ball_info.get("is_predicted", True)
    is_recovered = ball_info.get("occlusion_recovered", True)

    if is_predicted:
        color = (255, 165, 0)  # 주황색 (예측)
    elif is_recovered:
        color = (0, 0, 255)  # 파란색 (복구)
    else:
        color = (0, 0, 0)  # 노란색 (정상 감지)

    if radius > 0:
        # 1. 원본 이미지와 동일한 크기의 투명 레이어를 만듭니다.
        overlay = image.copy()
        # 2. 이 레이어에 속이 꽉 찬 검은색 원을 그립니다.
        cv2.circle(
            overlay, (center_x, center_y), radius, (0, 0, 0), -1
        )  # thickness=-1은 채우기

        # 3. 원본 이미지와 투명 레이어를 6:4 비율로 합성합니다.
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 0), 3)
        # 중심점은 합성된 이미지 위에 다시 그려서 항상 잘 보이도록 합니다.
        cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)


def draw_trajectory(image: np.ndarray, trajectory_points: List[tuple]):
    if len(trajectory_points) >= 2:
        for i in range(1, len(trajectory_points)):
            cv2.line(
                image, trajectory_points[i - 1], trajectory_points[i], (0, 255, 255), 2
            )


def draw_info_panel(image: np.ndarray, analysis_info: Dict[str, Any]):
    y_pos = 30
    display_items = {
        "Kicking Foot": analysis_info.get("kicking_foot", "N/A"),
        "Foot Speed (km/h)": analysis_info.get("kicking_foot_speed_kmh"),
        "Ball Speed (km/h)": analysis_info.get("ball_speed_kmh"),
        "Knee Angle": analysis_info.get("kicking_knee_angle"),
        "Hip Rotation": analysis_info.get("hip_rotation_angle"),
        "Backswing Knee": analysis_info.get("backswing_knee_angle"),
    }
    for key, value in display_items.items():
        if value is not None and value != "N/A":
            text = (
                f"{key}: {value:.1f}" if isinstance(value, float) else f"{key}: {value}"
            )
            _draw_text_with_alpha_bg(image, text, (15, y_pos))
            y_pos += 35


def draw_error_warning(image: np.ndarray, message: str):
    _draw_text_with_alpha_bg(
        image,
        f"⚠ {message}",
        (15, image.shape[0] - 30),
        font_scale=0.8,
        text_color=(0, 0, 255),
        alpha=0.6,
    )


def create_score_card_frame(
    base_frame: np.ndarray, score_data: Dict[str, Any], feedback_data: Dict[str, Any]
) -> np.ndarray:

    final_frame = base_frame.copy()
    h, w, _ = final_frame.shape

    # --- 1. 반투명 배경 생성 (카드 크기 확대) ---
    overlay = final_frame.copy()
    bg_x1, bg_y1 = int(w * 0.02), int(h * 0.05)  # 좌상단 여백 축소
    bg_x2, bg_y2 = int(w * 0.98), int(h * 0.95)  # 우하단 여백 축소
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, final_frame, 0.15, 0, final_frame)

    # --- 2. Pillow로 한글 텍스트 그리기 준비 ---
    pil_img = Image.fromarray(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        # --- ▼▼▼ 해상도 기반 적응형 크기 계산 ▼▼▼ ---
        # 기준 해상도 1080p에서의 적절한 크기를 정의하고, 현재 해상도에 맞게 스케일링
        base_height = 1080
        scale_factor = h / base_height

        font_title = ImageFont.truetype(
            KOREAN_FONT_PATH, max(24, int(48 * scale_factor))
        )  # 최소 24px
        font_main_score = ImageFont.truetype(
            KOREAN_FONT_PATH, max(22, int(42 * scale_factor))
        )  # 최소 22px
        font_category = ImageFont.truetype(
            KOREAN_FONT_PATH, max(18, int(32 * scale_factor))
        )  # 최소 18px
        font_gauge_label = ImageFont.truetype(
            KOREAN_FONT_PATH, max(16, int(26 * scale_factor))
        )  # 최소 16px
        font_feedback_title = ImageFont.truetype(
            KOREAN_FONT_PATH, max(18, int(30 * scale_factor))
        )  # 최소 18px
        font_feedback_body = ImageFont.truetype(
            KOREAN_FONT_PATH, max(14, int(24 * scale_factor))
        )  # 최소 14px
    except IOError:
        font_title = font_main_score = font_category = font_gauge_label = (
            font_feedback_title
        ) = font_feedback_body = ImageFont.load_default()
        logger.warning(
            f"폰트를 찾을 수 없습니다: {KOREAN_FONT_PATH}. 기본 폰트로 대체합니다."
        )

    # --- 3. 제목 및 총점 (스케일링된 간격 사용) ---
    y_pos = bg_y1 + max(25, int(35 * scale_factor))  # 스케일링된 여백
    # ▼▼▼ 이모티콘 제거 ▼▼▼
    title_text = "KICK ANALYSIS REPORT"
    title_w = draw.textlength(title_text, font=font_title)
    draw.text(
        ((w - title_w) / 2, y_pos), title_text, font=font_title, fill=(80, 255, 80)
    )
    y_pos += max(40, int(60 * scale_factor))  # 스케일링된 간격

    score_text = f"Total Score: {score_data.get('total_score', 0):.1f}"
    score_w = draw.textlength(score_text, font=font_main_score)
    draw.text(
        ((w - score_w) / 2, y_pos),
        score_text,
        font=font_main_score,
        fill=(255, 255, 255),
    )
    y_pos += max(35, int(55 * scale_factor))  # 스케일링된 간격

    # --- 4. 게이지 바 (2단 레이아웃) ---
    gauge_section_start_y = y_pos
    max_y_for_gauges = 0

    if "categories" in score_data:
        num_categories = len(score_data.get("categories", {}))
        # 카드 내부 여백을 고려한 전체 너비 계산 (열 간격 확보)
        available_width = bg_x2 - bg_x1 - 80  # 좌우 여백을 80px로 증가 (60→80)
        column_width = available_width / num_categories

        for i, (cat_name, cat_data) in enumerate(
            score_data.get("categories", {}).items()
        ):
            y_pos = gauge_section_start_y  # 각 열의 시작 높이를 동일하게 초기화
            current_x_start = (
                bg_x1 + 40 + (i * column_width)
            )  # 각 열의 시작 x좌표 (왼쪽 여백 40px)

            cat_title = f"--- {cat_name.replace('_', ' ').title()} ---"
            draw.text(
                (current_x_start, y_pos),
                cat_title,
                font=font_category,
                fill=(255, 255, 0),
            )
            y_pos += max(30, int(45 * scale_factor))  # 스케일링된 간격

            for item, details in cat_data.get("details", {}).items():
                label = FEEDBACK_MESSAGES.get(item, {}).get("name", item)
                score = details.get("score", 0)
                max_score = details.get("max_score", 10)

                # Pillow로 한글 라벨 그리기
                draw.text(
                    (current_x_start, y_pos + 5),
                    label,
                    font=font_gauge_label,
                    fill=(255, 255, 255),
                )

                # 게이지 바 그리기 - 열 너비에 맞게 조정
                label_width = max(
                    120, int(column_width * 0.25)
                )  # 라벨이 열 너비의 25% 차지
                bar_x = (
                    current_x_start + label_width + max(8, int(10 * scale_factor))
                )  # 간격
                bar_width = max(
                    200, min(int(column_width * 0.65), int(400 * scale_factor))
                )  # 열 너비의 65% 또는 최대 400px
                bar_height = max(20, int(32 * scale_factor))  # 스케일링된 바 높이

                # 바가 카드 경계를 넘지 않도록 제한
                max_bar_x = bg_x2 - 40  # 오른쪽 여백 40px
                if bar_x + bar_width > max_bar_x:
                    bar_width = max_bar_x - bar_x

                # 다음 열과 겹치지 않도록 제한
                next_column_start = current_x_start + column_width
                if (
                    bar_x + bar_width > next_column_start - 20
                ):  # 다음 열과 20px 간격 유지
                    bar_width = next_column_start - bar_x - 20

                _draw_gauge_bar(
                    draw,  # ← PIL ImageDraw 객체를 전달
                    (bar_x, y_pos),
                    score,
                    max_score,
                    bar_width,
                    bar_height,
                    font_gauge_label,
                )

                y_pos += max(35, int(50 * scale_factor))  # 스케일링된 간격

            if y_pos > max_y_for_gauges:
                max_y_for_gauges = y_pos  # 그려진 가장 낮은 y좌표를 기록

    # --- 5. 피드백 메시지 표시 (스케일링된 간격) ---
    y_pos = max_y_for_gauges + max(25, int(35 * scale_factor))  # 스케일링된 간격
    feedback_x_start = bg_x1 + 40  # 피드백 시작 위치 조정

    # ▼▼▼ 이모티콘 제거 ▼▼▼
    feedback_title = "종합 피드백"
    draw.text(
        (feedback_x_start, y_pos),
        feedback_title,
        font=font_feedback_title,
        fill=(255, 255, 0),
    )
    y_pos += max(30, int(40 * scale_factor))  # 스케일링된 간격

    # in create_score_card_frame function

    if feedback_data:
        strengths = (
            feedback_data.get("종합", {}).get("피드백", {}).get("매우 잘한점", [])
        )
        weaknesses = (
            feedback_data.get("종합", {}).get("피드백", {}).get("아쉬운 점", [])
        )

        if strengths:
            draw.text(
                (feedback_x_start, y_pos),
                "강점",
                font=font_feedback_title,
                fill=(0, 255, 127),
            )
            y_pos += max(25, int(35 * scale_factor))  # 스케일링된 간격
            for fb in strengths[:1]:  # 강점 1개만 표시
                # --- ▼▼▼ '개선점'과 동일한 지능형 줄바꿈 로직 추가 ▼▼▼ ---
                words = fb.split(" ")
                lines = []
                current_line = "•"  # 각 줄의 시작에 불릿 포인트 추가

                max_line_width = (bg_x2 - feedback_x_start) - 50

                for word in words:
                    test_line = f"{current_line} {word}".strip()
                    if (
                        draw.textlength(test_line, font=font_feedback_body)
                        <= max_line_width
                    ):
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = f"  {word}"

                lines.append(current_line)
                # --- ▲▲▲ 로직 추가 종료 ▲▲▲ ---

                # 계산된 줄(lines)을 화면에 그리기
                for line in lines:
                    draw.text(
                        (feedback_x_start + 20, y_pos),
                        line,
                        font=font_feedback_body,
                        fill=(220, 220, 220),
                    )
                    y_pos += max(20, int(32 * scale_factor))  # 스케일링된 줄 간격
            y_pos += max(15, int(25 * scale_factor))  # 스케일링된 섹션 간격

        if weaknesses:
            draw.text(
                (feedback_x_start, y_pos),
                "개선점",
                font=font_feedback_title,
                fill=(255, 165, 0),
            )
            y_pos += max(25, int(35 * scale_factor))  # 스케일링된 간격
            for fb in weaknesses[:1]:  # 개선점 1개만 표시
                words = fb.split(" ")
                lines = []
                current_line = "•"

                max_line_width = (bg_x2 - feedback_x_start) - 50

                for word in words:
                    test_line = f"{current_line} {word}".strip()
                    if (
                        draw.textlength(test_line, font=font_feedback_body)
                        <= max_line_width
                    ):
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = f"  {word}"

                lines.append(current_line)

                for line in lines:
                    draw.text(
                        (feedback_x_start + 20, y_pos),
                        line,
                        font=font_feedback_body,
                        fill=(220, 220, 220),
                    )
                    y_pos += max(20, int(32 * scale_factor))  # 스케일링된 줄 간격

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def generate_annotated_frame(
    frame: np.ndarray,
    pose_landmarks,
    ball_info: Dict[str, Any],
    analysis_info: Dict[str, Any],
    trajectory: Optional[List[tuple]] = None,
    warnings: Optional[List[str]] = None,
) -> np.ndarray:
    annotated = frame.copy()
    draw_pose_landmarks(annotated, pose_landmarks)
    draw_ball(annotated, ball_info)
    if trajectory:
        draw_trajectory(annotated, trajectory)
    kicking_foot = analysis_info.get("kicking_foot")
    if kicking_foot and kicking_foot != "N/A" and pose_landmarks:
        poly = get_foot_polygon(pose_landmarks, kicking_foot, annotated.shape)
        if poly is not None:
            draw_segmentation_mask(annotated, poly, color=config.FOOT_VIS_COLOR)
    draw_info_panel(annotated, analysis_info)
    # 백스윙 표시: 노란색 테두리와 큰 텍스트 (임팩트와 동일 스타일)
    if analysis_info.get("is_backswing_frame", False):
        cv2.rectangle(
            annotated,
            (0, 0),
            (frame.shape[1] - 1, frame.shape[0] - 1),
            (0, 255, 255),  # 노란색
            15,
        )
        _draw_text_with_alpha_bg(
            annotated,
            "BACKSWING!",
            (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
            font_scale=4.5,
            font_thickness=8,
            text_color=(0, 0, 255),
            alpha=0.4,
        )
        _draw_analysis_lines(annotated, pose_landmarks, kicking_foot)

    # 임팩트 표시: 노란색 테두리와 큰 텍스트
    if analysis_info.get("is_impact_frame", False):
        cv2.rectangle(
            annotated,
            (0, 0),
            (frame.shape[1] - 1, frame.shape[0] - 1),
            (0, 255, 255),  # 노란색
            15,
        )
        _draw_text_with_alpha_bg(
            annotated,
            "IMPACT!",
            (frame.shape[1] // 2 - 200, frame.shape[0] // 2),
            font_scale=4.5,
            font_thickness=8,
            text_color=(0, 0, 255),
            alpha=0.4,
        )
        _draw_analysis_lines(annotated, pose_landmarks, kicking_foot)

    draw_watermark(annotated)

    if warnings:
        for warning in warnings:
            draw_error_warning(annotated, warning)
    return annotated


def save_annotated_video(frames: List[np.ndarray], output_path: str, fps: int):
    if not frames:
        logger.warning("저장할 프레임이 없습니다.")
        return
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    logger.info(f"결과 영상 저장 완료: {output_path}")
