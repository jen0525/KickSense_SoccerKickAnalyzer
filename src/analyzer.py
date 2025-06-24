# src/analyzer.py
"""
분석의 전체 흐름을 지휘하는 메인 분석 모듈.
데이터 수집, 가공, 이벤트 탐지, 결과물 생성의 총괄 감독님!!
"""
import cv2
import pandas as pd
import mediapipe as mp
from typing import Dict, Any, Tuple, List, Optional
import os
import numpy as np

# 내부 모듈 임포트
from . import config, visualizer, scoring, impact_detection, feedback_generator, utils
from .ball import BallTracker
from .utils import (
    setup_logger,
    calculate_angle,
    get_specific_landmark_position,
)
from .impact_detection import _find_refined_impact_moment

logger = setup_logger(__name__)
mp_pose = mp.solutions.pose

# --- 내부 헬퍼 함수 정의 ---


def _initialize_components(video_path: str) -> dict:
    """분석에 필요한 모든 객체들을 초기화합니다."""
    logger.info("분석 컴포넌트 초기화 시작...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"비디오 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- FPS 값 검증 및 로깅 강화 ---
    logger.info(f"비디오 파일에서 읽어들인 원본 FPS: {fps}")

    # FPS 값이 비정상적인 범위에 있을 경우 (예: 10 미만 또는 120 초과) 경고 및 보정
    if fps < 10 or fps > 120:
        logger.warning(
            f"비정상적인 FPS 값({fps})이 감지되었습니다. 분석의 신뢰도를 위해 기본값 30.0으로 강제 설정합니다."
        )
        fps = 30.0
    # --- 검증 끝 ---

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # [추가] 90도 또는 270도 회전 시 너비와 높이를 교체합니다.
    if config.MANUAL_ROTATE_CODE in [
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ]:
        logger.info(
            f"90도 회전이 감지되어 너비와 높이를 교체합니다: ({width}, {height}) -> ({height}, {width})"
        )
        width, height = height, width

    components = {
        "cap": cap,
        "pose_detector": mp_pose.Pose(
            min_detection_confidence=config.MP_POSE_MIN_DETECTION_CONFIDENCE
        ),
        "ball_tracker": BallTracker(fps=fps),
        "metadata": {
            "fps": fps,
            "width": width,
            "height": height,
            "pixel_to_cm_scale": None,  # 아직 모름
        },
    }
    logger.info("분석 컴포넌트 초기화 완료.")
    return components


# analyzer.py에 추가할 새로운 함수


def _calculate_scale_from_dataframe(df: pd.DataFrame) -> float:
    """
    분석이 완료된 DataFrame 전체에서 유효한 공 반지름 데이터를 수집하고,
    하위 20% 그룹의 중앙값을 사용하여 최종 스케일을 계산합니다.
    """
    logger.info("DataFrame 기반 스케일 계산 시작 (하위 20% 중앙값 방식)...")

    # DataFrame에서 유효한 반지름 값만 모두 추출
    valid_radii_list = df["ball_radius"].dropna().tolist()

    pixel_to_cm_scale = None
    if (
        len(valid_radii_list) >= 10
    ):  # 영상 전체에서 최소 10개 이상의 유효 샘플이 있을 때

        # 1. 반지름 리스트를 오름차순으로 정렬
        sorted_radii = sorted(valid_radii_list)

        # 2. 하위 20%에 해당하는 인덱스 계산 (최소 1개는 있도록 보장)
        percentile_index = max(1, int(len(sorted_radii) * 0.2))

        # 3. 하위 20% 그룹의 데이터를 추출
        bottom_20_percent_radii = sorted_radii[:percentile_index]

        # 4. 이 그룹의 중앙값을 최종 기준으로 사용
        final_radius_px = float(np.median(bottom_20_percent_radii))

        logger.info(
            f"영상 전체 유효 반지름 {len(valid_radii_list)}개 중, 하위 20% 그룹(n={len(bottom_20_percent_radii)})의 중앙값 사용: {final_radius_px:.2f}px"
        )

        # 'utils' 모듈의 함수를 직접 호출하거나, 해당 공식을 여기에 구현
        # 가정: utils.calculate_dynamic_pixel_to_cm_scale(radius, diameter)
        pixel_to_cm_scale = utils.calculate_dynamic_pixel_to_cm_scale(
            final_radius_px, config.REAL_BALL_DIAMETER_CM
        )
    else:
        logger.error(
            f"영상 전체에서 유효한 공을 충분히 찾지 못해({len(valid_radii_list)}개) 스케일 계산에 실패했습니다."
        )

    # 최종적으로 안전한 값인지 확인 후 반환
    final_scale = utils.get_safe_pixel_to_cm_scale(pixel_to_cm_scale)
    logger.info(f"계산된 최종 스케일: {final_scale:.4f} cm/pixel")

    return final_scale


def _calculate_initial_scale(video_path: str, yolo_model) -> float:
    """
    (최종 개선 버전)
    화면 비율 필터링과 중앙값 계산을 조합하여 가장 안정적인 스케일을 계산합니다.
    """
    logger.info("초기 픽셀-cm 스케일 계산 시작 (비율/중앙값 방식)...")
    temp_cap = cv2.VideoCapture(video_path)
    if not temp_cap.isOpened():
        logger.error("스케일 계산을 위한 비디오를 열 수 없습니다.")
        return config.DEFAULT_PIXEL_TO_CM_SCALE

    # --- 프레임 정보 및 샘플링 설정 ---
    total_frames_in_video = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    initial_sampling_interval_scale = 1
    if (
        config.SAMPLING_RATE_TARGET_FRAMES > 0
        and total_frames_in_video > config.INITIAL_FRAMES_FOR_BALL_SIZE_EST * 2
    ):
        base_main_sampling = max(
            1, total_frames_in_video // config.SAMPLING_RATE_TARGET_FRAMES
        )
        initial_sampling_interval_scale = max(
            1, base_main_sampling // 2 if base_main_sampling > 1 else 1
        )

    # --- 초기 프레임 분석 루프 ---
    valid_radii_list = []  # 유효한 반지름 값들을 저장할 리스트
    frames_processed_for_scale_count = 0
    temp_frame_idx_scale = 0

    while (
        temp_cap.isOpened()
        and frames_processed_for_scale_count < config.INITIAL_FRAMES_FOR_BALL_SIZE_EST
    ):
        ret_temp, frame_temp = temp_cap.read()
        if not ret_temp:
            break

        if temp_frame_idx_scale % initial_sampling_interval_scale == 0:
            yolo_preds_temp = yolo_model.predict(
                source=frame_temp,
                classes=[0],
                conf=config.BALL_SIZE_EST_MIN_CONFIDENCE,
                verbose=False,
            )
            if yolo_preds_temp and yolo_preds_temp[0].boxes:
                # 해당 프레임에서 신뢰도가 가장 높은 공 하나만 선택
                best_box = max(yolo_preds_temp[0].boxes, key=lambda box: box.conf[0])
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
                radius_px = (x2 - x1) / 2.0

                # --- 1단계: 비율 기반 필터링 ---
                # 화면 높이 대비 공 반지름의 비율을 계산
                ratio = radius_px / frame_height
                MIN_RADIUS_RATIO = 0.01  # 화면 높이의 1% 이상
                MAX_RADIUS_RATIO = 0.20  # 화면 높이의 20% 이하 (너무 가까운 경우 제외)

                if MIN_RADIUS_RATIO < ratio < MAX_RADIUS_RATIO:
                    valid_radii_list.append(radius_px)
                else:
                    logger.warning(
                        f"비정상적인 공 크기 비율({ratio:.2f})이 감지되어 스케일 계산에서 제외합니다."
                    )

            frames_processed_for_scale_count += 1
        temp_frame_idx_scale += 1
    temp_cap.release()

    # --- 최종 스케일 계산 ---
    pixel_to_cm_scale = None
    if valid_radii_list:
        # --- 2단계: 중앙값(Median) 사용 ---
        # 수집된 모든 유효 반지름 값들의 중앙값을 최종 기준으로 사용
        median_radius_px = np.median(valid_radii_list)
        logger.info(
            f"수집된 유효 반지름 {len(valid_radii_list)}개의 중앙값: {median_radius_px:.2f}px"
        )

        pixel_to_cm_scale = utils.calculate_dynamic_pixel_to_cm_scale(
            median_radius_px, config.REAL_BALL_DIAMETER_CM
        )
    else:
        logger.error(
            "초기 프레임에서 유효한 공을 찾지 못해 스케일 계산에 실패했습니다."
        )

    # 최종적으로 안전한 값인지 확인 후 반환
    final_scale = utils.get_safe_pixel_to_cm_scale(pixel_to_cm_scale)
    logger.info(f"계산된 최종 스케일: {final_scale:.4f} cm/pixel")

    logger.info(
        f"[DEBUG] 유효 반지름 샘플 수 (valid_radii_list): {len(valid_radii_list)}"
    )
    if valid_radii_list:
        # valid_radii_list는 숫자 리스트이므로 그대로 출력합니다.
        logger.info(f"[DEBUG] 유효 반지름 리스트 내용: {valid_radii_list}")
    # ▲▲▲ 올바르게 수정된 디버깅 로그 ▲▲▲

    return final_scale


def _extract_frame_data(components: dict) -> dict:
    """비디오의 모든 프레임을 처리하여 기초 데이터를 수집합니다."""
    # (이전과 동일)
    logger.info("프레임별 데이터 수집 시작...")
    cap, pose_detector, ball_tracker = (
        components["cap"],
        components["pose_detector"],
        components["ball_tracker"],
    )
    all_frames, all_pose_landmarks, all_ball_infos, all_yolo_results = [], [], [], []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
            # [추가] 설정된 값이 있으면 프레임을 회전시킵니다.

        if config.MANUAL_ROTATE_CODE is not None:
            frame = cv2.rotate(frame, config.MANUAL_ROTATE_CODE)

        all_frames.append(frame.copy())
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose_detector.process(rgb_frame)
        all_pose_landmarks.append(pose_results.pose_landmarks)
        yolo_preds = ball_tracker.yolo_model.predict(
            frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False
        )
        all_yolo_results.append(yolo_preds)  # YOLO 결과 저장
        ball_info = ball_tracker.process_frame(frame, pose_results.pose_landmarks)
        all_ball_infos.append(ball_info)

    logger.info(f"총 {len(all_frames)}개의 프레임 데이터 수집 완료.")
    cap.release()
    pose_detector.close()
    return {
        "frames": all_frames,
        "poses": all_pose_landmarks,
        "balls": all_ball_infos,
        "yolo_results": all_yolo_results,
    }


def _create_analysis_dataframe(
    raw_data: dict, metadata: dict, kicking_foot_preference: str
) -> pd.DataFrame:
    """
    (최종 완전판 - 축약어 통일)
    프레임별 raw 데이터 리스트를 입력받아, 모든 분석 지표가 계산된
    최종 Pandas DataFrame을 생성하여 반환합니다.
    """
    # --- 1. 기초 데이터 생성 ---
    all_frames = raw_data["frames"]
    all_pose_landmarks = raw_data["poses"]
    all_ball_infos = raw_data["balls"]
    pixel_to_cm_scale = metadata["pixel_to_cm_scale"]
    fps = metadata.get("fps", 30.0)

    frame_analysis_data_list = []
    for i, (pose_data_item, ball_data_item) in enumerate(
        zip(all_pose_landmarks, all_ball_infos)
    ):
        frame_entry = {"original_idx": i}
        current_frame_shape = all_frames[i].shape

        landmarks = {
            name: utils.get_specific_landmark_position(
                pose_data_item, landmark_enum, current_frame_shape
            )
            for name, landmark_enum in [
                ("r_shoulder", mp_pose.PoseLandmark.RIGHT_SHOULDER),
                ("l_shoulder", mp_pose.PoseLandmark.LEFT_SHOULDER),
                ("r_hip", mp_pose.PoseLandmark.RIGHT_HIP),
                ("l_hip", mp_pose.PoseLandmark.LEFT_HIP),
                ("r_knee", mp_pose.PoseLandmark.RIGHT_KNEE),
                ("l_knee", mp_pose.PoseLandmark.LEFT_KNEE),
                ("r_ankle", mp_pose.PoseLandmark.RIGHT_ANKLE),
                ("l_ankle", mp_pose.PoseLandmark.LEFT_ANKLE),
                ("r_foot_idx", mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
                ("l_foot_idx", mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            ]
        }
        frame_entry.update({f"{name}_pos": pos for name, pos in landmarks.items()})

        angles = {
            "r_hip_angle": utils.calculate_angle(
                landmarks["r_shoulder"], landmarks["r_hip"], landmarks["r_knee"]
            ),
            "l_hip_angle": utils.calculate_angle(
                landmarks["l_shoulder"], landmarks["l_hip"], landmarks["l_knee"]
            ),
            "r_knee_angle": utils.calculate_angle(
                landmarks["r_hip"], landmarks["r_knee"], landmarks["r_ankle"]
            ),
            "l_knee_angle": utils.calculate_angle(
                landmarks["l_hip"], landmarks["l_knee"], landmarks["l_ankle"]
            ),
            "r_ankle_angle": utils.calculate_angle(
                landmarks["r_knee"], landmarks["r_ankle"], landmarks["r_foot_idx"]
            ),
            "l_ankle_angle": utils.calculate_angle(
                landmarks["l_knee"], landmarks["l_ankle"], landmarks["l_foot_idx"]
            ),
        }
        frame_entry.update(angles)

        ball_center = (
            ball_data_item.get("center") if isinstance(ball_data_item, dict) else None
        )
        dist_r, dist_l, min_dist, kicking_foot, foot_pos, kicking_foot_part = (
            np.nan,
            np.nan,
            np.nan,
            "N/A",
            None,
            "N/A",
        )
        contact_region_on_ball = "N/A"

        if ball_center and pose_data_item:
            if landmarks["r_foot_idx"]:
                dist_r = np.hypot(
                    landmarks["r_foot_idx"][0] - ball_center[0],
                    landmarks["r_foot_idx"][1] - ball_center[1],
                )
            if landmarks["l_foot_idx"]:
                dist_l = np.hypot(
                    landmarks["l_foot_idx"][0] - ball_center[0],
                    landmarks["l_foot_idx"][1] - ball_center[1],
                )

            foot_candidates = []
            if pd.notna(dist_r):
                foot_candidates.append(
                    (dist_r, "right", "toe", landmarks["r_foot_idx"])
                )
            if pd.notna(dist_l):
                foot_candidates.append((dist_l, "left", "toe", landmarks["l_foot_idx"]))

            if foot_candidates:
                candidates_to_check = foot_candidates
                if kicking_foot_preference != "auto":
                    preferred = [
                        c for c in foot_candidates if c[1] == kicking_foot_preference
                    ]
                    if preferred:
                        candidates_to_check = preferred
                min_dist, kicking_foot, _, foot_pos = min(
                    candidates_to_check, key=lambda x: x[0]
                )
                if kicking_foot != "N/A":
                    kicking_foot_part = impact_detection.determine_kick_part(
                        pose_landmarks=pose_data_item,
                        ball_center=ball_center,
                        frame_shape=current_frame_shape,
                        kicking_foot_side=kicking_foot,
                    )
                    if foot_pos and ball_center:
                        vec_x = ball_center[0] - foot_pos[0]
                        vec_y = ball_center[1] - foot_pos[1]
                        contact_region_on_ball = utils.get_ball_contact_region(
                            vec_x, vec_y
                        )

        frame_entry.update(
            {
                "min_dist_r_foot_to_ball": dist_r,
                "min_dist_l_foot_to_ball": dist_l,
                "distance": min_dist,
                "kicking_foot": kicking_foot,
                "kicking_foot_part": kicking_foot_part,
                "contact_region_on_ball": contact_region_on_ball,
                "foot_pos": foot_pos,
                "ball_center": ball_center,
                "ball_radius": (
                    ball_data_item.get("radius")
                    if isinstance(ball_data_item, dict)
                    else np.nan
                ),
                "ball_box": (
                    ball_data_item.get("box")
                    if isinstance(ball_data_item, dict)
                    else None
                ),
                "is_ball_predicted": ball_data_item.get("is_predicted", True),
                "pixel_to_cm_scale": pixel_to_cm_scale,
            }
        )
        frame_analysis_data_list.append(frame_entry)

    analysis_dataframe = pd.DataFrame(frame_analysis_data_list)
    if analysis_dataframe.empty:
        return analysis_dataframe

    # --- 2. 핵심 시계열 지표 계산 ---

    # 거리 스무딩 및 접근 속도/가속도
    analysis_dataframe["distance"].fillna(float("inf"), inplace=True)
    analysis_dataframe["distance_smoothed"] = (
        analysis_dataframe["distance"]
        .rolling(window=config.SMOOTHING_WINDOW_SIZE, center=True, min_periods=1)
        .mean()
    )
    analysis_dataframe["foot_ball_approach_velocity"] = analysis_dataframe[
        "distance_smoothed"
    ].diff()
    analysis_dataframe["foot_ball_approach_acceleration"] = analysis_dataframe[
        "foot_ball_approach_velocity"
    ].diff()

    # 각 발의 속도 (px/frame)
    # 각 발의 속도 (px/frame)
    for foot_side in ["right", "left"]:
        prefix = "r" if foot_side == "right" else "l"
        pos_col = f"{prefix}_foot_idx_pos"
        speed_col = f"{prefix}_foot_speed"

        # [수정] 해당 발의 좌표 데이터가 유효한지 먼저 확인합니다.
        valid_coords_series = analysis_dataframe[pos_col].dropna()

        if valid_coords_series.empty:
            # 유효한 좌표가 하나도 없으면, 속도를 0으로 설정하고 계산을 건너뜁니다.
            analysis_dataframe[speed_col] = 0.0
        else:
            # 유효한 좌표가 있을 때만 속도를 계산합니다.
            coords = pd.DataFrame(
                valid_coords_series.tolist(),
                index=valid_coords_series.index,
                columns=["x", "y"],
            )
            # [수정] .diff() 계산 결과에서 첫 NaN값을 제외하기 위해 [1:]을 추가합니다.
            velocity_values = np.linalg.norm(coords.diff().to_numpy(), axis=1)[1:]
            velocity = pd.Series(velocity_values, index=coords.index[1:])

            # 계산된 속도를 원래 데이터프레임에 합칩니다. (NaN으로 비는 부분은 나중에 채워집니다)
            analysis_dataframe[speed_col] = velocity

    # 주발 속도
    analysis_dataframe["kicking_foot_speed"] = analysis_dataframe.apply(
        lambda row: (
            row["r_foot_speed"]
            if row["kicking_foot"] == "right"
            else (row["l_foot_speed"] if row["kicking_foot"] == "left" else np.nan)
        ),
        axis=1,
    )

    # 롤링 평균을 사용하여 발 속도 데이터를 부드럽게 만듭니다.
    analysis_dataframe["kicking_foot_speed_smoothed"] = (
        analysis_dataframe["kicking_foot_speed"]
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )

    # 부드러워진 속도 데이터로 가속도를 계산합니다.
    analysis_dataframe["kicking_foot_acceleration_scalar"] = analysis_dataframe[
        "kicking_foot_speed_smoothed"  # [수정] _smoothed가 붙은 컬럼 사용
    ].diff()

    # # km/h 단위 변환
    # analysis_dataframe["kicking_foot_speed_kmh"] = analysis_dataframe[
    #     "kicking_foot_speed"
    # ].apply(lambda x: utils.convert_speed_to_kmh_normalized(x, pixel_to_cm_scale, fps))

    # 디딤발 안정성
    def get_supporting_foot_ankle_pos(row):
        # kicking_foot 이 유효하지 않으면 (nan,nan) 반환
        if row.get("kicking_foot") not in ("right", "left"):
            return (np.nan, np.nan)

        # 실제 포지션 가져오기 (축약어 컬럼명 사용)
        pos = (
            row["l_ankle_pos"] if row["kicking_foot"] == "right" else row["r_ankle_pos"]
        )

        # 튜플 형태가 아니면 (nan,nan) 으로 보정
        if not (isinstance(pos, tuple) and len(pos) == 2):
            return (np.nan, np.nan)
        return pos

    analysis_dataframe["supporting_foot_ankle_pos"] = analysis_dataframe.apply(
        get_supporting_foot_ankle_pos, axis=1
    )

    supp_coords = pd.DataFrame(
        analysis_dataframe["supporting_foot_ankle_pos"].tolist(),
        index=analysis_dataframe.index,
        columns=["x", "y"],
    )
    supp_coords_smooth = supp_coords.rolling(
        window=9, center=True, min_periods=1
    ).mean()

    displacement_values = np.linalg.norm(supp_coords_smooth.diff().to_numpy(), axis=1)[
        1:
    ]
    displacement_series = pd.Series(displacement_values, index=supp_coords.index[1:])
    analysis_dataframe["supporting_foot_displacement"] = displacement_series.where(
        displacement_series < 50, np.nan
    )

    # --- 3. 공의 운동 정보 계산 및 병합 ---
    df_ball_kinematics = pd.DataFrame()
    if (
        "ball_center" in analysis_dataframe.columns
        and not analysis_dataframe["ball_center"].isna().all()
    ):
        valid_indices = analysis_dataframe[
            ~analysis_dataframe["is_ball_predicted"]
            & analysis_dataframe["ball_center"].notna()
        ].index
        if len(valid_indices) > 1:
            ball_coords = pd.DataFrame(
                analysis_dataframe.loc[valid_indices, "ball_center"].tolist(),
                index=valid_indices,
                columns=["x", "y"],
            )
            ball_velocity = np.linalg.norm(ball_coords.diff().to_numpy(), axis=1)
            df_ball_kinematics = pd.DataFrame(
                {"ball_speed": ball_velocity[1:]}, index=ball_coords.index[1:]
            )
            df_ball_kinematics["ball_acceleration"] = df_ball_kinematics[
                "ball_speed"
            ].diff()

    analysis_dataframe = pd.merge(
        analysis_dataframe,
        df_ball_kinematics,
        left_index=True,
        right_index=True,
        how="left",
    )
    # ↓ 여기에 추가 ↓
    for col in ["ball_speed", "ball_acceleration"]:
        if col not in analysis_dataframe.columns:
            analysis_dataframe[col] = 0.0

    # # analysis_dataframe["ball_speed_kmh"] = analysis_dataframe["ball_speed"].apply(
    #     lambda x: utils.convert_speed_to_kmh_normalized(x, pixel_to_cm_scale, fps)
    # )

    # --- 4. 최종 후처리 및 반환 ---
    cols_to_fillna = {
        "foot_ball_approach_velocity": 0.0,
        "foot_ball_approach_acceleration": 0.0,
        "kicking_foot_speed": 0.0,
        "kicking_foot_acceleration_scalar": 0.0,
        "r_foot_speed": 0.0,
        "l_foot_speed": 0.0,
        "supporting_foot_displacement": 0.0,
        "ball_speed": 0.0,
        "ball_acceleration": 0.0,
        # "kicking_foot_speed_kmh": 0.0,
        # "ball_speed_kmh": 0.0,
    }
    analysis_dataframe.fillna(value=cols_to_fillna, inplace=True)

    return analysis_dataframe


def _find_key_events(analysis_df: pd.DataFrame, kicking_foot_preference: str) -> dict:
    """
    (수정된 최종 버전)
    impact_detection 모듈을 직접 호출하여 모든 핵심 이벤트를 탐지합니다.
    """
    logger.info("핵심 이벤트(임팩트, 백스윙) 탐지 시작...")

    # ✨ impact_detection.py의 메인 함수를 직접 호출
    key_events = impact_detection.find_key_events(analysis_df, kicking_foot_preference)

    # 여기서는 보간(refine) 로직을 제거하고 impact_detection에 위임합니다.
    # 필요하다면 key_events 딕셔너리에 refined_details를 포함시킬 수 있습니다.

    if not key_events:
        logger.error("impact_detection 모듈에서 이벤트를 찾지 못했습니다.")
        return {}

    logger.info(f"핵심 이벤트 탐지 완료: {key_events}")
    return key_events


# 1) 타격 발 부위, (2) 공 타격 지점, (3) 디딤발-공 거리, (4) 선수 자세 각도 등
# 위치와 관련된 모든 지표들은 이 정밀 시점에 맞춰 재계산하는 것이 분석의 정확도를 높이는 핵심


def _prepare_final_infos(
    analysis_df: pd.DataFrame, key_events: dict, metadata: dict
) -> Tuple[Optional[dict], Optional[dict]]:

    logger.info("점수 계산을 위한 최종 정보 패키지 준비 시작...")

    # --- metadata에서 두 종류의 스케일과 fps를 미리 가져옵니다. ---
    initial_scale = metadata.get("initial_scale")
    ball_flight_scale = metadata.get("ball_flight_scale")
    fps = metadata.get("fps", 30.0)

    impact_idx = key_events.get("impact_frame_index")
    backswing_idx = key_events.get("backswing_frame_index", -1)
    kicking_foot = key_events.get("kicking_foot")

    # --- 백스윙 정보 준비 ---
    final_backswing_info = {}
    if backswing_idx != -1 and backswing_idx in analysis_df.index:
        final_backswing_info = analysis_df.loc[backswing_idx].to_dict()

        # # [수정 1] 최대 발 스윙 속도를 계산하고, 'initial_scale'로 km/h 변환
        # if (
        #     kicking_foot
        #     and f"{'r' if kicking_foot == 'right' else 'l'}_foot_idx_pos"
        #     in analysis_df.columns
        # ):
        #     kicking_foot_pos_col = (
        #         f"{'r' if kicking_foot == 'right' else 'l'}_foot_idx_pos"
        #     )
        #     swing_window = analysis_df.loc[
        #         backswing_idx:impact_idx, kicking_foot_pos_col
        #     ].dropna()
        #     if len(swing_window) > 1:
        #         swing_speeds_px_fr = np.linalg.norm(
        #             np.diff(np.array(swing_window.to_list()), axis=0), axis=1
        #         )
        #         max_foot_speed_px_fr = (
        #             np.max(swing_speeds_px_fr) if swing_speeds_px_fr.size > 0 else 0
        #         )
        #         # km/h 변환 결과를 딕셔너리에 직접 추가
        #         final_backswing_info["max_foot_swing_speed_kmh"] = (
        #             utils.convert_speed_to_kmh_normalized(
        #                 max_foot_speed_px_fr, initial_scale, fps
        #             )
        #         )
        # in analyzer.py -> _prepare_final_infos

        # [최종 수정] 최대 발 스윙 속도를 '발목' 기준으로 계산하고 보정 계수를 적용
        if kicking_foot:
            # 1. 측정 기준점을 안정적인 '발목'으로 설정
            kicking_foot_pos_col = (
                f"{'r' if kicking_foot == 'right' else 'l'}_ankle_pos"
            )

            if kicking_foot_pos_col in analysis_df.columns:
                swing_window = analysis_df.loc[
                    backswing_idx:impact_idx, kicking_foot_pos_col
                ].dropna()
                if len(swing_window) > 1:
                    swing_speeds_px_fr = np.linalg.norm(
                        np.diff(np.array(swing_window.to_list()), axis=0), axis=1
                    )
                    max_foot_speed_px_fr = (
                        np.max(swing_speeds_px_fr) if swing_speeds_px_fr.size > 0 else 0
                    )

                    # 2. 안정적인 발목 속도를 km/h로 계산
                    ankle_speed_kmh = utils.convert_speed_to_kmh_normalized(
                        max_foot_speed_px_fr, initial_scale, fps
                    )

                    # 3. 보정 계수를 곱해 '추정 발끝 속도' 계산
                    estimated_foot_tip_speed_kmh = (
                        ankle_speed_kmh * config.CORRECTION_FACTOR
                    )

                    # 4. 최종 결과를 딕셔너리에 저장
                    final_backswing_info["max_foot_swing_speed_kmh"] = (
                        estimated_foot_tip_speed_kmh
                    )

                    logger.info(
                        f"측정된 발목 속도: {ankle_speed_kmh:.1f}km/h, 보정된 최종 발끝 속도: {estimated_foot_tip_speed_kmh:.1f}km/h"
                    )

        # [수정 2] 디딤발 안정성을 계산하고, 'initial_scale'로 cm/frame 변환
        supporting_foot_col = (
            "l_ankle_pos" if kicking_foot == "right" else "r_ankle_pos"
        )
        if supporting_foot_col in analysis_df.columns:
            stability_window_start = max(
                0, impact_idx - config.SUPPORTING_FOOT_STABILITY_WINDOW_SIZE
            )
            stability_window_df = analysis_df.loc[
                stability_window_start:impact_idx, supporting_foot_col
            ].dropna()
            if len(stability_window_df) > 1:
                positions = np.array(stability_window_df.to_list())
                displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                support_stability_px_fr = (
                    np.mean(displacements) if displacements.size > 0 else 0
                )
                # cm/frame 변환 결과를 딕셔너리에 직접 추가
                final_backswing_info["support_foot_stability_cm_fr"] = (
                    support_stability_px_fr * initial_scale
                )

    else:
        logger.warning("유효한 백스윙 정보를 찾지 못했습니다.")

    # --- 임팩트 정보 준비 ---
    if impact_idx is None:
        logger.error("임팩트 정보를 찾을 수 없어 최종 정보 준비에 실패했습니다.")
        return None, final_backswing_info

    final_impact_info = analysis_df.loc[impact_idx].to_dict()

    # [수정 3] 공 초기 속도를 계산할 때, 'ball_flight_scale' 사용
    search_start = impact_idx + 1
    search_end = min(
        len(analysis_df), impact_idx + config.MAX_FRAMES_FOR_MAX_BALL_SPEED_SEARCH
    )
    ball_positions = analysis_df.loc[search_start:search_end, "ball_center"].dropna()

    # utils.calculate_ball_speed 함수에 'ball_flight_scale'을 명시적으로 전달
    ball_speed_results = utils.calculate_ball_speed(
        positions=ball_positions, fps=fps, scale_cm_per_px=ball_flight_scale
    )
    final_impact_info["ball_speed_results"] = ball_speed_results

    # 👇 보간된 정밀 데이터가 있는지 확인
    refined_details = key_events.get("refined_impact_details")
    if refined_details and refined_details.get("refined_original_idx") is not None:
        logger.info("보간된 정밀 데이터를 기반으로 최종 임팩트 정보를 재생성합니다.")

        # 1. 보간된 정보를 담을 딕셔너리를 원본 정보의 복사본으로 만듭니다.
        refined_impact_info = final_impact_info.copy()
        refined_idx = refined_details["refined_original_idx"]

        # 2. 보간된 시점(refined_idx)의 각 신체 부위 좌표를 새로 계산합니다.
        point_columns_to_interpolate = [
            "r_shoulder_pos",
            "l_shoulder_pos",
            "r_hip_pos",
            "l_hip_pos",
            "r_knee_pos",
            "l_knee_pos",
            "r_ankle_pos",
            "l_ankle_pos",
            "r_foot_idx_pos",
            "l_foot_idx_pos",
            "ball_center",
            "foot_pos",
        ]

        interpolated_points = {}
        for col in point_columns_to_interpolate:
            if col in analysis_df.columns:
                result = utils.interpolate_point_data_with_quality_and_fallback(
                    analysis_df, col, refined_idx
                )
                interpolated_points[col] = result.get("interpolated_point")
                if interpolated_points[col] is None:
                    logger.warning(f"[보간 경고] {col} 보간 결과가 유효하지 않습니다.")

        # 3. 새로 계산된 좌표들로 각도, 거리 등 점수 항목을 업데이트합니다.
        # 3-1. 각도 재계산
        angles_to_refine = {
            "l_hip_angle": ("l_shoulder_pos", "l_hip_pos", "l_knee_pos"),
            "r_hip_angle": ("r_shoulder_pos", "r_hip_pos", "r_knee_pos"),
            "l_knee_angle": ("l_hip_pos", "l_knee_pos", "l_ankle_pos"),
            "r_knee_angle": ("r_hip_pos", "r_knee_pos", "r_ankle_pos"),
            "l_ankle_angle": ("l_knee_pos", "l_ankle_pos", "l_foot_idx_pos"),
            "r_ankle_angle": ("r_knee_pos", "r_ankle_pos", "r_foot_idx_pos"),
        }

        for angle_name, point_keys in angles_to_refine.items():
            refined_angle = utils.calculate_angle(
                interpolated_points.get(point_keys[0]),
                interpolated_points.get(point_keys[1]),
                interpolated_points.get(point_keys[2]),
            )
            if not pd.notna(refined_angle):
                refined_angle = final_impact_info.get(
                    angle_name
                )  # 보간 실패 시 원본 값 사용
            refined_impact_info[angle_name] = refined_angle

        # 3-2. 디딤발-공 거리 재계산
        kicking_foot_side = refined_impact_info.get("kicking_foot")
        supporting_foot_col_name = (
            "l_ankle_pos" if kicking_foot_side == "right" else "r_ankle_pos"
        )

        interp_support_ankle = interpolated_points.get(supporting_foot_col_name)
        interp_ball_center = interpolated_points.get("ball_center")
        if interp_support_ankle and interp_ball_center:
            dist_px = np.hypot(
                interp_support_ankle[0] - interp_ball_center[0],
                interp_support_ankle[1] - interp_ball_center[1],
            )
            # cm 변환 결과를 딕셔너리에 직접 추가
            refined_impact_info["dist_ball_to_supporting_foot_ankle_cm"] = (
                dist_px * initial_scale
            )

        # 3-3. 공 타격 지점 재계산
        interp_foot_pos = interpolated_points.get("foot_pos")
        if interp_foot_pos and interp_ball_center:
            vec_x = interp_ball_center[0] - interp_foot_pos[0]
            vec_y = interp_ball_center[1] - interp_foot_pos[1]
            refined_impact_info["contact_region_on_ball"] = (
                utils.get_ball_contact_region(vec_x, vec_y)
            )

        #  [수정된 핵심] 최종적으로 '보간된 정보'를 반환합니다.
        return refined_impact_info, final_backswing_info

    # 보간된 정보가 없으면, 원본 임팩트 정보를 반환합니다.
    return final_impact_info, final_backswing_info


def _generate_outputs(
    raw_data: dict,
    frames: List,
    analysis_df: pd.DataFrame,
    key_events: dict,
    score_data: dict,
    metadata: dict,
    job_id: str,
    # ▼▼▼ feedback_data 인자 추가 ▼▼▼
    feedback_data: dict,
) -> dict:
    """최종 결과물(영상, 캡쳐 이미지)을 생성하고 경로를 반환합니다."""
    logger.info(f"[{job_id}] 최종 결과물 생성 시작...")

    impact_idx = key_events.get("impact_frame_index", -1)
    backswing_idx = key_events.get("backswing_frame_index", -1)

    impact_img_path = os.path.join(config.OUTPUTS_DIR, f"{job_id}_impact.jpg")
    backswing_img_path = os.path.join(config.OUTPUTS_DIR, f"{job_id}_backswing.jpg")

    # # 1. 캡쳐 이미지 저장
    # impact_img_path = os.path.join(config.OUTPUTS_DIR, f"{job_id}_impact.jpg")
    # backswing_img_path = os.path.join(config.OUTPUTS_DIR, f"{job_id}_backswing.jpg")

    # if 0 <= impact_idx < len(frames):
    #     cv2.imwrite(impact_img_path, frames[impact_idx])
    # if 0 <= backswing_idx < len(frames):
    #     cv2.imwrite(backswing_img_path, frames[backswing_idx])

    # 2. 분석 영상 생성
    output_video_path = os.path.join(config.OUTPUTS_DIR, f"{job_id}_result.mp4")
    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata["fps"],
        (metadata["width"], metadata["height"]),
    )

    # 3. 모든 프레임을 순회하며 영상 제작 및 캡처 이미지 저장
    for i, frame in enumerate(frames):
        row = analysis_df.loc[i]
        pose_landmarks = raw_data["poses"][i]
        ball_info = raw_data["balls"][i]

        current_analysis_info = row.to_dict()
        current_analysis_info["is_impact_frame"] = i == impact_idx
        current_analysis_info["is_backswing_frame"] = i == backswing_idx

        # 모든 시각화 요소가 포함된 프레임 생성
        annotated_frame = visualizer.generate_annotated_frame(
            frame, pose_landmarks, ball_info, current_analysis_info
        )

        # 해당 프레임이 백스윙 또는 임팩트 시점이면, 이 'annotated_frame'을 이미지로 저장
        if current_analysis_info["is_backswing_frame"]:
            cv2.imwrite(backswing_img_path, annotated_frame)
            logger.info(f"백스윙 캡처 이미지 저장: {backswing_img_path}")

        if current_analysis_info["is_impact_frame"]:
            cv2.imwrite(impact_img_path, annotated_frame)
            logger.info(f"임팩트 캡처 이미지 저장: {impact_img_path}")

        # 영상에는 모든 프레임을 씀
        video_writer.write(annotated_frame)

    # 3. 점수 카드 프레임 추가
    if score_data:
        score_card_frame = visualizer.create_score_card_frame(
            frames[-1], score_data, feedback_data
        )
        for _ in range(int(metadata["fps"] * 3)):  # 3초간 보여주기
            video_writer.write(score_card_frame)

    video_writer.release()
    logger.info("최종 결과물 생성 완료.")

    return {
        "video": output_video_path,
        "impact_image": impact_img_path,
        "backswing_image": backswing_img_path,
    }


# in src/analyzer.py


# --- 메인 분석 함수 (수정된 최종 완성본) ---
def run_full_analysis(
    video_path: str, kicking_foot_preference: str, job_id: str
) -> Tuple[Dict, str, str, str]:
    """
    (최종 '투-스케일' 버전)
    분석의 전체 과정을 순서대로 지휘합니다.
    """
    logger.info(f"[{job_id}] 전체 분석 시작 ('투-스케일' 방식): {video_path}")

    try:
        # --- 1. 초기 설정 및 '초기 스케일' 계산 ---
        logger.info(
            f"[{job_id}] 1단계: 컴포넌트 초기화 및 근거리용 초기 스케일 계산..."
        )
        components = _initialize_components(video_path)
        metadata = components["metadata"]
        yolo_model = components["ball_tracker"].yolo_model

        # ▼▼▼ 첫 번째 스케일 계산 위치 ▼▼▼
        # 영상 초반의 데이터를 기반으로 '근거리용' 스케일을 먼저 계산합니다.
        initial_scale = _calculate_initial_scale(video_path, yolo_model)
        metadata["initial_scale"] = initial_scale
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # --- 2. 전체 프레임 데이터 추출 및 DataFrame 생성 ---
        logger.info(f"[{job_id}] 2단계: 전체 프레임 데이터 분석 및 DataFrame 생성...")
        raw_data = _extract_frame_data(components)
        analysis_df = _create_analysis_dataframe(
            raw_data, metadata, kicking_foot_preference
        )
        if analysis_df.empty:
            raise ValueError("데이터프레임 생성 실패")

        # --- 3. '비행 스케일' 계산 및 최종 분석 준비 ---
        logger.info(f"[{job_id}] 3단계: DataFrame 기반 원거리용 비행 스케일 계산...")

        # ▼▼▼ 두 번째 스케일 계산 위치 ▼▼▼
        # 생성된 DataFrame 전체를 바탕으로 '원거리용' 스케일을 계산합니다.
        ball_flight_scale = _calculate_scale_from_dataframe(analysis_df)
        metadata["ball_flight_scale"] = ball_flight_scale
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 이제 두 종류의 스케일이 모두 metadata에 저장되었습니다.
        logger.info(
            f"모든 스케일 계산 완료. Initial(근거리용): {initial_scale:.4f}, Flight(원거리용): {ball_flight_scale:.4f}"
        )
        # 4. 핵심 이벤트 탐지 (정확한 impact_idx를 여기서 찾아냄)
        key_events = impact_detection.find_key_events(
            analysis_df, kicking_foot_preference
        )
        if not key_events:
            raise ValueError("핵심 이벤트 탐지 실패")

        impact_idx = key_events.get("impact_frame_index", -1)
        logger.info(f"[{job_id}] 1차 분석 완료. 임팩트 프레임 탐지: {impact_idx}")

        # =================================================================
        # == 2차 분석 (Pass 2): 이벤트 기반 정밀 공 재추적 -> yolo 성능 향상으로 필요 없어짐 & 공 속도 오히려 왜곡 => off
        # =================================================================
        # logger.info(f"[{job_id}] 2차 분석 시작: 임팩트 정보 기반 정밀 공 재추적...")

        # # 새로운 BallTracker를 만들어 상태를 초기화
        # refined_ball_tracker = BallTracker(fps=metadata["fps"])
        # refined_ball_infos = []

        # # 메모리에 있는 raw_data를 다시 순회 (비디오 파일을 다시 읽지 않음)
        # for i, (frame, pose_landmarks, yolo_res) in enumerate(
        #     zip(raw_data["frames"], raw_data["poses"], raw_data["yolo_results"])
        # ):
        #     pose_landmarks = raw_data["poses"][i]

        #
        #     if i == impact_idx:
        #         # 1차 분석에서 찾은 정확한 임팩트 순간의 공 정보를 가져옵니다.
        #         impact_ball_info = key_events.get("initial_impact_info", {})
        #         impact_ball_center = impact_ball_info.get("ball_center")
        #         impact_ball_radius = impact_ball_info.get("ball_radius")

        #         # 해당 정보가 유효할 경우, 추적기를 이 공에 강제로 재초기화합니다.
        #         if impact_ball_center and impact_ball_radius:
        #             refined_ball_tracker.force_reinitialize(
        #                 impact_ball_center, impact_ball_radius
        #             )
        #             logger.info(
        #                 f"[{job_id}] Frame {i}: BallTracker를 액션 볼에 강제 재초기화 완료."
        #             )
        #         else:
        #             # 만약을 위한 대비책
        #             refined_ball_tracker.notify_impact_detected()
        #     # ...

        #     # 새로워진 상태로 프레임 처리
        #     ball_info = refined_ball_tracker.process_frame(
        #         frame, pose_landmarks, yolo_results=yolo_res
        #     )
        #     refined_ball_infos.append(ball_info)

        # # 1차 분석 때의 부정확한 공 정보를, 2차 분석의 정제된 정보로 교체
        # raw_data["balls"] = refined_ball_infos

        # # ✨ 정제된 공 정보로 데이터프레임을 다시 생성하거나 업데이트 (여기서는 재생성을 선택)
        # logger.info(f"[{job_id}] 정제된 공 정보로 분석 데이터프레임 재생성...")
        # analysis_df = _create_analysis_dataframe(
        #     raw_data, metadata, kicking_foot_preference
        # )
        # if analysis_df.empty:
        #     raise ValueError("정제된 데이터프레임 생성 실패")
        # # analyzer.py의 _create_analysis_dataframe 함수 마지막에 추가
        # analysis_df["distance_smoothed"] = pd.to_numeric(
        #     analysis_df["distance_smoothed"], errors="coerce"
        # )

        # =================================================================
        # == 최종 처리 단계 (Scoring, Feedback, Output)
        # =================================================================

        # 5. 점수 계산용 최종 정보 준비 (이제 정확한 공 궤적을 바탕으로 함)
        final_impact_info, final_backswing_info = _prepare_final_infos(
            analysis_df, key_events, metadata
        )

        # 6. 점수 및 피드백 생성
        score_data = scoring.calculate_kick_score(
            final_impact_info, final_backswing_info, metadata
        )
        feedback_data = feedback_generator.generate_feedback(score_data)

        # 7. 결과물 생성
        output_paths = _generate_outputs(
            raw_data,
            raw_data["frames"],
            analysis_df,
            key_events,
            score_data,
            metadata,
            job_id,
            # ▼▼▼ feedback_data 전달 ▼▼▼
            feedback_data,
        )

        # 8. 최종 결과 종합
        final_data_package = {
            "scores": score_data,
            "feedback": feedback_data,
            "key_events": key_events,
        }
        logger.info(f"[{job_id}] 모든 분석 과정 성공적으로 완료.")

        return (
            final_data_package,
            output_paths["video"],
            output_paths["backswing_image"],
            output_paths["impact_image"],
        )

    except Exception as e:
        logger.error(
            f"[{job_id}] 분석 파이프라인 실행 중 오류 발생: {e}", exc_info=True
        )
        return {}, "", "", ""
