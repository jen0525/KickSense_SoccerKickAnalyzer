# src/impact_detection.py
"""
분석 데이터프레임을 입력받아,
핵심 이벤트(임팩트, 백스윙)의 프레임 인덱스를 찾아내는 전문 모듈.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from . import config, utils
from .utils import setup_logger
from scipy.interpolate import interp1d

logger = setup_logger(__name__)


# --- 내부 헬퍼 함수 ---
# ⚽️ 현재 인덱스 기준 주어진 반경 내에서 해당 값이 최소값인지 확인
def is_distance_minimum_in_window(
    analysis_df_arg, current_idx, window_radius, col_name="distance_smoothed"
):
    start_idx = max(0, current_idx - window_radius)
    end_idx = min(len(analysis_df_arg) - 1, current_idx + window_radius)
    if not (0 <= current_idx < len(analysis_df_arg)):
        return False
    current_dist_val = analysis_df_arg.loc[current_idx, col_name]
    if pd.isna(current_dist_val):
        return False
    window_values = analysis_df_arg.loc[start_idx:end_idx, col_name].dropna()
    if window_values.empty:
        return True
    return current_dist_val <= window_values.min()


import pandas as pd
import numpy as np
from . import config, utils


def _find_impact_candidates(
    analysis_df: pd.DataFrame, kicking_foot_preference: str
) -> list:
    """
    최적화된 임팩트 후보 탐지 함수.
    Distance, Proximity, Contact, Dynamics 네 요소에 local-minimum 보너스를 더해
    가중합 점수로 상위 후보를 리턴합니다.
    """
    impact_candidate_list = []

    # 1) 1차 Distance 필터링으로 연산량 절감
    close_df = analysis_df[analysis_df["distance"] < config.IMPACT_DISTANCE_MAX].copy()
    if close_df.empty:
        logger.warning(
            "임팩트 후보로 판단될 만큼 가까운 프레임이 없습니다. Fallback 실행."
        )
        idx = analysis_df["distance"].idxmin()
        return [{"df_idx": int(idx), "score": 0.0}]

    for idx, row in close_df.iterrows():
        # 공 예측 프레임, 잘못된 foot_pos 스킵
        if row.get("is_ball_predicted", False):
            continue
        fp = row.get("foot_pos")
        if not (isinstance(fp, tuple) and len(fp) == 2):
            continue

        # 킥 발 정보 유효성 체크
        kicking_side = row.get("kicking_foot")
        if kicking_side not in ("right", "left"):
            continue

        # --- 2) Distance Score (0~1)
        dist = row.get("distance", float("inf"))
        MAX_D = config.IMPACT_DISTANCE_MAX
        distance_score = max(0.0, (MAX_D - dist) / MAX_D)

        # --- 3) Proximity Score (0~1, capped)
        ball_rad = row.get("ball_radius", 0.0) or 0.0
        prox_thresh = ball_rad * config.BETA_PROXIMITY_SCALE
        raw_prox = max(0.0, (prox_thresh - dist) / (prox_thresh + 1e-6))
        proximity_score = min(raw_prox, 1.0)

        # --- 4) Contact Score (IoU 기반 하드 필터 + 스코어)
        iou = 0.0
        bb = row.get("ball_box")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            x, y = fp
            foot_bb = [
                x - config.FOOT_BOX_RADIUS,
                y - config.FOOT_BOX_RADIUS,
                x + config.FOOT_BOX_RADIUS,
                y + config.FOOT_BOX_RADIUS,
            ]
            iou = utils.calculate_iou(foot_bb, bb)
        # 하드 필터: 일정 이하 IoU 는 후보에서 제외 -> 너무 엄격함
        # if iou < config.IOU_CONTACT_THRESHOLD:
        #     continue
        contact_score = min(max(iou, 0.0), 1.0)

        # --- 5) Dynamics Score (속도 차 0~1)
        kick_sp = (
            row.get(f"{ 'r' if kicking_side=='right' else 'l' }_foot_speed", 0.0) or 0.0
        )
        sup_sp = (
            row.get(f"{ 'l' if kicking_side=='right' else 'r' }_foot_speed", 0.0) or 0.0
        )
        diff = max(0.0, kick_sp - sup_sp)
        DYN_MAX = config.IMPACT_DYNAMICS_MAX_SPEED
        dynamics_score = min(diff / (DYN_MAX or 1.0), 1.0)

        # --- 6) Local Minimum Bonus
        bonus = 0.0
        if is_distance_minimum_in_window(
            analysis_df, idx, config.MINIMUM_DISTANCE_WINDOW, "distance_smoothed"
        ):
            bonus = config.IMPACT_WEIGHT_LOCAL_MIN_BONUS

        # --- 7) 최종 가중합
        total_score = (
            distance_score * config.IMPACT_WEIGHT_DISTANCE
            + proximity_score * config.IMPACT_WEIGHT_PROXIMITY
            + contact_score * config.IMPACT_WEIGHT_IOU
            + dynamics_score * config.IMPACT_WEIGHT_DYNAMICS
            + bonus
        )

        impact_candidate_list.append(
            {
                "df_idx": idx,
                "data_row": row,
                "score": total_score,
                "distance_score": distance_score,
                "proximity_score": proximity_score,
                "contact_score": contact_score,
                "dynamics_score": dynamics_score,
                "local_min_bonus": bonus,
            }
        )

    # 후보가 하나도 없으면 fallback
    if not impact_candidate_list:
        logger.warning(
            "후보 프레임이 모두 필터링되었습니다. distance 최저 프레임을 하나 반환합니다."
        )
        idx = analysis_df["distance"].idxmin()
        return [{"df_idx": int(idx), "score": 0.0}]

    # 점수 내림차순 정렬 후 상위 N개 반환
    impact_candidate_list.sort(key=lambda x: x["score"], reverse=True)
    top_n = getattr(config, "IMPACT_CANDIDATE_MAX", 5)
    return impact_candidate_list[:top_n]


def _find_backswing_peaks(analysis_df: pd.DataFrame, impact_idx: int) -> dict:
    """
    백스윙 탐지 최종 수정 버전.
    점수 상위 후보부터 데이터 유효성을 검사하여 가장 좋은 '실제 사용 가능한' 프레임을 찾습니다.
    """
    if impact_idx == -1:
        return {"l": -1, "r": -1}

    search_start_idx = max(0, impact_idx - config.BACKSWING_SEARCH_WINDOW)
    search_df = analysis_df.loc[search_start_idx:impact_idx].copy()

    logger.info(f"백스윙 탐색 (프레임 {search_start_idx} ~ {impact_idx})")
    backswing_peaks = {"l": -1, "r": -1}

    for side_prefix in ["l", "r"]:
        knee_angle_col = f"{side_prefix}_knee_angle"
        foot_dist_col = f"min_dist_{side_prefix}_foot_to_ball"
        foot_pos_col = f"{side_prefix}_ankle_pos"

        required_cols = [knee_angle_col, foot_dist_col, foot_pos_col]
        if not all(col in search_df.columns for col in required_cols):
            logger.warning(f"[{side_prefix}발] 백스윙 분석 스킵: 필수 컬럼 부족")
            continue

        candidates_df = search_df[
            search_df[knee_angle_col] < config.KNEE_ANGLE_THRESHOLD
        ].copy()

        if candidates_df.empty:
            logger.warning(
                f"[{side_prefix}발] 무릎 각도 < {config.KNEE_ANGLE_THRESHOLD} 조건 만족 프레임 없음. "
                "조건을 완화하여 탐색합니다."
            )
            # 조건 만족 프레임이 없으면 전체 탐색 범위에서 점수 기반으로 탐색
            candidates_df = search_df.copy()

        try:
            # === 점수 계산 (이전과 동일) ===
            scores = pd.DataFrame(index=candidates_df.index)
            # 1. 무릎 각도 점수
            knee_values = candidates_df[knee_angle_col].dropna()
            if not knee_values.empty and knee_values.std() > 0:
                scores["knee"] = 1 - (knee_values - knee_values.min()) / (
                    knee_values.max() - knee_values.min()
                )
            else:
                scores["knee"] = 0.5

            # 2. 거리 점수
            dist_values = candidates_df[foot_dist_col].dropna()
            if not dist_values.empty and dist_values.std() > 0:
                scores["dist"] = (dist_values - dist_values.min()) / (
                    dist_values.max() - dist_values.min()
                )
            else:
                scores["dist"] = 0.5

            # 3. 높이 점수
            y_values = candidates_df[foot_pos_col].apply(
                lambda pos: (
                    pos[1]
                    if isinstance(pos, (list, tuple)) and len(pos) > 1
                    else np.nan
                )
            )
            valid_y = y_values.dropna()
            if len(valid_y) > 0 and valid_y.std() > 0:
                normalized_y = 1 - (valid_y - valid_y.min()) / (
                    valid_y.max() - valid_y.min()
                )
                scores.loc[valid_y.index, "height"] = normalized_y
            else:
                scores["height"] = 0.5

            weights = {"knee": 0.80, "dist": 0.10, "height": 0.10}
            final_scores = (
                scores["knee"].fillna(0) * weights["knee"]
                + scores["dist"].fillna(0) * weights["dist"]
                + scores["height"].fillna(0) * weights["height"]
            )

            if final_scores.empty:
                logger.warning(
                    f"[{side_prefix}발] 최종 점수 계산에 실패하여 백스윙 후보가 없습니다."
                )
                continue

            # 1. 점수가 높은 순으로 후보 프레임을 정렬합니다.
            sorted_candidates = final_scores.sort_values(ascending=False)

            best_valid_idx = -1

            # 2. 정렬된 후보들을 하나씩 순회하며 '쓸만한 재료'인지 검증합니다.
            for idx, score in sorted_candidates.items():

                # 2-1. [검증 단계] 해당 프레임의 핵심 데이터(각도)가 유효한지 확인
                angle_cols_to_check = [
                    f"{side_prefix}_hip_angle",
                    f"{side_prefix}_knee_angle",
                    f"{side_prefix}_ankle_angle",
                ]
                frame_data = analysis_df.loc[idx]  # 전체 데이터프레임에서 검증

                # any()를 사용해 하나라도 NaN이 있으면 '썩은 재료'로 판단
                if frame_data[angle_cols_to_check].isnull().any():
                    logger.warning(
                        f"[{side_prefix}발] 백스윙 후보 프레임 {idx} (점수: {score:.2f})는 "
                        f"핵심 각도 데이터가 없어 건너뜁니다."
                    )
                    continue  # 다음으로 점수 높은 후보로 넘어감

                # 2-2. [성공] 유효성 검증 통과! 이 프레임을 최종 선택합니다.
                best_valid_idx = int(idx)
                logger.info(
                    f" [{side_prefix}발] 유효한 백스윙 탐지 성공: 프레임 {best_valid_idx} (점수: {score:.2f})"
                )
                break  # 최고의 '유효한' 프레임을 찾았으므로 루프 종료

            # 3. 루프가 끝난 후, 찾은 유효한 인덱스를 최종 결과에 할당합니다.
            backswing_peaks[side_prefix] = best_valid_idx

            if best_valid_idx == -1:
                logger.warning(
                    f"[{side_prefix}발] 모든 백스윙 후보 프레임의 데이터가 유효하지 않았습니다."
                )

        except Exception as e:
            logger.error(
                f"[{side_prefix}발] 백스윙 점수 계산 및 검증 중 오류: {e}",
                exc_info=True,
            )
            # 오류 발생 시 최소한의 fallback
            min_knee_idx = candidates_df[knee_angle_col].idxmin()
            if pd.notna(min_knee_idx):
                backswing_peaks[side_prefix] = int(min_knee_idx)

    return backswing_peaks


def _find_refined_impact_moment(
    analysis_df: pd.DataFrame,
    impact_idx: int,
    backswing_idx: int,
    # window_radius: int = 7,
    density_factor: int = 50,
) -> Optional[dict]:

    # === 1단계: 물리적 충돌 후보 탐색 ===
    # window_radius = config.INTERPOLATION_WINDOW_RADIUS
    start_idx = backswing_idx
    end_idx = min(
        len(analysis_df), impact_idx + 3
    )  # 임팩트 직후 프레임까지 약간의 여유
    if start_idx >= end_idx:
        logger.warning("보간 탐색 범위가 유효하지 않습니다.")
        return None
    window_df = analysis_df.loc[start_idx:end_idx].copy()

    physical_candidates = []

    # 각 프레임에서 실제 접촉 가능성 검사
    for i, row in window_df.iterrows():

        # 기본 데이터 확인
        foot_pos = row.get("foot_pos")
        ball_center = row.get("ball_center")
        ball_radius = row.get("ball_radius", 11)
        foot_speed = row.get("kicking_foot_speed", 0)

        if not all([foot_pos, ball_center, pd.notna(foot_speed), foot_speed >= 0.5]):
            continue

        distance = np.hypot(foot_pos[0] - ball_center[0], foot_pos[1] - ball_center[1])
        contact_threshold = ball_radius * config.BETA_PROXIMITY_SCALE

        # === 물리적 접촉 조건 확인 ===
        is_in_contact = distance <= contact_threshold

        if is_in_contact:
            print(
                f"   물리적 접촉 후보: Frame {i}, 거리={distance:.1f}px ≤ 임계값={contact_threshold:.1f}px"
            )

            # 발 속도 조건 (최소 임팩트 속도)
            min_impact_speed = config.MIN_IMPACT_SPEED_PX_FR
            if foot_speed < min_impact_speed:
                print(f"       속도 부족: {foot_speed:.2f} < {min_impact_speed}")
                continue

            # 접근 방향 확인
            approach_velocity = row.get("foot_ball_approach_velocity", 0)
            is_approaching = approach_velocity < -0.1  # 음수 = 접근

            # 간단한 접촉 비율 계산
            contact_ratio = max(0, (contact_threshold - distance) / contact_threshold)

            # 물리적 점수 계산
            contact_score = (
                max(0, (contact_threshold - distance) / contact_threshold) * 100
            )
            speed_score = min(foot_speed * 10, 50)
            contact_bonus = contact_ratio * 50  # 최대 50점
            approach_bonus = 30 if is_approaching else 0

            physical_score = (
                contact_score + speed_score + contact_bonus + approach_bonus
            )

            print(
                f"      물리 점수: {physical_score:.1f} (접촉:{contact_score:.1f} + 속도:{speed_score:.1f} + 비율:{contact_bonus:.1f} + 접근:{approach_bonus})"
            )

            physical_candidates.append(
                {
                    "original_idx": row["original_idx"],
                    "distance": distance,
                    "foot_speed": foot_speed,
                    "physical_score": physical_score,
                    "contact_ratio": contact_ratio,
                    "is_real_contact": True,
                }
            )

    # === 2단계: 물리적 충돌 후보가 있으면 최고 점수 선택 ===
    if physical_candidates:
        best_candidate = max(physical_candidates, key=lambda x: x["physical_score"])

        print(f"실제 충돌 순간 발견!")
        print(f"   Original Index: {best_candidate['original_idx']}")
        print(f"   접촉 거리: {best_candidate['distance']:.2f}px")
        print(f"   발 속도: {best_candidate['foot_speed']:.2f}px/fr")
        print(f"   물리 점수: {best_candidate['physical_score']:.1f}")

        return {
            "refined_original_idx": best_candidate["original_idx"],
            "min_interpolated_distance": best_candidate["distance"],
            "physical_impact_detected": True,
            "impact_method": "physical_contact",
        }

    # === 3단계: 물리적 충돌이 없으면 기존 SciPy 방법으로 Fallback ===
    print("물리적 충돌을 찾지 못했습니다. 기존 수학적 방법으로 fallback...")

    # 기존 SciPy 보간 로직
    interp_data = window_df[["original_idx", "distance_smoothed"]].dropna()
    interp_data = interp_data[np.isfinite(interp_data["distance_smoothed"])]
    interp_data = interp_data.drop_duplicates(subset=["original_idx"])

    if len(interp_data) < 2:
        print("SciPy 보간 실패: 데이터 부족")
        return None

    x = interp_data["original_idx"].values
    y = interp_data["distance_smoothed"].values
    kind = "cubic" if len(x) >= 4 else "linear"

    try:
        interp_func = interp1d(
            x, y, kind=kind, bounds_error=False, fill_value="extrapolate"
        )

        fine_x_step = 1.0 / max(1, density_factor)
        fine_x = np.arange(x.min(), x.max() + fine_x_step, fine_x_step)
        interpolated_y = interp_func(fine_x)

        min_idx = np.argmin(interpolated_y)
        refined_idx = fine_x[min_idx]
        min_distance = interpolated_y[min_idx]

        # === 4단계: SciPy 결과의 물리적 타당성 검증 ===
        print(f"🔍 SciPy 결과 검증: Idx={refined_idx:.2f}, Dist={min_distance:.2f}px")

        # 거리 임계값 확인
        avg_ball_radius = window_df["ball_radius"].mean()
        max_acceptable_distance = avg_ball_radius * 4.0  # 공 지름 정도까지만 허용

        if min_distance > max_acceptable_distance:
            print(
                f"SciPy 결과가 물리적으로 타당하지 않음: {min_distance:.1f}px > {max_acceptable_distance:.1f}px"
            )
            print("가장 가까운 프레임을 대신 사용")

            # 가장 가까운 거리의 실제 프레임 찾기
            closest_idx = window_df["distance_smoothed"].idxmin()
            closest_row = window_df.loc[closest_idx]

            return {
                "refined_original_idx": closest_row["original_idx"],
                "min_interpolated_distance": closest_row["distance_smoothed"],
                "physical_impact_detected": False,
                "impact_method": "closest_frame_fallback",
                "scipy_rejected": True,
            }

        print(f"SciPy 결과 물리적 타당성 통과")

        return {
            "refined_original_idx": refined_idx,
            "min_interpolated_distance": min_distance,
            "physical_impact_detected": False,
            "impact_method": "scipy_mathematical",
        }

    except Exception as e:
        print(f"SciPy 보간 중 에러: {e}")
        return None


# --- 메인 함수 ---


def find_key_events(analysis_df: pd.DataFrame, kicking_foot_preference: str) -> dict:
    """
    (사후 검증 추가된 최종 버전)
    1. 모든 조건을 종합하여 상위 후보군(top N)을 먼저 찾고,
    2. 각 후보의 '임팩트 이후 공의 실제 움직임'을 확인하여 최종 1명을 선택합니다.
    """
    if analysis_df.empty:
        return {}

    # 1. 정교화된 로직으로 상위 5개의 임팩트 후보를 찾습니다.
    impact_candidates = _find_impact_candidates(analysis_df, kicking_foot_preference)
    if not impact_candidates:
        logger.error("임팩트 후보를 찾을 수 없습니다.")
        return {}

    # 2. '사후 검증' 단계: 각 후보가 진짜 킥인지 검증합니다.
    verified_candidates = []
    for candidate in impact_candidates:
        impact_idx = candidate["df_idx"]

        # 3. 해당 후보의 임팩트 직후, 공의 최대 속도를 계산합니다.
        search_start = impact_idx + 1
        search_end = min(
            len(analysis_df), impact_idx + config.MAX_FRAMES_FOR_MAX_BALL_SPEED_SEARCH
        )

        max_ball_speed_after_impact = 0.0
        if search_start < search_end:
            ball_speeds = analysis_df.loc[search_start:search_end, "ball_speed"]
            if not ball_speeds.empty:
                max_ball_speed_after_impact = ball_speeds.max()

        # 4. '결과'에 기반한 최종 점수를 다시 매깁니다.
        verification_bonus = (
            max_ball_speed_after_impact * config.IMPACT_WEIGHT_POST_SPEED_BONUS
        )
        final_score = candidate["score"] + verification_bonus

        candidate["final_score"] = final_score
        verified_candidates.append(candidate)

    # 5. 검증된 후보들 중에서 최종 점수가 가장 높은 단 한 명을 최적의 임팩트로 선정합니다.
    if not verified_candidates:
        best_impact_candidate = impact_candidates[0]
    else:
        best_impact_candidate = max(verified_candidates, key=lambda x: x["final_score"])

    impact_idx = best_impact_candidate["df_idx"]
    # [수정] data_row가 없을 수 있으므로, df에서 직접 가져오도록 수정
    impact_info = analysis_df.loc[impact_idx].to_dict()
    kicking_foot = impact_info.get("kicking_foot")

    backswing_candidates = _find_backswing_peaks(analysis_df, impact_idx)
    final_backswing_idx = -1
    if kicking_foot and kicking_foot != "N/A":
        kicking_foot_prefix = "r" if kicking_foot == "right" else "l"
        final_backswing_idx = backswing_candidates.get(kicking_foot_prefix, -1)

    refined_impact_details = None
    if config.USE_FRAME_INTERPOLATION and final_backswing_idx != -1:
        refined_impact_details = _find_refined_impact_moment(
            analysis_df, impact_idx, final_backswing_idx
        )
    key_events = {
        "impact_frame_index": impact_idx,
        "backswing_frame_index": final_backswing_idx,
        "kicking_foot": kicking_foot,
        "initial_impact_info": impact_info,
        "refined_impact_details": refined_impact_details,
    }
    logger.info(f"핵심 이벤트 탐지 완료 (사후 검증 적용): {key_events}")
    return key_events


def determine_kick_part(
    pose_landmarks, ball_center: tuple, frame_shape: tuple, kicking_foot_side: str
) -> str:
    """
    (개선안) 2D 좌표 기반으로 공과 가장 가까운 발의 부위를 찾아 킥 부위를 추정합니다.
    """
    if not pose_landmarks or not ball_center:
        return "unknown"

    # 1. 발의 상세 모델 정보를 가져옵니다.
    foot_model = utils.enhance_foot_model(
        pose_landmarks, kicking_foot_side, frame_shape
    )
    if not foot_model:
        return "unknown"

    # 2. 공과 각 부위 사이의 2D 거리를 계산합니다.
    distances = {}
    parts_to_check = ["instep", "inside", "toe"]
    for part in parts_to_check:
        part_pos = foot_model.get(part)
        if part_pos:
            # utils.calculate_distance는 두 점 (x1, y1), (x2, y2)의 거리를 계산하는 간단한 함수
            dist = np.linalg.norm(np.array(part_pos) - np.array(ball_center))
            distances[part] = dist

    if not distances:
        return "unknown"

    # 3. 오직 2D 거리 기반으로 가장 가까운 부위를 판단합니다.
    best_guess = min(distances, key=distances.get)

    # (ToDo: 나중에 여기에 '궤적 분석' 결과를 추가하여 최종 보정할 수 있습니다)

    return best_guess
