# src/scoring.py
"""
분석된 데이터를 바탕으로 킥의 품질을 평가하고 점수를 매기는 모듈.
"""
import numpy as np
import pandas as pd
import json
from . import config, utils
from typing import Dict, Any, Optional
from .utils import setup_logger

logger = setup_logger(__name__)


def load_player_data() -> dict:
    """data/pro_player_stats.json 파일에서 선수 데이터를 불러옵니다."""
    try:
        with open(config.PLAYER_STATS_PATH, "r", encoding="utf-8") as f:
            player_data = json.load(f)
        return player_data
    except FileNotFoundError:
        # get_logger().error(...) 등으로 로깅 처리 가능
        print(
            f"에러: 선수 데이터 파일을 찾을 수 없습니다. 경로: {config.PLAYER_STATS_PATH}"
        )
        return {}
    except json.JSONDecodeError:
        print(f"에러: 선수 데이터 JSON 파일의 형식이 잘못되었습니다.")
        return {}


# 함수를 호출하여 전역 변수로 선수 데이터를 로드
PLAYER_DATA = load_player_data()


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


def calculate_all_limbs_angle_comparison_score(
    user_angles: dict, kicking_foot: str, phase: str
) -> dict:
    """
    (축약어 통일 버전)
    사용자의 주발 정보와 관절 각도를 받아, 적절한 프로 선수와 비교 점수를 계산합니다.
    """
    if kicking_foot == "left":
        player_model_name = "messi"
    elif kicking_foot == "right":
        player_model_name = "ronaldo"
    else:
        return {"score": 0, "message": "주발이 결정되지 않아 선수 비교를 스킵합니다."}

    player_stats_phase = PLAYER_DATA.get(player_model_name, {}).get(phase, {})
    if not player_stats_phase:
        return {}

    all_joint_similarities = {}
    all_joint_z_scores = {}

    for angle_key, user_angle_val in user_angles.items():
        if (
            not user_angle_val
            or not isinstance(angle_key, str)
            or "_angle" not in angle_key
        ):
            continue

        # 'r_knee_angle' -> ['r', 'knee', 'angle']
        parts = angle_key.split("_")
        if len(parts) != 3:
            continue

        side_prefix, joint_name = parts[0], parts[1]
        # 'r' -> 'R', 'l' -> 'L'
        player_joint_key = f"{side_prefix.upper()}_{joint_name}"  # 'R_knee', 'L_hip'

        if player_joint_key in player_stats_phase and pd.notna(user_angle_val):
            mean_val = player_stats_phase[player_joint_key].get("mean")
            std_val = player_stats_phase[player_joint_key].get("std")
            z = calculate_z_score(user_angle_val, mean_val, std_val)
            sim = z_score_to_similarity(z)
            all_joint_similarities[player_joint_key] = sim
            all_joint_z_scores[player_joint_key] = z

    player_dominant_foot_prefix = "L" if player_model_name == "messi" else "R"
    supporting_foot_prefix = "R" if player_model_name == "messi" else "L"

    weights = {
        f"{player_dominant_foot_prefix}_knee": 0.30,
        f"{player_dominant_foot_prefix}_hip": 0.20,
        f"{player_dominant_foot_prefix}_ankle": 0.10,
        f"{supporting_foot_prefix}_knee": 0.20,
        f"{supporting_foot_prefix}_hip": 0.10,
        f"{supporting_foot_prefix}_ankle": 0.10,
    }

    weighted_similarity_sum = 0
    total_weight_sum = 0
    for joint_key, similarity_val in all_joint_similarities.items():
        if joint_key in weights and pd.notna(similarity_val):
            weighted_similarity_sum += similarity_val * weights[joint_key]
            total_weight_sum += weights[joint_key]

    final_overall_similarity = 0
    if total_weight_sum > 0:
        final_overall_similarity = weighted_similarity_sum / total_weight_sum

    score = final_overall_similarity * 10
    if score >= 6.8:
        adjusted_score = 8 + (score - 6.8) * (2 / 3.2)
    elif score >= 3.7:
        adjusted_score = 5 + (score - 3.7) * (3 / 3.1)
    elif score >= 1.4:
        adjusted_score = 2 + (score - 1.4) * (3 / 2.3)
    else:
        adjusted_score = score * (2 / 1.4)
    adjusted_score = min(10, max(0, adjusted_score))

    return {
        "score": round(adjusted_score, 1),
        "all_joint_similarities": all_joint_similarities,
        "all_joint_z_scores": all_joint_z_scores,
        "weighted_overall_similarity": final_overall_similarity,
        "comparison_player": player_model_name,
        "phase": phase,
    }


def calculate_kick_score(
    final_impact_info: Dict[str, Any],
    final_backswing_info: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    analyzer가 전달해준 최종 impact/backswing 정보를 바탕으로 점수를 계산합니다.
    이 함수는 데이터의 출처(보간 여부)를 신경쓰지 않고 계산에만 집중합니다.
    """
    if not final_impact_info or not final_backswing_info:
        logger.warning("점수 계산에 필요한 impact 또는 backswing 정보가 부족합니다.")
        return None

    # --- 1. 필요한 전역 변수 및 스케일 값 추출 ---
    fps = metadata.get("fps", 30.0)
    # pixel_to_cm_scale = metadata.get("pixel_to_cm_scale")
    # scale = utils.get_safe_pixel_to_cm_scale(pixel_to_cm_scale)

    # 점수 변수 초기화
    foot_part_score = 0
    ball_contact_score = 0
    support_dist_score = 0
    ball_initial_speed_score = 0
    impact_player_angle_score = 0
    max_foot_swing_speed_score = 0
    backswing_knee_score = 0
    support_stability_score = 0
    backswing_player_angle_score = 0

    # --- 2. 점수 계산을 위한 변수 초기화 ---
    score_details = {
        "impact_evaluation": {"subtotal": 0, "details": {}},
        "backswing_evaluation": {"subtotal": 0, "details": {}},
    }

    # =================================================================
    # == 임팩트 평가 (Impact Evaluation)
    # =================================================================
    category_impact = "impact_evaluation"

    # 타격 부위 점수
    kicking_foot_part = final_impact_info.get("kicking_foot_part", "N/A")
    foot_part_score = (
        10
        if kicking_foot_part in ["instep", "inside"]
        else (3 if kicking_foot_part != "N/A" else 0)
    )
    score_details[category_impact]["details"]["hitting_foot_part"] = {
        "value": kicking_foot_part,
        "score": foot_part_score,
        "max_score": 10,
    }
    score_details[category_impact]["subtotal"] += foot_part_score

    # 공 접촉 부위 점수
    contact_region = final_impact_info.get("contact_region_on_ball", "N/A")
    ball_contact_score = (
        10
        if contact_region in ["Bottom", "Center"]
        else (
            8
            if contact_region in ["Bottom-Left", "Bottom-Right"]
            else (5 if contact_region in ["Left", "Right"] else 3)
        )
    )
    score_details[category_impact]["details"]["ball_contact_point"] = {
        "value": contact_region,
        "score": ball_contact_score,
        "max_score": 10,
    }
    score_details[category_impact]["subtotal"] += ball_contact_score

    # 디딤발 거리 점수
    support_dist_cm = final_impact_info.get("dist_ball_to_supporting_foot_ankle_cm")
    support_dist_cm_str = "N/A"
    support_dist_score = 0
    if pd.notna(support_dist_cm):
        support_dist_cm_str = f"{support_dist_cm:.1f}cm"
        if 10 <= support_dist_cm < 25:
            support_dist_score = 10
        elif 25 <= support_dist_cm < 40:
            support_dist_score = 7
        else:
            support_dist_score = 4
    score_details[category_impact]["details"]["support_foot_ball_distance"] = {
        "value": support_dist_cm_str,
        "score": support_dist_score,
        "max_score": 10,
    }
    score_details[category_impact]["subtotal"] += support_dist_score

    # 공 초기 속도 점수
    # analyzer.py에서 계산된 km/h 값을 직접 가져옵니다.
    ball_speed_km_h = final_impact_info.get("ball_speed_results", {}).get(
        "max_speed_kmh", 0.0
    )
    ball_initial_speed_str = (
        f"{ball_speed_km_h:.1f}km/h" if ball_speed_km_h > 0 else "N/A"
    )
    ball_initial_speed_score = 0

    if ball_speed_km_h > 0:
        if ball_speed_km_h >= 100:
            ball_initial_speed_score = 10
        elif ball_speed_km_h >= 85:
            ball_initial_speed_score = 8
        elif ball_speed_km_h >= 70:
            ball_initial_speed_score = 6
        elif ball_speed_km_h >= 50:
            ball_initial_speed_score = 4
        else:
            ball_initial_speed_score = 2

    score_details[category_impact]["details"]["ball_initial_speed"] = {
        "value": ball_initial_speed_str,
        "score": ball_initial_speed_score,
        "max_score": 10,
    }
    score_details[category_impact]["subtotal"] += ball_initial_speed_score
    # --- [수정] 끝 ---

    # 임팩트 자세 비교 점수
    # analyzer.py가 모든 각도(r_hip_angle 등)를 final_impact_info에 담아준다고 가정
    kicking_foot = final_impact_info.get("kicking_foot")
    angle_comp_result_impact = calculate_all_limbs_angle_comparison_score(
        final_impact_info, kicking_foot, "impact"
    )
    impact_player_angle_score = angle_comp_result_impact.get("score", 0)
    impact_player_angle_metric_val = (
        f"{impact_player_angle_score}/10 vs {angle_comp_result_impact.get('comparison_player', '').capitalize()}"
        if impact_player_angle_score > 0
        else "N/A"
    )
    score_details[category_impact]["details"]["impact_angle_comparison"] = {
        "value": impact_player_angle_metric_val,
        "score": impact_player_angle_score,
        "max_score": 10,
    }
    score_details[category_impact]["subtotal"] += impact_player_angle_score

    # =================================================================
    # == 백스윙 평가 (Backswing Evaluation)
    # =================================================================
    category_backswing = "backswing_evaluation"

    # 최대 발 스윙 속도 점수
    # max_foot_swing_speed_km_h = final_backswing_info.get(
    #     "max_foot_swing_speed_kmh", 0.0
    # )

    # # 화면 표시용 문자열 생성
    # max_foot_swing_speed_str = (
    #     f"{max_foot_swing_speed_km_h:.1f}km/h"
    #     if max_foot_swing_speed_km_h > 0
    #     else "N/A"
    # )

    # # 점수화 로직 (이 부분은 기존과 동일)
    # max_foot_swing_speed_score = 0
    # if max_foot_swing_speed_km_h > 0:
    #     if max_foot_swing_speed_km_h >= 95:
    #         max_foot_swing_speed_score = 10
    #     elif max_foot_swing_speed_km_h >= 80:
    #         max_foot_swing_speed_score = 8
    #     elif max_foot_swing_speed_km_h >= 65:
    #         max_foot_swing_speed_score = 6
    #     elif max_foot_swing_speed_km_h >= 45:
    #         max_foot_swing_speed_score = 4
    #     else:
    #         max_foot_swing_speed_score = 2
    # in calculate_kick_score

    # 최대 발 스윙 속도 점수
    max_foot_swing_speed_km_h = final_backswing_info.get(
        "max_foot_swing_speed_kmh", 0.0
    )
    max_foot_swing_speed_str = (
        f"{max_foot_swing_speed_km_h:.1f}km/h"
        if max_foot_swing_speed_km_h > 0
        else "N/A"
    )

    # --- 발끝 추정 속도에 맞는 너그러운 점수 기준 ---
    max_foot_swing_speed_score = 0
    if max_foot_swing_speed_km_h > 0:
        if max_foot_swing_speed_km_h >= 70:
            max_foot_swing_speed_score = 10
        elif max_foot_swing_speed_km_h >= 60:
            max_foot_swing_speed_score = 8
        elif max_foot_swing_speed_km_h >= 50:
            max_foot_swing_speed_score = 6
        elif max_foot_swing_speed_km_h >= 35:
            max_foot_swing_speed_score = 4
        else:
            max_foot_swing_speed_score = 2

    score_details[category_backswing]["details"]["max_foot_swing_speed"] = {
        "value": max_foot_swing_speed_str,
        "score": max_foot_swing_speed_score,
        "max_score": 10,
    }
    score_details[category_backswing]["subtotal"] += max_foot_swing_speed_score

    # 스윙 안정성 점수
    foot_accel_px_fr2 = final_impact_info.get("kicking_foot_acceleration_scalar")
    kick_foot_kinematics_score = 0
    kick_foot_kinematics_str = "N/A"
    if pd.notna(foot_accel_px_fr2):
        kick_foot_kinematics_str = f"{foot_accel_px_fr2:.2f} px/fr²"
        if abs(foot_accel_px_fr2) < 7.5:
            kick_foot_kinematics_score = 10
        elif abs(foot_accel_px_fr2) < 15.0:
            kick_foot_kinematics_score = 7
        elif abs(foot_accel_px_fr2) < 30.0:
            kick_foot_kinematics_score = 4
        elif abs(foot_accel_px_fr2) < 45.0:
            kick_foot_kinematics_score = 2
        else:
            kick_foot_kinematics_score = 0

    score_details[category_backswing]["details"]["kick_foot_kinematics_change"] = {
        "value": kick_foot_kinematics_str,
        "score": kick_foot_kinematics_score,
        "max_score": 10,
    }
    score_details[category_backswing]["subtotal"] += kick_foot_kinematics_score

    # 백스윙 무릎 각도 점수
    kicking_foot_prefix = "r" if kicking_foot == "right" else "l"
    backswing_knee_angle_key = f"{kicking_foot_prefix}_knee_angle"
    backswing_knee_angle = final_backswing_info.get(backswing_knee_angle_key)
    backswing_knee_str = "N/A (데이터 없음)"
    if pd.notna(backswing_knee_angle):
        backswing_knee_str = f"{backswing_knee_angle:.1f}도"
        player_model = PLAYER_DATA.get(
            "messi" if kicking_foot == "left" else "ronaldo", {}
        )
        target_data = player_model.get("backswing", {}).get(
            f"{kicking_foot_prefix.upper()}_knee"
        )
        if target_data:
            mean_val = target_data.get("mean")
            std_val = target_data.get("std")
            if std_val is not None and std_val > 0:
                z = abs((backswing_knee_angle - mean_val) / std_val)
                if z <= 0.75:
                    backswing_knee_score = 10
                elif z <= 1.5:
                    backswing_knee_score = 8
                elif z <= 2.5:
                    backswing_knee_score = 6
                elif z <= 3.5:
                    backswing_knee_score = 4
                else:
                    backswing_knee_score = 2

    score_details[category_backswing]["details"]["backswing_knee_angle_size"] = {
        "value": backswing_knee_str,
        "score": backswing_knee_score,
        "max_score": 10,
    }
    score_details[category_backswing]["subtotal"] += backswing_knee_score

    # 디딤발 안정성 점수
    # analyzer.py에서 이미 cm/frame으로 변환된 값을 직접 가져옵니다.
    support_stability_cm_fr = final_backswing_info.get(
        "support_foot_stability_cm_fr", 0.0
    )

    # 화면 표시용 문자열 생성
    support_stability_str = (
        f"{support_stability_cm_fr:.2f}cm/frame"
        if support_stability_cm_fr > 0
        else "N/A"
    )

    # 점수화 로직
    support_stability_score = 0
    if support_stability_cm_fr > 0:
        if support_stability_cm_fr < 2.0:
            support_stability_score = 10
        elif support_stability_cm_fr < 18.0:
            support_stability_score = max(
                0, int(10 - ((support_stability_cm_fr - 2.0) * (10 / 16)))
            )
        else:
            support_stability_score = 0

    score_details[category_backswing]["details"]["support_foot_stability"] = {
        "value": support_stability_str,
        "score": support_stability_score,
        "max_score": 10,
    }
    score_details[category_backswing]["subtotal"] += support_stability_score

    # 백스윙 자세 비교 점수
    angle_comp_result_bs = calculate_all_limbs_angle_comparison_score(
        final_backswing_info, kicking_foot, "backswing"
    )
    backswing_player_angle_score = angle_comp_result_bs.get("score", 0)
    backswing_player_angle_metric_val = (
        f"{backswing_player_angle_score:.1f}/10 vs {angle_comp_result_bs.get('comparison_player', 'N/A').capitalize()}"
        if backswing_player_angle_score > 0
        else "N/A"
    )
    score_details[category_backswing]["details"]["backswing_angle_comparison"] = {
        "value": backswing_player_angle_metric_val,
        "score": backswing_player_angle_score,
        "max_score": 10,
    }
    score_details[category_backswing]["subtotal"] += backswing_player_angle_score

    # --- 최종 점수 종합 ---
    total_score = (
        score_details[category_impact]["subtotal"]
        + score_details[category_backswing]["subtotal"]
    )
    MAX_POSSIBLE_SCORE = 100

    return {
        "total_score": round(total_score, 1),
        "max_score": MAX_POSSIBLE_SCORE,
        "percentage": round((total_score / MAX_POSSIBLE_SCORE) * 100, 1),
        "categories": score_details,
    }
