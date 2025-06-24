# src/feedback_generator.py
"""
(전문가 버전)
점수 분석 결과(score_data)를 바탕으로 사용자에게 제공될 심층 분석 피드백을 생성합니다.
각 항목에 대한 진단, 원인 분석, 개선 방안을 포함한 전문적인 코멘트를 제공합니다.
"""
from typing import Dict, Any, Tuple, List

# ==============================================================================
# เชิง FEEDBACK MESSAGE TEMPLATES (전문가 버전)
# ==============================================================================
# 각 템플릿에는 {value} 플레이스홀더를 사용하여 실제 측정값을 동적으로 삽입합니다.
FEEDBACK_MESSAGES = {
    # --- 임팩트 평가 항목 ---
    "hitting_foot_part": {
        "name": "타격 부위",
        "good": "이상적인 타격 부위 ({value}): 킥의 목적에 맞는 정확한 부위(인스텝/인사이드)로 공을 가격했습니다. 이는 슈팅의 정확도와 파워를 극대화하는 가장 기본적인 조건이며, 의도한 대로 공을 컨트롤하고 있다는 증거입니다.",
        "bad": "부정확한 타격 부위 ({value}): 공이 발의 의도치 않은 부위에 맞았습니다. 이는 킥의 방향이 흔들리거나 파워가 손실되는 직접적인 원인이 됩니다. 꾸준한 연습을 통해 발등 또는 인사이드의 정확한 면에 공을 맞추는 감각을 익히는 것이 매우 중요합니다.",
    },
    "ball_contact_point": {
        "name": "공 타격 지점",
        "good": "정확한 공 타격 지점 ({value}): 공의 중심(Center) 혹은 중심-하단(Bottom)을 정확히 가격했습니다. 이는 운동 에너지를 공에 온전히 전달하여 강력한 직선 또는 상승 궤적을 만드는 핵심 기술입니다.",
        "bad": "아쉬운 공 타격 지점 ({value}): 공의 중심에서 벗어난 곳을 타격하여 힘이 분산되었습니다. 예를 들어, 공의 상단을 맞추면 땅볼이 되고, 너무 하단을 맞추면 힘없이 뜰 수 있습니다. 목표로 하는 궤적에 따라 공의 어느 부분을 찰지 명확히 인지하는 훈련이 필요합니다.",
    },
    "support_foot_ball_distance": {
        "name": "디딤발과 공의 거리",
        "good": "이상적인 디딤발 거리 ({value}): 공 옆에 최적의 간격으로 디딤발을 위치시켰습니다. 이는 상체의 균형을 안정시키고, 킥을 하는 다리가 자유롭게 최대 스윙을 할 수 있는 공간을 확보하여 킥의 파워와 안정성을 모두 잡는 교과서적인 자세입니다.",
        "bad": "불안정한 디딤발 거리 ({value}): 디딤발이 공과 너무 가깝거나 멀었습니다. 너무 가까우면 스윙 공간이 부족해 강력한 킥이 어렵고, 너무 멀면 몸의 중심이 무너져 정확도와 파워가 모두 떨어집니다. 자신에게 가장 편안하고 강력한 킥이 나오는 디딤발 위치를 반복 훈련을 통해 찾아야 합니다.",
    },
    "ball_initial_speed": {
        "name": "공의 초기 속도",
        "good": "압도적인 볼 스피드 ({value}): 매우 빠른 초기 속도를 기록했습니다. 이는 강력한 스윙과 정확한 임팩트가 결합된 결과로, 상대 골키퍼가 반응하기 어려운 위협적인 슈팅 능력의 증거입니다.",
        "bad": "아쉬운 볼 스피드 ({value}): 공의 초기 속도를 더 높일 수 있습니다. 이는 주로 스윙 속도가 느리거나, 임팩트 시 힘 전달이 효율적이지 못할 때 발생합니다. 백스윙 크기를 키우고, 고관절 회전을 사용하여 스윙 속도를 높이는 훈련이 효과적입니다.",
    },
    "impact_angle_comparison": {
        "name": "임팩트 자세 정확도",
        "good": "프로 수준의 임팩트 자세 (유사도 점수: {value}): 프로 선수와 거의 흡사한 임팩트 자세입니다. 이는 신체의 각 관절(고관절, 무릎, 발목)이 이상적인 각도로 협응하며 운동 사슬(Kinetic Chain)을 통해 힘을 효율적으로 전달하고 있음을 의미합니다.",
        "bad": " 개선이 필요한 임팩트 자세 (유사도 점수: {value}): 임팩트 순간의 신체 각도가 프로 선수와 차이를 보입니다. 특히 무릎이나 상체 각도가 부적절하면, 생성된 힘이 엉뚱한 곳으로 새어 나가거나 부상의 위험을 높일 수 있습니다. 프로 선수의 자세를 참고하여 교정하는 것이 좋습니다.",
    },
    # --- 백스윙 평가 항목 ---
    "max_foot_swing_speed": {
        "name": "최대 스윙 속도",
        "good": "폭발적인 스윙 속도 ({value}): 프로 선수급의 매우 빠른 최대 스윙 속도를 보여줍니다. 이는 강력한 킥 파워의 핵심 원천으로, 잘 훈련된 하체와 코어 근력을 증명합니다.",
        "bad": "아쉬운 스윙 속도 ({value}): 스윙의 최대 속도를 더 끌어올릴 필요가 있습니다. 백스윙의 크기를 키우고, 허리와 고관절의 회전력을 적극적으로 사용하여 채찍처럼 휘두르는 느낌으로 스윙 스피드를 높이는 연습이 필요합니다.",
    },
    # --- 스윙안정성 평가항목 ---
    "kick_foot_kinematics_change": {
        "name": "스윙 가속 안정성",
        "good": "안정적인 가속 ({value}): 임팩트 직전까지 불필요한 가속/감속 없이, 백스윙의 힘을 끝까지 유지하는 매우 안정적인 스윙입니다.",
        "bad": " 불안정한 가속 ({value}): 임팩트 직전 스윙 속도가 급격하게 변했습니다. 힘을 주려다 자세가 흐트러지거나, 공을 맞추기 위해 억지로 속도를 조절하는 등 불안정한 스윙입니다.",
    },
    "backswing_knee_angle_size": {
        "name": "백스윙 시 무릎 각도",
        "good": "최적의 백스윙 무릎 각도 ({value}): 백스윙 시 무릎을 충분히 깊게 접어, 마치 활시위를 당기듯 강력한 탄성 에너지를 다리에 저장했습니다. 이는 폭발적인 스윙 속도로 이어지는 중요한 준비 동작입니다.",
        "bad": "부족한 백스윙 무릎 각도 ({value}): 백스윙 시 무릎이 충분히 접히지 않아, 킥에 사용될 에너지를 최대로 저장하지 못했습니다. 더 과감하고 큰 백스윙 동작으로 무릎을 더 접어주면 스윙 파워가 극적으로 향상될 수 있습니다.",
    },
    "support_foot_stability": {
        "name": "디딤발 안정성",
        "good": "반석 같은 디딤발 안정성 (평균 흔들림: {value}): 킥 모션 내내 디딤발이 흔들림 없이 지면에 단단히 고정되었습니다. 디딤발은 모든 힘을 지탱하는 '축'이므로, 이 안정성은 지면 반발력을 온전히 파워로 전환시키는 가장 중요한 요소입니다.",
        "bad": " unsteady: 불안정한 디딤발 (평균 흔들림: {value}): 킥을 하는 동안 디딤발이 흔들리는 모습이 감지되었습니다. 이는 코어 밸런스가 무너졌거나, 디딤발을 딛는 힘이 부족하다는 신호입니다. 파워 손실의 주된 원인이 되므로, 한 발 서기 등 밸런스 훈련과 디딤발을 땅에 '심는다'는 느낌으로 딛는 연습이 필요합니다.",
    },
    "backswing_angle_comparison": {
        "name": "백스윙 자세 정확도",
        "good": " 교과서적인 백스윙 자세 (유사도 점수: {value}): 프로 선수와 매우 유사한 백스윙 자세입니다. 불필요한 동작이 없는 간결하고 효율적인 자세로, 다음 동작인 임팩트로 힘을 손실 없이 연결하기에 최적화되어 있습니다.",
        "bad": "🦵 개선이 필요한 백스윙 자세 (유사도 점수: {value}): 백스윙 시 신체 각도가 프로 선수와 다소 차이가 있습니다. 예를 들어 상체가 너무 일찍 열리거나 팔의 위치가 부적절하면 균형이 무너져 스윙의 파워나 정확도에 나쁜 영향을 줄 수 있습니다.",
    },
}


# ==============================================================================
# เชิง FEEDBACK GENERATION LOGIC
# ==============================================================================
def _get_overall_comment(total_score: float) -> str:
    """총점에 따라 전문적인 총평을 반환합니다."""
    if total_score >= 90:
        return "⚽️ 프로 선수 수준의 킥 메커니즘입니다. 모든 평가 항목에서 최상위권의 기량을 보여주었으며, 힘의 생성, 전달, 정확성 면에서 거의 완벽에 가까운 모습을 보입니다. 현 상태를 유지하며 컨디션 관리에 집중하는 것을 추천합니다."
    elif total_score >= 75:
        return "👍 매우 뛰어난 킥입니다. 안정적인 신체 밸런스를 바탕으로 강력한 운동 에너지를 생성하고, 이를 효율적으로 공에 전달하는 능력을 갖추었습니다. 일부 사소한 단점만 보완한다면 경기에서 결정적인 차이를 만들어낼 수 있는 강력한 무기가 될 것입니다."
    elif total_score >= 60:
        return "🙂 훌륭한 잠재력을 가진 킥입니다. 전반적으로 좋은 기본기를 갖추고 있으나, 특정 구간에서 힘의 손실이나 자세의 불안정성이 관찰됩니다. 제시된 약점을 집중적으로 개선한다면 경기력을 한 단계 위로 끌어올릴 수 있습니다."
    elif total_score >= 40:
        return "🏃 개선해야 할 부분이 명확히 보이는 킥입니다. 잘 수행된 부분도 있지만, 킥의 파워와 정확성에 큰 영향을 미치는 핵심적인 부분에서 아쉬움이 남습니다. 피드백을 통해 자신의 약점을 명확히 인지하고, 목적을 가진 훈련을 통해 개선해 나가는 것이 중요합니다."
    else:
        return "👟 기본기부터 차근차근 다질 필요가 있습니다. 현재 킥 메커니즘은 힘의 전달 효율이 낮고 자세가 불안정하여, 잠재력을 충분히 발휘하지 못하고 있습니다. 괜찮습니다. 모든 전문가는 초보자 시절이 있었습니다. 피드백을 바탕으로 기본 동작부터 반복하여 몸에 익히는 것이 중요합니다."


def _find_key_strengths_and_weaknesses(
    score_data: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """가장 점수가 높은 항목(강점)과 낮은 항목(약점)을 찾아 구체적인 데이터와 함께 피드백을 생성합니다."""
    metrics = []
    for category_name, category_data in score_data["categories"].items():
        for key, details in category_data["details"].items():
            if isinstance(details.get("value"), str) and details.get(
                "value", ""
            ).startswith("N/A"):
                continue

            score_ratio = (
                details["score"] / details["max_score"]
                if details["max_score"] > 0
                else 0
            )
            metrics.append(
                {
                    "key": key,
                    "ratio": score_ratio,
                    "value": details.get("value", "N/A"),
                    "message_template": FEEDBACK_MESSAGES.get(key, {}),
                }
            )

    metrics.sort(key=lambda x: x["ratio"])

    weaknesses = []
    for metric in metrics[:2]:
        if metric["ratio"] < 0.6 and "bad" in metric["message_template"]:
            formatted_message = metric["message_template"]["bad"].format(
                value=metric["value"]
            )
            weaknesses.append(formatted_message)

    strengths = []
    for metric in reversed(metrics[-2:]):
        if metric["ratio"] >= 0.8 and "good" in metric["message_template"]:
            formatted_message = metric["message_template"]["good"].format(
                value=metric["value"]
            )
            strengths.append(formatted_message)

    return strengths, weaknesses


def _generate_phase_feedback(phase_name: str, phase_data: Dict[str, Any]) -> str:
    """백스윙/임팩트 단계별 점수와 세부 항목을 종합하여 상세 피드백을 생성합니다."""
    phase_score = phase_data.get("subtotal", 0)

    if phase_score >= 40:
        summary = f"전반적으로 매우 안정적이고 강력한 {phase_name} 동작을 보여줍니다."
    elif phase_score >= 30:
        summary = f"좋은 {phase_name} 동작입니다. 다만 몇 가지 세부 항목을 개선하면 완성도를 높일 수 있습니다."
    else:
        summary = f"{phase_name} 동작의 효율성과 안정성을 높이기 위한 집중적인 훈련이 필요해 보입니다."

    # 세부 항목 중 가장 점수가 낮은 항목을 찾아 추가 코멘트
    details = phase_data.get("details", {})
    if not details:
        return summary

    lowest_metric = min(
        details.items(),
        key=lambda item: (
            item[1]["score"] / item[1]["max_score"] if item[1]["max_score"] > 0 else 1
        ),
    )

    key, lowest_details = lowest_metric
    score_ratio = (
        lowest_details["score"] / lowest_details["max_score"]
        if lowest_details["max_score"] > 0
        else 1
    )

    if score_ratio < 0.6:
        metric_name = FEEDBACK_MESSAGES.get(key, {}).get("name", "이 항목")
        summary += f" 특히, '{metric_name}' 부분을 중점적으로 개선한다면 해당 단계의 완성도가 크게 향상될 것입니다."

    return summary


def generate_feedback(score_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    최종 점수 데이터를 입력받아 플러터 화면 형식에 맞는
    전문가 피드백 딕셔너리를 생성하여 반환합니다.
    """
    if not score_data or "total_score" not in score_data:
        return {"error": "분석 데이터를 찾을 수 없어 피드백을 생성할 수 없습니다."}

    total_score = score_data.get("total_score", 0)

    # 1. 종합 카드 데이터 생성
    overall_comment = _get_overall_comment(total_score)
    strengths, weaknesses = _find_key_strengths_and_weaknesses(score_data)

    overall_card = {
        "총점": round(total_score),
        "총평": overall_comment,
        "피드백": {
            "매우 잘한점": (
                strengths
                if strengths
                else [
                    "전반적으로 균형 잡힌 모습을 보여주셨습니다. 약점을 보완하는 데 집중해 보세요!"
                ]
            ),
            "아쉬운 점": (
                weaknesses
                if weaknesses
                else ["특별히 아쉬운 점 없이 모든 면에서 훌륭한 킥을 보여주셨습니다!"]
            ),
        },
    }

    # 2. 백스윙 카드 데이터 생성
    backswing_eval = score_data["categories"]["backswing_evaluation"]
    backswing_score = backswing_eval.get("subtotal", 0)
    backswing_feedback = _generate_phase_feedback("백스윙", backswing_eval)

    backswing_card = {
        "백스윙 점수": round(backswing_score),
        "백스윙 피드백": backswing_feedback,
    }

    # 3. 임팩트 카드 데이터 생성
    impact_eval = score_data["categories"]["impact_evaluation"]
    impact_score = impact_eval.get("subtotal", 0)
    impact_feedback = _generate_phase_feedback("임팩트", impact_eval)

    impact_card = {"임팩트 점수": round(impact_score), "임팩트 피드백": impact_feedback}

    # 최종 결과물 조합
    final_feedback_package = {
        "종합": overall_card,
        "백스윙": backswing_card,
        "임팩트": impact_card,
    }

    return final_feedback_package
