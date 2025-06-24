# # src/ball.py
# """
# 축구공의 탐지, 추적, 상태 예측을 담당하는 모듈 (ver. 3)
# - 'POST_IMPACT_TRACKING' 상태 추가
# - analyzer.py로부터 신호를 받아 물리 모델(중력) 및 파라미터 변경 기능 추가
# """
# import numpy as np
# from ultralytics import YOLO
# from typing import Dict, Any, Optional, List
# import mediapipe as mp

# # 내부 모듈 임포트
# from . import config
# from .utils import setup_logger

# mp_pose = mp.solutions.pose


# class ImprovedKalmanFilter:
#     """공의 물리적 움직임을 모델링하기 위한 칼만 필터"""

#     def __init__(self, fps: int = 30):
#         self.dt = 1.0 / fps
#         self.S = np.zeros((6, 1))
#         self.F = np.array(
#             [
#                 [1, self.dt, 0.5 * self.dt**2, 0, 0, 0],
#                 [0, 1, self.dt, 0, 0, 0],
#                 [0, 0, 1, 0, 0, 0],
#                 [0, 0, 0, 1, self.dt, 0.5 * self.dt**2],
#                 [0, 0, 0, 0, 1, self.dt],
#                 [0, 0, 0, 0, 0, 1],
#             ],
#             dtype=float,
#         )
#         self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=float)
#         self.Q = np.eye(6) * config.KF_PROCESS_NOISE
#         self.R = np.eye(2) * config.KF_MEASUREMENT_NOISE
#         self.P = np.eye(6) * 500.0

#         self.frames_without_measurement = 0
#         self.max_prediction_frames = int(fps * 0.7)  # 예측 허용 시간 소폭 증가

#         # ✨ Phase 3 추가: 물리 모델 관련 변수
#         self.gravity = config.GRAVITY_PIXELS_PER_SECOND_SQUARED
#         self.is_airborne = False
#         self.logger = setup_logger(__name__)

#     def initialize_state(self, initial_pos: tuple):
#         self.S.fill(0)
#         self.S[0, 0] = initial_pos[0]
#         self.S[3, 0] = initial_pos[1]
#         self.P = np.eye(6) * 100.0

#     def predict(self) -> tuple:
#         self.S = self.F @ self.S

#         # ✨ Phase 3 추가: 공중에 떴을 때만 중력 적용
#         if self.is_airborne:
#             self.S[5, 0] = self.gravity  # y축 가속도에 중력 적용

#         return (self.S[0, 0], self.S[3, 0])

#     def update(self, measurement: Optional[tuple]):
#         if measurement is None:
#             self.frames_without_measurement += 1
#             return
#         z = np.array(measurement).reshape(2, 1)
#         S_k = self.H @ self.P @ self.H.T + self.R
#         K = self.P @ self.H.T @ np.linalg.inv(S_k)
#         self.S = self.S + K @ (z - self.H @ self.S)
#         self.P = (np.eye(6) - K @ self.H) @ self.P
#         self.frames_without_measurement = 0

#     def get_current_state(self) -> dict:
#         return {
#             "position": (self.S[0, 0], self.S[3, 0]),
#             "velocity": (self.S[1, 0], self.S[4, 0]),
#             "confidence": max(
#                 0.0,
#                 1.0 - (self.frames_without_measurement / self.max_prediction_frames),
#             ),
#         }

#     # ✨ Phase 3 추가: 외부 신호에 따라 필터 파라미터를 변경하는 메서드
#     def set_airborne(self, airborne_status: bool):
#         """중력 적용 여부를 설정합니다."""
#         self.is_airborne = airborne_status
#         if airborne_status:
#             self.logger.info("Kalman Filter: 중력 모델 활성화.")

#     def increase_process_noise_for_kick(self):
#         """킥 직후의 불확실성을 반영하기 위해 프로세스 노이즈를 일시적으로 높입니다."""
#         self.Q *= config.KF_KICK_NOISE_MULTIPLIER
#         self.logger.info(
#             f"Kalman Filter: 킥으로 인해 프로세스 노이즈를 {config.KF_KICK_NOISE_MULTIPLIER}배 증가."
#         )

#     def reset_process_noise(self):
#         """프로세스 노이즈를 기본값으로 복원합니다."""
#         self.Q = np.eye(6) * config.KF_PROCESS_NOISE


# class BallTracker:
#     """YOLO와 칼만 필터를 사용하여 공을 탐지하고 추적하는 총괄 클래스"""

#     def __init__(self, fps: int = 30):
#         self.logger = setup_logger(__name__)
#         try:
#             self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
#         except Exception as e:
#             raise RuntimeError(f"YOLO 모델을 로드할 수 없습니다: {e}")

#         self.kf = ImprovedKalmanFilter(fps=fps)
#         self.last_known_ball_info = None
#         self.just_reinitialized = False

#         # ✨ Phase 2 추가: 추적 상태 관리
#         self.state = "SEARCHING"  # 초기 상태: 공을 찾는 중
#         self.occlusion_frames = 0

#     # ball.py

#     # in ball.py -> class BallTracker

#     def _select_best_ball(
#         self,
#         detections: List[Dict],
#         last_pos: tuple,
#         last_radius: Optional[float],
#         pose_landmarks: Optional[mp.solutions.pose.PoseLandmark],
#         frame_shape: tuple,
#     ) -> Optional[dict]:
#         """
#         (상태 인지 최종 버전)
#         '킥 직후' 상태에 따라 점수 가중치를 동적으로 변경하여 최적의 공을 선택합니다.
#         """
#         if not detections:
#             return None

#         best_ball = None
#         min_score = float("inf")

#         # [수정] 재초기화 직후인지, 킥 직후인지 상태를 확인하고, 그에 맞는 가중치를 선택합니다.
#         if self.just_reinitialized:
#             # 재초기화 직후 첫 프레임에는 예측이 부정확하므로, 예측 점수 가중치를 0으로 설정
#             w_pred = 0.0
#             w_size = config.BALL_SELECTION_WEIGHT_SIZE_POST_KICK
#             w_player = config.BALL_SELECTION_WEIGHT_PLAYER_POST_KICK
#         elif self.state == "POST_IMPACT_TRACKING":
#             # 재초기화 이후, 정상적인 킥 추적 상태일 때의 가중치
#             w_pred = config.BALL_SELECTION_WEIGHT_PREDICTION_POST_KICK
#             w_size = config.BALL_SELECTION_WEIGHT_SIZE_POST_KICK
#             w_player = config.BALL_SELECTION_WEIGHT_PLAYER_POST_KICK
#         else:  # 평상시
#             w_pred = config.BALL_SELECTION_WEIGHT_PREDICTION
#             w_size = config.BALL_SELECTION_WEIGHT_SIZE
#             w_player = config.BALL_SELECTION_WEIGHT_PLAYER

#         foot_positions = []
#         if pose_landmarks:
#             left_ankle = pose_landmarks.landmark[
#                 mp.solutions.pose.PoseLandmark.LEFT_ANKLE
#             ]
#             right_ankle = pose_landmarks.landmark[
#                 mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
#             ]
#             if left_ankle.visibility > config.MP_POSE_MIN_DETECTION_CONFIDENCE:
#                 foot_positions.append(
#                     (left_ankle.x * frame_shape[1], left_ankle.y * frame_shape[0])
#                 )
#             if right_ankle.visibility > config.MP_POSE_MIN_DETECTION_CONFIDENCE:
#                 foot_positions.append(
#                     (right_ankle.x * frame_shape[1], right_ankle.y * frame_shape[0])
#                 )

#         for det in detections:
#             dist_from_prediction = np.linalg.norm(
#                 np.array(det["center"]) - np.array(last_pos)
#             )
#             score_pred = min(
#                 dist_from_prediction / config.BALL_SELECT_MAX_PREDICTION_DIST, 1.0
#             )

#             size_diff = (
#                 abs(det["radius"] - last_radius) if last_radius is not None else 0
#             )
#             score_size = min(size_diff / config.BALL_SELECT_MAX_SIZE_DIFF, 1.0)

#             dist_to_foot = float("inf")
#             if foot_positions:
#                 distances_to_feet = [
#                     np.linalg.norm(np.array(det["center"]) - np.array(fp))
#                     for fp in foot_positions
#                 ]
#                 dist_to_foot = (
#                     min(distances_to_feet) if distances_to_feet else float("inf")
#                 )
#             score_foot = min(dist_to_foot / config.BALL_SELECT_MAX_FOOT_DIST, 1.0)

#             # 선택된 가중치를 사용하여 최종 점수를 계산합니다.
#             final_score = (
#                 score_pred * w_pred + score_size * w_size + score_foot * w_player
#             )

#             if final_score < min_score:
#                 min_score = final_score
#                 best_ball = det

#         return best_ball

#     # ✨ Phase 3 추가: analyzer.py가 호출할 메서드
#     def notify_impact_detected(self):
#         """
#         analyzer로부터 임팩트가 감지되었음을 통보받아, 추적 모드를 변경합니다.
#         """
#         if self.state != "POST_IMPACT_TRACKING":
#             self.state = "POST_IMPACT_TRACKING"
#             self.logger.info("상태 변경: -> POST_IMPACT_TRACKING")

#             # 칼만 필터의 파라미터를 비행 모드에 맞게 조정
#             self.kf.set_airborne(True)
#             self.kf.increase_process_noise_for_kick()

#     def _get_foot_bbox(self, pose_landmarks, frame_shape) -> Optional[tuple]:
#         if not pose_landmarks:
#             return None
#         l_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
#         r_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
#         if l_ankle.visibility < 0.5 and r_ankle.visibility < 0.5:
#             return None
#         ankle = l_ankle if l_ankle.visibility > r_ankle.visibility else r_ankle
#         x, y = int(ankle.x * frame_shape[1]), int(ankle.y * frame_shape[0])
#         return (x - 40, y - 40, x + 40, y + 40)

#     # 👇 다음 에러의 원인이 될 뻔한 누락된 함수
#     def _is_ball_in_impact_zone(self, ball_center, foot_bbox, margin=50) -> bool:
#         if not ball_center or not foot_bbox:
#             return False
#         fx1, fy1, fx2, fy2 = foot_bbox
#         iz = (fx1 - margin, fy1 - margin, fx2 + margin, fy2 + margin)
#         return iz[0] < ball_center[0] < iz[2] and iz[1] < ball_center[1] < iz[3]

#     def process_frame(
#         self, frame: np.ndarray, pose_landmarks, yolo_results: Optional[List] = None
#     ) -> Dict[str, Any]:
#         best_ball, measurement = None, None

#         if yolo_results is None:
#             yolo_results = self.yolo_model.predict(
#                 frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False
#             )

#         detected_balls = []
#         if yolo_results and yolo_results[0].boxes:
#             for box in yolo_results[0].boxes:

#                 if int(box.cls) == 0:
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                     detected_balls.append(
#                         {
#                             "center": ((x1 + x2) / 2, (y1 + y2) / 2),
#                             "radius": (x2 - x1) / 2,
#                             "box": [x1, y1, x2, y2],
#                         }
#                     )

#         if self.state == "SEARCHING":
#             if detected_balls:
#                 best_ball = detected_balls[0]
#                 measurement = best_ball["center"]
#                 self.kf.initialize_state(measurement)
#                 self.state = "TRACKING"
#                 self.logger.info(
#                     f"공 탐지 시작. 상태 -> TRACKING. 위치: {best_ball['center']}"
#                 )
#                 self.last_known_ball_info = best_ball

#         elif self.state in ["TRACKING", "OCCLUDED", "POST_IMPACT_TRACKING"]:
#             predicted_pos = self.kf.predict()
#             last_radius = (
#                 self.last_known_ball_info["radius"]
#                 if self.last_known_ball_info
#                 else None
#             )

#             best_ball = self._select_best_ball(
#                 detected_balls, predicted_pos, last_radius, pose_landmarks, frame.shape
#             )
#             # 플래그를 사용한 후 리셋
#             if self.just_reinitialized:
#                 self.just_reinitialized = False

#             measurement = best_ball["center"] if best_ball else None

#             self.kf.update(measurement)

#             if self.state == "TRACKING" and not best_ball:
#                 foot_bbox = self._get_foot_bbox(pose_landmarks, frame.shape)
#                 if self.last_known_ball_info and self._is_ball_in_impact_zone(
#                     self.last_known_ball_info["center"], foot_bbox
#                 ):
#                     self.state = "OCCLUDED"
#                     self.occlusion_frames = 1
#                     self.logger.warning("임팩트 존에서 공 소실. 상태 -> OCCLUDED")

#             elif self.state == "OCCLUDED":
#                 self.occlusion_frames += 1
#                 if best_ball:
#                     self.state = "TRACKING"
#                     self.occlusion_frames = 0
#                     self.logger.info("공 재탐지 성공. 상태 -> TRACKING")
#                 elif self.occlusion_frames > self.kf.max_prediction_frames:
#                     self.state = "SEARCHING"
#                     self.logger.error("추적 완전 실패. 상태 -> SEARCHING")

#         if self.state != "SEARCHING":
#             current_state = self.kf.get_current_state()
#             final_radius = (
#                 best_ball["radius"]
#                 if best_ball
#                 else (
#                     self.last_known_ball_info.get("radius")
#                     if self.last_known_ball_info
#                     else 15
#                 )
#             )
#             final_result = {
#                 "center": current_state["position"],
#                 "radius": final_radius,
#                 "confidence": current_state["confidence"],
#                 "is_predicted": measurement is None,
#                 "box": best_ball["box"] if best_ball else None,
#             }
#             if best_ball:
#                 self.last_known_ball_info = best_ball
#             return final_result
#         else:
#             return {
#                 "center": None,
#                 "radius": None,
#                 "confidence": 0.0,
#                 "is_predicted": True,
#                 "box": None,
#             }

#     def force_reinitialize(self, position: tuple, radius: float):
#         """
#         외부에서 주어진 정보로 칼만 필터와 추적기 상태를 강제로 재초기화합니다.
#         '선택' 과정을 생략하고 이 공을 새로운 추적 대상으로 확정합니다.
#         """
#         self.logger.info(f"강제 재초기화 실행. 타겟 위치: {position}")

#         # 1. 칼만 필터의 상태를 새로운 공의 위치로 초기화
#         self.kf.initialize_state(position)

#         # 2. 마지막으로 알려진 공 정보를 새로운 공으로 업데이트
#         self.last_known_ball_info = {
#             "center": position,
#             "radius": radius,
#             "box": None,  # box 정보는 이 시점에서는 중요하지 않음
#         }

#         # 3. 상태를 POST_IMPACT_TRACKING 으로 즉시 변경
#         self.notify_impact_detected()
#         self.just_reinitialized = True

# src/ball.py (단순화 버전)
"""
(단순화 버전)
YOLOv8 모델을 사용하여 각 프레임에서 가장 확실한 공을 탐지하는 역할만 수행합니다.
복잡한 칼만 필터나 상태 추적 로직을 모두 제거합니다.
"""
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any

# 내부 모듈 임포트
from . import config
from .utils import setup_logger


class BallTracker:
    """
    YOLO 모델을 이용해 프레임 내에서 가장 신뢰도 높은 공 하나를 탐지합니다.
    """

    def __init__(self, **kwargs):
        """
        YOLO 모델만 로드합니다. fps 등 다른 인자는 필요 없습니다.
        """
        self.logger = setup_logger(__name__)
        try:
            self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
            self.logger.info("YOLO 모델 로딩 성공.")
        except Exception as e:
            raise RuntimeError(f"YOLO 모델을 로드할 수 없습니다: {e}")

    def process_frame(self, frame: np.ndarray, *args, **kwargs) -> Dict[str, Any]:
        """
        주어진 프레임에서 YOLO를 실행하고, 가장 신뢰도(confidence) 높은 공 하나를 선택하여
        그 정보를 딕셔너리 형태로 반환합니다.
        """
        # YOLO 모델로 예측을 실행합니다. (공 클래스 '0'만 대상)
        yolo_preds = self.yolo_model.predict(
            frame, conf=config.YOLO_CONF_THRESHOLD, classes=[0], verbose=False
        )

        best_ball_info = None
        max_confidence = -1.0

        # 예측 결과가 있는지 확인합니다.
        if yolo_preds and yolo_preds[0].boxes:
            # 탐지된 모든 공 중에서 가장 신뢰도가 높은 공을 찾습니다.
            for box in yolo_preds[0].boxes:
                if box.conf[0] > max_confidence:
                    max_confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    best_ball_info = {
                        "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                        "radius": (x2 - x1) / 2,  # 반지름은 너비를 기준으로 계산
                        "box": [x1, y1, x2, y2],
                    }

        if best_ball_info:
            # 공을 찾았을 경우, 데이터를 반환합니다.
            # is_predicted는 'False'로 설정하여 실제 탐지값임을 명시합니다.
            return {
                "center": best_ball_info["center"],
                "radius": best_ball_info["radius"],
                "confidence": max_confidence,
                "is_predicted": False,
                "box": best_ball_info["box"],
            }
        else:
            # 공을 찾지 못했을 경우, 비어 있는 데이터를 반환합니다.
            # is_predicted는 'True'로 설정하여 탐지된 값이 아님을 명시합니다.
            return {
                "center": None,
                "radius": None,
                "confidence": 0.0,
                "is_predicted": True,
                "box": None,
            }
