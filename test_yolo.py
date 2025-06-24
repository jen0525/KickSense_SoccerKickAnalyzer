# test_yolo.py
import cv2
from ultralytics import YOLO
from pathlib import Path

# 1. 여기에 테스트할 모델의 정확한 경로를 입력하세요.
MODEL_PATH = "/Users/heojaeeun/Documents/soccer-kick-analyzer/models/ball_yolom.pt"

# 2. 공이 명확하게 보이는 이미지 파일의 경로를 입력하세요.
# 경로 끝에 공백이 있을 수 있으니 .strip()으로 제거합니다.
VIDEO_PATH = "/Users/heojaeeun/Documents/soccer-kick-analyzer/tests/custom_test_data/test5(inside_good).mp4".strip()

# 3. 모델 로드
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ 모델 로드 성공: {MODEL_PATH}")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    exit()

# 4. 예측 실행 및 동영상 출력
try:
    # stream=True 옵션을 사용하여 동영상을 프레임 단위로 처리합니다.
    results = model(VIDEO_PATH, classes=0, conf=0.4, stream=True, verbose=False)
    print("\n✅ 예측 실행 시작... 'q' 키를 누르면 종료됩니다.")

    # 각 프레임에 대한 결과를 반복합니다.
    for r in results:
        # YOLO 결과 객체에 내장된 .plot() 메소드를 사용하여
        # 감지된 객체의 경계 상자와 라벨이 그려진 프레임을 얻습니다.
        annotated_frame = r.plot()

        # "YOLOv8 Inference"라는 이름의 창에 결과 프레임을 보여줍니다.
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # 'q' 키를 누르면 반복문을 중단합니다.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"❌ 예측 실행 중 오류 발생: {e}")

finally:
    # 모든 작업이 끝나면 화면에 열린 창을 모두 닫습니다.
    cv2.destroyAllWindows()
    print("\n✅ 동영상 재생이 종료되었습니다.")
