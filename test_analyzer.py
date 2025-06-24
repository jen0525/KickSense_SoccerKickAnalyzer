# run_local_test.py
import os
import pprint  # 딕셔너리를 예쁘게 출력하기 위해 사용
import uuid  # 고유한 테스트 ID를 만들기 위해 사용

# 가장 중요한 부분: src 폴더의 analyzer 모듈에서 run_full_analysis 함수를 가져옵니다.
from src.analyzer import run_full_analysis
from src.config import OUTPUTS_DIR  # 설정 파일에서 출력 폴더 경로 가져오기

# ===============================================================
#                   테스트 설정
# ===============================================================

# 여기에 테스트하고 싶은 동영상 파일의 전체 경로를 정확하게 입력하세요.
# 예시: "C:/Users/YourName/Videos/my_kick.mp4" (윈도우)
# 예시: "/Users/YourName/Movies/my_kick.mp4" (맥)
VIDEO_PATH = "/Users/heojaeeun/Documents/soccer-kick-analyzer/tests/custom_test_data/test5(inside_good).mp4"
# 테스트는 7번, 5번
# 차는 발 선호도 설정 ('left', 'right', 또는 'auto')
KICKING_FOOT_PREFERENCE = "auto"

# ===============================================================


def main():
    """
    로컬에서 전체 분석 파이프라인을 테스트하기 위한 메인 함수
    """

    print("=" * 50)
    print("로컬 분석 파이프라인 테스트를 시작합니다.")
    print("=" * 50)

    # 1. 동영상 파일 존재 여부 확인
    if not os.path.exists(VIDEO_PATH):
        print(f"에러: 동영상 파일을 찾을 수 없습니다!")
        print(f"경로를 확인해주세요: {VIDEO_PATH}")
        return

    # 2. 고유한 테스트 작업 ID 생성
    job_id = f"local-test-{str(uuid.uuid4())[:8]}"

    print(f"▶작업 ID: {job_id}")
    print(f"▶분석 대상 동영상: {VIDEO_PATH}")
    print(f"▶선호하는 발: {KICKING_FOOT_PREFERENCE}")
    print(
        "\n분석 중... 잠시만 기다려주세요. (동영상 길이에 따라 몇 분 정도 소요될 수 있습니다)\n"
    )

    try:
        # 3. ✨ 핵심: 전체 분석 함수 호출!
        final_data, video_out_path, backswing_img_path, impact_img_path = (
            run_full_analysis(
                video_path=VIDEO_PATH,
                kicking_foot_preference=KICKING_FOOT_PREFERENCE,
                job_id=job_id,
            )
        )

        # 4. 결과 확인
        if not final_data:
            print("분석에 실패했습니다. 콘솔에 출력된 에러 메시지를 확인해주세요.")
            return

        print("\n" + "=" * 50)
        print("분석이 성공적으로 완료되었습니다!")
        print("=" * 50)

        # 5. 분석 결과(딕셔너리) 상세 출력
        print("\n 최종 분석 데이터]")
        pprint.pprint(final_data)

        # 6. 생성된 결과물 파일 경로 출력
        print("\n[생성된 결과 파일]")
        print(f"  - 분석 영상: {video_out_path}")
        print(f"  - 백스윙 캡처: {backswing_img_path}")
        print(f"  - 임팩트 캡처: {impact_img_path}")
        print(f"\n위 경로의 폴더({OUTPUTS_DIR})에서 결과물을 확인하세요.")

    except Exception as e:
        print(f"\n 분석 과정에서 심각한 오류가 발생했습니다: {e}")
        # 더 상세한 오류를 보고 싶을 경우 아래 줄의 주석을 해제하세요.
        # import traceback
        # traceback.print_exc()


if __name__ == "__main__":
    main()
