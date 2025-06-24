# KickSense Soccer Kick Analyzer Server 

## ⚽️ 축구 킥 동작을 AI로 분석하여 성능 지표와 개선 피드백을 제공하는 API 서버입니다.

## 주요 기능

- **킥 분석**: YOLO와 MediaPipe 포즈 추정으로 킥 동작 분석
- **성능 지표 계산**: 공의 초기 속도, 발의 스윙 속도, 디딤발 안정성 및 프로 선수와의 자세 유사도 등을 종합하여 킥의 완성도를 점수화
- **단계별 분석**: Phase 1 - 백스윙, Phase 2 - 임팩트 
- **개인 맞춤 피드백**: AI 기반 개선 제안사항 생성
- **시각화**: 분석 결과 차트 및 주요 장면에 대한 시각적 하이라이트를 담은 최종 분석 영상 생성
- **Firebase 연동**: 클라우드 기반 결과 저장 및 조회

## ⚽️ Soccer Kick Analyzer 프로젝트 폴더 구조

```
.
├── main.py                    # 🚀 FastAPI 웹 서버를 실행하는 시작점 파일입니다.
├── service.py                 # 🤖 AI 모델을 서빙하는 BentoML 서비스 로직을 정의합니다. (Optional)
├── bentofile.yaml            # 🍱 BentoML 서비스 배포를 위한 레시피 파일입니다. (Optional)
├── requirements.txt          # 📦 이 프로젝트에 필요한 파이썬 라이브러리 목록입니다.
├── .env                      # 🔐 Firebase 인증 정보 등 환경변수를 저장합니다.
│
├── src/                      # 🧠 프로젝트의 핵심 두뇌! 모든 주요 소스 코드가 들어있습니다.
│   ├── config.py            # ⚙️ 모든 설정을 한곳에서 관리하는 중앙 제어실입니다. (민감도, 경로 등)
│   ├── analyzer.py          # 🕵️‍♂️ 영상 분석의 전체 과정을 지휘하는 총감독입니다.
│   ├── ball.py              # ⚾️ 영상 속 축구공을 찾아냅니다.
│   ├── impact_detection.py  # 💥 킥의 임팩트와 백스윙 순간을 귀신같이 찾아냅니다.
│   ├── scoring.py           # 💯 분석 결과를 바탕으로 점수를 매기는 채점관입니다.
│   ├── visualizer.py        # 🎨 분석 결과를 영상 위에 그림으로 그려줍니다. (스켈레톤, 점수판 등)
│   ├── feedback_generator.py # ✍️ 분석 점수에 따라 사용자에게 맞춤형 피드백을 생성합니다.
│   └── firebase_client.py   # 🔥 Firebase와 통신하여 영상, 결과 등을 저장하고 불러옵니다.
│
├── models/                   # 🤖 훈련된 AI 모델을 보관하는 곳입니다.
│   └── ball_yolom.pt        # 축구공 탐지를 위해 파인튜닝된 YOLO 모델 파일입니다.
│
├── data/                     # 📊 분석에 사용되는 데이터를 저장합니다.
│   └── pro_player_stats.json # 프로 선수의 통계 데이터로, 사용자 자세와 비교하는 데 사용됩니다.
│
├── assets/                   # 🖼️ 로고, 폰트 등 앱에 필요한 리소스 파일이 들어있습니다.
├── outputs/                  # 🎬 분석이 완료된 비디오, 이미지 결과물이 저장되는 폴더입니다.
├── notebooks/                # 🧪 알고리즘을 실험하고 데이터를 분석하는 주피터 노트북 공간입니다.
└── tests/                    # 🧪 테스트 영상들이 있습니다.
```

## 🛠️ 설치 및 설정

### 1. 환경 준비

```bash
# Python 3.11+ 권장
python --version

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정

`.env` 파일을 생성하고 다음 정보를 입력하세요:

```bash
# Firebase 설정
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=your-bucket.appspot.com
FIREBASE_CREDENTIALS_PATH=path/to/serviceAccountKey.json

# 환경 설정
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
```

## 🚀 실행 방법

### Option 1: FastAPI 서버 실행 (권장)

```bash
# FastAPI 개발 서버 시작
python main.py

# 또는 uvicorn으로 직접 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**API 문서 확인:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Option 2: BentoML 서버 실행 (선택사항)

```bash
# BentoML 개발 서버 시작
bentoml serve service:svc --reload --host 0.0.0.0 --port 3000
```

**API 문서 확인:**
- Swagger UI: http://localhost:3000/docs
- ReDoc: http://localhost:3000/redoc

### 프로덕션 배포

```bash
# BentoML 빌드
bentoml build

# Docker 이미지 생성
bentoml containerize kick_analysis_service:latest

# 컨테이너 실행
docker run -p 3000:3000 kick_analysis_service:latest
```

## 📡 API 사용법

### 킥 분석 요청 (FastAPI)

```bash
curl -X POST "http://localhost:8000/analyze_kick" \
  -H "Content-Type: application/json" \
  -d '{
    "jobID": "job_123456",
    "videoURL": "https://example.com/kick_video.mp4",
    "kickingFoot": "right"
  }'
```

### 킥 분석 요청 (BentoML) (⭐️Optional)

```bash
curl -X POST "http://localhost:3000/analyze_kick" \
  -H "Content-Type: application/json" \
  -d '{
    "jobID": "job_123456",
    "videoURL": "https://example.com/kick_video.mp4",
    "kickingFoot": "right"
  }'
```

### 작업 상태 조회

```bash
# FastAPI
curl -X POST "http://localhost:8000/get_status" \
  -H "Content-Type: application/json" \
  -d '{"jobID": "job_123456"}'

# BentoML
curl -X POST "http://localhost:3000/get_status" \
  -H "Content-Type: application/json" \
  -d '{"jobID": "job_123456"}'
```

### 헬스 체크

```bash
# FastAPI
curl "http://localhost:8000/health_check"

# BentoML
curl "http://localhost:3000/health_check"
```

## 📊 분석 결과 구조

### 점수 데이터 예시

```python
{
  'scores': {
    'categories': {
      'backswing_evaluation': {
        'details': {
          'backswing_angle_comparison': {
            'max_score': 10,
            'score': 7.8,
            'value': '7.8/10 vs Ronaldo'
          },
          'backswing_knee_angle_size': {
            'max_score': 10,
            'score': 10,
            'value': '67.4도'
          },
          'kick_foot_kinematics_change': {
            'max_score': 10,
            'score': 4,
            'value': '25.57 px/fr²'
          },
          'max_foot_swing_speed': {
            'max_score': 10,
            'score': 8,
            'value': '60.7km/h'
          },
          'support_foot_stability': {
            'max_score': 10,
            'score': 4,
            'value': '10.81cm/frame'
          }
        },
        'subtotal': 33.8
      },
      'impact_evaluation': {
        'details': {
          'ball_contact_point': {
            'max_score': 10,
            'score': 8,
            'value': 'Bottom-Right'
          },
          'ball_initial_speed': {
            'max_score': 10,
            'score': 6,
            'value': '73.3km/h'
          },
          'hitting_foot_part': {
            'max_score': 10,
            'score': 10,
            'value': 'inside'
          },
          'impact_angle_comparison': {
            'max_score': 10,
            'score': 5.8,
            'value': '5.8/10 vs Ronaldo'
          },
          'support_foot_ball_distance': {
            'max_score': 10,
            'score': 10,
            'value': '20.2cm'
          }
        },
        'subtotal': 39.8
      }
    },
    'max_score': 100,
    'percentage': 73.6,
    'total_score': 73.6
  }
}
```

### 피드백 데이터

```python
{
  'feedback': {
    '백스윙': {
      '백스윙 점수': 34,
      '백스윙 피드백': '좋은 백스윙 동작입니다. 다만 몇 가지 세부 항목을 개선하면 완성도를 높일 수 있습니다. 특히, \'스윙 가속 안정성\' 부분을 중점적으로 개선한다면 해당 단계의 완성도가 크게 향상될 것입니다.'
    },
    '임팩트': {
      '임팩트 점수': 40,
      '임팩트 피드백': '좋은 임팩트 동작입니다. 다만 몇 가지 세부 항목을 개선하면 완성도를 높일 수 있습니다. 특히, \'임팩트 자세 정확도\' 부분을 중점적으로 개선한다면 해당 단계의 완성도가 크게 향상될 것입니다.'
    },
    '종합': {
      '총점': 74,
      '총평': '🙂 훌륭한 잠재력을 가진 킥입니다. 전반적으로 좋은 기본기를 갖추고 있으나, 특정 구간에서 힘의 손실이나 자세의 불안정성이 관찰됩니다. 제시된 약점을 집중적으로 개선한다면 경기력을 한 단계 위로 끌어올릴 수 있습니다.',
      '피드백': {
        '매우 잘한점': [
          '최적의 백스윙 무릎 각도 (67.4도): 백스윙 시 무릎을 충분히 깊게 접어, 마치 활시위를 당기듯 강력한 탄성 에너지를 다리에 저장했습니다. 이는 폭발적인 스윙 속도로 이어지는 중요한 준비 동작입니다.',
          '이상적인 디딤발 거리 (20.2cm): 공 옆에 최적의 간격으로 디딤발을 위치시켰습니다. 이는 상체의 균형을 안정시키고, 킥을 하는 다리가 자유롭게 최대 스윙을 할 수 있는 공간을 확보하여 킥의 파워와 안정성을 모두 잡는 교과서적인 자세입니다.'
        ],
        '아쉬운 점': [
          '불안정한 가속 (25.57 px/fr²): 임팩트 직전 스윙 속도가 급격하게 변했습니다. 힘을 주려다 자세가 흐트러지거나, 공을 맞추기 위해 억지로 속도를 조절하는 등 불안정한 스윙입니다.',
          '불안정한 디딤발 (평균 흔들림: 10.81cm/frame): 킥을 하는 동안 디딤발이 흔들리는 모습이 감지되었습니다. 이는 코어 밸런스가 무너졌거나, 디딤발을 딛는 힘이 부족하다는 신호입니다. 파워 손실의 주된 원인이 되므로, 한 발 서기 등 밸런스 훈련과 디딤발을 땅에 \'심는다\'는 느낌으로 딛는 연습이 필요합니다.'
        ]
      }
    }
  }
}
```

## ⚙️ 설정 사용자화

### config.py 주요 설정

```python
# 모델 설정
YOLO_MODEL_PATH = "models/ball_yolom.pt"
MODEL_CONFIDENCE_THRESHOLD = 0.4

# 영상 처리 설정
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']

# 물리적 상수
REAL_BALL_DIAMETER_CM = 22.0
REFERENCE_FPS = 30.0
```

## 🔧 개발 및 테스트

### 테스트 실행

```bash
# 특정 테스트 파일 실행
python test_analyzer.py
python test_yolo.py
```

### 코드 포맷팅

```bash
# Black으로 코드 포맷팅
black src/ *.py

# Flake8으로 린팅
flake8 src/ *.py
```

## 🐳 Docker 배포

### Dockerfile 예시

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 3000

CMD ["bentoml", "serve", "service:svc", "--host", "0.0.0.0", "--port", "3000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  kick-analyzer:
    build: .
    ports:
      - "3000:3000"
    environment:
      - FIREBASE_PROJECT_ID=your-project
      - ENVIRONMENT=production
    volumes:
      - ./models:/app/models
```

## 🚨 문제 해결

### 자주 발생하는 문제

**Q: 모델 파일을 찾을 수 없다는 오류**  
A: `models/` 폴더에 `ball_yolom.pt` 파일이 있는지 확인하세요.

**Q: Firebase 연결 오류**  
A: 환경변수 설정과 서비스 계정 키 파일 경로를 확인하세요.

**Q: 영상 처리 실패**  
A: 지원되는 영상 형식(.mp4, .avi, .mov, .mkv)인지 확인하세요.

**Q: 포트 충돌 오류**  
A: FastAPI는 8000번, BentoML은 3000번 포트를 사용합니다. 필요시 포트를 변경하세요.

## 📞 지원

- 이메일: wodms5744@gmail.com

---
