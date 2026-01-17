# MIRIP 기술 스택

> 미립(MIRIP) 전체 기술 스택 및 인프라 구성

---

## 1. 기술 스택 개요

| 영역 | 기술 | 버전 | 용도 |
|-----|------|------|------|
| Frontend | React | 18.x | UI 프레임워크 |
| Routing | React Router | 6.x | 클라이언트 라우팅 |
| Bundler | Create React App | 5.x | 빌드 도구 |
| Styling | CSS Modules | - | 컴포넌트 스타일링 |
| State | React Context -> Zustand | - | 상태 관리 |
| Charts | Recharts / Chart.js | - | 데이터 시각화 |

---

## 2. 프론트엔드 스택

### 2.1 React 18.x

메인 UI 프레임워크입니다.

**선택 이유:**
- 컴포넌트 기반 아키텍처로 재사용성 극대화
- 가상 DOM을 통한 효율적인 렌더링
- 풍부한 생태계 및 커뮤니티 지원
- Concurrent Features를 통한 향상된 사용자 경험

**주요 의존성:**
- react: ^18.3.1
- react-dom: ^18.3.1
- react-router-dom: ^6.2.1

### 2.2 React Router DOM 6.x

클라이언트 사이드 라우팅 솔루션입니다.

**선택 이유:**
- 선언적 라우팅으로 가독성 향상
- Nested Routes 지원
- 데이터 로딩 통합 (loader/action)
- 스크롤 복원 기능

### 2.3 CSS Modules

컴포넌트별 스타일 격리 솔루션입니다.

**선택 이유:**
- 클래스명 충돌 방지
- 컴포넌트 단위 스타일 캡슐화
- 빌드 시 자동 최적화
- CSS-in-JS보다 빠른 런타임 성능

**파일 구조:**
```
ComponentName.js
ComponentName.module.css
```

### 2.4 상태 관리

**현재:** React Context API
**계획:** Zustand 도입 예정

**Zustand 선택 이유:**
- 간단한 API
- 보일러플레이트 최소화
- TypeScript 지원 우수
- 작은 번들 사이즈

---

## 3. 백엔드/인프라 스택

### 3.1 서버 구성

| 구분 | 기술 | 용도 |
|------|------|------|
| API Server | FastAPI | ML Inference + REST API |
| Hosting | Firebase Hosting | 정적 파일 서빙 |
| Database | Firebase Firestore | 유저/공모전/결과 데이터 |
| Storage | Firebase Storage | 이미지 저장 |
| Auth | Firebase Auth | 소셜 로그인 (카카오/구글) |
| Cache | Redis | API 캐싱 |
| Payments | 토스페이먼츠 / 카카오페이 | 결제 연동 |

### 3.2 Firebase 구성

**Firebase Hosting:**
- 정적 웹사이트 호스팅
- 글로벌 CDN
- SSL 자동 설정
- 간편한 CLI 배포

**Firebase Firestore:**
- NoSQL 문서 데이터베이스
- 실시간 동기화
- 오프라인 지원
- 자동 확장

**Firebase Auth:**
- 소셜 로그인 (카카오, 구글)
- 이메일/비밀번호 인증
- 세션 관리
- 보안 규칙

**Firebase Storage:**
- 이미지 저장
- CDN 지원
- 보안 규칙 기반 접근 제어

### 3.3 FastAPI

Python 기반 고성능 웹 프레임워크입니다.

**선택 이유:**
- 비동기 지원으로 높은 처리량
- 자동 API 문서화 (Swagger/ReDoc)
- Pydantic 기반 데이터 검증
- ML 라이브러리와의 완벽한 호환

---

## 4. ML/AI 스택

### 4.1 Core ML

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| PyTorch | 2.1+ | 딥러닝 프레임워크 |
| Transformers | 4.35+ | DINOv2, CLIP 모델 |
| timm | 0.9+ | 비전 모델 라이브러리 |

### 4.2 Data Processing

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| opencv-python | 4.8+ | 이미지 처리 |
| albumentations | 1.3+ | 데이터 증강 |
| pillow | 10.0+ | 이미지 I/O |

### 4.3 Training

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| pytorch-lightning | 2.1+ | 학습 프레임워크 |
| wandb | 0.16+ | 실험 추적 |

### 4.4 Inference

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| fastapi | 0.104+ | API 서버 |
| onnxruntime | 1.16+ | 최적화된 추론 |

### 4.5 Storage

| 기술 | 버전 | 용도 |
|------|------|------|
| PostgreSQL | 15+ | 관계형 데이터 |
| Redis | 7+ | 캐싱 |
| MinIO | latest | 이미지 저장 |

### 4.6 모델 아키텍처

**Backbone 선택:**

| 모델 | 역할 | 선택 이유 |
|------|------|-----------|
| DINOv2 ViT-L | RGB 피처 추출 | visual similarity에서 CLIP 대비 2배 이상 정확도 (64% vs 28%) |
| PiDiNet | Edge 피처 추출 | HED 대비 28% 파라미터로 0.9% ODS 향상, 경량화 |
| CLIP ViT-L | 주제 해석 전용 | text-image alignment 최적화 |

---

## 5. 인프라 및 비용

### 5.1 학습 환경

| 항목 | 스펙 |
|------|------|
| GPU | RTX 4070 Ti Super 16GB |
| 예상 학습 시간 | 4~6시간 (backbone freeze) |
| VRAM 사용량 | ~12GB (batch 32, fp16) |

### 5.2 비용 구조

| 단계 | 비용 | 비고 |
|------|------|------|
| 학습 | 전기세 (~3,000원) | 1회 학습 기준 |
| Inference | 무료 | 로컬 처리 |
| 피드백 생성 | API 비용 (선택) | LLM 호출 시에만 |

### 5.3 Inference 서버 요구사항

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 16GB
- GPU: RTX 3060 12GB (또는 동급)
- Storage: 50GB SSD

**Latency Target:**
- 단일 이미지: < 500ms
- 배치 10개: < 2s

### 5.4 Docker 설정

**Dockerfile:**
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./weights:/app/weights:ro
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_WEIGHTS_PATH=/app/weights
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

## 6. 환경 변수

### 6.1 Frontend (.env)

```
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_FIREBASE_API_KEY=xxx
REACT_APP_FIREBASE_AUTH_DOMAIN=xxx
REACT_APP_FIREBASE_PROJECT_ID=xxx
REACT_APP_FIREBASE_STORAGE_BUCKET=xxx
REACT_APP_TOSS_CLIENT_KEY=xxx
```

### 6.2 Backend (.env)

```
HOST=0.0.0.0
PORT=8000
DEBUG=true
MODEL_WEIGHTS_PATH=/app/weights
DEVICE=cuda
OPENAI_API_KEY=xxx
REDIS_URL=redis://localhost:6379
FIREBASE_CREDENTIALS_PATH=/app/credentials/firebase.json
```

---

## 7. 체크리스트

### 7.1 개발 시작 전

- [ ] Node.js 18+, Python 3.10+
- [ ] Firebase 프로젝트 생성
- [ ] 토스페이먼츠 테스트 계정

### 7.2 ML 모델 가중치

- [ ] DINOv2 ViT-L (transformers 자동)
- [ ] PiDiNet pretrained weights
- [ ] CLIP ViT-L (transformers 자동)
- [ ] Trained fusion/rubric heads

### 7.3 Phase별 완료 확인

- [ ] Phase 1: 공모전 MVP 동작
- [ ] Phase 2: AI 진단 API 응답 < 3초
- [ ] Phase 3: 결제 플로우 테스트

---

## 8. 개발 환경 요구사항

### 8.1 필수 소프트웨어

| 소프트웨어 | 최소 버전 | 권장 버전 |
|-----------|----------|----------|
| Node.js | 16.x | 20.x LTS |
| npm | 8.x | 10.x |
| Python | 3.10 | 3.11 |
| Git | 2.x | 최신 |

### 8.2 IDE 권장사항

**Visual Studio Code 확장:**
- ESLint
- Prettier
- ES7+ React/Redux/React-Native snippets
- CSS Modules
- Python
- Pylance

---

## 9. 스크립트 명령어

### 9.1 React 앱 (`my-app/`)

```bash
# 개발 서버 시작
npm start

# 프로덕션 빌드
npm run build

# 테스트 실행
npm test
```

### 9.2 Backend (`backend/`)

```bash
# 개발 서버 시작
uvicorn app.main:app --reload

# Docker 빌드 및 실행
docker-compose up --build

# 테스트 실행
pytest
```

### 9.3 Firebase 배포

```bash
# landing 디렉토리에서
cd landing
firebase deploy --only hosting
```

---

## 10. 브라우저 지원

### 프로덕션 환경

지원 브라우저:
- Chrome (최근 2개 버전)
- Firefox (최근 2개 버전)
- Safari (최근 2개 버전)
- Edge (최근 2개 버전)

### Browserslist 설정

```
> 0.2%
not dead
not op_mini all
```

---

## 11. 참고 자료

### ML 모델
- DINOv2: https://github.com/facebookresearch/dinov2
- PiDiNet: https://github.com/hellozhuo/pidinet
- CLIP: https://github.com/openai/CLIP

### 기술 문서
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/
- Firebase: https://firebase.google.com/docs
- 토스페이먼츠: https://docs.tosspayments.com/

---

*문서 버전: 2.1*
*최종 업데이트: 2025년 1월*
