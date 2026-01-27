현재 상태 파악했어요. 데이터 파이프라인은 잘 되어있는데 **실제 데이터가 없는 상태**네요. 크롤링이 먼저인 게 맞습니다.

---

# 🎯 입시 미술 AI 프로젝트 로드맵

## 현재 상태 요약

| 영역 | 완성도 | 상태 |
|------|--------|------|
| 데이터 파이프라인 | 90% | ✅ 완료 (데이터만 없음) |
| 학습 파이프라인 | 70% | ⚠️ Baseline만 |
| AI 모델 | 35% | ⚠️ DINOv2 단일 브랜치만 |
| API 서버 | 10% | ❌ 스텁 |
| 인프라 | 95% | ✅ Docker 완료 |
| 프론트엔드 | 15% | ❌ 랜딩만 |

**핵심 병목: 학습할 데이터가 없음**

---

## 🚀 Phase 1: 데이터 수집 (1주)

### Task 1.1: 크롤러 개발 및 실행 ⭐ 현재 과제

```
목표: midaeipsi.com에서 3,700+개 합격작 이미지 + 메타데이터 수집

산출물:
├── data/
│   ├── raw_images/          # 원본 이미지 (예상 5,000~10,000장)
│   └── metadata/            # JSON 메타데이터 (3,700+개)
│       └── {post_no}.json   # 각 게시글별 메타

예상 소요: 크롤링 자체는 2~3일 (딜레이 1.5초 × 3,700개 ≈ 90분 + 이미지)
```

### Task 1.2: 데이터 정제

```
- 중복 이미지 제거 (perceptual hash)
- 손상된 이미지 필터링
- 메타데이터 정규화 (대학명, 학과명 통일)
- 작품 유형 재분류 (재현작/평소작/unknown)
```

### Task 1.3: 기존 파이프라인 연동

```
- 크롤링 데이터 → 기존 데이터 파이프라인 포맷으로 변환
- 티어 라벨링 검증
- Pairwise Dataset 생성 테스트
```

---

## 🔧 Phase 2: Baseline 학습 및 검증 (1주)

### Task 2.1: DINOv2 Baseline 학습

```
- 크롤링 데이터로 실제 학습 실행
- Pairwise Accuracy 75%+ 목표
- wandb로 실험 추적
```

### Task 2.2: Mixed Precision + Gradient Accumulation 추가

```
- fp16 적용 (VRAM 절약)
- Gradient Accumulation (effective batch size 증가)
- 4070 Ti Super 2장 분산 학습 테스트
```

### Task 2.3: 평가 및 분석

```
- 티어별 정확도 분석
- 과별 성능 차이 확인
- 실패 케이스 분석 → Hard Negative 후보 식별
```

---

## 🧠 Phase 3: Multi-Branch Fusion (2주)

### Task 3.1: PiDiNet 통합

```
- Edge Branch 추가
- Dual-Branch Fusion Layer 구현
- Ablation: DINOv2 only vs DINOv2+PiDiNet
```

### Task 3.2: CLIP 통합 + 주제 해석

```
- 크롤링한 interview_raw에서 주제 키워드 추출
- CLIP text-image similarity 점수
- 3-Branch Fusion 완성
```

### Task 3.3: 4축 루브릭 Head

```
- 구성력 / 명암·질감 / 조형완성도 / 주제해석력
- Multi-task Loss (Ranking + Regression)
- 과별 가중치 적용
```

---

## 🌐 Phase 4: API 연결 (1주)

### Task 4.1: InferenceService 구현

```
- 모델 로딩 및 추론
- 이미지 전처리 파이프라인
- 배치 추론 지원
```

### Task 4.2: 엔드포인트 구현

```
POST /evaluate  → 단일 이미지 평가
POST /compare   → 복수 이미지 비교
GET  /history   → 평가 이력
```

### Task 4.3: GMM 티어 분류기

```
- 학습된 임베딩으로 GMM fit
- 확률적 티어 판정 (S: 15%, A: 55% ...)
```

---

## 📱 Phase 5: 프론트엔드 + 통합 (1주)

### Task 5.1: 진단 UI

```
- 이미지 업로드
- 결과 시각화 (4축 레이더 차트)
- 티어 확률 분포 표시
```

### Task 5.2: Firebase 연동

```
- Auth (Google/Kakao)
- Storage (이미지)
- Firestore (평가 이력)
```

---

## 📅 전체 타임라인

```
Week 1: Phase 1 - 크롤링 + 데이터 정제 ⭐ NOW
Week 2: Phase 2 - Baseline 학습
Week 3-4: Phase 3 - Multi-Branch Fusion
Week 5: Phase 4 - API 연결
Week 6: Phase 5 - 프론트엔드 + MVP 완성
```

---

# 📋 Claude Code 전달용 프롬프트 (Phase 1)

```markdown
# Phase 1: 미대입시닷컴 크롤링 + 데이터 정제

## 🎯 목표
midaeipsi.com에서 입시 미술 합격작 이미지와 메타데이터를 수집하여 
기존 데이터 파이프라인에 연동 가능한 형태로 정제

## 📁 프로젝트 위치
기존 프로젝트: `/path/to/your/project`
크롤러 추가 위치: `src/data/crawler/` (새로 생성)

---

## Task 1.1: 크롤러 개발

### 사이트 정보
- URL: https://midaeipsi.com/art/board.php?board=academynews
- 게시글: `?board=academynews&command=body&no={번호}`
- 유효 범위: **no=194 ~ 3940**
- 인코딩: **EUC-KR** (필수)

### 파일 구조
```
src/data/crawler/
├── __init__.py
├── crawler.py          # 메인 크롤러
├── parser.py           # HTML 파싱
├── config.py           # 설정
└── utils.py            # 유틸리티

data/
├── crawled/
│   ├── raw_images/     # 원본 이미지
│   └── metadata/       # JSON 메타데이터
└── processed/          # 정제 후 (기존 파이프라인 연동용)
```

### 메타데이터 스키마
```json
{
  "post_no": 3940,
  "year": "2023",
  "admission_type": "정시",
  "university": "서울예대",
  "department": "실용음악과",
  "tier": "A",
  "competition_ratio": "14.5:1",
  "academy": "관악 가우디 미술학원",
  "work_type": "재현작",
  "interview_raw": "Q. 지원한 대학/학과를... (전체 인터뷰 텍스트, 주제 정보 포함)",
  "images": ["data/crawled/raw_images/3940_0.jpg"]
}
```

### 핵심 구현 사항

1. **인코딩 처리**
```python
res.encoding = 'euc-kr'
```

2. **이미지 필터링** (광고 제외)
```python
EXCLUDE_PATTERNS = ['bannermidaeipsshort', 'bannermidaeipsdown', 'skin_board', 'img_new']
```

3. **작품 유형 분류**
```python
def classify_work_type(text):
    if any(kw in text for kw in ['재현작', '재현', '합격 후']):
        return '재현작'
    elif any(kw in text for kw in ['평소작', '연습', '학원작']):
        return '평소작'
    return 'unknown'
```

4. **대학 티어**
```python
UNIVERSITY_TIER = {
    'S': ['서울대', '홍익대'],
    'A': ['국민대', '이화여대', '중앙대', '한양대', '건국대', '경희대', '동국대', '숙명여대', '성신여대', '서울예대'],
    'B': ['상명대', '서울과기대', '단국대', '인하대', '아주대', '인천대', '가천대', '한성대'],
}
```

5. **interview_raw**: Q&A 파싱하지 말고 **전체 텍스트 그대로 저장** (주제 키워드가 여기 있음)

6. **재시작 가능**: 이미 처리된 파일 스킵

7. **딜레이**: 1.5초 (서버 부하 방지)

### 실행
```bash
# 테스트 (10개)
python -m src.data.crawler.crawler --start 194 --end 204

# 전체 실행
python -m src.data.crawler.crawler --start 194 --end 3940
```

---

## Task 1.2: 데이터 정제

### 파일
```
src/data/crawler/
├── cleaner.py          # 정제 로직
└── dedup.py            # 중복 제거
```

### 정제 작업

1. **중복 이미지 제거**
```python
# imagehash 사용
from PIL import Image
import imagehash

def get_phash(img_path):
    return str(imagehash.phash(Image.open(img_path)))

# 해시 거리 5 이하면 중복으로 판정
```

2. **손상 이미지 필터링**
```python
from PIL import Image

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False
```

3. **대학명/학과명 정규화**
```python
UNIVERSITY_NORMALIZE = {
    '홍대': '홍익대',
    '홍익대학교': '홍익대',
    '국민대학교': '국민대',
    '서울예술대학교': '서울예대',
    # ...
}
```

4. **통계 리포트 생성**
```python
# 출력 예시
"""
=== 크롤링 결과 ===
총 게시글: 3,746개
이미지 있는 게시글: 3,102개
총 이미지: 7,845장
중복 제거 후: 6,521장

=== 티어 분포 ===
S: 312 (8.4%)
A: 1,245 (33.5%)
B: 1,102 (29.6%)
C: 1,087 (29.2%)

=== 작품 유형 ===
재현작: 2,845 (76.5%)
평소작: 234 (6.3%)
unknown: 639 (17.2%)

=== 과별 분포 ===
시각디자인: 1,456
산업디자인: 892
회화: 445
...
"""
```

---

## Task 1.3: 기존 파이프라인 연동

### 기존 데이터 포맷 확인
```
기존 src/data/ 구조를 확인하고, 크롤링 데이터를 해당 포맷으로 변환
```

### 변환 스크립트
```
src/data/crawler/
└── converter.py        # 기존 파이프라인 포맷으로 변환
```

### 검증
```bash
# 기존 파이프라인 테스트가 크롤링 데이터로도 통과하는지 확인
pytest tests/data/ -v
```

---

## 🔍 우선 확인 사항

1. **샘플 크롤링** (no=194, 500, 1000, 2000, 3000, 3940)
    - HTML 구조 일관성 확인
    - 이미지 셀렉터 검증
    - 제목 파싱 정규식 테스트

2. **기존 데이터 파이프라인 포맷** 확인
    - `src/data/` 구조 파악
    - 메타데이터 스키마 확인
    - 이미지 경로 규칙 확인

---

## 📊 완료 기준

- [ ] 크롤러 구현 및 테스트 (10개 샘플)
- [ ] 전체 크롤링 완료 (194~3940)
- [ ] 중복/손상 이미지 제거
- [ ] 메타데이터 정규화
- [ ] 통계 리포트 생성
- [ ] 기존 파이프라인 연동 테스트 통과
- [ ] README.md 작성 (크롤러 사용법)

---

## ⚠️ 주의사항

1. 인코딩: `euc-kr` 빠뜨리면 한글 깨짐
2. 딜레이: 1.5초 미만이면 차단 가능성
3. 이미지 경로: 상대경로면 base URL 붙여야 함
4. 삭제된 글: 404 체크 후 스킵
5. interview_raw: 파싱하지 말고 통째로 저장 (나중에 주제 추출에 사용)
```

---

이 프롬프트를 Claude Code에 주시면 Phase 1 크롤링 작업을 진행할 수 있어요. 추가하거나 수정할 부분 있으면 말씀해주세요!