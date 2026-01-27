# -*- coding: utf-8 -*-
"""
크롤러 설정 모듈

midaeipsi.com 크롤링에 필요한 상수, URL 패턴, 대학 매핑 등의 설정을 정의합니다.
"""

from typing import Dict, List, Tuple

# ======================
# URL 설정
# ======================
BASE_URL: str = "https://midaeipsi.com/art/board.php"
SITE_ORIGIN: str = "https://midaeipsi.com"
BOARD_NAME: str = "academynews"

# ======================
# 크롤링 범위
# ======================
POST_RANGE: Tuple[int, int] = (194, 3940)

# ======================
# 요청 설정
# ======================
REQUEST_DELAY: float = 1.5  # 요청 간 대기 시간 (초)
REQUEST_TIMEOUT: int = 30  # 요청 타임아웃 (초)
MAX_RETRIES: int = 3  # 이미지 다운로드 최대 재시도 횟수
RETRY_DELAY: float = 2.0  # 재시도 간 대기 시간 (초)

# ======================
# HTTP 헤더
# ======================
REQUEST_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# ======================
# 이미지 필터링 - 광고/배너 제외 패턴
# ======================
EXCLUDE_IMAGE_PATTERNS: List[str] = [
    "bannermidaeipsshort",
    "bannermidaeipsdown",
    "skin_board",
    "img_new",
]

# ======================
# 지원 이미지 확장자
# ======================
IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]

# ======================
# 대학교 티어 분류
# ======================
UNIVERSITY_TIER: Dict[str, List[str]] = {
    "S": ["서울대", "홍익대"],
    "A": [
        "국민대",
        "이화여대",
        "중앙대",
        "한양대",
        "건국대",
        "경희대",
        "동국대",
        "숙명여대",
        "성신여대",
        "서울예대",
    ],
    "B": [
        "상명대",
        "서울과기대",
        "단국대",
        "인하대",
        "아주대",
        "인천대",
        "가천대",
        "한성대",
    ],
}
# C 티어 = S, A, B에 포함되지 않는 모든 대학

# ======================
# 대학명 정규화 매핑
# ======================
UNIVERSITY_NORMALIZE: Dict[str, str] = {
    "홍대": "홍익대",
    "홍익대학교": "홍익대",
    "국민대학교": "국민대",
    "서울예술대학교": "서울예대",
    "이화여자대학교": "이화여대",
    "중앙대학교": "중앙대",
    "한양대학교": "한양대",
    "건국대학교": "건국대",
    "경희대학교": "경희대",
    "동국대학교": "동국대",
    "숙명여자대학교": "숙명여대",
    "성신여자대학교": "성신여대",
    "상명대학교": "상명대",
    "서울과학기술대학교": "서울과기대",
    "단국대학교": "단국대",
    "인하대학교": "인하대",
    "아주대학교": "아주대",
    "인천대학교": "인천대",
    "가천대학교": "가천대",
    "한성대학교": "한성대",
    "서울대학교": "서울대",
}

# ======================
# 학과명 정규화 매핑
# ======================
DEPARTMENT_NORMALIZE: Dict[str, str] = {
    "시각디자인": "visual_design",
    "시각디자인과": "visual_design",
    "시디": "visual_design",
    "시각디자인학과": "visual_design",
    "산업디자인": "industrial_design",
    "산업디자인과": "industrial_design",
    "산디": "industrial_design",
    "산업디자인학과": "industrial_design",
    "공예": "craft",
    "공예과": "craft",
    "공예디자인": "craft",
    "금속공예": "craft",
    "도자공예": "craft",
    "섬유공예": "craft",
    "회화": "fine_art",
    "회화과": "fine_art",
    "서양화": "fine_art",
    "동양화": "fine_art",
    "한국화": "fine_art",
    "조소": "fine_art",
    "조소과": "fine_art",
    "판화": "fine_art",
}

# ======================
# 출력 디렉토리 설정
# ======================
DATA_DIR: str = "data"
RAW_IMAGES_DIR: str = f"{DATA_DIR}/crawled/raw_images"
METADATA_DIR: str = f"{DATA_DIR}/crawled/metadata"

# ======================
# 진행 로그 설정
# ======================
PROGRESS_LOG_INTERVAL: int = 100  # N개 포스트마다 진행 상황 출력
