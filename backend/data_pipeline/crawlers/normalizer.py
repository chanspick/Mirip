# -*- coding: utf-8 -*-
"""
정규식 기반 대학명/학과명 정규화 모듈

한국 대학교명은 매우 다양한 형태로 표기됩니다:
- "홍익대학교", "홍대", "홍익대", "홍익"
- "서울예술대학교", "서울예대", "서예대"
- "한국예술종합학교", "한예종"

하드코딩 매핑으로는 모든 변형을 커버할 수 없으므로,
정규식 패턴으로 대학명을 추출하고 정규화합니다.
"""

import re
from typing import Dict, List, Tuple

import structlog

logger = structlog.get_logger("crawler.normalizer")


# ============================================================
# 대학 티어 분류 (정규화된 축약형 기준)
# S, A, B에 포함되지 않는 대학은 자동으로 C 티어
# ============================================================
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
        "한예종",
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
        "한양에리카",
        "세종대",
        "광운대",
        "명지대",
        "서울여대",
        "덕성여대",
        "동덕여대",
    ],
}


# ============================================================
# 알려진 축약어 매핑 (정규식으로 처리할 수 없는 특수 축약어)
# 제목에서 이 축약어를 발견하면 바로 정규화된 형태로 변환
# ============================================================
KNOWN_ABBREVIATIONS: Dict[str, str] = {
    # 1~2글자 축약형
    "홍대": "홍익대",
    "국대": "국민대",
    "이대": "이화여대",
    "중대": "중앙대",
    "한대": "한양대",
    "건대": "건국대",
    "경대": "경희대",
    "동대": "동국대",
    "숙대": "숙명여대",
    "성대": "성신여대",
    "서예대": "서울예대",
    "상대": "상명대",
    "서울과기": "서울과기대",
    # 특수 명칭
    "한예종": "한예종",
    "한국예술종합학교": "한예종",
    # 캠퍼스 구분
    "한양에리카": "한양에리카",
}


# ============================================================
# 정규식 정규화 규칙 (순서 중요 - 구체적인 것부터)
# 입력 대학명을 표준 축약형으로 변환
# ============================================================
_NORMALIZATION_RULES: List[Tuple[re.Pattern, str]] = [
    # "XX예술대학교" -> "XX예대" (예: 서울예술대학교 -> 서울예대)
    (re.compile(r"^(.+?)예술대학교$"), r"\1예대"),
    # "XX과학기술대학교" -> "XX과기대" (예: 서울과학기술대학교 -> 서울과기대)
    (re.compile(r"^(.+?)과학기술대학교$"), r"\1과기대"),
    # "XX여자대학교" -> "XX여대" (예: 이화여자대학교 -> 이화여대)
    (re.compile(r"^(.+?)여자대학교$"), r"\1여대"),
    # "XX대학교" -> "XX대" (예: 홍익대학교 -> 홍익대)
    (re.compile(r"^(.+?)대학교$"), r"\1대"),
]


# ============================================================
# 제목에서 대학명을 추출하기 위한 정규식 패턴 (순서 중요)
# 가장 구체적인 패턴부터 매칭하여 부분매칭 방지
# ============================================================
_EXTRACTION_PATTERNS: List[re.Pattern] = [
    # 1. 특수 명칭: 한국예술종합학교
    re.compile(r"(한국예술종합학교)"),
    # 2. 정식 명칭 (접미사 포함): XX예술대학교, XX과학기술대학교, XX여자대학교
    re.compile(r"([가-힣]{2,10}(?:예술|과학기술|여자)대학교)"),
    # 3. 정식 명칭: XX대학교
    re.compile(r"([가-힣]{2,10}대학교)"),
    # 4. 축약형 (접미사 포함): XX예대, XX여대, XX과기대
    re.compile(r"([가-힣]{2,6}(?:예대|여대|과기대))"),
    # 5. 축약형: XX대 (뒤에 '학'이 오면 제외하여 "XX대학교" 부분매칭 방지)
    re.compile(r"([가-힣]{2,6}대)(?!학)"),
]


# ============================================================
# 캠퍼스 구분 패턴 (대학명 추출 전 전처리)
# "한양대(ERICA)", "한양대 에리카" 등을 한양에리카로 인식
# ============================================================
_CAMPUS_PATTERN: re.Pattern = re.compile(
    r"(한양대(?:학교)?)\s*[\(\[]*\s*(?:에리카|ERICA|erica)\s*[\)\]]*"
)


# ============================================================
# 학과명 정규화 매핑 (학과 수가 제한적이므로 딕셔너리 방식 유지)
# ============================================================
DEPARTMENT_NORMALIZE: Dict[str, str] = {
    # 시각디자인 계열
    "시각디자인": "visual_design",
    "시각디자인과": "visual_design",
    "시각디자인학과": "visual_design",
    "시디": "visual_design",
    "시각": "visual_design",
    "커뮤니케이션디자인": "visual_design",
    "커뮤니케이션디자인과": "visual_design",
    "커뮤니케이션디자인학과": "visual_design",
    "영상디자인": "visual_design",
    "영상디자인과": "visual_design",
    # 산업디자인 계열
    "산업디자인": "industrial_design",
    "산업디자인과": "industrial_design",
    "산업디자인학과": "industrial_design",
    "산디": "industrial_design",
    "제품디자인": "industrial_design",
    "제품디자인과": "industrial_design",
    "제품디자인학과": "industrial_design",
    # 공예 계열
    "공예": "craft",
    "공예과": "craft",
    "공예학과": "craft",
    "공예디자인": "craft",
    "공예디자인과": "craft",
    "금속공예": "craft",
    "금속공예과": "craft",
    "도자공예": "craft",
    "도자공예과": "craft",
    "섬유공예": "craft",
    "섬유공예과": "craft",
    "섬유미술": "craft",
    "섬유미술과": "craft",
    "도예": "craft",
    "도예과": "craft",
    # 회화/순수미술 계열
    "회화": "fine_art",
    "회화과": "fine_art",
    "회화학과": "fine_art",
    "서양화": "fine_art",
    "서양화과": "fine_art",
    "서양화학과": "fine_art",
    "동양화": "fine_art",
    "동양화과": "fine_art",
    "동양화학과": "fine_art",
    "한국화": "fine_art",
    "한국화과": "fine_art",
    "한국화학과": "fine_art",
    "조소": "fine_art",
    "조소과": "fine_art",
    "조소학과": "fine_art",
    "판화": "fine_art",
    "판화과": "fine_art",
    "판화학과": "fine_art",
    "순수미술": "fine_art",
    "순수미술과": "fine_art",
    "미술학과": "fine_art",
}

# 학과 추출 정규식 패턴
_DEPARTMENT_PATTERNS: List[re.Pattern] = [
    # "XX디자인학과", "XX디자인과" (디자인 계열 우선)
    re.compile(r"([가-힣]{2,10}디자인(?:학과|과))"),
    # "XX공예", "XX공예과" (공예 계열)
    re.compile(r"([가-힣]{2,8}공예(?:학과|과)?)"),
    # "XX학과", "XX과" (일반)
    re.compile(r"([가-힣]{2,8}(?:학과|과))"),
]


# ============================================================
# 공개 함수
# ============================================================


def normalize_university(raw: str) -> str:
    """
    대학명을 표준 축약형으로 정규화합니다.

    정규화 순서:
    1. 알려진 축약어 매핑 확인 (홍대 -> 홍익대)
    2. 정규식 규칙 적용 (홍익대학교 -> 홍익대)
    3. 규칙 미매칭 시 원본 반환

    Args:
        raw: 정규화할 대학명 원본 텍스트

    Returns:
        str: 정규화된 대학명 축약형

    Examples:
        >>> normalize_university("홍익대학교")
        '홍익대'
        >>> normalize_university("서울예술대학교")
        '서울예대'
        >>> normalize_university("이화여자대학교")
        '이화여대'
        >>> normalize_university("서울과학기술대학교")
        '서울과기대'
        >>> normalize_university("홍대")
        '홍익대'
        >>> normalize_university("홍익대")
        '홍익대'
    """
    if not raw:
        return ""

    raw = raw.strip()

    # 1단계: 알려진 축약어 확인
    if raw in KNOWN_ABBREVIATIONS:
        return KNOWN_ABBREVIATIONS[raw]

    # 2단계: 정규식 규칙 순차 적용 (구체적인 것부터)
    for pattern, replacement in _NORMALIZATION_RULES:
        result = pattern.sub(replacement, raw)
        if result != raw:
            logger.debug(
                "대학명 정규화 적용",
                raw=raw,
                normalized=result,
            )
            return result

    # 3단계: 규칙 미매칭 - 원본 반환 (이미 축약형이거나 알 수 없는 형태)
    return raw


def extract_university_from_title(title: str) -> Tuple[str, str]:
    """
    게시글 제목에서 대학명을 추출하고 정규화합니다.

    추출 우선순위 (구체적인 패턴부터):
    1. 캠퍼스 구분 특수 패턴 (한양대 에리카 -> 한양에리카)
    2. 알려진 축약어 (홍대, 국대, 한예종 등)
    3. 정규식 패턴 (XX대학교, XX대 등)

    Args:
        title: 게시글 제목 텍스트

    Returns:
        Tuple[str, str]: (원본 대학명, 정규화된 대학명)
                         추출 실패 시 ("", "")

    Examples:
        >>> extract_university_from_title("2024 홍익대학교 시각디자인과 합격")
        ('홍익대학교', '홍익대')
        >>> extract_university_from_title("2024 홍대 시디 합격")
        ('홍대', '홍익대')
        >>> extract_university_from_title("한양대(ERICA) 산디 합격")
        ('한양대(ERICA)', '한양에리카')
        >>> extract_university_from_title("2024 성균관대학교 미술학과")
        ('성균관대학교', '성균관대')
    """
    if not title:
        return ("", "")

    # 1단계: 캠퍼스 구분 특수 패턴 확인 (한양대 에리카 등)
    campus_match = _CAMPUS_PATTERN.search(title)
    if campus_match:
        return (campus_match.group(0).strip(), "한양에리카")

    # 2단계: 알려진 축약어 확인 (긴 것부터 검색하여 부분매칭 방지)
    sorted_abbrevs = sorted(KNOWN_ABBREVIATIONS.keys(), key=len, reverse=True)
    for abbrev in sorted_abbrevs:
        if abbrev in title:
            return (abbrev, KNOWN_ABBREVIATIONS[abbrev])

    # 3단계: 정규식 패턴으로 추출 (구체적인 것부터)
    for pattern in _EXTRACTION_PATTERNS:
        match = pattern.search(title)
        if match:
            raw_name = match.group(1)
            normalized = normalize_university(raw_name)
            return (raw_name, normalized)

    return ("", "")


def determine_tier(university: str) -> str:
    """
    정규화된 대학명으로 티어를 결정합니다.

    S, A, B 티어에 포함되지 않는 대학은 모두 C 티어입니다.

    Args:
        university: 정규화된 대학명

    Returns:
        str: 티어 등급 ("S", "A", "B", "C")

    Examples:
        >>> determine_tier("홍익대")
        'S'
        >>> determine_tier("국민대")
        'A'
        >>> determine_tier("가천대")
        'B'
        >>> determine_tier("한밭대")
        'C'
        >>> determine_tier("")
        'C'
    """
    if not university:
        return "C"

    for tier, universities in UNIVERSITY_TIER.items():
        if university in universities:
            return tier

    return "C"


def normalize_department(raw: str) -> str:
    """
    학과명을 표준 코드로 정규화합니다.

    학과 수가 제한적이므로 딕셔너리 매핑 방식을 사용합니다.
    매핑되지 않는 학과명은 원본 그대로 반환합니다.

    Args:
        raw: 정규화할 학과명 원본 텍스트

    Returns:
        str: 정규화된 학과 코드 (visual_design, industrial_design, craft,
             fine_art 중 하나) 또는 매핑 실패 시 원본 텍스트

    Examples:
        >>> normalize_department("시각디자인과")
        'visual_design'
        >>> normalize_department("산업디자인학과")
        'industrial_design'
        >>> normalize_department("공예")
        'craft'
        >>> normalize_department("회화과")
        'fine_art'
    """
    if not raw:
        return ""

    raw = raw.strip()
    return DEPARTMENT_NORMALIZE.get(raw, raw)


def extract_department_from_title(title: str) -> Tuple[str, str]:
    """
    게시글 제목에서 학과명을 추출하고 정규화합니다.

    추출 우선순위:
    1. 딕셔너리에 등록된 학과 키워드 매칭 (긴 것부터)
    2. 정규식 패턴으로 추출

    Args:
        title: 게시글 제목 텍스트

    Returns:
        Tuple[str, str]: (원본 학과명, 정규화된 학과 코드)
                         추출 실패 시 ("", "")

    Examples:
        >>> extract_department_from_title("홍익대 시각디자인과 합격")
        ('시각디자인과', 'visual_design')
        >>> extract_department_from_title("국민대 공예 합격")
        ('공예', 'craft')
        >>> extract_department_from_title("이화여대 서양화과 합격")
        ('서양화과', 'fine_art')
    """
    if not title:
        return ("", "")

    # 1단계: 딕셔너리 키워드 매칭 (긴 것부터 - 부분매칭 방지)
    sorted_depts = sorted(DEPARTMENT_NORMALIZE.keys(), key=len, reverse=True)
    for dept_name in sorted_depts:
        if dept_name in title:
            return (dept_name, DEPARTMENT_NORMALIZE[dept_name])

    # 2단계: 정규식 패턴으로 추출
    for pattern in _DEPARTMENT_PATTERNS:
        match = pattern.search(title)
        if match:
            raw_dept = match.group(1)
            normalized = DEPARTMENT_NORMALIZE.get(raw_dept, raw_dept)
            return (raw_dept, normalized)

    return ("", "")
