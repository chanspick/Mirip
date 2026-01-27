# -*- coding: utf-8 -*-
"""
HTML 파서 모듈

midaeipsi.com 게시글 HTML을 파싱하여 제목, 이미지, 인터뷰 텍스트,
학원명 등의 메타데이터를 추출합니다.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import structlog
from bs4 import BeautifulSoup, Tag

from data_pipeline.crawlers.config import (
    DEPARTMENT_NORMALIZE,
    EXCLUDE_IMAGE_PATTERNS,
    IMAGE_EXTENSIONS,
    UNIVERSITY_NORMALIZE,
    UNIVERSITY_TIER,
)
from data_pipeline.crawlers.utils import resolve_image_url

logger = structlog.get_logger("crawler.parser")


@dataclass
class ParsedPost:
    """
    파싱된 게시글 데이터 구조

    HTML에서 추출한 모든 메타데이터와 이미지 URL을 담는 데이터 클래스입니다.
    """

    post_no: int
    title: str = ""
    year: str = ""
    admission_type: str = ""
    university: str = ""
    university_raw: str = ""
    department: str = ""
    department_raw: str = ""
    tier: str = "C"
    academy: str = ""
    work_type: str = "unknown"
    interview_raw: str = ""
    image_urls: List[str] = field(default_factory=list)
    source_url: str = ""
    crawled_at: str = ""

    def to_metadata(self, saved_image_paths: List[str]) -> Dict:
        """
        메타데이터 딕셔너리로 변환합니다.

        Args:
            saved_image_paths: 저장된 이미지 파일 경로 목록

        Returns:
            Dict: JSON 저장용 메타데이터 딕셔너리
        """
        return {
            "post_no": self.post_no,
            "year": self.year,
            "admission_type": self.admission_type,
            "university": self.university,
            "department": self.department,
            "department_raw": self.department_raw,
            "tier": self.tier,
            "academy": self.academy,
            "work_type": self.work_type,
            "interview_raw": self.interview_raw,
            "images": saved_image_paths,
            "crawled_at": self.crawled_at or datetime.now(timezone.utc).isoformat(),
            "source_url": self.source_url,
        }


class PostParser:
    """
    게시글 HTML 파서

    midaeipsi.com의 게시글 HTML에서 제목, 이미지, 인터뷰 텍스트,
    학원명 등의 정보를 추출합니다.
    """

    # 제목에서 연도를 추출하는 정규식 패턴
    # 예: "2023학년도", "2023년도", "2023"
    YEAR_PATTERN: re.Pattern = re.compile(r"(20[0-9]{2})\s*(?:학년도|년도|년)?")

    # 입시 유형 패턴
    # 예: "정시", "수시", "편입"
    ADMISSION_TYPE_PATTERN: re.Pattern = re.compile(r"(정시|수시|편입)")

    # 학원명 추출 패턴 (다양한 형식 지원)
    ACADEMY_PATTERNS: List[re.Pattern] = [
        re.compile(r"(?:학원\s*[:：]\s*)(.+?)(?:\s|$|<)"),
        re.compile(r"(?:출신\s*학원\s*[:：]\s*)(.+?)(?:\s|$|<)"),
        re.compile(r"([가-힣]+\s*(?:미술|입시|예술)\s*학원)"),
    ]

    def parse(self, html: str, post_no: int, source_url: str) -> ParsedPost:
        """
        게시글 HTML을 파싱하여 구조화된 데이터로 변환합니다.

        Args:
            html: 게시글 HTML 문자열
            post_no: 게시글 번호
            source_url: 원본 URL

        Returns:
            ParsedPost: 파싱된 게시글 데이터
        """
        soup = BeautifulSoup(html, "html.parser")

        parsed = ParsedPost(
            post_no=post_no,
            source_url=source_url,
            crawled_at=datetime.now(timezone.utc).isoformat(),
        )

        # 1. 제목 추출 및 파싱
        parsed.title = self._extract_title(soup)
        self._parse_title_metadata(parsed)

        # 2. 본문 콘텐츠 영역 탐색
        body_element = self._find_body_element(soup)

        if body_element:
            # 3. 이미지 URL 추출
            parsed.image_urls = self._extract_images(body_element)

            # 4. 인터뷰 원문 텍스트 추출
            parsed.interview_raw = self._extract_interview_text(body_element)

            # 5. 학원명 추출
            parsed.academy = self._extract_academy(body_element, parsed.title)

            # 6. 작품 유형 분류
            full_text = parsed.title + " " + parsed.interview_raw
            parsed.work_type = classify_work_type(full_text)
        else:
            logger.warning("게시글 본문 영역을 찾을 수 없음", post_no=post_no)

        return parsed

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        게시글 제목을 추출합니다.

        여러 가지 HTML 구조를 시도하여 제목을 찾습니다.

        Args:
            soup: BeautifulSoup 파싱 결과

        Returns:
            str: 추출된 제목 (찾지 못한 경우 빈 문자열)
        """
        # 방법 1: 일반적인 게시판 제목 구조 (td.subject, div.subject 등)
        title_selectors = [
            "td.subject",
            "div.subject",
            ".board_subject",
            ".view_subject",
            "h2.title",
            "h3.title",
            ".title_area",
        ]

        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title_text = element.get_text(strip=True)
                if title_text:
                    return title_text

        # 방법 2: title 태그에서 추출
        title_tag = soup.find("title")
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            # 사이트 이름 등 불필요한 부분 제거
            if " - " in title_text:
                title_text = title_text.split(" - ")[0].strip()
            if title_text:
                return title_text

        # 방법 3: 첫 번째 큰 텍스트 블록에서 추출
        for tag_name in ["h1", "h2", "h3", "strong"]:
            element = soup.find(tag_name)
            if element:
                title_text = element.get_text(strip=True)
                if len(title_text) > 5:
                    return title_text

        return ""

    def _parse_title_metadata(self, parsed: ParsedPost) -> None:
        """
        제목에서 연도, 입시 유형, 대학교, 학과 정보를 추출합니다.

        제목 예시: "2023학년도 정시 서울예대 시각디자인과 합격"

        Args:
            parsed: 메타데이터를 채울 ParsedPost 인스턴스 (in-place 수정)
        """
        title = parsed.title

        if not title:
            return

        # 연도 추출
        year_match = self.YEAR_PATTERN.search(title)
        if year_match:
            parsed.year = year_match.group(1)

        # 입시 유형 추출
        admission_match = self.ADMISSION_TYPE_PATTERN.search(title)
        if admission_match:
            parsed.admission_type = admission_match.group(1)

        # 대학교명 추출 및 정규화
        parsed.university_raw, parsed.university = self._extract_university(title)

        # 티어 결정
        parsed.tier = self._determine_tier(parsed.university)

        # 학과명 추출 및 정규화
        parsed.department_raw, parsed.department = self._extract_department(title)

    def _extract_university(self, title: str) -> tuple:
        """
        제목에서 대학교명을 추출하고 정규화합니다.

        정규화 매핑에 등록된 모든 대학명(정규화 전/후)을 우선 검색하고,
        그 외에 일반 패턴으로 대학명을 추출합니다.

        Args:
            title: 게시글 제목

        Returns:
            tuple: (원본 대학명, 정규화된 대학명)
        """
        # 정규화 매핑의 키(원본명)에서 먼저 검색 (긴 이름부터 매칭)
        sorted_names = sorted(UNIVERSITY_NORMALIZE.keys(), key=len, reverse=True)
        for raw_name in sorted_names:
            if raw_name in title:
                return raw_name, UNIVERSITY_NORMALIZE[raw_name]

        # 정규화 매핑의 값(정규화명)에서 검색
        all_normalized = set(UNIVERSITY_NORMALIZE.values())
        sorted_normalized = sorted(all_normalized, key=len, reverse=True)
        for norm_name in sorted_normalized:
            if norm_name in title:
                return norm_name, norm_name

        # 일반 패턴: "XX대" 또는 "XX대학교" 형태
        uni_pattern = re.compile(r"([가-힣]{2,6}(?:대학교|대))")
        match = uni_pattern.search(title)
        if match:
            raw = match.group(1)
            normalized = UNIVERSITY_NORMALIZE.get(raw, raw)
            return raw, normalized

        return "", ""

    def _extract_department(self, title: str) -> tuple:
        """
        제목에서 학과명을 추출하고 정규화합니다.

        Args:
            title: 게시글 제목

        Returns:
            tuple: (원본 학과명, 정규화된 학과 코드)
        """
        # 정규화 매핑의 키에서 검색 (긴 이름부터 매칭)
        sorted_depts = sorted(DEPARTMENT_NORMALIZE.keys(), key=len, reverse=True)
        for raw_dept in sorted_depts:
            if raw_dept in title:
                return raw_dept, DEPARTMENT_NORMALIZE[raw_dept]

        # 일반 패턴: "XX과", "XX학과" 형태
        dept_pattern = re.compile(r"([가-힣]{2,8}(?:학과|과))")
        match = dept_pattern.search(title)
        if match:
            raw = match.group(1)
            normalized = DEPARTMENT_NORMALIZE.get(raw, raw)
            return raw, normalized

        return "", ""

    def _determine_tier(self, university: str) -> str:
        """
        대학교명으로 티어를 결정합니다.

        Args:
            university: 정규화된 대학교명

        Returns:
            str: 티어 등급 (S, A, B, C)
        """
        if not university:
            return "C"

        for tier, universities in UNIVERSITY_TIER.items():
            if university in universities:
                return tier

        return "C"

    def _find_body_element(self, soup: BeautifulSoup) -> Optional[Tag]:
        """
        게시글 본문 콘텐츠 영역을 찾습니다.

        게시판 구조에 맞는 다양한 셀렉터를 시도합니다.

        Args:
            soup: BeautifulSoup 파싱 결과

        Returns:
            Optional[Tag]: 본문 요소 (찾지 못한 경우 None)
        """
        body_selectors = [
            "td.board_content",
            "div.board_content",
            ".view_content",
            ".board_view_content",
            "#board_content",
            "td.content",
            "div.content",
        ]

        for selector in body_selectors:
            element = soup.select_one(selector)
            if element:
                return element

        # 폴백: 이미지가 가장 많이 포함된 td 또는 div 요소 검색
        candidates = soup.find_all(["td", "div"])
        best = None
        max_images = 0

        for candidate in candidates:
            imgs = candidate.find_all("img")
            filtered_imgs = [
                img for img in imgs if not self._is_excluded_image(img.get("src", ""))
            ]
            if len(filtered_imgs) > max_images:
                max_images = len(filtered_imgs)
                best = candidate

        if best and max_images > 0:
            return best

        return None

    def _extract_images(self, body: Tag) -> List[str]:
        """
        본문에서 이미지 URL을 추출합니다.

        광고/배너 이미지를 필터링하고, 상대 URL을 절대 URL로 변환합니다.

        Args:
            body: 본문 HTML 요소

        Returns:
            List[str]: 필터링 및 변환된 이미지 URL 목록
        """
        image_urls: List[str] = []
        img_tags = body.find_all("img")

        for img in img_tags:
            src = img.get("src", "").strip()

            if not src:
                continue

            # 광고/배너 이미지 필터링
            if self._is_excluded_image(src):
                continue

            # 지원되는 이미지 확장자 확인 (확장자가 없는 URL도 허용)
            if not self._has_valid_extension(src):
                continue

            # 절대 URL로 변환
            absolute_url = resolve_image_url(src)
            image_urls.append(absolute_url)

        # 중복 제거 (순서 유지)
        seen = set()
        unique_urls: List[str] = []
        for url in image_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls

    def _is_excluded_image(self, src: str) -> bool:
        """
        이미지 URL이 제외 패턴에 해당하는지 확인합니다.

        Args:
            src: 이미지 URL 또는 경로

        Returns:
            bool: 제외 대상이면 True
        """
        src_lower = src.lower()
        return any(pattern.lower() in src_lower for pattern in EXCLUDE_IMAGE_PATTERNS)

    def _has_valid_extension(self, src: str) -> bool:
        """
        이미지 URL의 확장자가 유효한지 확인합니다.

        확장자가 없는 URL은 서버에서 이미지를 동적으로 제공하는 것일 수 있으므로
        허용합니다.

        Args:
            src: 이미지 URL 또는 경로

        Returns:
            bool: 유효한 확장자이거나 확장자가 없으면 True
        """
        # 쿼리 파라미터 제거 후 확장자 확인
        path = src.split("?")[0].split("#")[0]
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""

        if not ext:
            # 확장자 없음 - 동적 이미지일 수 있으므로 허용
            return True

        return f".{ext}" in IMAGE_EXTENSIONS

    def _extract_interview_text(self, body: Tag) -> str:
        """
        본문에서 인터뷰 원문 텍스트를 추출합니다.

        Q&A 형식의 인터뷰 텍스트 전체를 파싱하지 않고 원문 그대로 저장합니다.
        나중에 CLIP 모델에서 주제/테마 정보를 추출할 때 사용됩니다.

        Args:
            body: 본문 HTML 요소

        Returns:
            str: 인터뷰 원문 텍스트 (전체)
        """
        # 모든 텍스트 추출 (태그 간 공백 유지)
        text_parts: List[str] = []

        for element in body.descendants:
            if isinstance(element, str):
                text = element.strip()
                if text:
                    text_parts.append(text)
            elif hasattr(element, "name") and element.name in ["br", "p", "div"]:
                # 줄바꿈 요소를 개행으로 변환
                text_parts.append("\n")

        raw_text = " ".join(text_parts)

        # 연속된 공백/개행 정리
        raw_text = re.sub(r"\n\s*\n", "\n\n", raw_text)
        raw_text = re.sub(r" {2,}", " ", raw_text)

        return raw_text.strip()

    def _extract_academy(self, body: Tag, title: str) -> str:
        """
        본문 또는 제목에서 학원명을 추출합니다.

        Args:
            body: 본문 HTML 요소
            title: 게시글 제목

        Returns:
            str: 학원명 (찾지 못한 경우 빈 문자열)
        """
        # 본문 텍스트에서 학원명 검색
        body_text = body.get_text(separator=" ", strip=True)
        full_text = title + " " + body_text

        for pattern in self.ACADEMY_PATTERNS:
            match = pattern.search(full_text)
            if match:
                academy_name = match.group(1).strip()
                # 너무 짧거나 긴 학원명 필터링
                if 2 <= len(academy_name) <= 30:
                    return academy_name

        return ""


def classify_work_type(text: str) -> str:
    """
    텍스트에서 작품 유형을 분류합니다.

    재현작, 평소작 등의 키워드를 검색하여 작품 유형을 결정합니다.

    Args:
        text: 작품 유형을 판별할 텍스트 (제목 + 본문)

    Returns:
        str: 작품 유형 ('재현작', '평소작', 'unknown')
    """
    if any(kw in text for kw in ["재현작", "재현", "합격 후"]):
        return "재현작"
    elif any(kw in text for kw in ["평소작", "연습", "학원작"]):
        return "평소작"
    return "unknown"
