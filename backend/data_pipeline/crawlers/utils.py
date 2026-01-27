# -*- coding: utf-8 -*-
"""
크롤러 유틸리티 모듈

URL 생성, 파일 경로 관리, 이미지 다운로드, 로거 설정 등
크롤러에서 공통으로 사용하는 헬퍼 함수를 제공합니다.
"""

import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urljoin

import requests
import structlog

from data_pipeline.crawlers.config import (
    BASE_URL,
    BOARD_NAME,
    IMAGE_EXTENSIONS,
    MAX_RETRIES,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
    SITE_ORIGIN,
)


def setup_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    structlog 로거를 설정하고 반환합니다.

    Args:
        name: 로거 이름 (모듈 식별용)

    Returns:
        structlog.stdlib.BoundLogger: 설정된 로거 인스턴스
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger(name)


def build_post_url(post_no: int) -> str:
    """
    게시글 번호로 상세 페이지 URL을 생성합니다.

    Args:
        post_no: 게시글 번호

    Returns:
        str: 완성된 게시글 URL

    Examples:
        >>> build_post_url(3940)
        'https://midaeipsi.com/art/board.php?board=academynews&command=body&no=3940'
    """
    params = {
        "board": BOARD_NAME,
        "command": "body",
        "no": post_no,
    }
    return f"{BASE_URL}?{urlencode(params)}"


def build_list_url(page: int = 1) -> str:
    """
    게시판 목록 페이지 URL을 생성합니다.

    Args:
        page: 페이지 번호 (기본값: 1)

    Returns:
        str: 게시판 목록 URL
    """
    params = {
        "board": BOARD_NAME,
    }
    if page > 1:
        params["page"] = page  # type: ignore[assignment]
    return f"{BASE_URL}?{urlencode(params)}"


def resolve_image_url(src: str) -> str:
    """
    이미지 src 속성을 절대 URL로 변환합니다.

    상대 경로인 경우 사이트 원본 URL을 앞에 붙여 절대 URL로 만듭니다.

    Args:
        src: img 태그의 src 속성 값

    Returns:
        str: 절대 URL로 변환된 이미지 경로
    """
    src = src.strip()

    # 이미 절대 URL인 경우
    if src.startswith("http://") or src.startswith("https://"):
        return src

    # 프로토콜 상대 URL인 경우
    if src.startswith("//"):
        return f"https:{src}"

    # 상대 경로인 경우 사이트 원본 URL 기준으로 합성
    return urljoin(SITE_ORIGIN + "/", src.lstrip("/"))


def get_image_save_path(output_dir: Path, post_no: int, index: int, ext: str = ".jpg") -> Path:
    """
    이미지 저장 경로를 생성합니다.

    Args:
        output_dir: 이미지 저장 기본 디렉토리
        post_no: 게시글 번호
        index: 이미지 인덱스 (0부터 시작)
        ext: 파일 확장자 (기본값: .jpg)

    Returns:
        Path: 이미지 저장 경로

    Examples:
        >>> get_image_save_path(Path("data/raw"), 3940, 0)
        PosixPath('data/raw/3940_0.jpg')
    """
    # 확장자 정규화
    ext = ext.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
    if not ext.startswith("."):
        ext = f".{ext}"

    return output_dir / f"{post_no}_{index}{ext}"


def get_metadata_path(metadata_dir: Path, post_no: int) -> Path:
    """
    메타데이터 JSON 파일 경로를 생성합니다.

    Args:
        metadata_dir: 메타데이터 저장 디렉토리
        post_no: 게시글 번호

    Returns:
        Path: 메타데이터 JSON 파일 경로
    """
    return metadata_dir / f"{post_no}.json"


def extract_extension_from_url(url: str) -> str:
    """
    URL에서 이미지 파일 확장자를 추출합니다.

    쿼리 파라미터를 제거한 후 경로에서 확장자를 추출합니다.
    유효하지 않은 확장자인 경우 기본값(.jpg)을 반환합니다.

    Args:
        url: 이미지 URL

    Returns:
        str: 파일 확장자 (점 포함, 예: '.jpg')
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    path = parsed.path

    # 경로에서 확장자 추출
    ext = Path(path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return ext

    return ".jpg"


def download_image(
    url: str,
    save_path: Path,
    session: Optional[requests.Session] = None,
) -> bool:
    """
    URL에서 이미지를 다운로드하여 지정 경로에 저장합니다.

    최대 MAX_RETRIES 횟수까지 재시도하며, 실패 시 False를 반환합니다.

    Args:
        url: 이미지 다운로드 URL
        save_path: 이미지 저장 경로
        session: HTTP 세션 (기존 세션 재사용 시 전달)

    Returns:
        bool: 다운로드 성공 여부
    """
    logger = structlog.get_logger("crawler.utils")

    # 저장 디렉토리 생성
    save_path.parent.mkdir(parents=True, exist_ok=True)

    requester = session or requests
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requester.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
                stream=True,
            )
            response.raise_for_status()

            # 콘텐츠 타입 확인
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/") and "octet-stream" not in content_type:
                logger.warning(
                    "이미지가 아닌 콘텐츠 타입",
                    url=url,
                    content_type=content_type,
                )

            # 파일 저장 (스트리밍 방식)
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # 파일 크기 확인 (빈 파일 체크)
            if save_path.stat().st_size == 0:
                save_path.unlink(missing_ok=True)
                logger.warning("빈 이미지 파일 삭제", url=url, path=str(save_path))
                return False

            logger.debug(
                "이미지 다운로드 완료",
                url=url,
                path=str(save_path),
                size=save_path.stat().st_size,
            )
            return True

        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning(
                "이미지 다운로드 실패 (재시도 예정)",
                url=url,
                attempt=attempt,
                max_retries=MAX_RETRIES,
                error=str(e),
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    logger.error(
        "이미지 다운로드 최종 실패",
        url=url,
        error=str(last_error),
    )
    return False


def is_already_crawled(metadata_dir: Path, post_no: int) -> bool:
    """
    해당 게시글이 이미 크롤링되었는지 확인합니다.

    메타데이터 JSON 파일이 존재하면 이미 크롤링된 것으로 판단합니다.

    Args:
        metadata_dir: 메타데이터 저장 디렉토리
        post_no: 게시글 번호

    Returns:
        bool: 이미 크롤링된 경우 True
    """
    metadata_path = get_metadata_path(metadata_dir, post_no)
    return metadata_path.exists()
