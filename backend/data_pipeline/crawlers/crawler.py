# -*- coding: utf-8 -*-
"""
메인 크롤러 오케스트레이터

midaeipsi.com에서 미대 입시 합격작 이미지와 메타데이터를 수집하는
크롤러의 전체 실행 흐름을 관리합니다.

사용법:
    python -m data_pipeline.crawlers.crawler --start 194 --end 3940
    python -m data_pipeline.crawlers.crawler --start 3900 --end 3940 --delay 2.0
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests
import structlog

from data_pipeline.crawlers.config import (
    METADATA_DIR,
    POST_RANGE,
    PROGRESS_LOG_INTERVAL,
    RAW_IMAGES_DIR,
    REQUEST_DELAY,
    REQUEST_HEADERS,
    REQUEST_TIMEOUT,
)
from data_pipeline.crawlers.parser import ParsedPost, PostParser
from data_pipeline.crawlers.utils import (
    build_post_url,
    download_image,
    extract_extension_from_url,
    get_image_save_path,
    get_metadata_path,
    is_already_crawled,
    setup_logger,
)


@dataclass
class CrawlStats:
    """
    크롤링 통계 데이터

    전체 크롤링 진행 상황과 결과를 추적합니다.
    """

    total: int = 0
    success: int = 0
    skipped: int = 0
    failed: int = 0
    images_downloaded: int = 0

    def summary(self) -> str:
        """
        통계 요약 문자열을 반환합니다.

        Returns:
            str: 통계 요약
        """
        return (
            f"전체: {self.total}, "
            f"성공: {self.success}, "
            f"스킵: {self.skipped}, "
            f"실패: {self.failed}, "
            f"이미지: {self.images_downloaded}"
        )


class MidaeipsiCrawler:
    """
    midaeipsi.com 크롤러

    게시글 범위를 순회하며 이미지와 메타데이터를 수집합니다.
    이미 크롤링된 게시글은 자동으로 건너뛰며, 오류 발생 시 로그를 남기고
    다음 게시글로 계속 진행합니다.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        delay: float = REQUEST_DELAY,
    ):
        """
        크롤러 초기화

        Args:
            output_dir: 데이터 저장 기본 디렉토리 (기본값: config.DATA_DIR 기준)
            delay: 요청 간 대기 시간 (초, 기본값: config.REQUEST_DELAY)
        """
        self.logger = setup_logger("crawler.main")
        self.parser = PostParser()
        self.delay = delay

        # 출력 디렉토리 설정
        if output_dir:
            base_dir = Path(output_dir)
            self.images_dir = base_dir / "crawled" / "raw_images"
            self.metadata_dir = base_dir / "crawled" / "metadata"
        else:
            self.images_dir = Path(RAW_IMAGES_DIR)
            self.metadata_dir = Path(METADATA_DIR)

        # 디렉토리 생성
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # HTTP 세션 설정 (커넥션 풀 재사용)
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)

        self.logger.info(
            "크롤러 초기화 완료",
            images_dir=str(self.images_dir),
            metadata_dir=str(self.metadata_dir),
            delay=self.delay,
        )

    def crawl(self, start: int, end: int) -> CrawlStats:
        """
        지정 범위의 게시글을 크롤링합니다.

        Args:
            start: 시작 게시글 번호
            end: 종료 게시글 번호 (포함)

        Returns:
            CrawlStats: 크롤링 결과 통계
        """
        stats = CrawlStats()
        stats.total = end - start + 1

        self.logger.info(
            "크롤링 시작",
            start=start,
            end=end,
            total=stats.total,
        )

        for post_no in range(start, end + 1):
            # 진행 상황 로그 출력
            current = post_no - start + 1
            if current % PROGRESS_LOG_INTERVAL == 0 or current == 1:
                self.logger.info(
                    "크롤링 진행 중",
                    current=current,
                    total=stats.total,
                    progress=f"{current}/{stats.total} ({100 * current / stats.total:.1f}%)",
                    stats=stats.summary(),
                )

            # 이미 크롤링된 게시글 건너뛰기
            if is_already_crawled(self.metadata_dir, post_no):
                stats.skipped += 1
                self.logger.debug("이미 크롤링됨, 건너뜀", post_no=post_no)
                continue

            # 개별 게시글 크롤링
            try:
                images_count = self._crawl_single_post(post_no)
                if images_count >= 0:
                    stats.success += 1
                    stats.images_downloaded += images_count
                else:
                    stats.failed += 1
            except Exception as e:
                stats.failed += 1
                self.logger.error(
                    "게시글 크롤링 중 예외 발생",
                    post_no=post_no,
                    error=str(e),
                    exc_info=True,
                )

            # 요청 간 대기
            time.sleep(self.delay)

        self.logger.info(
            "크롤링 완료",
            stats=stats.summary(),
        )

        return stats

    def _crawl_single_post(self, post_no: int) -> int:
        """
        단일 게시글을 크롤링합니다.

        HTML을 가져와 파싱하고, 이미지를 다운로드하며, 메타데이터를 저장합니다.

        Args:
            post_no: 게시글 번호

        Returns:
            int: 다운로드된 이미지 수 (실패 시 -1)
        """
        url = build_post_url(post_no)

        # 1. HTML 페이지 가져오기
        html = self._fetch_page(url)
        if html is None:
            return -1

        # 2. HTML 파싱
        parsed = self.parser.parse(html, post_no, url)

        if not parsed.image_urls:
            self.logger.debug(
                "이미지가 없는 게시글",
                post_no=post_no,
                title=parsed.title,
            )

        # 3. 이미지 다운로드
        saved_paths = self._download_images(parsed)

        # 4. 메타데이터 저장
        self._save_metadata(parsed, saved_paths)

        self.logger.debug(
            "게시글 크롤링 완료",
            post_no=post_no,
            title=parsed.title[:50] if parsed.title else "",
            images=len(saved_paths),
            university=parsed.university,
            tier=parsed.tier,
        )

        return len(saved_paths)

    def _fetch_page(self, url: str) -> Optional[str]:
        """
        URL에서 HTML 페이지를 가져옵니다.

        EUC-KR 인코딩을 명시적으로 설정하여 한글이 깨지지 않도록 합니다.

        Args:
            url: 요청할 페이지 URL

        Returns:
            Optional[str]: HTML 문자열 (실패 시 None)
        """
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)

            # EUC-KR 인코딩 설정 (중요: 한글 깨짐 방지)
            response.encoding = "euc-kr"

            if response.status_code == 404:
                self.logger.debug("페이지 없음 (404)", url=url)
                return None

            response.raise_for_status()
            return response.text

        except requests.exceptions.Timeout:
            self.logger.warning("요청 타임아웃", url=url)
            return None
        except requests.exceptions.ConnectionError:
            self.logger.warning("연결 오류", url=url)
            return None
        except requests.exceptions.HTTPError as e:
            self.logger.warning(
                "HTTP 오류",
                url=url,
                status_code=e.response.status_code if e.response else "N/A",
            )
            return None
        except requests.exceptions.RequestException as e:
            self.logger.warning("요청 실패", url=url, error=str(e))
            return None

    def _download_images(self, parsed: ParsedPost) -> List[str]:
        """
        파싱된 게시글의 이미지를 다운로드합니다.

        Args:
            parsed: 파싱된 게시글 데이터

        Returns:
            List[str]: 저장된 이미지의 상대 경로 목록
        """
        saved_paths: List[str] = []

        for index, image_url in enumerate(parsed.image_urls):
            # 확장자 추출
            ext = extract_extension_from_url(image_url)

            # 저장 경로 생성
            save_path = get_image_save_path(
                self.images_dir,
                parsed.post_no,
                index,
                ext,
            )

            # 이미지 다운로드
            success = download_image(image_url, save_path, session=self.session)

            if success:
                # 상대 경로로 저장 (메타데이터에 기록용)
                relative_path = str(save_path).replace("\\", "/")
                saved_paths.append(relative_path)
            else:
                self.logger.warning(
                    "이미지 다운로드 실패",
                    post_no=parsed.post_no,
                    index=index,
                    url=image_url,
                )

        return saved_paths

    def _save_metadata(self, parsed: ParsedPost, saved_image_paths: List[str]) -> None:
        """
        게시글 메타데이터를 JSON 파일로 저장합니다.

        Args:
            parsed: 파싱된 게시글 데이터
            saved_image_paths: 저장된 이미지 파일 경로 목록
        """
        metadata = parsed.to_metadata(saved_image_paths)
        metadata_path = get_metadata_path(self.metadata_dir, parsed.post_no)

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            self.logger.debug(
                "메타데이터 저장 완료",
                post_no=parsed.post_no,
                path=str(metadata_path),
            )
        except OSError as e:
            self.logger.error(
                "메타데이터 저장 실패",
                post_no=parsed.post_no,
                path=str(metadata_path),
                error=str(e),
            )

    def close(self) -> None:
        """
        크롤러 리소스를 정리합니다.

        HTTP 세션을 종료합니다.
        """
        self.session.close()
        self.logger.info("크롤러 세션 종료")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    CLI 인자를 파싱합니다.

    Args:
        args: 커맨드라인 인자 목록 (테스트 시 주입용, 기본값: sys.argv)

    Returns:
        argparse.Namespace: 파싱된 인자
    """
    parser = argparse.ArgumentParser(
        description="midaeipsi.com 미대 입시 합격작 크롤러",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 범위 크롤링
  python -m data_pipeline.crawlers.crawler

  # 특정 범위 크롤링
  python -m data_pipeline.crawlers.crawler --start 3900 --end 3940

  # 대기 시간 변경
  python -m data_pipeline.crawlers.crawler --start 3900 --end 3940 --delay 2.0

  # 출력 디렉토리 변경
  python -m data_pipeline.crawlers.crawler --output-dir ./my_data
        """,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=POST_RANGE[0],
        help=f"시작 게시글 번호 (기본값: {POST_RANGE[0]})",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=POST_RANGE[1],
        help=f"종료 게시글 번호 (기본값: {POST_RANGE[1]})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"요청 간 대기 시간 - 초 단위 (기본값: {REQUEST_DELAY})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="데이터 저장 기본 디렉토리 (기본값: config 설정 사용)",
    )

    return parser.parse_args(args)


def main() -> None:
    """
    크롤러 메인 진입점

    CLI 인자를 파싱하고 크롤러를 실행합니다.
    """
    args = parse_args()

    # 입력값 유효성 검증
    if args.start > args.end:
        print(f"오류: start({args.start})가 end({args.end})보다 클 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    if args.delay < 0:
        print("오류: delay는 0 이상이어야 합니다.", file=sys.stderr)
        sys.exit(1)

    crawler = MidaeipsiCrawler(
        output_dir=args.output_dir,
        delay=args.delay,
    )

    try:
        stats = crawler.crawl(args.start, args.end)
        print(f"\n크롤링 완료: {stats.summary()}")
    except KeyboardInterrupt:
        print("\n사용자에 의해 크롤링이 중단되었습니다.")
    finally:
        crawler.close()


if __name__ == "__main__":
    main()
