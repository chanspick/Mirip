#!/usr/bin/env python3
"""
파일럿 샘플 선정 + 이미지 다운로드 스크립트

50건을 티어별 균등 샘플링하고, source_url에서 이미지를 재다운로드합니다.

사용법:
  python backend/scripts/sample_and_download.py
  python backend/scripts/sample_and_download.py --n-samples 50 --dry-run
  python backend/scripts/sample_and_download.py --n-samples 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

BACKEND_ROOT = Path(__file__).parents[1]
PROJECT_ROOT = Path(__file__).parents[2]
METADATA_DIR = BACKEND_ROOT / "data" / "crawled" / "metadata"
RAW_IMAGES_DIR = BACKEND_ROOT / "data" / "crawled" / "raw_images"
SAMPLE_OUTPUT = BACKEND_ROOT / "data" / "pilot_samples.json"

TIERS = ["S", "A", "B", "C"]

# midaeipsi.com에서 이미지 추출용
SITE_ORIGIN = "https://midaeipsi.com"
EXCLUDE_PATTERNS = ["bannermidaeipsshort", "bannermidaeipsdown", "skin_board", "img_new"]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def load_all_metadata() -> list[dict]:
    """모든 메타데이터 로드"""
    files = sorted(METADATA_DIR.glob("*.json"), key=lambda p: int(p.stem))
    data_list = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d["_path"] = str(f)
            d["_post_no"] = int(f.stem)
            data_list.append(d)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            continue
    return data_list


def stratified_sample(data_list: list[dict], n_total: int, seed: int) -> list[dict]:
    """티어별 균등 샘플링 (exam_topic 유무, work_type 다양성 고려)"""
    random.seed(seed)

    # 티어별 분류
    by_tier: dict[str, list[dict]] = {t: [] for t in TIERS}
    for d in data_list:
        tier = d.get("tier", "")
        if tier in by_tier:
            by_tier[tier].append(d)

    n_per_tier = n_total // len(TIERS)
    remainder = n_total % len(TIERS)

    samples = []
    for i, tier in enumerate(TIERS):
        pool = by_tier[tier]
        n = n_per_tier + (1 if i < remainder else 0)

        if len(pool) <= n:
            selected = pool
        else:
            # exam_topic이 있는 항목 우선 (주제해석 축 채점 가능)
            with_topic = [d for d in pool if d.get("interview_parsed", {}).get("exam_topic")]
            without_topic = [d for d in pool if not d.get("interview_parsed", {}).get("exam_topic")]

            # 비율 유지: 70% with_topic, 30% without_topic (가능한 범위에서)
            n_with = min(len(with_topic), int(n * 0.7))
            n_without = min(len(without_topic), n - n_with)
            n_with = n - n_without  # 나머지는 with_topic에서

            selected = random.sample(with_topic, min(n_with, len(with_topic)))
            if n_without > 0 and without_topic:
                selected += random.sample(without_topic, min(n_without, len(without_topic)))

            # 부족분 채우기
            if len(selected) < n:
                remaining = [d for d in pool if d not in selected]
                selected += random.sample(remaining, min(n - len(selected), len(remaining)))

        samples.extend(selected)
        print(f"  {tier}티어: {len(selected)}건 선정 (풀: {len(pool)}건)")

    return samples


def fetch_image_url_from_page(source_url: str, session: requests.Session) -> Optional[str]:
    """게시글 페이지에서 작품 이미지 URL 추출

    midaeipsi.com의 이미지는 동적 PHP를 통해 서빙됨:
    - makeimgs_interview.php: 인터뷰 작품 이미지 (워터마크 포함)
    - 0wmark_ss.php: 리사이즈된 이미지
    우선순위: makeimgs_interview.php > 0wmark_ss.php (본문 첫 번째)
    """
    try:
        resp = session.get(source_url, timeout=30)
        resp.encoding = "euc-kr"
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        SKIP_PATTERNS = [
            "bannermidaeipsshort", "bannermidaeipsdown", "skin_board",
            "img_new", "dot.gif", "new.gif", "nt.gif", "pint.png",
            "insta.png", "md-ipsi.png", "cafe.png",
        ]

        # 1순위: makeimgs_interview.php (작품 이미지)
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if "makeimgs_interview.php" in src:
                if src.startswith("./"):
                    src = f"{SITE_ORIGIN}/art/{src[2:]}"
                elif not src.startswith("http"):
                    from urllib.parse import urljoin
                    src = urljoin(SITE_ORIGIN + "/art/", src)
                return src

        # 2순위: 0wmark_ss.php (본문 내 첫 번째 작품 이미지)
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if "0wmark_ss.php" in src:
                if src.startswith("./"):
                    src = f"{SITE_ORIGIN}/art/{src[2:]}"
                elif src.startswith("/"):
                    src = f"{SITE_ORIGIN}{src}"
                elif not src.startswith("http"):
                    from urllib.parse import urljoin
                    src = urljoin(SITE_ORIGIN + "/art/", src)
                return src

        # 3순위: 일반 이미지 (필터링 후)
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if not src:
                continue
            if any(pat in src for pat in SKIP_PATTERNS):
                continue

            if src.startswith("//"):
                src = f"https:{src}"
            elif not src.startswith("http"):
                from urllib.parse import urljoin
                src = urljoin(SITE_ORIGIN + "/art/", src)

            lower_src = src.lower()
            if any(ext in lower_src for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                return src

        return None
    except Exception as e:
        print(f"    페이지 파싱 오류: {e}")
        return None


def download_image(url: str, save_path: Path, session: requests.Session) -> bool:
    """이미지 다운로드"""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        resp = session.get(url, timeout=30, stream=True)
        resp.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if save_path.stat().st_size < 1000:  # 1KB 미만은 유효하지 않음
            save_path.unlink(missing_ok=True)
            return False

        return True
    except Exception as e:
        print(f"    다운로드 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="파일럿 샘플 선정 + 이미지 다운로드")
    parser.add_argument("--n-samples", type=int, default=50, help="샘플 수 (기본값: 50)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--dry-run", action="store_true", help="샘플 선정만 하고 다운로드하지 않음")
    parser.add_argument("--delay", type=float, default=1.5, help="요청 간 대기 시간 (초)")
    args = parser.parse_args()

    print("메타데이터 로드 중...")
    data_list = load_all_metadata()
    print(f"  총 {len(data_list)}건 로드")

    print(f"\n{args.n_samples}건 샘플 선정 중...")
    samples = stratified_sample(data_list, args.n_samples, args.seed)
    print(f"  총 {len(samples)}건 선정 완료")

    # 샘플 요약
    print("\n샘플 요약:")
    for tier in TIERS:
        tier_samples = [s for s in samples if s.get("tier") == tier]
        with_topic = sum(1 for s in tier_samples if s.get("interview_parsed", {}).get("exam_topic"))
        print(f"  {tier}: {len(tier_samples)}건 (exam_topic: {with_topic}건)")

    # 샘플 정보 저장
    sample_info = []
    for s in samples:
        sample_info.append({
            "post_no": s["_post_no"],
            "tier": s.get("tier"),
            "university": s.get("university"),
            "department_raw": s.get("department_raw"),
            "work_type": s.get("work_type"),
            "source_url": s.get("source_url"),
            "has_exam_topic": bool(s.get("interview_parsed", {}).get("exam_topic")),
            "exam_topic": (s.get("interview_parsed", {}).get("exam_topic") or "")[:100],
            "tier_score": s.get("tier_refined", {}).get("tier_score"),
            "competition_ratio": s.get("interview_parsed", {}).get("competition_ratio"),
        })

    SAMPLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(sample_info, f, ensure_ascii=False, indent=2)
    print(f"\n샘플 목록 저장: {SAMPLE_OUTPUT}")

    if args.dry_run:
        print("\n[DRY RUN] 이미지 다운로드 건너뜀")
        return

    # 이미지 다운로드
    print(f"\n이미지 다운로드 시작 ({len(samples)}건)...")
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    success = 0
    failed = 0

    for i, sample in enumerate(samples):
        post_no = sample["_post_no"]
        source_url = sample.get("source_url", "")
        save_path = RAW_IMAGES_DIR / f"{post_no}_0.jpg"

        # 이미 다운로드된 이미지가 있으면 건너뛰기
        if save_path.exists() and save_path.stat().st_size > 1000:
            print(f"  [{i+1}/{len(samples)}] {post_no} - 이미 존재, 건너뜀")
            success += 1
            continue

        print(f"  [{i+1}/{len(samples)}] {post_no} ({sample.get('tier')}) - ", end="")

        if not source_url:
            print("source_url 없음, 건너뜀")
            failed += 1
            continue

        # 페이지에서 이미지 URL 추출
        image_url = fetch_image_url_from_page(source_url, session)
        if not image_url:
            print("이미지 URL 추출 실패")
            failed += 1
            time.sleep(args.delay)
            continue

        # 이미지 다운로드
        if download_image(image_url, save_path, session):
            print(f"완료 ({save_path.stat().st_size // 1024}KB)")
            success += 1
        else:
            print("다운로드 실패")
            failed += 1

        time.sleep(args.delay)

    session.close()

    print(f"\n다운로드 결과: 성공 {success}, 실패 {failed}, 총 {len(samples)}건")


if __name__ == "__main__":
    main()
