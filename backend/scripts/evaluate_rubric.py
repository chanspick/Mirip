#!/usr/bin/env python3
"""
Pseudo Rubric 평가 스크립트 (Anthropic Claude Vision API)

파일럿 샘플 이미지를 Claude Opus Vision으로 5축 채점합니다.

5축 평가:
  1. formative (조형력): 형태 정확도, 비례, 구조 파악
  2. technique (표현기법): 렌더링, 질감, 명암, 매체 활용
  3. composition (구도/구성): 화면 배치, 시선 흐름, 공간감
  4. topic_interpretation (주제해석): 출제 의도 파악, 발상의 독창성
  5. completeness (완성도): 전체적 마무리, 시간 배분 반영

사용법:
  python backend/scripts/evaluate_rubric.py
  python backend/scripts/evaluate_rubric.py --limit 5 --dry-run
  python backend/scripts/evaluate_rubric.py --model claude-sonnet-4-6

필수 환경변수:
  ANTHROPIC_API_KEY: Anthropic API 키
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# dotenv 자동 로드
try:
    from dotenv import load_dotenv
    for env_path in [
        Path(__file__).parents[1] / ".env",
        Path(__file__).parents[2] / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

BACKEND_ROOT = Path(__file__).parents[1]
PROJECT_ROOT = Path(__file__).parents[2]
RAW_IMAGES_DIR = BACKEND_ROOT / "data" / "crawled" / "raw_images"
METADATA_DIR = BACKEND_ROOT / "data" / "crawled" / "metadata"
SAMPLE_FILE = BACKEND_ROOT / "data" / "pilot_samples.json"
OUTPUT_FILE = BACKEND_ROOT / "data" / "rubric_scores.json"
REPORT_FILE = BACKEND_ROOT / "data" / "rubric_pilot_report.json"

# 5축 루브릭 정의
RUBRIC_AXES = [
    "formative",          # 조형력
    "technique",          # 표현기법
    "composition",        # 구도/구성
    "topic_interpretation",  # 주제해석
    "completeness",       # 완성도
]

SYSTEM_PROMPT = """당신은 한국 미대입시 실기 전문 평가자입니다.
서울대, 국민대, 홍익대 등 최상위 미대부터 지방 대학까지 폭넓은 입시 실기 평가 경험이 있는
20년 경력의 전문가로서, 작품의 본질적 실력을 정확히 변별해야 합니다.

반드시 JSON 형식으로만 응답하세요. 각 축은 0~100 정수입니다.

## ⚠️ 평가 시 반드시 주의할 편향

다음 편향에 빠지지 않도록 각별히 주의하세요:

1. **매체 편향 금지**: 마커/과슈로 그린 기초디자인이 연필 소묘나 동양화보다
   "더 잘 그린 것"이 아닙니다. 각 매체의 특성에 맞게 평가하세요.
   - 마커 렌더링의 선명한 색채와 광택은 매체 특성이지, 그 자체로 높은 기법 점수를
     의미하지 않습니다.
   - 수묵/연필의 절제된 표현도 동일한 수준의 기법 점수를 받을 수 있습니다.

2. **시각적 화려함 ≠ 실력**: 색이 선명하고 대비가 강한 작품이 자동으로 높은 점수를
   받아서는 안 됩니다. 조형적 이해, 관찰력, 구조적 사고가 더 중요합니다.

3. **점수 범위 전체 사용**: 0~100 전체 범위를 적극 활용하세요.
   - 80~100에만 점수를 몰지 마세요.
   - 평균적인 입시 작품은 50~65점대입니다.
   - 70점 이상은 상위권, 85점 이상은 최상위권으로 매우 드물어야 합니다.

## 매체별 평가 기준

### 기초디자인 (마커/과슈 기반)
- 조형력: 사물의 3차원 구조 이해, 투시/비례의 정확도
- 기법: 마커 블렌딩, 색채 조화, 하이라이트/그림자 처리의 자연스러움
- 주의: 단순히 "깔끔한 마커 렌더링"만으로 높은 점수를 주지 마세요.
  구조적 이해 없이 표면만 매끈한 작품은 기법 60점대입니다.

### 동양화 (수묵/채색)
- 조형력: 필획의 농담 조절, 형태의 구조적 파악
- 기법: 먹의 번짐/갈필 활용, 채색의 적절성, 여백의 미
- 주의: 동양화는 "비워두는 것"도 표현입니다. 여백을 미완성으로 보지 마세요.

### 소묘/드로잉
- 조형력: 해부학적/구조적 정확도, 비례감
- 기법: 톤 변화의 풍부함, 선의 정제도, 질감 표현
- 주의: 흑백이라고 해서 기법 점수를 낮추지 마세요.

### 조소/입체
- 조형력: 3차원 형태의 정확성, 비례와 균형
- 기법: 재료 다루기의 숙련도, 표면 처리

## 5축 평가 기준

### 1. formative (조형력) 0~100
대상의 형태를 정확히 파악하고 표현하는 능력. 비례, 구조, 공간 이해.
- 85~100: 최상위. 형태 정확도가 탁월하고 3차원 구조를 완벽히 이해.
  관찰력이 뛰어나고 비례가 자연스러움. (상위 5% 수준)
- 70~84: 우수. 형태가 정확하고 구조 파악 양호. 일부 비례 편차 존재.
- 55~69: 평균. 기본적 형태 파악은 되나 비례 오류 다수.
- 40~54: 미흡. 형태 왜곡이 있고 구조 이해 부족.
- 0~39: 기초 부족. 형태 인식 자체가 어려움.

### 2. technique (표현기법) 0~100
선택한 매체를 다루는 숙련도. 질감, 명암, 색채 활용 능력.
매체 종류(마커/연필/수묵 등)에 관계없이 해당 매체 내에서의 숙련도를 평가.
- 85~100: 최상위. 매체를 자유자재로 다루며, 표현의 깊이와 정교함이 돋보임.
- 70~84: 우수. 기법이 안정적이고 매체 특성을 잘 활용.
- 55~69: 평균. 기본적 기법 구사. 매체 활용이 단조로움.
- 40~54: 미흡. 기법이 미숙하고 매체 다루기가 어색.
- 0~39: 기초 부족. 기법적 이해가 부재.

### 3. composition (구도/구성) 0~100
화면 구성의 효과성. 시선 유도, 여백, 밸런스, 공간감.
- 85~100: 최상위. 독창적이면서 안정적. 시선 흐름이 자연스럽고 의도적.
- 70~84: 우수. 안정적 구도. 시선 유도가 명확.
- 55~69: 평균. 기본적 구도. 일부 불균형.
- 40~54: 미흡. 불안정한 구도. 화면 활용 비효율적.
- 0~39: 구도 의식 부재.

### 4. topic_interpretation (주제해석) 0~100
주제에 대한 해석의 깊이, 발상의 독창성, 표현 의도의 명확성.
주제가 제공되면 주제와의 연관성을, 없으면 작품 자체의 의도와 독창성을 평가.
- 85~100: 최상위. 깊고 독창적인 해석. 발상이 참신하면서 설득력 있음.
- 70~84: 우수. 주제를 잘 반영하고 적절한 발상.
- 55~69: 평균. 주제를 기본적으로 반영. 발상이 관습적.
- 40~54: 미흡. 주제 연관성 약함. 발상 부족.
- 0~39: 주제 해석 부재.

### 5. completeness (완성도) 0~100
전체적 마무리 수준. 화면 전체에 걸친 균등한 밀도와 마무리.
단, 동양화의 의도적 여백은 미완성이 아님.
- 85~100: 최상위. 화면 전체가 균등하게 완성되고 밀도 편차가 없음.
- 70~84: 우수. 전반적으로 잘 마무리. 주변부 약간의 밀도 차이.
- 55~69: 평균. 주요 부분은 완성되었으나 부분적 미흡.
- 40~54: 미흡. 미완성 영역이 눈에 띔. 마무리 부족.
- 0~39: 대부분 미완성.

## 응답 형식 (반드시 이 JSON만 출력, 다른 텍스트 금지)

```json
{
  "formative": 62,
  "technique": 58,
  "composition": 55,
  "topic_interpretation": 48,
  "completeness": 60,
  "comment": "한 줄 종합 코멘트"
}
```

위 예시처럼 50~60점대가 평균적 작품의 점수입니다. 80점 이상은 신중하게 부여하세요."""


def build_user_message(
    image_b64: str,
    media_type: str,
    metadata: dict,
) -> list[dict]:
    """사용자 메시지 구성 (이미지 + 컨텍스트)"""
    # 컨텍스트 정보 구성
    context_parts = []

    uni = metadata.get("university", "")
    dept = metadata.get("department_raw", "")
    work_type = metadata.get("work_type", "")
    year = metadata.get("year", "")

    if uni or dept:
        context_parts.append(f"대학/학과: {uni} {dept}")
    if work_type:
        context_parts.append(f"작품 유형: {work_type}")
    if year:
        context_parts.append(f"연도: {year}학년도")

    # 주제 정보 (있으면 포함)
    parsed = metadata.get("interview_parsed", {})
    exam_topic = parsed.get("exam_topic") if parsed else None
    if exam_topic and len(exam_topic.strip()) > 5:
        context_parts.append(f"출제 주제: {exam_topic[:200]}")
    else:
        context_parts.append("출제 주제: (정보 없음 - 작품 자체의 의도와 독창성으로 평가)")

    context_text = "\n".join(context_parts)

    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_b64,
            },
        },
        {
            "type": "text",
            "text": f"위 작품을 5축 루브릭으로 평가해 주세요.\n\n[작품 정보]\n{context_text}\n\n반드시 지정된 JSON 형식으로만 응답하세요.",
        },
    ]

    return content


def call_claude_vision(
    image_b64: str,
    media_type: str,
    metadata: dict,
    api_key: str,
    model: str = "claude-sonnet-4-6",
) -> Optional[dict]:
    """Claude Vision API 호출"""
    import httpx

    content = build_user_message(image_b64, media_type, metadata)

    payload = {
        "model": model,
        "max_tokens": 512,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": content},
        ],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()

        # 응답 텍스트 추출
        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        # JSON 파싱
        # ```json ... ``` 블록이 있으면 추출
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        scores = json.loads(text)

        # 유효성 검증
        for axis in RUBRIC_AXES:
            if axis not in scores:
                print(f"    경고: {axis} 축 누락")
                return None
            val = scores[axis]
            if not isinstance(val, (int, float)) or val < 0 or val > 100:
                print(f"    경고: {axis} 값 범위 오류 ({val})")
                return None

        # 토큰 사용량 기록
        usage = result.get("usage", {})
        scores["_usage"] = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

        return scores

    except httpx.HTTPStatusError as e:
        print(f"    API 오류: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except json.JSONDecodeError as e:
        print(f"    JSON 파싱 오류: {e}")
        print(f"    원문: {text[:300]}")
        return None
    except Exception as e:
        print(f"    예외: {e}")
        return None


def load_image_as_base64(image_path: Path) -> tuple[str, str]:
    """이미지를 base64로 인코딩"""
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")

    ext = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/jpeg")

    return b64, media_type


def main():
    parser = argparse.ArgumentParser(description="Pseudo Rubric 평가 (Claude Vision)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                        help="Claude 모델 (기본값: claude-sonnet-4-6)")
    parser.add_argument("--limit", type=int, default=0, help="처리할 최대 건수 (0=전체)")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 구조만 확인")
    parser.add_argument("--delay", type=float, default=2.0, help="API 호출 간 대기 시간 (초)")
    parser.add_argument("--resume", action="store_true", help="이전 결과 이어서 진행")
    args = parser.parse_args()

    # API 키 확인
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        print("오류: ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    # 샘플 목록 로드
    if not SAMPLE_FILE.exists():
        print(f"오류: {SAMPLE_FILE} 파일이 없습니다.")
        print("  먼저 sample_and_download.py를 실행하세요.")
        sys.exit(1)

    samples = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    print(f"샘플 {len(samples)}건 로드")

    if args.limit > 0:
        samples = samples[:args.limit]
        print(f"  → {len(samples)}건으로 제한")

    # 이전 결과 로드 (resume 모드)
    existing_results = {}
    if args.resume and OUTPUT_FILE.exists():
        prev = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        for r in prev:
            existing_results[r["post_no"]] = r
        print(f"  이전 결과 {len(existing_results)}건 로드")

    # 채점 실행
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    success = 0
    failed = 0
    skipped = 0

    start_time = time.time()

    for i, sample in enumerate(samples):
        post_no = sample["post_no"]

        # 이미 채점된 항목 건너뛰기 (resume)
        if post_no in existing_results:
            results.append(existing_results[post_no])
            skipped += 1
            print(f"  [{i+1}/{len(samples)}] #{post_no} - 이전 결과 사용")
            continue

        # 이미지 파일 확인
        image_path = RAW_IMAGES_DIR / f"{post_no}_0.jpg"
        if not image_path.exists():
            # png도 시도
            image_path = RAW_IMAGES_DIR / f"{post_no}_0.png"
            if not image_path.exists():
                print(f"  [{i+1}/{len(samples)}] #{post_no} - 이미지 없음, 건너뜀")
                failed += 1
                continue

        # 메타데이터 로드
        meta_path = METADATA_DIR / f"{post_no}.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            metadata = sample

        print(f"  [{i+1}/{len(samples)}] #{post_no} ({sample.get('tier')}) - ", end="", flush=True)

        if args.dry_run:
            img_size = image_path.stat().st_size // 1024
            has_topic = "주제있음" if sample.get("has_exam_topic") else "주제없음"
            print(f"[DRY RUN] 이미지 {img_size}KB, {has_topic}")
            continue

        # 이미지 로드
        image_b64, media_type = load_image_as_base64(image_path)

        # Claude Vision API 호출
        scores = call_claude_vision(image_b64, media_type, metadata, api_key, args.model)

        if scores:
            usage = scores.pop("_usage", {})
            total_input_tokens += usage.get("input_tokens", 0)
            total_output_tokens += usage.get("output_tokens", 0)

            result_entry = {
                "post_no": post_no,
                "tier": sample.get("tier"),
                "university": sample.get("university"),
                "work_type": sample.get("work_type"),
                "has_exam_topic": sample.get("has_exam_topic"),
                "tier_score": sample.get("tier_score"),
                "scores": {axis: scores[axis] for axis in RUBRIC_AXES},
                "comment": scores.get("comment", ""),
            }
            results.append(result_entry)
            success += 1

            avg = sum(scores[a] for a in RUBRIC_AXES) / len(RUBRIC_AXES)
            print(f"평균 {avg:.0f} [{scores['formative']}/{scores['technique']}/{scores['composition']}/{scores['topic_interpretation']}/{scores['completeness']}]")
        else:
            failed += 1
            print("채점 실패")

        # 중간 저장 (5건마다)
        if (i + 1) % 5 == 0 and results:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(args.delay)

    elapsed = time.time() - start_time

    # 최종 결과 저장
    if results and not args.dry_run:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {OUTPUT_FILE}")

    # 리포트 생성
    if results and not args.dry_run:
        report = generate_report(results, elapsed, total_input_tokens, total_output_tokens)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"리포트 저장: {REPORT_FILE}")
        print_report(report)

    print(f"\n완료: 성공 {success}, 실패 {failed}, 건너뜀 {skipped}, 소요 {elapsed:.1f}초")
    if total_input_tokens > 0:
        est_cost = (total_input_tokens * 15 + total_output_tokens * 75) / 1_000_000
        print(f"토큰 사용: 입력 {total_input_tokens:,}, 출력 {total_output_tokens:,}")
        print(f"예상 비용: ~${est_cost:.3f}")


def generate_report(results: list[dict], elapsed: float,
                    input_tokens: int, output_tokens: int) -> dict:
    """파일럿 결과 분석 리포트 생성"""
    import statistics

    report = {
        "summary": {
            "total": len(results),
            "elapsed_seconds": round(elapsed, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "axes_stats": {},
        "tier_correlation": {},
        "axes_discrimination": {},
    }

    # 축별 통계
    for axis in RUBRIC_AXES:
        values = [r["scores"][axis] for r in results if axis in r["scores"]]
        if values:
            report["axes_stats"][axis] = {
                "mean": round(statistics.mean(values), 1),
                "median": round(statistics.median(values), 1),
                "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
            }

    # 티어별 평균 점수 (상관 관계 확인)
    for tier in ["S", "A", "B", "C"]:
        tier_results = [r for r in results if r["tier"] == tier]
        if tier_results:
            tier_stats = {}
            for axis in RUBRIC_AXES:
                values = [r["scores"][axis] for r in tier_results if axis in r["scores"]]
                if values:
                    tier_stats[axis] = round(statistics.mean(values), 1)
            tier_stats["_count"] = len(tier_results)
            tier_stats["_overall_mean"] = round(
                statistics.mean(
                    sum(r["scores"][a] for a in RUBRIC_AXES) / len(RUBRIC_AXES)
                    for r in tier_results
                ), 1
            )
            report["tier_correlation"][tier] = tier_stats

    # 축 간 변별력 (tier S vs C 차이)
    s_results = [r for r in results if r["tier"] == "S"]
    c_results = [r for r in results if r["tier"] == "C"]
    if s_results and c_results:
        for axis in RUBRIC_AXES:
            s_mean = statistics.mean(r["scores"][axis] for r in s_results)
            c_mean = statistics.mean(r["scores"][axis] for r in c_results)
            report["axes_discrimination"][axis] = {
                "s_mean": round(s_mean, 1),
                "c_mean": round(c_mean, 1),
                "gap": round(s_mean - c_mean, 1),
            }

    return report


def print_report(report: dict) -> None:
    """리포트 콘솔 출력"""
    print("\n" + "=" * 60)
    print("PSEUDO RUBRIC PILOT REPORT")
    print("=" * 60)

    # 축별 통계
    print("\n[축별 점수 분포]")
    print(f"{'축':<25} {'평균':>6} {'중앙값':>6} {'표준편차':>6} {'범위':>12}")
    print("-" * 60)
    for axis in RUBRIC_AXES:
        s = report["axes_stats"].get(axis, {})
        if s:
            print(f"{axis:<25} {s['mean']:>6.1f} {s['median']:>6.1f} "
                  f"{s['stdev']:>6.1f} [{s['min']:>3}-{s['max']:>3}]")

    # 티어-점수 상관
    print("\n[티어별 평균 점수]")
    print(f"{'티어':<6}", end="")
    for axis in RUBRIC_AXES:
        label = axis[:8]
        print(f" {label:>10}", end="")
    print(f" {'전체평균':>10}")
    print("-" * 70)
    for tier in ["S", "A", "B", "C"]:
        tc = report["tier_correlation"].get(tier, {})
        if tc:
            print(f"{tier:<6}", end="")
            for axis in RUBRIC_AXES:
                print(f" {tc.get(axis, 0):>10.1f}", end="")
            print(f" {tc.get('_overall_mean', 0):>10.1f}")

    # S vs C 변별력
    disc = report.get("axes_discrimination", {})
    if disc:
        print("\n[S vs C 변별력]")
        print(f"{'축':<25} {'S평균':>8} {'C평균':>8} {'차이':>8}")
        print("-" * 50)
        for axis in RUBRIC_AXES:
            d = disc.get(axis, {})
            if d:
                gap = d["gap"]
                marker = "***" if gap > 20 else "**" if gap > 10 else "*" if gap > 5 else ""
                print(f"{axis:<25} {d['s_mean']:>8.1f} {d['c_mean']:>8.1f} {gap:>7.1f} {marker}")


if __name__ == "__main__":
    main()
