#!/usr/bin/env python3
"""
인터뷰 로컬 파싱 스크립트 (LLM 불필요)

midaeipsi.com 인터뷰의 일관된 Q&A 형식을 활용하여
regex/heuristic으로 구조화된 필드를 추출합니다.

사용법:
  python backend/scripts/parse_interviews_local.py
  python backend/scripts/parse_interviews_local.py --limit 10
  python backend/scripts/parse_interviews_local.py --no-resume  # 재파싱
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

BACKEND_ROOT = Path(__file__).parents[1]
METADATA_DIR = BACKEND_ROOT / "data" / "crawled" / "metadata"


def extract_practice_years(text: str) -> Optional[float]:
    """실기경력 추출: "실기경력: 2년 4개월" → 2.33"""
    patterns = [
        r'실기경력\s*[:：]?\s*(\d+)\s*년\s*(\d+)\s*개월',
        r'실기경력\s*[:：]?\s*(\d+)\s*년',
        r'실기\s*경력\s*[:：]?\s*(\d+)\s*년\s*(\d+)\s*개월',
        r'실기\s*경력\s*[:：]?\s*(\d+)\s*년',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            years = int(m.group(1))
            months = int(m.group(2)) if m.lastindex >= 2 else 0
            return round(years + months / 12, 2)
    return None


def extract_competition_ratio(text: str) -> Optional[float]:
    """경쟁률 추출: "경쟁률 81.63:1" → 81.63"""
    patterns = [
        r'경쟁률\s*[:：]?\s*(\d+\.?\d*)\s*[:：]\s*1',
        r'경쟁률\s*(\d+\.?\d*)\s*[:：]\s*1',
        r'경쟁률\s*[:：]?\s*(\d+\.?\d*)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                val = float(m.group(1))
                if val > 0:
                    return round(val, 2)
            except ValueError:
                continue
    return None


def extract_qa_answer(text: str, question_pattern: str, max_length: int = 500) -> Optional[str]:
    """Q&A 형식에서 특정 질문의 답변 추출"""
    # 질문 위치 찾기
    m = re.search(question_pattern, text, re.IGNORECASE)
    if not m:
        return None

    # 질문 끝(?)까지 스킵하여 답변 시작 위치 결정
    start = m.end()
    remaining_after_match = text[start:]

    # 질문 텍스트가 아직 남아있으면 "?" 까지 스킵
    q_end = re.search(r'\?', remaining_after_match)
    if q_end and q_end.start() < 200:  # 200자 이내에 ?가 있으면 질문의 끝
        start = start + q_end.end()

    # 다음 Q. 또는 ★ 또는 특정 패턴까지가 답변
    remaining = text[start:]
    end_patterns = [
        r'\n\s*Q\.',
        r'\n\s*★',
        r'\n\s*●',
        r'\n\s*자료제공',
        r'\n\s*미대입시닷컴 합격생',
        r'\n\s*본문내용',
    ]

    end_pos = len(remaining)
    for ep in end_patterns:
        em = re.search(ep, remaining)
        if em and em.start() < end_pos:
            end_pos = em.start()

    answer = remaining[:end_pos].strip()

    # 사이트 워터마크 제거
    answer = re.sub(r'미대입시닷컴\s*midaeipsi\.com', '', answer)
    answer = re.sub(r'홍대\s+\S+\s*(프리미엄[^미]*)?(미술학원|학원)?', '', answer)
    answer = re.sub(r'\S+\s*(미술학원|학원)\s*$', '', answer, flags=re.MULTILINE)
    answer = re.sub(r'\s+', ' ', answer).strip()

    if len(answer) < 10:
        return None

    # 너무 길면 자르기
    if len(answer) > max_length:
        answer = answer[:max_length].rsplit('.', 1)[0] + '.'

    return answer if answer else None


def extract_exam_topic(text: str) -> Optional[str]:
    """출제 주제 추출"""
    # Q. 실기고사 당일 출제문제는... 패턴
    answer = extract_qa_answer(
        text,
        r'Q\.\s*실기고사\s*당일\s*출제\s*문제',
        max_length=300,
    )
    if answer:
        # 첫 1~2문장만 (주제 설명 부분)
        sentences = re.split(r'(?<=[.다요!])\s+', answer)
        topic = ' '.join(sentences[:2]).strip()
        # 불필요한 접두사 제거
        topic = re.sub(r'^(저는|저희|우리|제가)\s*', '', topic)
        return topic if len(topic) >= 5 else None

    return None


def extract_student_strength(text: str) -> Optional[str]:
    """합격 주요인 추출"""
    return extract_qa_answer(
        text,
        r'Q\.\s*합격의?\s*주요인은?\s*무엇이\s*라고',
        max_length=200,
    )


def extract_ideation_process(text: str) -> Optional[str]:
    """발상 과정 추출 (출제문제 답변에서 발상/풀이 부분)"""
    answer = extract_qa_answer(
        text,
        r'Q\.\s*실기고사\s*당일\s*출제\s*문제',
        max_length=500,
    )
    if not answer:
        return None

    # 발상/접근 관련 키워드 이후 텍스트 추출
    ideation_keywords = [
        r'(발상.{10,300})',
        r'(접근.{10,300})',
        r'(생각해서.{10,300})',
        r'(생각했습니다.{0,300})',
        r'(풀어나.{10,300})',
        r'(표현하.{10,300})',
        r'(구상.{10,300})',
    ]
    for pattern in ideation_keywords:
        m = re.search(pattern, answer)
        if m:
            result = m.group(1).strip()
            sentences = re.split(r'(?<=[.다요!])\s+', result)
            return ' '.join(sentences[:2]).strip()

    # 키워드 못 찾으면 두 번째 문장부터 (첫 문장은 주제)
    sentences = re.split(r'(?<=[.다요!])\s+', answer)
    if len(sentences) > 1:
        return ' '.join(sentences[1:3]).strip()

    return None


def extract_teacher_feedback(text: str) -> Optional[str]:
    """강사 피드백 추출 (실기준비 답변에서)"""
    answer = extract_qa_answer(
        text,
        r'Q\.\s*실기준비는?\s*어떻게\s*했나요',
        max_length=400,
    )
    if not answer:
        return None

    # 선생님/강사 관련 언급 찾기
    teacher_patterns = [
        r'(선생님\S*\s*.{10,100})',
        r'(강사\S*\s*.{10,100})',
        r'(원장님\S*\s*.{10,100})',
        r'(피드백.{10,100})',
    ]
    for pattern in teacher_patterns:
        m = re.search(pattern, answer)
        if m:
            feedback = m.group(1).strip()
            # 문장 단위로 자르기
            sentences = re.split(r'[.!?]\s+', feedback)
            return sentences[0].strip() + '.' if sentences else feedback

    return None


def parse_single_interview(interview_raw: str) -> dict:
    """단일 인터뷰 텍스트에서 구조화 필드 추출"""
    if not interview_raw or len(interview_raw.strip()) < 50:
        return {
            "exam_topic": None,
            "practice_years": None,
            "competition_ratio": None,
            "student_strength": None,
            "teacher_feedback": None,
            "ideation_process": None,
        }

    return {
        "exam_topic": extract_exam_topic(interview_raw),
        "practice_years": extract_practice_years(interview_raw),
        "competition_ratio": extract_competition_ratio(interview_raw),
        "student_strength": extract_student_strength(interview_raw),
        "teacher_feedback": extract_teacher_feedback(interview_raw),
        "ideation_process": extract_ideation_process(interview_raw),
    }


def main():
    parser = argparse.ArgumentParser(description="인터뷰 로컬 파싱 (regex/heuristic)")
    parser.add_argument("--limit", type=int, default=0, help="처리할 최대 건수 (0=전체)")
    parser.add_argument("--no-resume", action="store_true", help="이미 파싱된 건도 다시 처리")
    parser.add_argument("--overwrite-local", action="store_true",
                        help="LLM 파싱 결과를 로컬 파싱으로 덮어씀")
    args = parser.parse_args()

    resume = not args.no_resume

    files = sorted(METADATA_DIR.glob("*.json"), key=lambda p: int(p.stem))
    print(f"총 메타데이터 파일: {len(files)}건")

    stats = {
        "total": len(files),
        "processed": 0,
        "skipped_already_parsed": 0,
        "skipped_no_interview": 0,
        "skipped_parse_error": 0,
        "success": 0,
        "fields": {f: 0 for f in [
            "exam_topic", "practice_years", "competition_ratio",
            "student_strength", "teacher_feedback", "ideation_process",
        ]},
    }

    if args.limit > 0:
        files = files[:args.limit]

    start_time = time.time()

    for i, filepath in enumerate(files):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"[{i+1}/{len(files)}] 성공={stats['success']}, 속도={stats['processed']/elapsed:.0f}건/초")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats["skipped_parse_error"] += 1
            continue

        # 이미 파싱된 건 건너뜀 (LLM 결과가 있으면 보존)
        if resume and "interview_parsed" in data and not args.overwrite_local:
            stats["skipped_already_parsed"] += 1
            continue

        interview_raw = data.get("interview_raw", "")
        if not interview_raw or len(interview_raw.strip()) < 50:
            stats["skipped_no_interview"] += 1
            continue

        stats["processed"] += 1

        # 파싱
        parsed = parse_single_interview(interview_raw)

        # 저장
        data["interview_parsed"] = parsed
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        stats["success"] += 1

        # 필드별 통계
        for field in stats["fields"]:
            if parsed.get(field) is not None:
                stats["fields"][field] += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("로컬 파싱 완료 리포트")
    print("=" * 60)
    print(f"총 파일: {stats['total']}건")
    print(f"처리됨: {stats['processed']}건")
    print(f"성공: {stats['success']}건")
    print(f"건너뜀 (이미 파싱됨): {stats['skipped_already_parsed']}건")
    print(f"건너뜀 (인터뷰 없음): {stats['skipped_no_interview']}건")
    print(f"건너뜀 (파일 오류): {stats['skipped_parse_error']}건")
    print(f"소요 시간: {elapsed:.1f}초")
    print()
    print("필드별 추출률:")
    total = stats["success"] if stats["success"] > 0 else 1
    for field, count in stats["fields"].items():
        pct = count / total * 100
        print(f"  {field}: {count}/{stats['success']} ({pct:.1f}%)")

    # 리포트 저장
    report_path = BACKEND_ROOT / "data" / "crawled" / "interview_parse_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
