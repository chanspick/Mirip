#!/usr/bin/env python3
"""
인터뷰 데이터 정규식 파싱 스크립트

메타데이터 JSON의 interview_raw 텍스트에서 구조화된 필드를 추출합니다.
LLM 없이 정규식 기반으로 동작합니다.

추출 필드:
  - exam_topic: 출제 주제 원문
  - practice_years: 실기 경력 (숫자, 단위: 년)
  - competition_ratio: 경쟁률 (숫자)
  - student_strength: 합격 주요인 (학생 자평)
  - teacher_feedback: 강사 피드백
  - ideation_process: 발상 과정

사용법:
  python backend/scripts/parse_interviews.py
  python backend/scripts/parse_interviews.py --stems 579,580,581
  python backend/scripts/parse_interviews.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parents[2]
BACKEND_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

METADATA_DIR = BACKEND_ROOT / "data" / "crawled" / "metadata"

# 기본 처리 대상 stem 목록
DEFAULT_STEMS = [
    579, 580, 581, 582, 583, 584, 585, 586, 587, 588,
    589, 590, 591, 592, 593, 594, 595, 596, 597, 598,
    599, 600, 601, 604, 605, 606, 607, 608, 609, 610,
    611, 612, 613, 614, 615, 616, 617, 618, 619, 620,
    621, 622, 623, 624, 625, 626, 627, 628, 629, 630,
]


def parse_practice_years(text: str) -> float | None:
    """실기경력 파싱: '2년4개월' -> 2.3"""
    m = re.search(r'실기경력\s*[:：]\s*(\d+)\s*년\s*(\d+)\s*개?월', text)
    if m:
        years = int(m.group(1))
        months = int(m.group(2))
        return round(years + months / 12, 1)
    m = re.search(r'실기경력\s*[:：]\s*(\d+)\s*년', text)
    if m:
        return float(int(m.group(1)))
    m = re.search(r'실기경력\s*[:：]\s*(\d+)\s*개?월', text)
    if m:
        return round(int(m.group(1)) / 12, 1)
    return None


def parse_competition_ratio(text: str) -> float | None:
    """경쟁률 파싱: '9.41:1' -> 9.41"""
    m = re.search(r'경쟁률\s*[:：]?\s*([\d.]+)\s*[:：]\s*1', text)
    if m:
        return float(m.group(1))
    return None


def extract_answer(text: str, question_pattern: str) -> str | None:
    """질문 패턴 이후 답변 텍스트 추출"""
    m = re.search(question_pattern, text)
    if not m:
        return None
    start = m.end()
    # 다음 질문이나 섹션 마커까지
    end_patterns = [
        r'Q\.\s',
        r'★',
        r'미대입시닷컴 합격생 인터뷰에',
        r'본문내용',
        r'자료제공',
        r'-●',
    ]
    min_end = len(text)
    for ep in end_patterns:
        em = re.search(ep, text[start:])
        if em and em.start() > 0:
            if start + em.start() < min_end:
                min_end = start + em.start()
    answer = text[start:min_end].strip()
    # 공통 아티팩트 제거
    answer = re.sub(r'미대입시닷컴\s*midaeipsi\.com', '', answer).strip()
    answer = re.sub(r'홍대\s+\S+\s*(프리미엄[^미]*)?미술학원', '', answer).strip()
    answer = re.sub(r'\s+', ' ', answer).strip()
    if len(answer) < 3:
        return None
    return answer


def summarize(text: str | None, max_sentences: int = 2) -> str | None:
    """텍스트를 1~2문장으로 요약 (잘라내기)"""
    if text is None:
        return None
    sentences = re.split(r'(?<=[.다요!])\s+', text)
    result = ' '.join(sentences[:max_sentences]).strip()
    return result if result else None


def process_file(stem: int) -> tuple[int, dict]:
    """단일 파일 처리"""
    fpath = METADATA_DIR / f'{stem}.json'
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw = data.get('interview_raw', '') or ''

    practice_years = parse_practice_years(raw)
    competition_ratio = parse_competition_ratio(raw)

    # 실기고사 당일 출제문제 질문에서 exam_topic + ideation_process 추출
    exam_q = extract_answer(
        raw, r'Q\.\s*실기고사 당일 출제문제는 무엇이었으며[^?]*\?'
    )
    if exam_q is None:
        exam_q = extract_answer(raw, r'Q\.\s*실기고사 당일')

    exam_topic = None
    ideation_process = None
    if exam_q:
        exam_topic = exam_q[:200]
        ideation_process = summarize(exam_q)

    # 합격의 주요인
    student_strength_raw = extract_answer(
        raw, r'Q\.\s*합격의 주요인은 무엇이라고 생각하나요\?'
    )
    student_strength = summarize(student_strength_raw)

    # 후배들이 유의해야 할 점 (강사/선배 피드백)
    teacher_feedback_raw = extract_answer(
        raw, r'Q\.\s*입시를 준비하는 후배들이 유의해야 할 점[^.?\n]*(?:\.|적어주세요\.?|\?)\s*'
    )
    teacher_feedback = summarize(teacher_feedback_raw)

    parsed = {
        'exam_topic': exam_topic,
        'practice_years': practice_years,
        'competition_ratio': competition_ratio,
        'student_strength': student_strength,
        'teacher_feedback': teacher_feedback,
        'ideation_process': ideation_process,
    }

    data['interview_parsed'] = parsed

    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return stem, parsed


def main():
    parser = argparse.ArgumentParser(description="인터뷰 정규식 배치 파싱")
    parser.add_argument(
        "--stems", type=str, default=None,
        help="처리할 stem 목록 (쉼표 구분, 예: 579,580,581)"
    )
    parser.add_argument("--dry-run", action="store_true", help="파싱만 하고 저장하지 않음")
    args = parser.parse_args()

    if args.stems:
        stems = [int(s.strip()) for s in args.stems.split(',')]
    else:
        stems = DEFAULT_STEMS

    print(f"총 대상: {len(stems)}건")
    start_time = time.time()

    results = {}
    for i in range(0, len(stems), 10):
        batch = stems[i:i + 10]
        print(f'Batch {i // 10 + 1}: {batch[0]}-{batch[-1]}')
        for stem in batch:
            try:
                s, p = process_file(stem)
                results[s] = p
                fields_found = sum(1 for v in p.values() if v is not None)
                print(f'  {s}: {fields_found}/6 fields')
            except Exception as e:
                print(f'  {s}: ERROR - {e}')

    elapsed = time.time() - start_time
    print(f'\nTotal: {len(results)} files ({elapsed:.1f}s)')

    total_fields = {
        k: 0 for k in [
            'exam_topic', 'practice_years', 'competition_ratio',
            'student_strength', 'teacher_feedback', 'ideation_process',
        ]
    }
    for p in results.values():
        for k in total_fields:
            if p.get(k) is not None:
                total_fields[k] += 1
    print('Stats:')
    for k, v in total_fields.items():
        print(f'  {k}: {v}/{len(results)}')


if __name__ == '__main__':
    main()
