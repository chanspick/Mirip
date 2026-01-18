# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: DataPipeline 구현

수집 -> 전처리 -> 태깅 -> 라벨링 -> 저장의
전체 데이터 파이프라인을 통합하는 클래스입니다.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from data_pipeline.collectors.image_collector import ImageCollector
from data_pipeline.labelers.auto_labeler import AutoLabeler
from data_pipeline.labelers.base_labeler import LabelingResult
from data_pipeline.models.metadata import Department, ImageMetadata, Tier
from data_pipeline.preprocessors.base_preprocessor import BasePreprocessor
from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor
from data_pipeline.storage.metadata_storage import MetadataStorage


class DataPipeline:
    """
    데이터 파이프라인

    이미지 수집, 전처리, 라벨링, 저장의 전체 워크플로우를 관리합니다.
    """

    def __init__(
        self,
        source_dir: Union[str, Path],
        output_dir: Union[str, Path],
        preprocessors: Optional[List[BasePreprocessor]] = None,
        target_size: Tuple[int, int] = (224, 224),
        confidence_threshold: float = 0.6,
        recursive: bool = True,
    ):
        """
        DataPipeline 초기화

        Args:
            source_dir: 소스 이미지 디렉토리
            output_dir: 출력 디렉토리
            preprocessors: 전처리기 리스트 (None이면 기본 전처리기 사용)
            target_size: 타겟 이미지 크기
            confidence_threshold: 신뢰도 임계값
            recursive: 재귀적 수집 여부
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.recursive = recursive

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 컴포넌트 초기화
        self.collector = ImageCollector(
            output_dir=self.output_dir / "images",
            recursive=recursive,
        )

        # 전처리기 설정
        if preprocessors is not None:
            self.preprocessors = preprocessors
        else:
            self.preprocessors = [
                ResizePreprocessor(target_size=target_size),
            ]

        # 라벨러 초기화
        self.labeler = AutoLabeler(confidence_threshold=confidence_threshold)

        # 스토리지 초기화
        self.storage = MetadataStorage(base_path=self.output_dir)

        # 처리 결과 저장
        self._results: List[LabelingResult] = []

    def process_single(
        self, image_path: Union[str, Path]
    ) -> LabelingResult:
        """
        단일 이미지를 처리합니다.

        Args:
            image_path: 이미지 파일 경로

        Returns:
            LabelingResult: 라벨링 결과
        """
        image_path = Path(image_path)

        # 이미지 로드
        image = Image.open(image_path)

        # 전처리 적용
        processed = image
        for preprocessor in self.preprocessors:
            processed = preprocessor.process(processed)

        # 이미지 ID 생성
        image_id = f"img_{image_path.stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 라벨링
        result = self.labeler.label(image_id, processed)

        # 메타데이터 생성 및 저장
        metadata = ImageMetadata(
            image_id=image_id,
            original_filename=image_path.name,
            file_path=str(image_path),
            width=image.width,
            height=image.height,
            file_size=image_path.stat().st_size,
            format=image.format or "UNKNOWN",
            department=result.department,
            tier=result.tier,
            tier_score=result.tier_score,
            is_manual_label=False,
        )
        self.storage.save_metadata(metadata)

        return result

    def run(self) -> List[LabelingResult]:
        """
        전체 파이프라인을 실행합니다.

        Returns:
            LabelingResult 리스트
        """
        self._results = []

        # 이미지 수집 및 처리
        for collection_result in self.collector.collect(str(self.source_dir)):
            if collection_result.success:
                try:
                    # 경로 변환 (문자열 -> Path)
                    source_path = Path(collection_result.source_path)
                    local_path = Path(collection_result.local_path) if collection_result.local_path else source_path

                    # 이미지 로드
                    image = Image.open(source_path)

                    # 전처리 적용
                    processed = image
                    for preprocessor in self.preprocessors:
                        processed = preprocessor.process(processed)

                    # 라벨링
                    result = self.labeler.label(
                        collection_result.image_id, processed
                    )

                    # 메타데이터 생성 및 저장
                    metadata = ImageMetadata(
                        image_id=collection_result.image_id,
                        original_filename=source_path.name,
                        file_path=str(local_path),
                        width=image.width,
                        height=image.height,
                        file_size=source_path.stat().st_size,
                        format=image.format or "UNKNOWN",
                        department=result.department,
                        tier=result.tier,
                        tier_score=result.tier_score,
                        is_manual_label=False,
                    )
                    self.storage.save_metadata(metadata)

                    self._results.append(result)

                except Exception as e:
                    # 처리 실패한 이미지는 건너뜀
                    print(f"이미지 처리 실패: {collection_result.source_path} - {e}")
                    continue

        return self._results

    def get_statistics(self) -> Dict[str, Any]:
        """
        파이프라인 실행 통계를 반환합니다.

        Returns:
            통계 딕셔너리
        """
        if not self._results:
            return {
                "total_processed": 0,
                "department_distribution": {},
                "tier_distribution": {},
                "average_confidence": {
                    "department": 0.0,
                    "tier": 0.0,
                },
            }

        # 과별 분포
        dept_counts = Counter(r.department.value for r in self._results)
        dept_distribution = {
            dept: count / len(self._results)
            for dept, count in dept_counts.items()
        }

        # 티어 분포
        tier_counts = Counter(r.tier.value for r in self._results)
        tier_distribution = {
            tier: count / len(self._results)
            for tier, count in tier_counts.items()
        }

        # 평균 신뢰도
        avg_dept_conf = sum(r.department_confidence for r in self._results) / len(self._results)
        avg_tier_conf = sum(r.tier_confidence for r in self._results) / len(self._results)

        return {
            "total_processed": len(self._results),
            "department_distribution": dept_distribution,
            "tier_distribution": tier_distribution,
            "average_confidence": {
                "department": round(avg_dept_conf, 4),
                "tier": round(avg_tier_conf, 4),
            },
        }

    def filter_results(
        self,
        results: List[LabelingResult],
        min_confidence: float = 0.5,
    ) -> List[LabelingResult]:
        """
        신뢰도로 결과를 필터링합니다.

        Args:
            results: 필터링할 결과 리스트
            min_confidence: 최소 신뢰도

        Returns:
            필터링된 결과 리스트
        """
        return [
            r for r in results
            if r.department_confidence >= min_confidence
            or r.tier_confidence >= min_confidence
        ]

    def get_items_needing_review(
        self,
        results: List[LabelingResult],
        threshold: float = 0.6,
    ) -> List[LabelingResult]:
        """
        검토가 필요한 항목을 반환합니다.

        Args:
            results: 결과 리스트
            threshold: 신뢰도 임계값

        Returns:
            검토 필요 항목 리스트
        """
        return [r for r in results if r.needs_review(threshold)]

    def export_report(
        self,
        filename: str = "pipeline_report.json",
    ) -> Path:
        """
        처리 결과 리포트를 내보냅니다.

        Args:
            filename: 리포트 파일명

        Returns:
            리포트 파일 경로
        """
        report_path = self.output_dir / filename

        report_data = {
            "generated_at": datetime.now().isoformat(),
            "source_dir": str(self.source_dir),
            "output_dir": str(self.output_dir),
            "statistics": self.get_statistics(),
            "results": [
                {
                    "image_id": r.image_id,
                    "department": r.department.value,
                    "department_confidence": r.department_confidence,
                    "tier": r.tier.value,
                    "tier_confidence": r.tier_confidence,
                    "tier_score": r.tier_score,
                }
                for r in self._results
            ],
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return report_path
