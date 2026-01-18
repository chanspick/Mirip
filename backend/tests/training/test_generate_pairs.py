# test_generate_pairs.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: GREEN - Implementation
"""
generate_pairs 스크립트 테스트 모듈.

Acceptance Criteria (AC-005):
- Generate pairs from metadata CSV
- Save pairs to JSON/CSV for reproducibility
- Support stratified train/val/test split
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest

from training.scripts.generate_pairs import (
    load_metadata,
    generate_pairs_from_metadata,
    save_pairs_json,
    save_pairs_csv,
)


class TestLoadMetadata:
    """load_metadata 함수 테스트."""

    def test_load_metadata_success(self, tmp_path: Path) -> None:
        """메타데이터 CSV 파일을 성공적으로 로드한다."""
        # Given: 올바른 형식의 메타데이터 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "tier": ["S", "A", "B"]
        })
        df.to_csv(metadata_path, index=False)

        # When: 메타데이터 로드
        result = load_metadata(metadata_path)

        # Then: DataFrame이 올바르게 로드됨
        assert len(result) == 3
        assert "image_path" in result.columns
        assert "tier" in result.columns

    def test_load_metadata_file_not_found(self, tmp_path: Path) -> None:
        """존재하지 않는 파일을 로드할 때 FileNotFoundError를 발생시킨다."""
        # Given: 존재하지 않는 파일 경로
        metadata_path = tmp_path / "nonexistent.csv"

        # When/Then: FileNotFoundError 발생
        with pytest.raises(FileNotFoundError):
            load_metadata(metadata_path)

    def test_load_metadata_missing_columns(self, tmp_path: Path) -> None:
        """필수 컬럼이 누락된 경우 ValueError를 발생시킨다."""
        # Given: image_path 컬럼만 있는 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["img1.jpg", "img2.jpg"]
            # tier 컬럼 누락
        })
        df.to_csv(metadata_path, index=False)

        # When/Then: ValueError 발생
        with pytest.raises(ValueError, match="Missing required columns"):
            load_metadata(metadata_path)

    def test_load_metadata_extra_columns_allowed(self, tmp_path: Path) -> None:
        """추가 컬럼이 있어도 정상적으로 로드한다."""
        # Given: 추가 컬럼이 있는 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["img1.jpg", "img2.jpg"],
            "tier": ["S", "A"],
            "extra_column": ["value1", "value2"]
        })
        df.to_csv(metadata_path, index=False)

        # When: 메타데이터 로드
        result = load_metadata(metadata_path)

        # Then: 성공적으로 로드됨
        assert len(result) == 2
        assert "extra_column" in result.columns


class TestGeneratePairsFromMetadata:
    """generate_pairs_from_metadata 함수 테스트."""

    def test_generate_pairs_basic(self) -> None:
        """기본 페어 생성 테스트."""
        # Given: 2개 티어의 메타데이터
        metadata_df = pd.DataFrame({
            "image_path": ["s1.jpg", "c1.jpg"],
            "tier": ["S", "C"]
        })

        # When: 페어 생성
        pairs = generate_pairs_from_metadata(metadata_df)

        # Then: 2개 페어 생성 (양방향)
        assert len(pairs) == 2
        # S > C 페어
        assert ("s1.jpg", "c1.jpg", 1) in pairs
        # C < S 페어
        assert ("c1.jpg", "s1.jpg", -1) in pairs

    def test_generate_pairs_multiple_tiers(self) -> None:
        """여러 티어에서 페어 생성 테스트."""
        # Given: 4개 티어의 메타데이터
        metadata_df = pd.DataFrame({
            "image_path": ["s1.jpg", "a1.jpg", "b1.jpg", "c1.jpg"],
            "tier": ["S", "A", "B", "C"]
        })

        # When: 페어 생성
        pairs = generate_pairs_from_metadata(metadata_df)

        # Then: 6 cross-tier 조합 * 2 (양방향) = 12 페어
        # S-A, S-B, S-C, A-B, A-C, B-C
        assert len(pairs) == 12

    def test_generate_pairs_multiple_images_per_tier(self) -> None:
        """티어당 여러 이미지가 있을 때 페어 생성 테스트."""
        # Given: 각 티어에 2개 이미지
        metadata_df = pd.DataFrame({
            "image_path": ["s1.jpg", "s2.jpg", "c1.jpg", "c2.jpg"],
            "tier": ["S", "S", "C", "C"]
        })

        # When: 페어 생성
        pairs = generate_pairs_from_metadata(metadata_df)

        # Then: 2*2 = 4 기본 페어 * 2 (양방향) = 8 페어
        assert len(pairs) == 8

    def test_generate_pairs_verbose_mode(self, capsys) -> None:
        """verbose 모드에서 출력 확인."""
        # Given: 메타데이터
        metadata_df = pd.DataFrame({
            "image_path": ["s1.jpg", "c1.jpg"],
            "tier": ["S", "C"]
        })

        # When: verbose=True로 페어 생성
        pairs = generate_pairs_from_metadata(metadata_df, verbose=True)

        # Then: 출력 확인
        captured = capsys.readouterr()
        assert "Tiers found" in captured.out
        assert "Generated" in captured.out

    def test_generate_pairs_label_logic(self) -> None:
        """라벨 로직 검증: 높은 티어가 더 좋음."""
        # Given: S와 A 티어
        metadata_df = pd.DataFrame({
            "image_path": ["s1.jpg", "a1.jpg"],
            "tier": ["S", "A"]
        })

        # When: 페어 생성
        pairs = generate_pairs_from_metadata(metadata_df)

        # Then: S > A이므로 label=1
        for img1, img2, label in pairs:
            if img1 == "s1.jpg" and img2 == "a1.jpg":
                assert label == 1  # S > A
            elif img1 == "a1.jpg" and img2 == "s1.jpg":
                assert label == -1  # A < S


class TestSavePairsJson:
    """save_pairs_json 함수 테스트."""

    def test_save_pairs_json_basic(self, tmp_path: Path) -> None:
        """기본 JSON 저장 테스트."""
        # Given: 페어 리스트
        pairs = [
            ("img1.jpg", "img2.jpg", 1),
            ("img3.jpg", "img4.jpg", -1)
        ]
        output_path = tmp_path / "pairs.json"

        # When: JSON으로 저장
        save_pairs_json(pairs, output_path)

        # Then: 파일이 생성되고 내용이 올바름
        assert output_path.exists()
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["metadata"]["num_pairs"] == 2
        assert data["metadata"]["format_version"] == "1.0"
        assert len(data["pairs"]) == 2
        assert data["pairs"][0]["img1"] == "img1.jpg"
        assert data["pairs"][0]["img2"] == "img2.jpg"
        assert data["pairs"][0]["label"] == 1

    def test_save_pairs_json_creates_directory(self, tmp_path: Path) -> None:
        """존재하지 않는 디렉토리를 생성한다."""
        # Given: 존재하지 않는 디렉토리 경로
        pairs = [("img1.jpg", "img2.jpg", 1)]
        output_path = tmp_path / "subdir" / "pairs.json"

        # When: JSON으로 저장
        save_pairs_json(pairs, output_path)

        # Then: 디렉토리와 파일이 생성됨
        assert output_path.exists()

    def test_save_pairs_json_empty_list(self, tmp_path: Path) -> None:
        """빈 페어 리스트 저장 테스트."""
        # Given: 빈 페어 리스트
        pairs: List[Tuple[str, str, int]] = []
        output_path = tmp_path / "pairs.json"

        # When: JSON으로 저장
        save_pairs_json(pairs, output_path)

        # Then: 파일이 생성되고 num_pairs가 0
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data["metadata"]["num_pairs"] == 0
        assert len(data["pairs"]) == 0


class TestSavePairsCsv:
    """save_pairs_csv 함수 테스트."""

    def test_save_pairs_csv_basic(self, tmp_path: Path) -> None:
        """기본 CSV 저장 테스트."""
        # Given: 페어 리스트
        pairs = [
            ("img1.jpg", "img2.jpg", 1),
            ("img3.jpg", "img4.jpg", -1)
        ]
        output_path = tmp_path / "pairs.csv"

        # When: CSV로 저장
        save_pairs_csv(pairs, output_path)

        # Then: 파일이 생성되고 내용이 올바름
        assert output_path.exists()
        df = pd.read_csv(output_path)

        assert len(df) == 2
        assert list(df.columns) == ["img1", "img2", "label"]
        assert df.iloc[0]["img1"] == "img1.jpg"
        assert df.iloc[0]["img2"] == "img2.jpg"
        assert df.iloc[0]["label"] == 1

    def test_save_pairs_csv_creates_directory(self, tmp_path: Path) -> None:
        """존재하지 않는 디렉토리를 생성한다."""
        # Given: 존재하지 않는 디렉토리 경로
        pairs = [("img1.jpg", "img2.jpg", 1)]
        output_path = tmp_path / "subdir" / "pairs.csv"

        # When: CSV로 저장
        save_pairs_csv(pairs, output_path)

        # Then: 디렉토리와 파일이 생성됨
        assert output_path.exists()

    def test_save_pairs_csv_empty_list(self, tmp_path: Path) -> None:
        """빈 페어 리스트 저장 테스트."""
        # Given: 빈 페어 리스트
        pairs: List[Tuple[str, str, int]] = []
        output_path = tmp_path / "pairs.csv"

        # When: CSV로 저장
        save_pairs_csv(pairs, output_path)

        # Then: 파일이 생성되고 빈 DataFrame
        df = pd.read_csv(output_path)
        assert len(df) == 0
        assert list(df.columns) == ["img1", "img2", "label"]


class TestIntegration:
    """통합 테스트: load -> generate -> save 파이프라인."""

    def test_full_pipeline_json(self, tmp_path: Path) -> None:
        """전체 파이프라인 JSON 출력 테스트."""
        # Given: 메타데이터 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["s1.jpg", "s2.jpg", "a1.jpg", "b1.jpg", "c1.jpg"],
            "tier": ["S", "S", "A", "B", "C"]
        })
        df.to_csv(metadata_path, index=False)

        # When: 전체 파이프라인 실행
        metadata_df = load_metadata(metadata_path)
        pairs = generate_pairs_from_metadata(metadata_df)
        output_path = tmp_path / "output" / "pairs.json"
        save_pairs_json(pairs, output_path)

        # Then: 결과 검증
        assert output_path.exists()
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data["metadata"]["num_pairs"] > 0
        assert len(data["pairs"]) == data["metadata"]["num_pairs"]

    def test_full_pipeline_csv(self, tmp_path: Path) -> None:
        """전체 파이프라인 CSV 출력 테스트."""
        # Given: 메타데이터 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["s1.jpg", "a1.jpg", "c1.jpg"],
            "tier": ["S", "A", "C"]
        })
        df.to_csv(metadata_path, index=False)

        # When: 전체 파이프라인 실행
        metadata_df = load_metadata(metadata_path)
        pairs = generate_pairs_from_metadata(metadata_df)
        output_path = tmp_path / "output" / "pairs.csv"
        save_pairs_csv(pairs, output_path)

        # Then: 결과 검증
        assert output_path.exists()
        result_df = pd.read_csv(output_path)
        assert len(result_df) > 0
        assert all(col in result_df.columns for col in ["img1", "img2", "label"])

    def test_pipeline_preserves_image_paths(self, tmp_path: Path) -> None:
        """파이프라인이 이미지 경로를 보존하는지 확인."""
        # Given: 특정 이미지 경로
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["path/to/image1.jpg", "path/to/image2.jpg"],
            "tier": ["S", "C"]
        })
        df.to_csv(metadata_path, index=False)

        # When: 파이프라인 실행
        metadata_df = load_metadata(metadata_path)
        pairs = generate_pairs_from_metadata(metadata_df)

        # Then: 경로가 보존됨
        all_paths = set()
        for img1, img2, _ in pairs:
            all_paths.add(img1)
            all_paths.add(img2)

        assert "path/to/image1.jpg" in all_paths
        assert "path/to/image2.jpg" in all_paths


class TestParseArgs:
    """parse_args 함수 테스트."""

    def test_parse_args_required_arguments(self, monkeypatch, tmp_path: Path) -> None:
        """필수 인자가 올바르게 파싱된다."""
        from training.scripts.generate_pairs import parse_args

        # Given: 필수 인자
        metadata_path = str(tmp_path / "metadata.csv")
        output_path = str(tmp_path / "pairs.json")

        monkeypatch.setattr(
            "sys.argv",
            ["generate_pairs", "--metadata", metadata_path, "--output", output_path]
        )

        # When: 인자 파싱
        args = parse_args()

        # Then: 올바르게 파싱됨
        assert args.metadata == Path(metadata_path)
        assert args.output == Path(output_path)
        assert args.format == "json"  # 기본값
        assert args.split is False
        assert args.verbose is False

    def test_parse_args_all_options(self, monkeypatch, tmp_path: Path) -> None:
        """모든 옵션이 올바르게 파싱된다."""
        from training.scripts.generate_pairs import parse_args

        metadata_path = str(tmp_path / "metadata.csv")
        output_path = str(tmp_path / "output")

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", metadata_path,
                "--output", output_path,
                "--format", "both",
                "--split",
                "--train-ratio", "0.7",
                "--val-ratio", "0.2",
                "--seed", "123",
                "--verbose"
            ]
        )

        # When: 인자 파싱
        args = parse_args()

        # Then: 모든 옵션이 올바르게 파싱됨
        assert args.format == "both"
        assert args.split is True
        assert args.train_ratio == 0.7
        assert args.val_ratio == 0.2
        assert args.seed == 123
        assert args.verbose is True

    def test_parse_args_short_options(self, monkeypatch, tmp_path: Path) -> None:
        """짧은 옵션이 올바르게 파싱된다."""
        from training.scripts.generate_pairs import parse_args

        metadata_path = str(tmp_path / "metadata.csv")
        output_path = str(tmp_path / "pairs.csv")

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "-m", metadata_path,
                "-o", output_path,
                "-f", "csv",
                "-v"
            ]
        )

        # When: 인자 파싱
        args = parse_args()

        # Then: 짧은 옵션이 올바르게 파싱됨
        assert args.metadata == Path(metadata_path)
        assert args.output == Path(output_path)
        assert args.format == "csv"
        assert args.verbose is True


class TestMainFunction:
    """main() 함수 테스트."""

    def test_main_basic_json_output(self, monkeypatch, tmp_path: Path, capsys) -> None:
        """기본 JSON 출력 테스트."""
        from training.scripts.generate_pairs import main

        # Given: 메타데이터 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["s1.jpg", "a1.jpg", "c1.jpg"],
            "tier": ["S", "A", "C"]
        })
        df.to_csv(metadata_path, index=False)

        output_path = tmp_path / "pairs"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_path),
                "--format", "json"
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: 성공적으로 완료
        assert exit_code == 0
        assert (tmp_path / "pairs.json").exists()

        # JSON 파일 내용 확인
        with open(tmp_path / "pairs.json", 'r') as f:
            data = json.load(f)
        assert data["metadata"]["num_pairs"] > 0

    def test_main_csv_output(self, monkeypatch, tmp_path: Path) -> None:
        """CSV 출력 테스트."""
        from training.scripts.generate_pairs import main

        # Given: 메타데이터 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["s1.jpg", "c1.jpg"],
            "tier": ["S", "C"]
        })
        df.to_csv(metadata_path, index=False)

        output_path = tmp_path / "pairs"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_path),
                "--format", "csv"
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: CSV 파일 생성
        assert exit_code == 0
        assert (tmp_path / "pairs.csv").exists()

    def test_main_both_formats(self, monkeypatch, tmp_path: Path) -> None:
        """JSON과 CSV 모두 출력 테스트."""
        from training.scripts.generate_pairs import main

        # Given: 메타데이터 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["s1.jpg", "c1.jpg"],
            "tier": ["S", "C"]
        })
        df.to_csv(metadata_path, index=False)

        output_path = tmp_path / "pairs"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_path),
                "--format", "both"
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: 두 형식 모두 생성
        assert exit_code == 0
        assert (tmp_path / "pairs.json").exists()
        assert (tmp_path / "pairs.csv").exists()

    def test_main_with_split(self, monkeypatch, tmp_path: Path) -> None:
        """train/val/test 분할 테스트."""
        from training.scripts.generate_pairs import main

        # Given: 충분한 데이터가 있는 메타데이터 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": [f"img{i}.jpg" for i in range(20)],
            "tier": ["S"] * 5 + ["A"] * 5 + ["B"] * 5 + ["C"] * 5
        })
        df.to_csv(metadata_path, index=False)

        output_dir = tmp_path / "output"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_dir),
                "--split",
                "--format", "json"
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: 분할된 파일들이 생성됨
        assert exit_code == 0
        assert output_dir.exists()
        assert (output_dir / "train_pairs.json").exists()
        assert (output_dir / "val_pairs.json").exists()
        assert (output_dir / "test_pairs.json").exists()
        assert (output_dir / "train_metadata.csv").exists()

    def test_main_verbose_mode(self, monkeypatch, tmp_path: Path, capsys) -> None:
        """verbose 모드 테스트."""
        from training.scripts.generate_pairs import main

        # Given: 메타데이터 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["s1.jpg", "c1.jpg"],
            "tier": ["S", "C"]
        })
        df.to_csv(metadata_path, index=False)

        output_path = tmp_path / "pairs"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_path),
                "--verbose"
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: verbose 출력 확인
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Loading metadata" in captured.out
        assert "Loaded" in captured.out

    def test_main_with_extension(self, monkeypatch, tmp_path: Path) -> None:
        """확장자가 있는 출력 경로 테스트."""
        from training.scripts.generate_pairs import main

        # Given: 메타데이터 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["s1.jpg", "c1.jpg"],
            "tier": ["S", "C"]
        })
        df.to_csv(metadata_path, index=False)

        output_path = tmp_path / "pairs.json"  # 확장자 포함

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_path)
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: 지정된 경로에 파일 생성
        assert exit_code == 0
        assert output_path.exists()


class TestMainErrors:
    """main() 함수 에러 처리 테스트."""

    def test_main_file_not_found(self, monkeypatch, tmp_path: Path, capsys) -> None:
        """존재하지 않는 파일 에러 처리."""
        from training.scripts.generate_pairs import main

        # Given: 존재하지 않는 메타데이터 경로
        metadata_path = tmp_path / "nonexistent.csv"
        output_path = tmp_path / "pairs.json"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_path)
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: 에러 코드 1 반환
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_main_missing_columns(self, monkeypatch, tmp_path: Path, capsys) -> None:
        """필수 컬럼 누락 에러 처리."""
        from training.scripts.generate_pairs import main

        # Given: 잘못된 형식의 CSV
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["img1.jpg", "img2.jpg"]
            # tier 컬럼 누락
        })
        df.to_csv(metadata_path, index=False)

        output_path = tmp_path / "pairs.json"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_path)
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: 에러 코드 2 반환
        assert exit_code == 2
        captured = capsys.readouterr()
        assert "Error" in captured.err


class TestMainSplitEdgeCases:
    """main() 함수의 split 모드 엣지 케이스 테스트."""

    def test_main_split_with_both_formats(self, monkeypatch, tmp_path: Path) -> None:
        """split 모드에서 두 형식 모두 출력."""
        from training.scripts.generate_pairs import main

        # Given: 충분한 데이터
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": [f"img{i}.jpg" for i in range(30)],
            "tier": ["S"] * 10 + ["A"] * 10 + ["C"] * 10
        })
        df.to_csv(metadata_path, index=False)

        output_dir = tmp_path / "output"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_dir),
                "--split",
                "--format", "both",
                "--verbose"
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: JSON과 CSV 모두 생성
        assert exit_code == 0
        assert (output_dir / "train_pairs.json").exists()
        assert (output_dir / "train_pairs.csv").exists()
        assert (output_dir / "val_pairs.json").exists()
        assert (output_dir / "val_pairs.csv").exists()

    def test_main_split_custom_ratios(self, monkeypatch, tmp_path: Path) -> None:
        """사용자 정의 분할 비율 테스트."""
        from training.scripts.generate_pairs import main

        # Given: 충분한 데이터
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": [f"img{i}.jpg" for i in range(40)],
            "tier": ["S"] * 10 + ["A"] * 10 + ["B"] * 10 + ["C"] * 10
        })
        df.to_csv(metadata_path, index=False)

        output_dir = tmp_path / "output"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_dir),
                "--split",
                "--train-ratio", "0.7",
                "--val-ratio", "0.15",
                "--seed", "123"
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: 성공
        assert exit_code == 0
        assert (output_dir / "train_pairs.json").exists()

    def test_main_split_verbose_with_skipped_splits(
        self, monkeypatch, tmp_path: Path, capsys
    ) -> None:
        """분할 데이터가 부족할 때 verbose 출력 테스트."""
        from training.scripts.generate_pairs import main

        # Given: 적은 데이터 (일부 분할에 이미지가 없을 수 있음)
        metadata_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "image_path": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "tier": ["S", "A", "C"]
        })
        df.to_csv(metadata_path, index=False)

        output_dir = tmp_path / "output"

        monkeypatch.setattr(
            "sys.argv",
            [
                "generate_pairs",
                "--metadata", str(metadata_path),
                "--output", str(output_dir),
                "--split",
                "--verbose"
            ]
        )

        # When: main 실행
        exit_code = main()

        # Then: 성공 (일부 분할이 스킵될 수 있음)
        assert exit_code == 0
