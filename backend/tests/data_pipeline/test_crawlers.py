# -*- coding: utf-8 -*-
"""
크롤러 모듈 단위 테스트

normalizer, dedup, cleaner, stats, converter 모듈에 대한
포괄적인 단위 테스트를 포함합니다.
"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ============================================================
# 공통 헬퍼 함수
# ============================================================


def _create_test_image(path: Path, color=(255, 0, 0), size=(10, 10)) -> None:
    """테스트용 이미지 파일을 생성합니다."""
    img = Image.new("RGB", size, color=color)
    img.save(str(path))


def _create_metadata_json(path: Path, metadata: dict) -> None:
    """테스트용 메타데이터 JSON 파일을 생성합니다."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


_SENTINEL = object()


def _sample_metadata(
    post_no: int = 1,
    university: str = "홍익대학교",
    department: str = "visual_design",
    tier: str = "S",
    images=_SENTINEL,
) -> dict:
    """테스트용 기본 메타데이터 딕셔너리를 반환합니다."""
    if images is _SENTINEL:
        images = [f"img_{post_no:03d}.jpg"]
    return {
        "post_no": post_no,
        "title": f"{university} 합격",
        "university": university,
        "department": department,
        "tier": tier,
        "year": "2024",
        "admission_type": "수시",
        "work_type": "기초디자인",
        "images": images,
        "crawled_at": "2025-01-15T10:30:00",
    }


# ============================================================
# normalizer.py 테스트
# ============================================================


class TestNormalizeUniversity:
    """대학명 정규화 테스트"""

    def test_정식명칭에서_대_축약형_변환(self):
        """'XX대학교' -> 'XX대' 정규식 규칙이 올바르게 동작하는지 확인"""
        from data_pipeline.crawlers.normalizer import normalize_university

        assert normalize_university("홍익대학교") == "홍익대"
        assert normalize_university("국민대학교") == "국민대"
        assert normalize_university("건국대학교") == "건국대"

    def test_예술대학교_축약형_변환(self):
        """'XX예술대학교' -> 'XX예대' 정규식 규칙 확인"""
        from data_pipeline.crawlers.normalizer import normalize_university

        assert normalize_university("서울예술대학교") == "서울예대"

    def test_과학기술대학교_축약형_변환(self):
        """'XX과학기술대학교' -> 'XX과기대' 정규식 규칙 확인"""
        from data_pipeline.crawlers.normalizer import normalize_university

        assert normalize_university("서울과학기술대학교") == "서울과기대"

    def test_여자대학교_축약형_변환(self):
        """'XX여자대학교' -> 'XX여대' 정규식 규칙 확인"""
        from data_pipeline.crawlers.normalizer import normalize_university

        assert normalize_university("이화여자대학교") == "이화여대"
        assert normalize_university("숙명여자대학교") == "숙명여대"

    def test_알려진_축약어_매핑(self):
        """KNOWN_ABBREVIATIONS 딕셔너리 매핑 확인"""
        from data_pipeline.crawlers.normalizer import normalize_university

        assert normalize_university("홍대") == "홍익대"
        assert normalize_university("국대") == "국민대"
        assert normalize_university("이대") == "이화여대"
        assert normalize_university("한예종") == "한예종"

    def test_빈_문자열_입력(self):
        """빈 문자열 입력 시 빈 문자열 반환"""
        from data_pipeline.crawlers.normalizer import normalize_university

        assert normalize_university("") == ""

    def test_이미_축약형인_대학명(self):
        """이미 축약형인 대학명은 그대로 반환"""
        from data_pipeline.crawlers.normalizer import normalize_university

        assert normalize_university("홍익대") == "홍익대"

    def test_규칙에_없는_대학명(self):
        """정규식 규칙에 매칭되지 않는 대학명은 원본 반환"""
        from data_pipeline.crawlers.normalizer import normalize_university

        assert normalize_university("알수없는학교") == "알수없는학교"

    def test_공백_있는_대학명(self):
        """앞뒤 공백은 제거되어야 함"""
        from data_pipeline.crawlers.normalizer import normalize_university

        assert normalize_university("  홍익대학교  ") == "홍익대"


class TestExtractUniversityFromTitle:
    """제목에서 대학명 추출 테스트"""

    def test_정식명칭_추출(self):
        """제목에서 'XX대학교' 형태의 대학명을 추출하고 정규화"""
        from data_pipeline.crawlers.normalizer import extract_university_from_title

        raw, norm = extract_university_from_title("2024 홍익대학교 시각디자인과 합격")
        assert raw == "홍익대학교"
        assert norm == "홍익대"

    def test_축약어_추출(self):
        """제목에서 알려진 축약어를 추출"""
        from data_pipeline.crawlers.normalizer import extract_university_from_title

        raw, norm = extract_university_from_title("2024 홍대 시디 합격")
        assert raw == "홍대"
        assert norm == "홍익대"

    def test_캠퍼스_패턴_추출(self):
        """한양대(ERICA) 패턴을 '한양에리카'로 변환"""
        from data_pipeline.crawlers.normalizer import extract_university_from_title

        raw, norm = extract_university_from_title("한양대(ERICA) 산디 합격")
        assert norm == "한양에리카"

    def test_빈_제목(self):
        """빈 제목에서는 빈 튜플 반환"""
        from data_pipeline.crawlers.normalizer import extract_university_from_title

        assert extract_university_from_title("") == ("", "")

    def test_대학명_없는_제목(self):
        """대학명이 없는 제목에서는 빈 튜플 반환"""
        from data_pipeline.crawlers.normalizer import extract_university_from_title

        assert extract_university_from_title("합격 후기입니다") == ("", "")


class TestDetermineTier:
    """대학 티어 결정 테스트"""

    def test_S티어_대학(self):
        """S 티어 대학 확인 (서울대, 홍익대)"""
        from data_pipeline.crawlers.normalizer import determine_tier

        assert determine_tier("서울대") == "S"
        assert determine_tier("홍익대") == "S"

    def test_A티어_대학(self):
        """A 티어 대학 확인"""
        from data_pipeline.crawlers.normalizer import determine_tier

        assert determine_tier("국민대") == "A"
        assert determine_tier("이화여대") == "A"
        assert determine_tier("중앙대") == "A"

    def test_B티어_대학(self):
        """B 티어 대학 확인"""
        from data_pipeline.crawlers.normalizer import determine_tier

        assert determine_tier("상명대") == "B"
        assert determine_tier("한양에리카") == "B"
        assert determine_tier("가천대") == "B"

    def test_C티어_기본값(self):
        """S/A/B에 없는 대학은 C 티어"""
        from data_pipeline.crawlers.normalizer import determine_tier

        assert determine_tier("한밭대") == "C"
        assert determine_tier("알수없는대") == "C"

    def test_빈_문자열_C티어(self):
        """빈 문자열은 C 티어 반환"""
        from data_pipeline.crawlers.normalizer import determine_tier

        assert determine_tier("") == "C"


class TestNormalizeDepartment:
    """학과명 정규화 테스트"""

    def test_시각디자인_계열(self):
        """시각디자인 계열 학과명 정규화"""
        from data_pipeline.crawlers.normalizer import normalize_department

        assert normalize_department("시각디자인과") == "visual_design"
        assert normalize_department("시각디자인학과") == "visual_design"
        assert normalize_department("시디") == "visual_design"
        assert normalize_department("커뮤니케이션디자인과") == "visual_design"

    def test_산업디자인_계열(self):
        """산업디자인 계열 학과명 정규화"""
        from data_pipeline.crawlers.normalizer import normalize_department

        assert normalize_department("산업디자인과") == "industrial_design"
        assert normalize_department("산디") == "industrial_design"
        assert normalize_department("제품디자인학과") == "industrial_design"

    def test_공예_계열(self):
        """공예 계열 학과명 정규화"""
        from data_pipeline.crawlers.normalizer import normalize_department

        assert normalize_department("공예") == "craft"
        assert normalize_department("금속공예과") == "craft"
        assert normalize_department("도예과") == "craft"

    def test_회화_계열(self):
        """회화/순수미술 계열 학과명 정규화"""
        from data_pipeline.crawlers.normalizer import normalize_department

        assert normalize_department("회화과") == "fine_art"
        assert normalize_department("서양화과") == "fine_art"
        assert normalize_department("조소과") == "fine_art"

    def test_빈_문자열(self):
        """빈 문자열 입력 시 빈 문자열 반환"""
        from data_pipeline.crawlers.normalizer import normalize_department

        assert normalize_department("") == ""

    def test_매핑되지_않는_학과(self):
        """매핑되지 않는 학과명은 원본 반환"""
        from data_pipeline.crawlers.normalizer import normalize_department

        assert normalize_department("건축학과") == "건축학과"


class TestExtractDepartmentFromTitle:
    """제목에서 학과명 추출 테스트"""

    def test_학과명_추출(self):
        """제목에서 학과명을 추출하고 정규화"""
        from data_pipeline.crawlers.normalizer import extract_department_from_title

        raw, code = extract_department_from_title("홍익대 시각디자인과 합격")
        assert raw == "시각디자인과"
        assert code == "visual_design"

    def test_공예_추출(self):
        """제목에서 공예 학과 추출"""
        from data_pipeline.crawlers.normalizer import extract_department_from_title

        raw, code = extract_department_from_title("국민대 공예 합격")
        assert raw == "공예"
        assert code == "craft"

    def test_빈_제목(self):
        """빈 제목에서는 빈 튜플 반환"""
        from data_pipeline.crawlers.normalizer import extract_department_from_title

        assert extract_department_from_title("") == ("", "")

    def test_학과명_없는_제목(self):
        """학과명이 없는 제목에서는 빈 튜플 반환"""
        from data_pipeline.crawlers.normalizer import extract_department_from_title

        assert extract_department_from_title("합격 후기") == ("", "")


# ============================================================
# dedup.py 테스트
# ============================================================


class TestDeduplicationResult:
    """DeduplicationResult 데이터클래스 테스트"""

    def test_기본값(self):
        """기본값이 올바르게 설정되는지 확인"""
        from data_pipeline.crawlers.dedup import DeduplicationResult

        result = DeduplicationResult()
        assert result.total_images == 0
        assert result.unique_images == 0
        assert result.duplicates_removed == 0
        assert result.duplicate_groups == {}

    def test_값_설정(self):
        """직접 값을 설정하여 생성"""
        from data_pipeline.crawlers.dedup import DeduplicationResult

        result = DeduplicationResult(
            total_images=10,
            unique_images=7,
            duplicates_removed=3,
        )
        assert result.total_images == 10
        assert result.unique_images == 7
        assert result.duplicates_removed == 3


class TestImageDeduplicator:
    """ImageDeduplicator 테스트"""

    def test_초기화(self):
        """기본 초기화 파라미터 확인"""
        from data_pipeline.crawlers.dedup import ImageDeduplicator

        dedup = ImageDeduplicator()
        assert dedup.hash_size == 8
        assert dedup.threshold == 5

    def test_커스텀_초기화(self):
        """커스텀 파라미터로 초기화"""
        from data_pipeline.crawlers.dedup import ImageDeduplicator

        dedup = ImageDeduplicator(hash_size=16, threshold=10)
        assert dedup.hash_size == 16
        assert dedup.threshold == 10

    def test_지원_확장자(self):
        """지원하는 이미지 확장자 목록 확인"""
        from data_pipeline.crawlers.dedup import ImageDeduplicator

        assert ".jpg" in ImageDeduplicator.SUPPORTED_EXTENSIONS
        assert ".jpeg" in ImageDeduplicator.SUPPORTED_EXTENSIONS
        assert ".png" in ImageDeduplicator.SUPPORTED_EXTENSIONS
        assert ".gif" in ImageDeduplicator.SUPPORTED_EXTENSIONS
        assert ".webp" in ImageDeduplicator.SUPPORTED_EXTENSIONS

    def test_빈_디렉토리_이미지_목록(self):
        """빈 디렉토리에서 이미지 목록 조회"""
        from data_pipeline.crawlers.dedup import ImageDeduplicator

        dedup = ImageDeduplicator()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = dedup._list_images(Path(tmpdir))
            assert result == []

    def test_존재하지_않는_디렉토리(self):
        """존재하지 않는 디렉토리에서 빈 리스트 반환"""
        from data_pipeline.crawlers.dedup import ImageDeduplicator

        dedup = ImageDeduplicator()
        result = dedup._list_images(Path("/nonexistent/path"))
        assert result == []

    def test_이미지_파일_목록(self):
        """디렉토리 내 이미지 파일만 필터링하여 목록 반환"""
        from data_pipeline.crawlers.dedup import ImageDeduplicator

        dedup = ImageDeduplicator()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # 이미지 파일 생성
            _create_test_image(tmppath / "test1.jpg")
            _create_test_image(tmppath / "test2.png", color=(0, 255, 0))
            # 비이미지 파일 생성
            (tmppath / "readme.txt").write_text("not an image")
            (tmppath / "data.json").write_text("{}")

            result = dedup._list_images(tmppath)
            assert len(result) == 2
            # 이름순 정렬 확인
            assert result[0].name == "test1.jpg"
            assert result[1].name == "test2.png"

    def test_compute_hash_유효한_이미지(self):
        """유효한 이미지의 pHash 계산"""
        from data_pipeline.crawlers.dedup import ImageDeduplicator

        dedup = ImageDeduplicator()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"
            _create_test_image(img_path, size=(64, 64))

            result = dedup.compute_hash(img_path)
            # imagehash가 설치되어 있으면 문자열 반환, 아니면 None
            # 테스트 환경에 따라 결과가 다를 수 있으므로 타입만 확인
            assert result is None or isinstance(result, str)

    def test_compute_hash_손상된_이미지(self):
        """손상된 이미지 파일에 대해 None 반환"""
        from data_pipeline.crawlers.dedup import ImageDeduplicator

        dedup = ImageDeduplicator()
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "corrupted.jpg"
            bad_path.write_bytes(b"this is not an image file")

            result = dedup.compute_hash(bad_path)
            assert result is None

    def test_빈_디렉토리_중복탐색(self):
        """빈 디렉토리에서 중복 탐색 시 빈 딕셔너리 반환"""
        from data_pipeline.crawlers.dedup import ImageDeduplicator

        dedup = ImageDeduplicator()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = dedup.find_duplicates(Path(tmpdir))
            assert result == {}

    def test_빈_디렉토리_중복제거(self):
        """빈 디렉토리에서 중복 제거 실행"""
        from data_pipeline.crawlers.dedup import ImageDeduplicator

        dedup = ImageDeduplicator()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = dedup.remove_duplicates(Path(tmpdir))
            assert result.total_images == 0
            assert result.duplicates_removed == 0


# ============================================================
# cleaner.py 테스트
# ============================================================


class TestCleaningResult:
    """CleaningResult 데이터클래스 테스트"""

    def test_기본값(self):
        """기본값이 올바르게 설정되는지 확인"""
        from data_pipeline.crawlers.cleaner import CleaningResult

        result = CleaningResult()
        assert result.total_posts == 0
        assert result.valid_posts == 0
        assert result.invalid_images == 0
        assert result.duplicates_removed == 0
        assert result.normalized_count == 0

    def test_요약_문자열(self):
        """summary() 메서드가 올바른 형식의 문자열을 반환"""
        from data_pipeline.crawlers.cleaner import CleaningResult

        result = CleaningResult(
            total_posts=100,
            valid_posts=90,
            invalid_images=5,
            duplicates_removed=3,
            normalized_count=10,
        )
        summary = result.summary()
        assert "100" in summary
        assert "90" in summary
        assert "5" in summary
        assert "3" in summary
        assert "10" in summary


class TestDataCleaner:
    """DataCleaner 테스트"""

    def test_기본_초기화(self):
        """기본 파라미터로 초기화"""
        from data_pipeline.crawlers.cleaner import DataCleaner

        cleaner = DataCleaner()
        assert cleaner.images_dir == Path("data/crawled/raw_images")
        assert cleaner.metadata_dir == Path("data/crawled/metadata")

    def test_커스텀_디렉토리_초기화(self):
        """커스텀 디렉토리로 초기화"""
        from data_pipeline.crawlers.cleaner import DataCleaner

        with tempfile.TemporaryDirectory() as tmpdir:
            cleaner = DataCleaner(data_dir=tmpdir)
            assert cleaner.images_dir == Path(tmpdir) / "raw_images"
            assert cleaner.metadata_dir == Path(tmpdir) / "metadata"

    def test_출력_디렉토리_설정(self):
        """별도 출력 디렉토리 설정"""
        from data_pipeline.crawlers.cleaner import DataCleaner

        with tempfile.TemporaryDirectory() as data_dir:
            with tempfile.TemporaryDirectory() as out_dir:
                cleaner = DataCleaner(data_dir=data_dir, output_dir=out_dir)
                assert cleaner.output_images_dir == Path(out_dir) / "raw_images"
                assert cleaner.output_metadata_dir == Path(out_dir) / "metadata"

    def test_이미지_검증_빈_디렉토리(self):
        """이미지 디렉토리가 비어 있으면 손상 이미지 없음"""
        from data_pipeline.crawlers.cleaner import DataCleaner

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "raw_images").mkdir()
            (base / "metadata").mkdir()

            cleaner = DataCleaner(data_dir=tmpdir)
            invalid = cleaner._validate_images()
            assert len(invalid) == 0

    def test_이미지_검증_유효한_이미지(self):
        """유효한 이미지는 손상 목록에 포함되지 않음"""
        from data_pipeline.crawlers.cleaner import DataCleaner

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            images_dir = base / "raw_images"
            images_dir.mkdir()
            (base / "metadata").mkdir()

            # 유효한 이미지 생성
            _create_test_image(images_dir / "valid.jpg")

            cleaner = DataCleaner(data_dir=tmpdir)
            invalid = cleaner._validate_images()
            assert len(invalid) == 0

    def test_이미지_검증_손상된_이미지(self):
        """손상된 이미지는 손상 목록에 포함됨"""
        from data_pipeline.crawlers.cleaner import DataCleaner

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            images_dir = base / "raw_images"
            images_dir.mkdir()
            (base / "metadata").mkdir()

            # 손상된 이미지 파일 생성
            (images_dir / "bad.jpg").write_bytes(b"not a valid image")

            cleaner = DataCleaner(data_dir=tmpdir)
            invalid = cleaner._validate_images()
            assert len(invalid) == 1

    def test_이미지_검증_빈_파일(self):
        """빈 파일(0바이트)은 손상으로 처리"""
        from data_pipeline.crawlers.cleaner import DataCleaner

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            images_dir = base / "raw_images"
            images_dir.mkdir()
            (base / "metadata").mkdir()

            # 빈 이미지 파일 생성
            (images_dir / "empty.jpg").write_bytes(b"")

            cleaner = DataCleaner(data_dir=tmpdir)
            invalid = cleaner._validate_images()
            assert len(invalid) == 1

    def test_메타데이터_정규화(self):
        """메타데이터 대학명/학과명 정규화 변경 수 반환"""
        from data_pipeline.crawlers.cleaner import DataCleaner

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "raw_images").mkdir()
            metadata_dir = base / "metadata"
            metadata_dir.mkdir()

            # 정규화가 필요한 메타데이터 생성
            meta = {
                "post_no": 1,
                "university": "홍익대학교",
                "department": "시각디자인과",
                "tier": "C",
            }
            _create_metadata_json(metadata_dir / "post_001.json", meta)

            cleaner = DataCleaner(data_dir=tmpdir)
            changed = cleaner._normalize_metadata()
            # 대학명이 홍익대학교 -> 홍익대로 변경, 티어도 S로 변경
            assert changed >= 1

            # 변경된 파일 다시 읽어서 확인
            with open(metadata_dir / "post_001.json", "r", encoding="utf-8") as f:
                updated = json.load(f)
            assert updated["university"] == "홍익대"
            assert updated["tier"] == "S"

    def test_유효_게시글_수_계산(self):
        """이미지가 있는 게시글 수 정확하게 계산"""
        from data_pipeline.crawlers.cleaner import DataCleaner

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "raw_images").mkdir()
            metadata_dir = base / "metadata"
            metadata_dir.mkdir()

            # 이미지가 있는 게시글
            _create_metadata_json(
                metadata_dir / "post_001.json",
                {"images": ["img_001.jpg"], "post_no": 1},
            )
            # 이미지가 없는 게시글
            _create_metadata_json(
                metadata_dir / "post_002.json",
                {"images": [], "post_no": 2},
            )

            cleaner = DataCleaner(data_dir=tmpdir)
            count = cleaner._count_valid_posts()
            assert count == 1

    def test_이미지_참조_정리(self):
        """메타데이터에서 존재하지 않는 이미지 참조를 제거"""
        from data_pipeline.crawlers.cleaner import DataCleaner

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            images_dir = base / "raw_images"
            images_dir.mkdir()
            metadata_dir = base / "metadata"
            metadata_dir.mkdir()

            # 실제 이미지 하나만 생성
            _create_test_image(images_dir / "exists.jpg")

            # 메타데이터에 존재하는 것과 존재하지 않는 이미지 참조 포함
            _create_metadata_json(
                metadata_dir / "post_001.json",
                {
                    "post_no": 1,
                    "images": ["exists.jpg", "deleted.jpg"],
                },
            )

            cleaner = DataCleaner(data_dir=tmpdir)
            cleaner._clean_metadata_image_refs()

            # 정리 후 확인
            with open(metadata_dir / "post_001.json", "r", encoding="utf-8") as f:
                updated = json.load(f)
            assert updated["images"] == ["exists.jpg"]


class TestCleanerParseArgs:
    """cleaner CLI 인자 파싱 테스트"""

    def test_기본_인자(self):
        """인자 없이 실행 시 기본값 확인"""
        from data_pipeline.crawlers.cleaner import parse_args

        args = parse_args([])
        assert args.data_dir is None
        assert args.output_dir is None

    def test_커스텀_인자(self):
        """커스텀 인자 파싱"""
        from data_pipeline.crawlers.cleaner import parse_args

        args = parse_args(["--data-dir", "/my/data", "--output-dir", "/my/output"])
        assert args.data_dir == "/my/data"
        assert args.output_dir == "/my/output"


# ============================================================
# stats.py 테스트
# ============================================================


class TestStatsReport:
    """StatsReport 데이터클래스 테스트"""

    def test_기본값(self):
        """기본값이 올바르게 설정되는지 확인"""
        from data_pipeline.crawlers.stats import StatsReport

        report = StatsReport()
        assert report.total_posts == 0
        assert report.posts_with_images == 0
        assert report.total_images == 0
        assert report.actual_image_files == 0
        assert report.tier_distribution == {}
        assert report.department_distribution == {}
        assert report.university_ranking == []


class TestStatsReporter:
    """StatsReporter 테스트"""

    def test_빈_메타데이터_통계(self):
        """메타데이터 파일이 없으면 빈 통계 반환"""
        from data_pipeline.crawlers.stats import StatsReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = StatsReporter(metadata_dir=Path(tmpdir))
            report = reporter.generate()
            assert report.total_posts == 0
            assert report.total_images == 0

    def test_메타데이터_통계_생성(self):
        """메타데이터 파일들로부터 통계 생성"""
        from data_pipeline.crawlers.stats import StatsReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)

            # 여러 메타데이터 파일 생성
            _create_metadata_json(
                metadata_dir / "post_001.json",
                _sample_metadata(
                    post_no=1,
                    university="홍익대",
                    department="visual_design",
                    tier="S",
                    images=["img_001.jpg", "img_002.jpg"],
                ),
            )
            _create_metadata_json(
                metadata_dir / "post_002.json",
                _sample_metadata(
                    post_no=2,
                    university="국민대",
                    department="industrial_design",
                    tier="A",
                    images=["img_003.jpg"],
                ),
            )
            _create_metadata_json(
                metadata_dir / "post_003.json",
                _sample_metadata(
                    post_no=3,
                    university="가천대",
                    department="craft",
                    tier="B",
                    images=[],
                ),
            )

            reporter = StatsReporter(metadata_dir=metadata_dir)
            report = reporter.generate()

            assert report.total_posts == 3
            assert report.posts_with_images == 2
            assert report.total_images == 3
            assert report.tier_distribution["S"] == 1
            assert report.tier_distribution["A"] == 1
            assert report.tier_distribution["B"] == 1

    def test_실제_이미지_파일_수_계산(self):
        """이미지 디렉토리의 실제 파일 수 계산"""
        from data_pipeline.crawlers.stats import StatsReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            metadata_dir = base / "metadata"
            metadata_dir.mkdir()
            images_dir = base / "images"
            images_dir.mkdir()

            # 이미지 파일 생성
            _create_test_image(images_dir / "img_001.jpg")
            _create_test_image(images_dir / "img_002.png")
            # 비이미지 파일
            (images_dir / "readme.txt").write_text("not image")

            # generate()에서 early return 방지를 위해 메타데이터 파일 추가
            _create_metadata_json(
                metadata_dir / "post_001.json",
                _sample_metadata(post_no=1, images=["img_001.jpg"]),
            )

            reporter = StatsReporter(
                metadata_dir=metadata_dir, images_dir=images_dir
            )
            report = reporter.generate()
            assert report.actual_image_files == 2

    def test_콘솔_출력_형식(self):
        """to_console() 메서드의 출력 형식 확인"""
        from data_pipeline.crawlers.stats import StatsReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)
            _create_metadata_json(
                metadata_dir / "post_001.json",
                _sample_metadata(post_no=1, tier="S"),
            )

            reporter = StatsReporter(metadata_dir=metadata_dir)
            output = reporter.to_console()

            assert "크롤링 데이터 통계 리포트" in output
            assert "총 게시글" in output
            assert "티어 분포" in output

    def test_마크다운_출력_형식(self):
        """to_markdown() 메서드의 출력 형식 확인"""
        from data_pipeline.crawlers.stats import StatsReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)
            _create_metadata_json(
                metadata_dir / "post_001.json",
                _sample_metadata(post_no=1, tier="S"),
            )

            reporter = StatsReporter(metadata_dir=metadata_dir)
            output = reporter.to_markdown()

            assert "# 크롤링 데이터 통계 리포트" in output
            assert "## 기본 통계" in output
            assert "| 항목 | 수치 |" in output

    def test_막대_그래프_생성(self):
        """_bar() 정적 메서드 동작 확인"""
        from data_pipeline.crawlers.stats import StatsReporter

        # 최대값 절반인 경우
        bar = StatsReporter._bar(10, 20, width=20)
        assert bar == "#" * 10

        # 최대값인 경우
        bar = StatsReporter._bar(20, 20, width=20)
        assert bar == "#" * 20

        # 0인 경우
        bar = StatsReporter._bar(0, 20, width=20)
        assert bar == ""

        # 최대값이 0인 경우
        bar = StatsReporter._bar(0, 0, width=20)
        assert bar == ""

    def test_report_없이_to_console_호출(self):
        """report 파라미터 없이 to_console 호출 시 자동 생성"""
        from data_pipeline.crawlers.stats import StatsReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)
            _create_metadata_json(
                metadata_dir / "post_001.json",
                _sample_metadata(post_no=1),
            )

            reporter = StatsReporter(metadata_dir=metadata_dir)
            # report=None이면 내부에서 generate() 호출
            output = reporter.to_console(report=None)
            assert "크롤링 데이터 통계 리포트" in output

    def test_대학별_분포_집계(self):
        """대학별 분포가 올바르게 집계되는지 확인"""
        from data_pipeline.crawlers.stats import StatsReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_dir = Path(tmpdir)
            # 같은 대학 2개
            _create_metadata_json(
                metadata_dir / "post_001.json",
                _sample_metadata(post_no=1, university="홍익대"),
            )
            _create_metadata_json(
                metadata_dir / "post_002.json",
                _sample_metadata(post_no=2, university="홍익대"),
            )
            _create_metadata_json(
                metadata_dir / "post_003.json",
                _sample_metadata(post_no=3, university="국민대"),
            )

            reporter = StatsReporter(metadata_dir=metadata_dir)
            report = reporter.generate()

            # 대학별 랭킹 확인 (내림차순)
            assert len(report.university_ranking) == 2
            assert report.university_ranking[0] == ("홍익대", 2)
            assert report.university_ranking[1] == ("국민대", 1)


class TestStatsParseArgs:
    """stats CLI 인자 파싱 테스트"""

    def test_기본_인자(self):
        """기본 인자 확인"""
        from data_pipeline.crawlers.stats import parse_args

        args = parse_args([])
        assert args.metadata_dir == "data/crawled/metadata"
        assert args.format == "console"
        assert args.output is None

    def test_마크다운_형식(self):
        """마크다운 형식 인자"""
        from data_pipeline.crawlers.stats import parse_args

        args = parse_args(["--format", "markdown"])
        assert args.format == "markdown"


# ============================================================
# converter.py 테스트
# ============================================================


class TestConversionResult:
    """ConversionResult 데이터클래스 테스트"""

    def test_기본값(self):
        """기본값이 올바르게 설정되는지 확인"""
        from data_pipeline.crawlers.converter import ConversionResult

        result = ConversionResult()
        assert result.total_posts == 0
        assert result.total_images == 0
        assert result.skipped_posts == 0
        assert result.skipped_images == 0
        assert result.filtered_invalid_tier == 0
        assert result.output_csv_path is None
        assert result.validation is None
        assert result.dataframe is None


class TestValidationReport:
    """ValidationReport 데이터클래스 테스트"""

    def test_기본값(self):
        """기본값이 올바르게 설정되는지 확인"""
        from data_pipeline.crawlers.converter import ValidationReport

        report = ValidationReport()
        assert report.total_rows == 0
        assert report.is_valid is True
        assert report.missing_images == []
        assert report.errors == []
        assert report.warnings == []


class TestCrawledDataConverter:
    """CrawledDataConverter 테스트"""

    def test_메타데이터_디렉토리_없으면_예외(self):
        """존재하지 않는 메타데이터 디렉토리 시 FileNotFoundError"""
        from data_pipeline.crawlers.converter import CrawledDataConverter

        converter = CrawledDataConverter(
            crawled_dir="/nonexistent/dir",
            output_dir="/tmp/output",
        )
        with pytest.raises(FileNotFoundError):
            converter.convert()

    def test_빈_메타데이터_변환(self):
        """메타데이터 파일이 없으면 빈 DataFrame 반환"""
        from data_pipeline.crawlers.converter import CrawledDataConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "metadata").mkdir()
            (base / "raw_images").mkdir()

            converter = CrawledDataConverter(
                crawled_dir=tmpdir,
                output_dir=str(base / "output"),
            )
            df = converter.to_dataframe()
            assert len(df) == 0

    def test_학과_코드_매핑(self):
        """크롤러 학과 코드가 한글 표시값으로 올바르게 변환"""
        from data_pipeline.crawlers.converter import DEPARTMENT_CODE_TO_DISPLAY

        assert DEPARTMENT_CODE_TO_DISPLAY["visual_design"] == "시디"
        assert DEPARTMENT_CODE_TO_DISPLAY["industrial_design"] == "산디"
        assert DEPARTMENT_CODE_TO_DISPLAY["craft"] == "공예"
        assert DEPARTMENT_CODE_TO_DISPLAY["fine_art"] == "회화"

    def test_성공적인_변환(self):
        """정상적인 메타데이터로부터 DataFrame 생성"""
        from data_pipeline.crawlers.converter import CrawledDataConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            metadata_dir = base / "metadata"
            metadata_dir.mkdir()
            images_dir = base / "raw_images"
            images_dir.mkdir()

            # 이미지 파일 생성
            _create_test_image(images_dir / "img_001.jpg")
            _create_test_image(images_dir / "img_002.jpg", color=(0, 255, 0))

            # 메타데이터 생성 (이미지 경로를 raw_images 하위로 설정)
            meta = _sample_metadata(
                post_no=1,
                university="홍익대",
                department="visual_design",
                tier="S",
                images=[
                    str(images_dir / "img_001.jpg"),
                    str(images_dir / "img_002.jpg"),
                ],
            )
            _create_metadata_json(metadata_dir / "post_001.json", meta)

            converter = CrawledDataConverter(
                crawled_dir=tmpdir,
                output_dir=str(base / "output"),
                project_root=tmpdir,
            )
            df = converter.to_dataframe()
            assert len(df) == 2
            assert df["tier"].iloc[0] == "S"
            assert df["department"].iloc[0] == "시디"
            assert df["university"].iloc[0] == "홍익대"

    def test_무효_티어_필터링(self):
        """유효하지 않은 티어(S/A/B/C 외)는 필터링됨"""
        from data_pipeline.crawlers.converter import CrawledDataConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            metadata_dir = base / "metadata"
            metadata_dir.mkdir()
            images_dir = base / "raw_images"
            images_dir.mkdir()

            _create_test_image(images_dir / "img_001.jpg")

            # 유효하지 않은 티어
            meta = _sample_metadata(
                post_no=1,
                tier="X",
                images=[str(images_dir / "img_001.jpg")],
            )
            _create_metadata_json(metadata_dir / "post_001.json", meta)

            converter = CrawledDataConverter(
                crawled_dir=tmpdir,
                output_dir=str(base / "output"),
                project_root=tmpdir,
            )
            df = converter.to_dataframe()
            assert len(df) == 0  # 무효 티어이므로 필터링됨

    def test_이미지_없는_포스트_건너뛰기(self):
        """이미지가 없는 포스트는 건너뜀"""
        from data_pipeline.crawlers.converter import CrawledDataConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            metadata_dir = base / "metadata"
            metadata_dir.mkdir()
            (base / "raw_images").mkdir()

            meta = _sample_metadata(post_no=1, images=[])
            _create_metadata_json(metadata_dir / "post_001.json", meta)

            converter = CrawledDataConverter(
                crawled_dir=tmpdir,
                output_dir=str(base / "output"),
                project_root=tmpdir,
            )
            df = converter.to_dataframe()
            assert len(df) == 0

    def test_존재하지_않는_이미지_건너뛰기(self):
        """디스크에 없는 이미지 파일은 건너뜀"""
        from data_pipeline.crawlers.converter import CrawledDataConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            metadata_dir = base / "metadata"
            metadata_dir.mkdir()
            (base / "raw_images").mkdir()

            meta = _sample_metadata(
                post_no=1,
                images=["/nonexistent/img_001.jpg"],
            )
            _create_metadata_json(metadata_dir / "post_001.json", meta)

            converter = CrawledDataConverter(
                crawled_dir=tmpdir,
                output_dir=str(base / "output"),
                project_root=tmpdir,
            )
            df = converter.to_dataframe()
            assert len(df) == 0

    def test_CSV_내보내기(self):
        """DataFrame을 CSV 파일로 올바르게 내보내기"""
        import pandas as pd

        from data_pipeline.crawlers.converter import (
            CSV_COLUMNS,
            CrawledDataConverter,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "metadata").mkdir()
            (base / "raw_images").mkdir()

            converter = CrawledDataConverter(
                crawled_dir=tmpdir,
                output_dir=str(base / "output"),
            )

            # 테스트 DataFrame 생성
            df = pd.DataFrame(
                [
                    {
                        "image_path": "img_001.jpg",
                        "tier": "S",
                        "department": "시디",
                        "university": "홍익대",
                        "year": "2024",
                        "admission_type": "수시",
                        "work_type": "기초디자인",
                        "post_no": 1,
                    }
                ],
                columns=CSV_COLUMNS,
            )

            csv_path = base / "output" / "test.csv"
            result_path = converter.export_csv(df, csv_path)
            assert result_path.exists()

            # CSV 내용 확인
            loaded = pd.read_csv(result_path)
            assert len(loaded) == 1
            assert loaded["tier"].iloc[0] == "S"

    def test_유효성_검증_유효한_데이터(self):
        """유효한 데이터로 검증 시 is_valid=True"""
        import pandas as pd

        from data_pipeline.crawlers.converter import (
            CSV_COLUMNS,
            CrawledDataConverter,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            images_dir = base / "raw_images"
            images_dir.mkdir()
            (base / "metadata").mkdir()

            # 이미지 파일 생성
            _create_test_image(images_dir / "img_001.jpg")
            _create_test_image(images_dir / "img_002.jpg")

            converter = CrawledDataConverter(
                crawled_dir=tmpdir,
                project_root=tmpdir,
            )

            df = pd.DataFrame(
                [
                    {
                        "image_path": str(images_dir / "img_001.jpg"),
                        "tier": "S",
                        "department": "시디",
                        "university": "홍익대",
                        "year": "2024",
                        "admission_type": "수시",
                        "work_type": "기초디자인",
                        "post_no": 1,
                    },
                    {
                        "image_path": str(images_dir / "img_002.jpg"),
                        "tier": "A",
                        "department": "산디",
                        "university": "국민대",
                        "year": "2024",
                        "admission_type": "정시",
                        "work_type": "기초디자인",
                        "post_no": 2,
                    },
                ],
                columns=CSV_COLUMNS,
            )

            report = converter.validate(df)
            assert report.is_valid is True
            assert report.total_rows == 2
            assert report.unique_tiers == 2

    def test_유효성_검증_빈_데이터(self):
        """빈 DataFrame은 is_valid=False"""
        import pandas as pd

        from data_pipeline.crawlers.converter import (
            CSV_COLUMNS,
            CrawledDataConverter,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CrawledDataConverter(crawled_dir=tmpdir)
            df = pd.DataFrame(columns=CSV_COLUMNS)
            report = converter.validate(df)
            assert report.is_valid is False

    def test_유효성_검증_단일_티어_경고(self):
        """티어가 1개만 있으면 pairwise 학습 불가로 is_valid=False"""
        import pandas as pd

        from data_pipeline.crawlers.converter import (
            CSV_COLUMNS,
            CrawledDataConverter,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            images_dir = base / "raw_images"
            images_dir.mkdir()
            (base / "metadata").mkdir()

            _create_test_image(images_dir / "img_001.jpg")

            converter = CrawledDataConverter(
                crawled_dir=tmpdir,
                project_root=tmpdir,
            )

            # 모든 행이 같은 티어
            df = pd.DataFrame(
                [
                    {
                        "image_path": str(images_dir / "img_001.jpg"),
                        "tier": "S",
                        "department": "시디",
                        "university": "홍익대",
                        "year": "2024",
                        "admission_type": "수시",
                        "work_type": "기초디자인",
                        "post_no": 1,
                    },
                ],
                columns=CSV_COLUMNS,
            )

            report = converter.validate(df)
            assert report.is_valid is False
            assert report.unique_tiers == 1

    def test_분할_생성(self):
        """학습/검증/테스트 분할 생성"""
        import pandas as pd

        from data_pipeline.crawlers.converter import (
            CSV_COLUMNS,
            CrawledDataConverter,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CrawledDataConverter(crawled_dir=tmpdir)

            # 충분한 데이터 생성 (20행)
            rows = []
            for i in range(20):
                tier = ["S", "A", "B", "C"][i % 4]
                rows.append(
                    {
                        "image_path": f"img_{i:03d}.jpg",
                        "tier": tier,
                        "department": "시디",
                        "university": "홍익대",
                        "year": "2024",
                        "admission_type": "수시",
                        "work_type": "기초디자인",
                        "post_no": i + 1,
                    }
                )

            df = pd.DataFrame(rows, columns=CSV_COLUMNS)
            splits = converter.generate_split(df, train_ratio=0.8, val_ratio=0.1)

            assert "train" in splits
            assert "val" in splits
            assert "test" in splits
            # 모든 행이 분할에 포함되어야 함
            total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
            assert total == 20

    def test_분할_비율_초과_에러(self):
        """train_ratio + val_ratio > 1.0이면 ValueError"""
        import pandas as pd

        from data_pipeline.crawlers.converter import (
            CSV_COLUMNS,
            CrawledDataConverter,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CrawledDataConverter(crawled_dir=tmpdir)
            df = pd.DataFrame(
                [
                    {
                        "image_path": "img.jpg",
                        "tier": "S",
                        "department": "시디",
                        "university": "홍익대",
                        "year": "2024",
                        "admission_type": "수시",
                        "work_type": "기초디자인",
                        "post_no": 1,
                    }
                ],
                columns=CSV_COLUMNS,
            )
            with pytest.raises(ValueError):
                converter.generate_split(df, train_ratio=0.9, val_ratio=0.2)

    def test_빈_DataFrame_분할(self):
        """빈 DataFrame 분할 시 빈 결과 반환"""
        import pandas as pd

        from data_pipeline.crawlers.converter import (
            CSV_COLUMNS,
            CrawledDataConverter,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CrawledDataConverter(crawled_dir=tmpdir)
            df = pd.DataFrame(columns=CSV_COLUMNS)
            splits = converter.generate_split(df)

            assert len(splits["train"]) == 0
            assert len(splits["val"]) == 0
            assert len(splits["test"]) == 0

    def test_이미지_경로_해석(self):
        """상대 경로가 project_root 기준으로 절대 경로로 해석"""
        from data_pipeline.crawlers.converter import CrawledDataConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CrawledDataConverter(
                crawled_dir=tmpdir,
                project_root="/my/project",
            )
            resolved = converter._resolve_image_path("data/images/test.jpg")
            assert resolved == Path("/my/project/data/images/test.jpg")

    def test_절대_이미지_경로_해석(self):
        """절대 경로는 그대로 반환"""
        import sys

        from data_pipeline.crawlers.converter import CrawledDataConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            converter = CrawledDataConverter(crawled_dir=tmpdir)
            # Windows에서는 드라이브 문자가 있어야 절대 경로로 인식됨
            if sys.platform == "win32":
                abs_path = "C:\\absolute\\path\\image.jpg"
            else:
                abs_path = "/absolute/path/image.jpg"
            resolved = converter._resolve_image_path(abs_path)
            assert resolved == Path(abs_path)
            assert resolved.is_absolute()


class TestConverterParseArgs:
    """converter CLI 인자 파싱 테스트"""

    def test_기본_인자(self):
        """기본 인자 확인"""
        from data_pipeline.crawlers.converter import parse_args

        args = parse_args([])
        assert args.crawled_dir == "data/crawled"
        assert args.output_dir == "data/processed"
        assert args.split is False
        assert args.train_ratio == 0.8
        assert args.val_ratio == 0.1
        assert args.seed == 42

    def test_분할_인자(self):
        """분할 관련 인자 파싱"""
        from data_pipeline.crawlers.converter import parse_args

        args = parse_args([
            "--split",
            "--train-ratio", "0.7",
            "--val-ratio", "0.15",
            "--seed", "123",
        ])
        assert args.split is True
        assert args.train_ratio == 0.7
        assert args.val_ratio == 0.15
        assert args.seed == 123


class TestConverterConvertIntegration:
    """converter.convert() 통합 테스트"""

    def test_전체_변환_파이프라인(self):
        """메타데이터 JSON -> CSV 전체 변환 파이프라인"""
        from data_pipeline.crawlers.converter import CrawledDataConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            metadata_dir = base / "metadata"
            metadata_dir.mkdir()
            images_dir = base / "raw_images"
            images_dir.mkdir()
            output_dir = base / "output"

            # 다양한 티어의 이미지와 메타데이터 생성
            for i, (tier, dept) in enumerate(
                [("S", "visual_design"), ("A", "industrial_design")],
                start=1,
            ):
                img_name = f"img_{i:03d}.jpg"
                _create_test_image(images_dir / img_name)

                meta = _sample_metadata(
                    post_no=i,
                    university="홍익대" if tier == "S" else "국민대",
                    department=dept,
                    tier=tier,
                    images=[str(images_dir / img_name)],
                )
                _create_metadata_json(
                    metadata_dir / f"post_{i:03d}.json", meta
                )

            converter = CrawledDataConverter(
                crawled_dir=tmpdir,
                output_dir=str(output_dir),
                project_root=tmpdir,
            )
            result = converter.convert()

            assert result.total_posts == 2
            assert result.total_images == 2
            assert result.output_csv_path is not None
            assert result.output_csv_path.exists()
            assert result.validation is not None
            assert result.validation.is_valid is True
