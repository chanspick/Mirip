# -*- coding: utf-8 -*-
"""
데이터 파이프라인 테스트용 conftest.py

data_pipeline 모듈 테스트를 위한 pytest 설정 및 공통 fixtures
상위 conftest.py의 의존성을 피하기 위해 별도로 정의합니다.
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

# backend 디렉토리를 Python 경로에 추가
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


# ======================
# 테스트 이미지 fixtures
# ======================
@pytest.fixture
def sample_image_bytes() -> bytes:
    """샘플 이미지 바이트 (10x10 RGB PNG)"""
    # Pillow를 사용하여 실제 유효한 PNG 생성
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """샘플 JPEG 바이트 (Pillow로 생성된 유효한 JPEG)"""
    # Pillow를 사용하여 실제 유효한 JPEG 생성
    img = Image.new("RGB", (10, 10), color=(0, 128, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


# ======================
# 마커
# ======================
def pytest_configure(config):
    """pytest 마커 등록"""
    config.addinivalue_line(
        "markers",
        "slow: 느린 테스트 (GPU 필요 등)",
    )
    config.addinivalue_line(
        "markers",
        "integration: 통합 테스트 (외부 서비스 필요)",
    )
