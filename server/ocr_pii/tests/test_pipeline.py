"""
Integration tests for the full OCR + PII pipeline.
Tests the complete flow: image bytes → DetectResponse (5-field clean output).
All PII is SYNTHETIC — no real personal data.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from server.ocr_pii.pipeline import (
    OCRPIIPipeline,
    _md5_hash,
    _bytes_to_image,
    _compute_word_offsets,
    _merge_bboxes,
)
from server.ocr_pii.schemas import (
    DetectResponse,
    OCRResult,
    OCRWord,
    BoundingBox,
    PIIEntity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline() -> OCRPIIPipeline:
    return OCRPIIPipeline()


@pytest.fixture(scope="module")
def pii_image_bytes() -> bytes:
    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10),  "Email: test@example.com",  fill=(0, 0, 0))
    draw.text((10, 50),  "Phone: 9876543210",         fill=(0, 0, 0))
    draw.text((10, 90),  "PAN: ABCDE1234F",           fill=(0, 0, 0))
    draw.text((10, 130), "Password: TestPass@2024",   fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def empty_image_bytes() -> bytes:
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_md5_hash_returns_32_chars(self, pii_image_bytes):
        h = _md5_hash(pii_image_bytes)
        assert isinstance(h, str)
        assert len(h) == 32

    def test_md5_hash_deterministic(self, pii_image_bytes):
        assert _md5_hash(pii_image_bytes) == _md5_hash(pii_image_bytes)

    def test_md5_hash_different_images(self, pii_image_bytes, empty_image_bytes):
        assert _md5_hash(pii_image_bytes) != _md5_hash(empty_image_bytes)

    def test_bytes_to_image_returns_rgb(self, pii_image_bytes):
        img = _bytes_to_image(pii_image_bytes)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_compute_word_offsets_basic(self):
        words = [
            OCRWord(text="Hello", bounding_box=None, confidence=0.9),
            OCRWord(text="World", bounding_box=None, confidence=0.9),
        ]
        offsets = _compute_word_offsets("Hello World", words)
        assert offsets[0] == (0, 5)
        assert offsets[1] == (6, 11)

    def test_merge_bboxes_returns_enclosing_box(self):
        words = [
            OCRWord(text="a", bounding_box=BoundingBox(x=10, y=20, width=30, height=15), confidence=0.9),
            OCRWord(text="b", bounding_box=BoundingBox(x=50, y=22, width=40, height=12), confidence=0.9),
        ]
        bbox = _merge_bboxes(words)
        assert bbox == [10, 20, 90, 35]  # [min_x, min_y, max_x2, max_y2]

    def test_merge_bboxes_no_boxes_returns_none(self):
        words = [OCRWord(text="a", bounding_box=None, confidence=0.9)]
        assert _merge_bboxes(words) is None


# ---------------------------------------------------------------------------
# Pipeline — ocr_only
# ---------------------------------------------------------------------------

class TestPipelineOCROnly:

    @pytest.mark.asyncio
    async def test_ocr_only_returns_ocr_result(self, pipeline, pii_image_bytes):
        result = await pipeline.ocr_only(pii_image_bytes)
        assert isinstance(result, OCRResult)

    @pytest.mark.asyncio
    async def test_ocr_only_engine_is_valid(self, pipeline, pii_image_bytes):
        result = await pipeline.ocr_only(pii_image_bytes)
        assert result.engine_used in ("tesseract", "easyocr", "none")

    @pytest.mark.asyncio
    async def test_ocr_only_words_is_list(self, pipeline, pii_image_bytes):
        result = await pipeline.ocr_only(pii_image_bytes)
        assert isinstance(result.words, list)

    @pytest.mark.asyncio
    async def test_ocr_only_full_text_is_string(self, pipeline, pii_image_bytes):
        result = await pipeline.ocr_only(pii_image_bytes)
        assert isinstance(result.full_text, str)

    @pytest.mark.asyncio
    async def test_ocr_only_empty_image_no_crash(self, pipeline, empty_image_bytes):
        result = await pipeline.ocr_only(empty_image_bytes)
        assert isinstance(result, OCRResult)


# ---------------------------------------------------------------------------
# Pipeline — analyze → DetectResponse
# ---------------------------------------------------------------------------

class TestPipelineAnalyze:

    @pytest.mark.asyncio
    async def test_analyze_returns_detect_response(self, pipeline, pii_image_bytes):
        result = await pipeline.analyze(pii_image_bytes)
        assert isinstance(result, DetectResponse)

    @pytest.mark.asyncio
    async def test_analyze_detections_is_list(self, pipeline, pii_image_bytes):
        result = await pipeline.analyze(pii_image_bytes)
        assert isinstance(result.detections, list)

    @pytest.mark.asyncio
    async def test_analyze_each_detection_is_pii_entity(self, pipeline, pii_image_bytes):
        result = await pipeline.analyze(pii_image_bytes)
        for entity in result.detections:
            assert isinstance(entity, PIIEntity)

    @pytest.mark.asyncio
    async def test_analyze_entity_has_five_fields(self, pipeline, pii_image_bytes):
        result = await pipeline.analyze(pii_image_bytes)
        for entity in result.detections:
            assert hasattr(entity, "type")
            assert hasattr(entity, "text")
            assert hasattr(entity, "bbox")
            assert hasattr(entity, "ocr_confidence")
            assert hasattr(entity, "pii_score")

    @pytest.mark.asyncio
    async def test_analyze_type_is_uppercase_string(self, pipeline, pii_image_bytes):
        result = await pipeline.analyze(pii_image_bytes)
        for entity in result.detections:
            assert isinstance(entity.type, str)
            assert entity.type == entity.type.upper()

    @pytest.mark.asyncio
    async def test_analyze_pii_score_in_range(self, pipeline, pii_image_bytes):
        result = await pipeline.analyze(pii_image_bytes)
        for entity in result.detections:
            assert 0.0 <= entity.pii_score <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_ocr_confidence_in_range(self, pipeline, pii_image_bytes):
        result = await pipeline.analyze(pii_image_bytes)
        for entity in result.detections:
            assert 0.0 <= entity.ocr_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_text_is_nonempty_string(self, pipeline, pii_image_bytes):
        result = await pipeline.analyze(pii_image_bytes)
        for entity in result.detections:
            assert isinstance(entity.text, str)
            assert len(entity.text.strip()) > 0

    @pytest.mark.asyncio
    async def test_analyze_bbox_is_list_or_none(self, pipeline, pii_image_bytes):
        result = await pipeline.analyze(pii_image_bytes)
        for entity in result.detections:
            if entity.bbox is not None:
                assert isinstance(entity.bbox, list)
                assert len(entity.bbox) == 4
                x1, y1, x2, y2 = entity.bbox
                assert x2 >= x1
                assert y2 >= y1

    @pytest.mark.asyncio
    async def test_analyze_no_internal_fields_exposed(self, pipeline, pii_image_bytes):
        """Verify internal fields are not in the PIIEntity output."""
        result = await pipeline.analyze(pii_image_bytes)
        for entity in result.detections:
            assert not hasattr(entity, "source")
            assert not hasattr(entity, "char_start")
            assert not hasattr(entity, "char_end")
            assert not hasattr(entity, "category")
            assert not hasattr(entity, "confidence")

    @pytest.mark.asyncio
    async def test_analyze_no_duplicate_text_type_pairs(self, pipeline, pii_image_bytes):
        """Same text+type should not appear twice after deduplication."""
        result = await pipeline.analyze(pii_image_bytes)
        pairs = [(e.text.strip().lower(), e.type) for e in result.detections]
        assert len(pairs) == len(set(pairs)), f"Duplicate detections: {pairs}"

    @pytest.mark.asyncio
    async def test_analyze_empty_image_no_crash(self, pipeline, empty_image_bytes):
        result = await pipeline.analyze(empty_image_bytes)
        assert isinstance(result, DetectResponse)
        assert isinstance(result.detections, list)

    @pytest.mark.asyncio
    async def test_analyze_same_image_same_result(self, pipeline, pii_image_bytes):
        """Deterministic: same image produces same detections."""
        r1 = await pipeline.analyze(pii_image_bytes)
        r2 = await pipeline.analyze(pii_image_bytes)
        types1 = sorted(e.type for e in r1.detections)
        types2 = sorted(e.type for e in r2.detections)
        assert types1 == types2
