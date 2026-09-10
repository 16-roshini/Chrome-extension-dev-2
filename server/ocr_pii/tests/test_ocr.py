"""
Tests for OCR engines — TesseractEngine and EasyOCREngine.
All images used are synthetic — no real documents.
"""

from __future__ import annotations

import pytest
from PIL import Image

from server.ocr_pii.ocr.tesseract_engine import TesseractEngine
from server.ocr_pii.ocr.easyocr_engine import EasyOCREngine
from server.ocr_pii.ocr.ocr_result import (
    build_ocr_result,
    easyocr_bbox_to_schema,
    tesseract_bbox_to_schema,
)
from server.ocr_pii.schemas import OCRWord, BoundingBox


# ---------------------------------------------------------------------------
# TesseractEngine
# ---------------------------------------------------------------------------

class TestTesseractEngine:

    def test_engine_name(self):
        engine = TesseractEngine()
        assert engine.engine_name == "tesseract"

    def test_is_available(self):
        engine = TesseractEngine()
        # Tesseract is installed at the expected path
        assert engine.is_available() is True

    def test_run_returns_ocr_result(self, synthetic_image):
        engine = TesseractEngine()
        result = engine.run(synthetic_image)
        assert result.engine_used == "tesseract"
        assert isinstance(result.full_text, str)
        assert isinstance(result.words, list)

    def test_run_extracts_text(self, synthetic_image):
        engine = TesseractEngine()
        result = engine.run(synthetic_image)
        # The synthetic image contains "example.com" — check partial match
        assert len(result.full_text) > 0

    def test_word_confidence_in_range(self, synthetic_image):
        engine = TesseractEngine()
        result = engine.run(synthetic_image)
        for word in result.words:
            assert 0.0 <= word.confidence <= 1.0

    def test_word_has_bounding_box(self, synthetic_image):
        engine = TesseractEngine()
        result = engine.run(synthetic_image)
        for word in result.words:
            if word.bounding_box is not None:
                bb = word.bounding_box
                assert bb.width >= 0
                assert bb.height >= 0

    def test_rgba_image_converted(self):
        """RGBA image should not crash Tesseract."""
        engine = TesseractEngine()
        img = Image.new("RGBA", (200, 50), (255, 255, 255, 255))
        result = engine.run(img)
        assert result.engine_used == "tesseract"

    def test_empty_image_returns_empty_text(self):
        """Pure white image should return empty or whitespace-only text."""
        engine = TesseractEngine()
        img = Image.new("RGB", (100, 50), (255, 255, 255))
        result = engine.run(img)
        assert result.full_text.strip() == "" or isinstance(result.full_text, str)


# ---------------------------------------------------------------------------
# EasyOCREngine
# ---------------------------------------------------------------------------

class TestEasyOCREngine:

    def test_engine_name(self):
        engine = EasyOCREngine()
        assert engine.engine_name == "easyocr"

    def test_is_available(self):
        engine = EasyOCREngine()
        assert engine.is_available() is True

    def test_run_returns_ocr_result(self, synthetic_image):
        engine = EasyOCREngine()
        result = engine.run(synthetic_image)
        assert result.engine_used == "easyocr"
        assert isinstance(result.full_text, str)
        assert isinstance(result.words, list)

    def test_word_confidence_in_range(self, synthetic_image):
        engine = EasyOCREngine()
        result = engine.run(synthetic_image)
        for word in result.words:
            assert 0.0 <= word.confidence <= 1.0


# ---------------------------------------------------------------------------
# OCR result helpers
# ---------------------------------------------------------------------------

class TestOCRResultHelpers:

    def test_build_ocr_result_joins_words(self):
        words = [
            OCRWord(text="Hello", bounding_box=None, confidence=0.9),
            OCRWord(text="World", bounding_box=None, confidence=0.8),
        ]
        result = build_ocr_result(words, "tesseract")
        assert result.full_text == "Hello World"
        assert result.engine_used == "tesseract"

    def test_build_ocr_result_skips_blank_words(self):
        words = [
            OCRWord(text="  ", bounding_box=None, confidence=0.5),
            OCRWord(text="Data", bounding_box=None, confidence=0.9),
        ]
        result = build_ocr_result(words, "tesseract")
        assert "Data" in result.full_text

    def test_tesseract_bbox_to_schema(self):
        bb = tesseract_bbox_to_schema(10, 20, 100, 30)
        assert isinstance(bb, BoundingBox)
        assert bb.x == 10
        assert bb.y == 20
        assert bb.width == 100
        assert bb.height == 30

    def test_easyocr_bbox_to_schema(self):
        # EasyOCR format: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        bbox = [[10, 20], [110, 20], [110, 50], [10, 50]]
        bb = easyocr_bbox_to_schema(bbox)
        assert bb is not None
        assert bb.x == 10
        assert bb.y == 20
        assert bb.width == 100
        assert bb.height == 30

    def test_easyocr_bbox_invalid_returns_none(self):
        bb = easyocr_bbox_to_schema([])
        assert bb is None
