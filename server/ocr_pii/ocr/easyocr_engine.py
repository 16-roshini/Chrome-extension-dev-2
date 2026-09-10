"""
EasyOCR engine (fallback).
Used when Tesseract is unavailable or returns low-confidence results.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from server.ocr_pii.schemas import OCRResult, OCRWord
from .base import BaseOCREngine
from .ocr_result import build_ocr_result, easyocr_bbox_to_schema

# EasyOCR is imported lazily so the module loads even if torch isn't warm yet
_reader = None


def _get_reader():
    """Lazily initialise the EasyOCR reader (downloads model on first use)."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


class EasyOCREngine(BaseOCREngine):
    """
    Fallback OCR engine using EasyOCR.

    EasyOCR returns (bbox, text, confidence) tuples per detected text block.
    Confidence is already in 0.0–1.0 range.
    """

    @property
    def engine_name(self) -> str:
        return "easyocr"

    def is_available(self) -> bool:
        """Check easyocr package is importable."""
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, image: Image.Image) -> OCRResult:
        """
        Run EasyOCR on a PIL Image.

        Converts PIL Image → numpy array, runs the reader,
        then maps results to OCRWord objects.

        Args:
            image: PIL Image.

        Returns:
            OCRResult populated with word/phrase-level data.
        """
        # Convert PIL → numpy array (EasyOCR expects numpy)
        image_np = np.array(image.convert("RGB"))

        reader = _get_reader()
        # detail=1 returns (bbox, text, confidence)
        results = reader.readtext(image_np, detail=1)

        words: list[OCRWord] = []
        for bbox, text, confidence in results:
            text = text.strip()
            if not text:
                continue

            bbox_schema = easyocr_bbox_to_schema(bbox)

            words.append(OCRWord(
                text=text,
                bounding_box=bbox_schema,
                confidence=float(confidence),
            ))

        return build_ocr_result(words, self.engine_name)
