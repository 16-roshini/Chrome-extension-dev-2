"""
Tesseract OCR engine (primary).
Uses pytesseract with the system Tesseract installation.
"""

from __future__ import annotations

import pytesseract
from PIL import Image

from server.ocr_pii.schemas import OCRResult, OCRWord
from .base import BaseOCREngine
from .ocr_result import build_ocr_result, tesseract_bbox_to_schema

# Hard-coded Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Tesseract config: OEM 3 (default, LSTM), PSM 3 (fully automatic page segmentation)
_TESSERACT_CONFIG = "--oem 3 --psm 3"


class TesseractEngine(BaseOCREngine):
    """
    Primary OCR engine using Tesseract via pytesseract.

    Returns word-level results with bounding boxes and confidence scores.
    Confidence from Tesseract is 0–100; we normalise it to 0.0–1.0.
    """

    @property
    def engine_name(self) -> str:
        return "tesseract"

    def is_available(self) -> bool:
        """Check Tesseract binary is accessible."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def run(self, image: Image.Image) -> OCRResult:
        """
        Run Tesseract on a PIL Image.

        Uses image_to_data() to get word-level text, confidence,
        and bounding boxes in one call.

        Args:
            image: PIL Image (will be converted to RGB if needed).

        Returns:
            OCRResult populated with word-level data.
        """
        # Ensure RGB — Tesseract handles RGB best
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # image_to_data returns a TSV-style dict with per-word info
        data = pytesseract.image_to_data(
            image,
            config=_TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
        )

        words: list[OCRWord] = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            word_text = data["text"][i].strip()
            if not word_text:
                continue  # skip empty tokens

            raw_conf = int(data["conf"][i])
            if raw_conf < 0:
                continue  # Tesseract returns -1 for non-text blocks

            confidence = raw_conf / 100.0  # normalise to 0.0–1.0

            bbox = tesseract_bbox_to_schema(
                left=data["left"][i],
                top=data["top"][i],
                width=data["width"][i],
                height=data["height"][i],
            )

            words.append(OCRWord(
                text=word_text,
                bounding_box=bbox,
                confidence=confidence,
            ))

        return build_ocr_result(words, self.engine_name)
