"""
OCR subpackage.
Provides Tesseract (primary) and EasyOCR (fallback) engines.
"""

from .tesseract_engine import TesseractEngine
from .easyocr_engine import EasyOCREngine
from .ocr_result import build_ocr_result

__all__ = ["TesseractEngine", "EasyOCREngine", "build_ocr_result"]
