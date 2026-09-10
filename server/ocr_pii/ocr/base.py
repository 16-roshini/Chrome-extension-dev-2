"""
Abstract base class for all OCR engines.
Every engine must implement `run()` and return an OCRResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from PIL import Image

from server.ocr_pii.schemas import OCRResult


class BaseOCREngine(ABC):
    """
    Abstract OCR engine.
    Subclasses: TesseractEngine, EasyOCREngine
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Human-readable engine identifier, e.g. 'tesseract' or 'easyocr'."""
        ...

    @abstractmethod
    def run(self, image: Image.Image) -> OCRResult:
        """
        Run OCR on a PIL Image.

        Args:
            image: PIL Image object (RGB or grayscale).

        Returns:
            OCRResult with words, full_text, and engine_used populated.
        """
        ...

    def is_available(self) -> bool:
        """
        Check whether this engine's dependencies are available at runtime.
        Override in subclasses if the engine may be absent (e.g. EasyOCR).
        """
        return True
