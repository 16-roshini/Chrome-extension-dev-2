"""
Abstract base class for all PII detectors.
Every detector takes full_text and returns a list of PIIDetection objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from server.ocr_pii.schemas import PIIDetection


class BaseDetector(ABC):
    """
    Abstract PII detector.
    Subclasses: RegexDetector, NERDetector, ContextDetector
    """

    @property
    @abstractmethod
    def detector_name(self) -> str:
        """Human-readable name for this detector."""
        ...

    @abstractmethod
    def detect(self, text: str) -> List[PIIDetection]:
        """
        Run detection on a plain-text string.

        Args:
            text: The full OCR-extracted text to scan.

        Returns:
            List of PIIDetection objects found in the text.
            Returns empty list if nothing found.
        """
        ...
