"""
Regex-based PII detector.
Handles all deterministic patterns: email, phone, Aadhaar, PAN,
credit card (with Luhn), IP, URL, IFSC, DOB, passport, bank account.
"""

from __future__ import annotations

import re
from typing import List

from server.ocr_pii.schemas import DetectionSource, PIICategory, PIIDetection
from .base_detector import BaseDetector
from .patterns import (
    AADHAAR,
    BANK_ACCOUNT,
    CREDIT_CARD,
    DOB,
    EMAIL,
    IFSC,
    IP_ADDRESS,
    PASSPORT,
    PAN,
    PHONE,
    URL,
    luhn_check,
)


class RegexDetector(BaseDetector):
    """
    Deterministic PII detection using compiled regex patterns.

    Each pattern produces PIIDetection objects with:
    - source = DetectionSource.REGEX
    - confidence based on pattern strength and optional validation
    """

    @property
    def detector_name(self) -> str:
        return "regex"

    def detect(self, text: str) -> List[PIIDetection]:
        """
        Scan text with all regex patterns and return detections.

        Args:
            text: Full OCR-extracted text.

        Returns:
            List of PIIDetection, one per match found.
        """
        detections: List[PIIDetection] = []

        detections.extend(self._detect_email(text))
        detections.extend(self._detect_phone(text))
        detections.extend(self._detect_aadhaar(text))
        detections.extend(self._detect_pan(text))
        detections.extend(self._detect_credit_card(text))
        detections.extend(self._detect_ip(text))
        detections.extend(self._detect_url(text))
        detections.extend(self._detect_ifsc(text))
        detections.extend(self._detect_dob(text))
        detections.extend(self._detect_passport(text))
        detections.extend(self._detect_bank_account(text))

        return detections

    # ------------------------------------------------------------------
    # Private helpers — one method per PII category
    # ------------------------------------------------------------------

    def _make_detection(
        self,
        match: re.Match,
        category: PIICategory,
        confidence: float,
    ) -> PIIDetection:
        """Build a PIIDetection from a regex match."""
        return PIIDetection(
            text=match.group(),
            category=category,
            confidence=confidence,
            source=DetectionSource.REGEX,
            bounding_box=None,  # regex has no spatial info
            char_start=match.start(),
            char_end=match.end(),
        )

    def _detect_email(self, text: str) -> List[PIIDetection]:
        return [
            self._make_detection(m, PIICategory.EMAIL, 0.97)
            for m in EMAIL.finditer(text)
        ]

    def _detect_phone(self, text: str) -> List[PIIDetection]:
        return [
            self._make_detection(m, PIICategory.PHONE, 0.90)
            for m in PHONE.finditer(text)
        ]

    def _detect_aadhaar(self, text: str) -> List[PIIDetection]:
        detections = []
        for m in AADHAAR.finditer(text):
            # Remove spaces/hyphens to count digits
            digits = re.sub(r"[\s\-]", "", m.group())
            if len(digits) == 12:
                detections.append(
                    self._make_detection(m, PIICategory.AADHAAR, 0.95)
                )
        return detections

    def _detect_pan(self, text: str) -> List[PIIDetection]:
        return [
            self._make_detection(m, PIICategory.PAN, 0.95)
            for m in PAN.finditer(text)
        ]

    def _detect_credit_card(self, text: str) -> List[PIIDetection]:
        detections = []
        for m in CREDIT_CARD.finditer(text):
            digits_only = re.sub(r"[\s\-]", "", m.group())
            if 13 <= len(digits_only) <= 19 and luhn_check(digits_only):
                detections.append(
                    self._make_detection(m, PIICategory.CREDIT_CARD, 0.93)
                )
        return detections

    def _detect_ip(self, text: str) -> List[PIIDetection]:
        return [
            self._make_detection(m, PIICategory.IP_ADDRESS, 0.95)
            for m in IP_ADDRESS.finditer(text)
        ]

    def _detect_url(self, text: str) -> List[PIIDetection]:
        return [
            self._make_detection(m, PIICategory.URL, 0.90)
            for m in URL.finditer(text)
        ]

    def _detect_ifsc(self, text: str) -> List[PIIDetection]:
        return [
            self._make_detection(m, PIICategory.IFSC, 0.95)
            for m in IFSC.finditer(text)
        ]

    def _detect_dob(self, text: str) -> List[PIIDetection]:
        return [
            self._make_detection(m, PIICategory.DATE_OF_BIRTH, 0.85)
            for m in DOB.finditer(text)
        ]

    def _detect_passport(self, text: str) -> List[PIIDetection]:
        return [
            self._make_detection(m, PIICategory.PASSPORT, 0.88)
            for m in PASSPORT.finditer(text)
        ]

    def _detect_bank_account(self, text: str) -> List[PIIDetection]:
        """
        Bank account detection requires BOTH:
          1. A 12–18 digit sequence (regex match)
          2. A context keyword nearby (within 80 chars before the match)

        This prevents false positives from QR codes, barcodes, and other
        digit sequences on ID cards that have no account-related label.

        Context keywords checked: account, acc, acct, bank
        """
        _ACCOUNT_CONTEXT = re.compile(
            r"\b(?:account|acc|acct|bank)\b",
            re.IGNORECASE,
        )
        detections = []
        for m in BANK_ACCOUNT.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if len(digits) < 12:
                continue
            # Check for a context keyword within 80 chars before the match
            window_start = max(0, m.start() - 80)
            window = text[window_start:m.start()]
            if not _ACCOUNT_CONTEXT.search(window):
                continue
            detections.append(
                self._make_detection(m, PIICategory.BANK_ACCOUNT, 0.70)
            )
        return detections
