"""
Context-based PII detector.
Detects sensitive fields by looking for label keywords in the surrounding text.

Useful for catching:
  - Password fields (label: "Password:", "Enter password")
  - Date of birth fields (label: "DOB:", "Date of Birth")
  - Address fields (label: "Address:", "Street:")
  - Bank account fields (label: "Account No:", "Acc No:")
  - Aadhaar/PAN/IFSC (label context confirmation)
  - Credit card fields (label: "Card Number:")

Strategy:
  For each line in the text, check if it contains a context keyword.
  If yes, extract the rest of that line (after the keyword) as the
  sensitive value, and emit a PIIDetection.

  Window-based: also scans a small window of characters after the keyword
  for the actual value (handles "Password:   secretvalue" on the same line).
"""

from __future__ import annotations

import re
from typing import List

from server.ocr_pii.schemas import DetectionSource, PIICategory, PIIDetection
from .base_detector import BaseDetector
from .patterns import CONTEXT_KEYWORDS

# Characters to look ahead after a keyword for the value.
# Kept short so we don't capture entire sentences.
_LOOKAHEAD = 40

# Minimum length for a detected value to be meaningful
_MIN_VALUE_LEN = 3

# Maximum number of whitespace-separated words in a context-captured value.
# Reduced to 4 to prevent capturing OCR noise and section headers.
_MAX_VALUE_WORDS = 4


class ContextDetector(BaseDetector):
    """
    Context-based PII detector.

    Scans text line-by-line for label keywords, then captures
    the text that follows as the sensitive value.
    """

    @property
    def detector_name(self) -> str:
        return "context"

    def detect(self, text: str) -> List[PIIDetection]:
        """
        Scan text for label keywords and extract following values.

        Args:
            text: Full OCR-extracted text.

        Returns:
            List of PIIDetection for contextually identified PII.
        """
        if not text.strip():
            return []

        detections: List[PIIDetection] = []

        for category_key, keywords in CONTEXT_KEYWORDS.items():
            category = PIICategory(category_key)
            for keyword in keywords:
                detections.extend(
                    self._scan_for_keyword(text, keyword, category)
                )

        # Deduplicate by (char_start, char_end) to avoid the same span
        # being reported from multiple keyword hits
        seen: set[tuple[int, int]] = set()
        unique: List[PIIDetection] = []
        for d in detections:
            key = (d.char_start, d.char_end)
            if key not in seen:
                seen.add(key)
                unique.append(d)

        return unique

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_for_keyword(
        self,
        text: str,
        keyword: str,
        category: PIICategory,
    ) -> List[PIIDetection]:
        """
        Find all occurrences of `keyword` in text (case-insensitive).
        For each occurrence, capture the value that follows.
        """
        detections: List[PIIDetection] = []

        # Build a pattern: keyword followed by optional separator and value.
        # Separator allows: colon, equals, hyphen, spaces, and the word "is"
        # (handles "password is secretvalue" → captures "secretvalue").
        # Value capture stops at newline, sentence-ending punctuation,
        # or OCR noise characters (<, >, {, }) which indicate a section break.
        escaped = re.escape(keyword)
        pattern = re.compile(
            rf"\b{escaped}\b"
            r"(?:\s+is)?"         # optional "is" word (e.g. "password is")
            r"[\s:=\-]*"          # optional separators
            r"([^\n\r.!?<>{{}}]{{1,{}}})".format(_LOOKAHEAD),
            re.IGNORECASE,
        )

        for m in pattern.finditer(text):
            value = m.group(1).strip()

            # Strip trailing punctuation / extra whitespace
            value = re.sub(r"[\s,;.]+$", "", value)

            if len(value) < _MIN_VALUE_LEN:
                continue

            # Reject values that are too long (sentence captures)
            if len(value.split()) > _MAX_VALUE_WORDS:
                continue

            # char positions of the value (group 1), not the full match
            value_start = m.start(1)
            value_end = value_start + len(value)

            detections.append(PIIDetection(
                text=value,
                category=category,
                confidence=0.75,  # context-based: moderate confidence
                source=DetectionSource.CONTEXT,
                bounding_box=None,
                char_start=value_start,
                char_end=value_end,
            ))

        return detections
