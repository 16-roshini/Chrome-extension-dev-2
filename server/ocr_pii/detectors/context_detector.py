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
  - Person name fields (label: "Full Name:", "Name:")

Strategy:
  For each line in the text, check if it contains a context keyword.
  If yes, extract the rest of that line (after the keyword) as the
  sensitive value, and emit a PIIDetection.

  Window-based: also scans a small window of characters after the keyword
  for the actual value (handles "Password:   secretvalue" on the same line).

  For person_name: capture stops at the next field separator (colon) to
  avoid absorbing adjacent label text when OCR flattens fields to one line.
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

# Categories where value capture must stop at a colon (next field label).
# On single-line OCR output, colons mark the boundary between fields.
_STOP_AT_COLON: frozenset[str] = frozenset({
    "person_name",
    "password",
})


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
            stop_at_colon = category_key in _STOP_AT_COLON
            for keyword in keywords:
                detections.extend(
                    self._scan_for_keyword(text, keyword, category, stop_at_colon)
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
        stop_at_colon: bool = False,
    ) -> List[PIIDetection]:
        """
        Find all occurrences of `keyword` in text (case-insensitive).
        For each occurrence, capture the value that follows.

        Args:
            stop_at_colon: When True, value capture stops at the first colon
                           after the keyword. This prevents absorbing the next
                           field label when OCR flattens multiple fields onto
                           one line (e.g. "Full Name : Ravi Kumar Date of Birth : ...").
        """
        detections: List[PIIDetection] = []

        escaped = re.escape(keyword)

        if stop_at_colon:
            # Standard pattern — then we post-trim at label boundaries
            pattern = re.compile(
                rf"\b{escaped}\b"
                r"(?:\s+is)?"
                r"[\s:=\-]*"
                r"([^\n\r.!?<>{{}}]{{1,{}}})".format(_LOOKAHEAD),
                re.IGNORECASE,
            )
        else:
            # Standard capture — stops at sentence boundaries
            pattern = re.compile(
                rf"\b{escaped}\b"
                r"(?:\s+is)?"
                r"[\s:=\-]*"
                r"([^\n\r.!?<>{{}}]{{1,{}}})".format(_LOOKAHEAD),
                re.IGNORECASE,
            )

        for m in pattern.finditer(text):
            value = m.group(1).strip()

            # Strip trailing punctuation / extra whitespace
            value = re.sub(r"[\s,;.]+$", "", value)

            # For categories that must stop at label boundaries:
            # truncate at any word followed by " :" (next field label).
            # e.g. "Ravi Kumar Date of Birth :" → stop at "Kumar"
            # e.g. "MySecret123 CONTACT DETAILS Email :" → stop at "MySecret123"
            if stop_at_colon:
                # Stop at a SINGLE word immediately followed by space+colon
                label_boundary = re.search(r'\s+\w+\s+:', value)
                if label_boundary:
                    value = value[:label_boundary.start()].strip()
                # Also stop at all-caps section headers (CONTACT, LOGIN, etc.)
                allcaps_boundary = re.search(r'\s+[A-Z]{4,}', value)
                if allcaps_boundary:
                    value = value[:allcaps_boundary.start()].strip()
                # For person_name: cap at 3 tokens maximum
                if category == PIICategory.PERSON_NAME:
                    tokens = value.split()
                    if len(tokens) > 3:
                        value = " ".join(tokens[:3])
                    # Also reject if any token is a date-related word
                    _DATE_WORDS = frozenset({
                        "date", "of", "birth", "dob", "born",
                        "day", "month", "year", "age",
                    })
                    # Form field label words — stop capture at first occurrence
                    _FIELD_LABEL_WORDS = frozenset({
                        "employer", "employee", "job", "title", "designation",
                        "occupation", "department", "username", "password",
                        "email", "phone", "mobile", "address", "gender",
                        "nationality", "religion", "login", "contact",
                    })
                    tokens = value.split()
                    clean_tokens = []
                    for tok in tokens:
                        tok_l = tok.lower()
                        if tok_l in _DATE_WORDS or tok_l in _FIELD_LABEL_WORDS:
                            break
                        clean_tokens.append(tok)
                    value = " ".join(clean_tokens)

            value = re.sub(r"[\s,;.]+$", "", value)

            if len(value) < _MIN_VALUE_LEN:
                continue

            # Reject values that are too long (sentence captures)
            if len(value.split()) > _MAX_VALUE_WORDS:
                continue

            # For person_name: reject values that are clearly not names
            # (all-caps section headers, single common words, etc.)
            if category == PIICategory.PERSON_NAME:
                # Skip if value looks like a section header (all caps)
                if value.isupper() and len(value.split()) <= 2:
                    continue
                # Skip single-word values that start lowercase (likely OCR noise)
                if len(value.split()) == 1 and not value[0].isupper():
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
