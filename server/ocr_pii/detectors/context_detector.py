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

# Per-category lookahead overrides.
# Organisation names can be long (e.g. "Indian Space Research Organisation")
# so the default 40-char lookahead is not enough.
_LOOKAHEAD_BY_CATEGORY: dict[str, int] = {
    "organization": 80,
    "address":      80,   # addresses are also long
}

# Minimum length for a detected value to be meaningful
_MIN_VALUE_LEN = 3

# Maximum number of whitespace-separated words in a context-captured value.
# Overridden per category below.
_MAX_VALUE_WORDS = 4

# Per-category max-word overrides.
_MAX_VALUE_WORDS_BY_CATEGORY: dict[str, int] = {
    "organization": 8,    # "Indian Space Research Organisation (ISRO)" = 5 words
    "address":      12,   # full address can be 8+ tokens
}

# Categories where value capture must stop at the next field boundary
# (colon or all-caps section header).
# On single-line OCR output, colons mark the boundary between fields.
_STOP_AT_COLON: frozenset[str] = frozenset({
    "person_name",
    "password",
    "organization",
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
            lookahead = _LOOKAHEAD_BY_CATEGORY.get(category_key, _LOOKAHEAD)
            max_words = _MAX_VALUE_WORDS_BY_CATEGORY.get(category_key, _MAX_VALUE_WORDS)
            for keyword in keywords:
                detections.extend(
                    self._scan_for_keyword(
                        text, keyword, category,
                        stop_at_colon=stop_at_colon,
                        lookahead=lookahead,
                        max_words=max_words,
                    )
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
        lookahead: int = _LOOKAHEAD,
        max_words: int = _MAX_VALUE_WORDS,
    ) -> List[PIIDetection]:
        """
        Find all occurrences of `keyword` in text (case-insensitive).
        For each occurrence, capture the value that follows.

        Args:
            stop_at_colon: When True, value capture stops at the next field
                           boundary (colon or all-caps section header).
            lookahead:     Max characters to capture after the keyword.
            max_words:     Max whitespace-separated words in the value.
        """
        detections: List[PIIDetection] = []

        escaped = re.escape(keyword)

        pattern = re.compile(
            rf"\b{escaped}\b"
            r"(?:\s+is)?"
            r"[\s:=\-]*"
            r"([^\n\r.!?<>{{}}]{{1,{}}})".format(lookahead),
            re.IGNORECASE,
        )

        for m in pattern.finditer(text):
            value = m.group(1).strip()

            # Strip trailing punctuation / extra whitespace
            value = re.sub(r"[\s,;.]+$", "", value)

            # ----------------------------------------------------------------
            # Field-boundary trimming for categories that need it
            # ----------------------------------------------------------------
            if stop_at_colon:
                # Stop at a single word immediately followed by space+colon
                # e.g. "Ravi Kumar Date :" → stop before "Date"
                label_boundary = re.search(r'\s+\w+\s+:', value)
                if label_boundary:
                    value = value[:label_boundary.start()].strip()
                # Stop at all-caps section headers (CONTACT, LOGIN, etc.)
                allcaps_boundary = re.search(r'\s+[A-Z]{4,}', value)
                if allcaps_boundary:
                    value = value[:allcaps_boundary.start()].strip()

                if category == PIICategory.PERSON_NAME:
                    # Cap person names at 3 tokens
                    tokens = value.split()
                    if len(tokens) > 3:
                        value = " ".join(tokens[:3])
                    # Stop at date/field-label words
                    _DATE_WORDS = frozenset({
                        "date", "of", "birth", "dob", "born",
                        "day", "month", "year", "age",
                    })
                    _FIELD_LABEL_WORDS = frozenset({
                        "employer", "employee", "job", "title", "designation",
                        "occupation", "department", "username", "password",
                        "email", "phone", "mobile", "address", "gender",
                        "nationality", "religion", "login", "contact",
                    })
                    tokens = value.split()
                    clean_tokens = []
                    for tok in tokens:
                        if tok.lower() in _DATE_WORDS or tok.lower() in _FIELD_LABEL_WORDS:
                            break
                        clean_tokens.append(tok)
                    value = " ".join(clean_tokens)

                if category == PIICategory.ORGANIZATION:
                    # Strip parenthetical abbreviations like "(ISRO)", "(NASA)"
                    value = re.sub(r'\s*\([A-Z]{2,8}\)\s*$', '', value).strip()
                    # Strip trailing field labels
                    _ORG_FIELD_LABELS = frozenset({
                        "job", "title", "designation", "department",
                        "position", "role", "employee", "contact",
                    })
                    org_tokens = value.split()
                    cut_at = len(org_tokens)
                    for i, tok in enumerate(org_tokens):
                        if tok.lower().rstrip(".:,") in _ORG_FIELD_LABELS:
                            cut_at = i
                            break
                    value = " ".join(org_tokens[:cut_at]).strip()
                    # Also strip trailing punctuation again after cleaning
                    value = re.sub(r"[\s,;.:]+$", "", value)

            value = re.sub(r"[\s,;.]+$", "", value)

            if len(value) < _MIN_VALUE_LEN:
                continue

            # Reject values that are too long
            if len(value.split()) > max_words:
                continue

            # ----------------------------------------------------------------
            # Category-specific quality gates
            # ----------------------------------------------------------------
            if category == PIICategory.PERSON_NAME:
                # Skip all-caps section headers
                if value.isupper() and len(value.split()) <= 2:
                    continue
                # Skip single-word values that start lowercase (OCR noise)
                if len(value.split()) == 1 and not value[0].isupper():
                    continue

            if category == PIICategory.ORGANIZATION:
                # Must contain at least one uppercase letter (real org name)
                if not any(c.isupper() for c in value):
                    continue
                # Reject obvious section headers (all-caps short words)
                if value.isupper() and len(value.split()) <= 2:
                    continue
                # Reject if the value is only a known UI/browser word
                _ORG_BLOCKLIST = frozenset({
                    "youtube", "github", "gmail", "google", "chatgpt",
                    "twitter", "facebook", "instagram", "linkedin",
                    "details", "information", "notes", "settings",
                })
                if value.lower().strip() in _ORG_BLOCKLIST:
                    continue

            # char positions of the value (group 1)
            value_start = m.start(1)
            value_end = value_start + len(value)

            detections.append(PIIDetection(
                text=value,
                category=category,
                confidence=0.80 if category == PIICategory.ORGANIZATION else 0.75,
                source=DetectionSource.CONTEXT,
                bounding_box=None,
                char_start=value_start,
                char_end=value_end,
            ))

        return detections
