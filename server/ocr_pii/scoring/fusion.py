"""
FusionLayer — promotes DetectorEvidence to PIIDetection.

Architecture principle:
  "Detectors generate evidence; fusion makes the final PII decision
   using OCR text, position, and confidence."

Only ambiguous NER labels (DATE, GPE) go through this layer.
Committed detections (PERSON, ORG, and all regex detections) bypass it.

Promotion rules
---------------

SPACY_DATE  → DATE_OF_BIRTH
  Promoted when ANY of the following is true:
    1. A DOB context keyword (dob, date of birth, born, birth date, d.o.b)
       appears within _CONTEXT_WINDOW chars before the entity.
    2. The entity text itself matches the DOB regex (a date in a birth-year
       range 1900–2099 with separators — the same pattern used by RegexDetector).
    3. The entity text is corroborated by a regex DATE_OF_BIRTH detection
       at an overlapping char span.
  NOT promoted when:
    - The text is obviously not a birthdate (e.g. "Monday", "last week",
      "January 2022" in a non-DOB context, "3 months ago").
    - OCR confidence of the matched words is below _MIN_OCR_CONF_DATE.

SPACY_GPE  → ADDRESS
  Promoted when ANY of the following is true:
    1. An address context keyword (address, addr, street, locality, pincode,
       city, state) appears within _CONTEXT_WINDOW chars before the entity.
    2. A PIN code / zip code pattern appears within _PIN_WINDOW chars after
       the entity.
    3. The entity is corroborated by a CONTEXT ADDRESS detection at an
       overlapping span.
  NOT promoted when:
    - The GPE entity appears in a purely factual/descriptive sentence with
      no address-forming context.
    - OCR confidence is below _MIN_OCR_CONF_GPE.

Both labels:
  - If neither condition is met, the evidence is silently discarded.
  - The final confidence = combine_scores([raw_confidence, boosters]).
"""

from __future__ import annotations

import re
from typing import List, Optional

from server.ocr_pii.schemas import (
    DetectionSource,
    DetectorEvidence,
    EvidenceLabel,
    OCRResult,
    OCRWord,
    PIICategory,
    PIIDetection,
)
from .scorer import combine_scores

# ---------------------------------------------------------------------------
# Window sizes (character count)
# ---------------------------------------------------------------------------

# How far before an evidence span to look for a context keyword
_CONTEXT_WINDOW = 120

# How far after a GPE span to look for a PIN / zip code pattern
_PIN_WINDOW = 60

# ---------------------------------------------------------------------------
# Minimum per-entity OCR confidence for promotion
# Below these thresholds, evidence is discarded regardless of context.
# ---------------------------------------------------------------------------
_MIN_OCR_CONF_DATE = 0.65
_MIN_OCR_CONF_GPE  = 0.60

# ---------------------------------------------------------------------------
# Context keyword patterns
# ---------------------------------------------------------------------------

_DOB_CONTEXT_PATTERN = re.compile(
    r"\b(?:dob|d\.o\.b|date\s+of\s+birth|birth\s+date|birthdate|born\s+on|born)\b",
    re.IGNORECASE,
)

_ADDRESS_CONTEXT_PATTERN = re.compile(
    r"\b(?:address|addr|street|locality|pincode|pin\s+code|zip|residence|"
    r"current\s+address|permanent\s+address|city|state)\b",
    re.IGNORECASE,
)

# Indian PIN code or generic 5–6 digit postal code immediately after GPE
_PIN_CODE_PATTERN = re.compile(r"\b\d{5,6}\b")

# DOB date formats (same logic as patterns.py DOB but as a standalone check)
_DOB_FORMAT_PATTERN = re.compile(
    r"\b(?:"
    r"(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.]((?:19|20)\d{2})"
    r"|"
    r"((?:19|20)\d{2})[/\-](?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])"
    r"|"
    r"(?:0?[1-9]|[12]\d|3[01])\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"\s+((?:19|20)\d{2})"
    r")\b",
    re.IGNORECASE,
)

# Month names that indicate a calendar date (not a relative date like "last Monday")
_MONTH_NAME_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
    re.IGNORECASE,
)

# Relative date words — these are NOT birth dates
_RELATIVE_DATE_WORDS = re.compile(
    r"\b(?:today|yesterday|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|week|month|year|ago|last|next|recent|"
    r"morning|evening|night|now|soon|later)\b",
    re.IGNORECASE,
)


class FusionLayer:
    """
    Validates and promotes DetectorEvidence to PIIDetection.

    Receives:
      - evidence: List[DetectorEvidence] from NERDetector.detect_evidence()
      - full_text: str — the complete OCR text
      - ocr_result: OCRResult — for per-entity OCR confidence
      - regex_detections: List[PIIDetection] — for corroboration
      - context_detections: List[PIIDetection] — for corroboration

    Returns:
      - List[PIIDetection] — only promoted evidence
    """

    def fuse(
        self,
        evidence: List[DetectorEvidence],
        full_text: str,
        ocr_result: OCRResult,
        regex_detections: List[PIIDetection],
        context_detections: List[PIIDetection],
    ) -> List[PIIDetection]:
        """
        Evaluate each evidence candidate and promote those that meet
        the promotion criteria.
        """
        if not evidence:
            return []

        # Pre-compute OCR word offsets once for the whole batch
        word_offsets = _compute_word_offsets(full_text, ocr_result.words)

        promoted: List[PIIDetection] = []

        for ev in evidence:
            # Per-entity OCR confidence
            ocr_conf = _entity_ocr_confidence(
                ev.char_start, ev.char_end,
                ocr_result.words, word_offsets,
            )

            if ev.label == EvidenceLabel.SPACY_DATE:
                det = self._try_promote_date(
                    ev, full_text, ocr_conf, regex_detections
                )
            elif ev.label == EvidenceLabel.SPACY_GPE:
                det = self._try_promote_gpe(
                    ev, full_text, ocr_conf, context_detections
                )
            else:
                det = None

            if det is not None:
                promoted.append(det)

        return promoted

    # ------------------------------------------------------------------
    # DATE promotion
    # ------------------------------------------------------------------

    def _try_promote_date(
        self,
        ev: DetectorEvidence,
        full_text: str,
        ocr_conf: float,
        regex_detections: List[PIIDetection],
    ) -> Optional[PIIDetection]:
        """
        Promote SPACY_DATE to DATE_OF_BIRTH if:
          - OCR confidence is sufficient, AND
          - At least one of:
              (a) DOB context keyword found before the span
              (b) Entity text matches the DOB date format
              (c) Regex corroborated a DATE_OF_BIRTH at this span
          - AND the text is not a relative date expression
        """
        if ocr_conf < _MIN_OCR_CONF_DATE:
            return None

        # Reject relative date expressions immediately
        if _RELATIVE_DATE_WORDS.search(ev.text):
            return None

        # Reject if the text has no digit at all — a real date always has one
        if not any(c.isdigit() for c in ev.text):
            return None

        boosters: list[float] = [ev.raw_confidence]

        # Signal (a): DOB context keyword in the preceding window
        window_start = max(0, ev.char_start - _CONTEXT_WINDOW)
        preceding = full_text[window_start:ev.char_start]
        if _DOB_CONTEXT_PATTERN.search(preceding):
            boosters.append(0.85)

        # Signal (b): entity text itself looks like a formatted date
        if _DOB_FORMAT_PATTERN.search(ev.text):
            boosters.append(0.80)

        # Signal (c): regex already confirmed this as DATE_OF_BIRTH
        if _has_overlapping_detection(
            ev.char_start, ev.char_end,
            regex_detections, PIICategory.DATE_OF_BIRTH,
        ):
            boosters.append(0.90)

        # If only the raw evidence confidence — no signal fired — discard
        if len(boosters) == 1:
            return None

        final_conf = combine_scores(boosters)
        # Apply OCR quality factor
        if ocr_conf < 0.75:
            final_conf *= ocr_conf / 0.75
        final_conf = max(0.0, min(1.0, final_conf))

        return PIIDetection(
            text=ev.text,
            category=PIICategory.DATE_OF_BIRTH,
            confidence=final_conf,
            source=DetectionSource.AGGREGATED,
            bounding_box=None,
            char_start=ev.char_start,
            char_end=ev.char_end,
        )

    # ------------------------------------------------------------------
    # GPE promotion
    # ------------------------------------------------------------------

    def _try_promote_gpe(
        self,
        ev: DetectorEvidence,
        full_text: str,
        ocr_conf: float,
        context_detections: List[PIIDetection],
    ) -> Optional[PIIDetection]:
        """
        Promote SPACY_GPE to ADDRESS if:
          - OCR confidence is sufficient, AND
          - At least one of:
              (a) Address context keyword found before the span
              (b) A PIN/postal code found immediately after the span
              (c) Context detector already confirmed an ADDRESS at this span
        """
        if ocr_conf < _MIN_OCR_CONF_GPE:
            return None

        boosters: list[float] = [ev.raw_confidence]

        # Signal (a): address context keyword in the preceding window
        window_start = max(0, ev.char_start - _CONTEXT_WINDOW)
        preceding = full_text[window_start:ev.char_start]
        if _ADDRESS_CONTEXT_PATTERN.search(preceding):
            boosters.append(0.82)

        # Signal (b): PIN code appears shortly after the GPE entity
        window_end = min(len(full_text), ev.char_end + _PIN_WINDOW)
        following = full_text[ev.char_end:window_end]
        if _PIN_CODE_PATTERN.search(following):
            boosters.append(0.78)

        # Signal (c): context detector confirmed ADDRESS at this span
        if _has_overlapping_detection(
            ev.char_start, ev.char_end,
            context_detections, PIICategory.ADDRESS,
        ):
            boosters.append(0.80)

        # No signal — discard
        if len(boosters) == 1:
            return None

        final_conf = combine_scores(boosters)
        if ocr_conf < 0.75:
            final_conf *= ocr_conf / 0.75
        final_conf = max(0.0, min(1.0, final_conf))

        return PIIDetection(
            text=ev.text,
            category=PIICategory.ADDRESS,
            confidence=final_conf,
            source=DetectionSource.AGGREGATED,
            bounding_box=None,
            char_start=ev.char_start,
            char_end=ev.char_end,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_overlapping_detection(
    char_start: int,
    char_end: int,
    detections: List[PIIDetection],
    category: PIICategory,
) -> bool:
    """Return True if any detection of the given category overlaps this span."""
    for d in detections:
        if d.category != category:
            continue
        if d.char_start < char_end and d.char_end > char_start:
            return True
    return False


def _compute_word_offsets(
    full_text: str,
    words: List[OCRWord],
) -> List[tuple[int, int]]:
    """Forward cursor scan to find each OCR word's (start, end) in full_text."""
    offsets: List[tuple[int, int]] = []
    cursor = 0
    for word in words:
        token = word.text.strip()
        if not token:
            offsets.append((cursor, cursor))
            continue
        idx = full_text.find(token, cursor)
        if idx == -1:
            offsets.append((cursor, cursor + len(token)))
        else:
            offsets.append((idx, idx + len(token)))
            cursor = idx + len(token)
    return offsets


def _entity_ocr_confidence(
    char_start: int,
    char_end: int,
    words: List[OCRWord],
    word_offsets: List[tuple[int, int]],
) -> float:
    """Average OCR confidence of words overlapping the entity's char span."""
    matched = [
        words[i] for i, (ws, we) in enumerate(word_offsets)
        if ws < char_end and we > char_start
    ]
    if not matched:
        return 1.0
    return sum(w.confidence for w in matched) / len(matched)
