"""
Confidence scoring for PII detections.

Design principles (precision-first):
  - Regex is the most trusted source — deterministic patterns.
  - spaCy NER is statistical — lower base weight.
  - Context is heuristic — moderate weight.
  - OCR quality penalty kicks in at 0.75 threshold (raised from 0.60),
    so any detection where Tesseract was less than 75% confident gets
    penalised proportionally. This is the main gate against garbled-text
    false positives.
  - Final score is always clamped to [0.0, 1.0].
"""

from __future__ import annotations

from server.ocr_pii.schemas import DetectionSource, PIIDetection

# ---------------------------------------------------------------------------
# Source weight — how much we trust each detection method
# Regex = 1.0 (deterministic), NER = 0.80 (statistical), Context = 0.78
# AGGREGATED keeps the combined score as-is
# ---------------------------------------------------------------------------
_SOURCE_WEIGHT: dict[DetectionSource, float] = {
    DetectionSource.REGEX:      1.0,
    DetectionSource.SPACY_NER:  0.80,
    DetectionSource.CONTEXT:    0.78,
    DetectionSource.AGGREGATED: 1.0,
}

# ---------------------------------------------------------------------------
# OCR confidence penalty threshold (raised from 0.60 → 0.75)
#
# If the average OCR confidence of the words covering a detection is below
# this threshold, the detection confidence is scaled down proportionally.
#
# At 0.75: a detection whose OCR words averaged 0.50 confidence gets
# penalty = 0.50 / 0.75 = 0.67 applied to its score.
# At 0.30 OCR confidence: penalty = 0.30 / 0.75 = 0.40 — heavy reduction.
# This is intentional: garbled OCR text should not produce high-confidence
# PII detections.
# ---------------------------------------------------------------------------
_OCR_PENALTY_THRESHOLD = 0.75


def compute_confidence(
    detection: PIIDetection,
    ocr_confidence: float = 1.0,
) -> float:
    """
    Compute final confidence for a single PIIDetection.

    Args:
        detection:      The raw PIIDetection from a detector.
        ocr_confidence: Average OCR word confidence for the matched span
                        (0.0–1.0). Defaults to 1.0 (no penalty).

    Returns:
        Final confidence score in [0.0, 1.0].
    """
    base = detection.confidence
    source_weight = _SOURCE_WEIGHT.get(detection.source, 0.78)

    # Apply source weight
    score = base * source_weight

    # Apply OCR quality penalty when OCR was uncertain
    if ocr_confidence < _OCR_PENALTY_THRESHOLD:
        penalty = ocr_confidence / _OCR_PENALTY_THRESHOLD
        score *= penalty

    return _clamp(score)


def combine_scores(scores: list[float]) -> float:
    """
    Combine multiple confidence scores from different sources using noisy-OR.

    Formula: 1 - product(1 - s for s in scores)

    The more independent sources agree on the same entity, the higher
    the combined confidence — but it never exceeds 1.0.
    """
    if not scores:
        return 0.0

    combined = 1.0
    for s in scores:
        combined *= (1.0 - _clamp(s))

    return _clamp(1.0 - combined)


def _clamp(value: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))
