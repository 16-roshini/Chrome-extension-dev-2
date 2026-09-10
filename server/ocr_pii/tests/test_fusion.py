"""
Tests for FusionLayer.

The FusionLayer promotes DetectorEvidence to PIIDetection only when
supporting signals (context keywords, format match, regex corroboration,
or OCR confidence) are present.

All text and values used are SYNTHETIC — no real personal data.
"""

from __future__ import annotations

import pytest
from server.ocr_pii.scoring.fusion import FusionLayer
from server.ocr_pii.schemas import (
    BoundingBox,
    DetectionSource,
    DetectorEvidence,
    EvidenceLabel,
    OCRResult,
    OCRWord,
    PIICategory,
    PIIDetection,
)


@pytest.fixture
def fusion() -> FusionLayer:
    return FusionLayer()


def make_ocr_result(full_text: str, conf: float = 0.92) -> OCRResult:
    """Build a synthetic OCRResult with one word per token."""
    words = []
    cursor = 0
    for token in full_text.split():
        idx = full_text.find(token, cursor)
        bbox = BoundingBox(x=idx * 10, y=10, width=len(token) * 8, height=16)
        words.append(OCRWord(text=token, bounding_box=bbox, confidence=conf))
        cursor = idx + len(token)
    return OCRResult(words=words, full_text=full_text, engine_used="tesseract")


def make_evidence(
    text: str,
    label: EvidenceLabel,
    char_start: int,
    raw_confidence: float = 0.55,
) -> DetectorEvidence:
    return DetectorEvidence(
        text=text,
        label=label,
        raw_confidence=raw_confidence,
        char_start=char_start,
        char_end=char_start + len(text),
    )


def make_detection(
    text: str,
    category: PIICategory,
    char_start: int,
    source: DetectionSource = DetectionSource.REGEX,
) -> PIIDetection:
    return PIIDetection(
        text=text,
        category=category,
        confidence=0.85,
        source=source,
        bounding_box=None,
        char_start=char_start,
        char_end=char_start + len(text),
    )


# ---------------------------------------------------------------------------
# DATE evidence → DATE_OF_BIRTH promotion
# ---------------------------------------------------------------------------

class TestDatePromotion:

    def test_promotes_date_with_dob_keyword(self, fusion):
        """DOB keyword before the span should promote the evidence."""
        text = "Date of birth: 15/08/1990"
        ev = make_evidence("15/08/1990", EvidenceLabel.SPACY_DATE, 15)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        dobs = [d for d in result if d.category == PIICategory.DATE_OF_BIRTH]
        assert len(dobs) == 1

    def test_promotes_date_with_dob_format(self, fusion):
        """A well-formatted date (DD/MM/YYYY) should promote even without keyword."""
        text = "Applicant details: 22/07/1985"
        ev = make_evidence("22/07/1985", EvidenceLabel.SPACY_DATE, 19)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        dobs = [d for d in result if d.category == PIICategory.DATE_OF_BIRTH]
        assert len(dobs) == 1

    def test_promotes_date_with_regex_corroboration(self, fusion):
        """Regex already found the same span as DATE_OF_BIRTH — promote."""
        text = "Record shows 10/10/1996"
        ev = make_evidence("10/10/1996", EvidenceLabel.SPACY_DATE, 13)
        regex_det = make_detection("10/10/1996", PIICategory.DATE_OF_BIRTH, 13)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [regex_det], [])
        dobs = [d for d in result if d.category == PIICategory.DATE_OF_BIRTH]
        assert len(dobs) == 1

    def test_discards_date_without_any_signal(self, fusion):
        """A bare date entity with no DOB context, no format match, no regex → discard."""
        # "January 2022" — no DOB keyword, not a birth-year formatted date, no regex
        text = "The policy was updated in January 2022"
        ev = make_evidence("January 2022", EvidenceLabel.SPACY_DATE, 27)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        dobs = [d for d in result if d.category == PIICategory.DATE_OF_BIRTH]
        assert len(dobs) == 0

    def test_discards_relative_date(self, fusion):
        """Relative date expressions must never become DATE_OF_BIRTH."""
        for relative in ["last Monday", "yesterday", "next week", "3 months ago"]:
            text = f"The meeting was {relative}"
            ev = make_evidence(relative, EvidenceLabel.SPACY_DATE, len("The meeting was "))
            ocr = make_ocr_result(text)
            result = fusion.fuse([ev], text, ocr, [], [])
            dobs = [d for d in result if d.category == PIICategory.DATE_OF_BIRTH]
            assert len(dobs) == 0, f"Relative date '{relative}' should not be DATE_OF_BIRTH"

    def test_discards_date_with_low_ocr_confidence(self, fusion):
        """DATE evidence with very low OCR confidence must be discarded."""
        text = "DOB: 15/08/1990"
        ev = make_evidence("15/08/1990", EvidenceLabel.SPACY_DATE, 5)
        # OCR confidence below minimum (0.65)
        ocr = make_ocr_result(text, conf=0.40)
        result = fusion.fuse([ev], text, ocr, [], [])
        assert len(result) == 0

    def test_promoted_date_has_correct_category(self, fusion):
        text = "DOB: 22/07/1985"
        ev = make_evidence("22/07/1985", EvidenceLabel.SPACY_DATE, 5)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        assert all(d.category == PIICategory.DATE_OF_BIRTH for d in result)

    def test_promoted_date_confidence_higher_than_raw(self, fusion):
        """Promotion should boost confidence above the raw evidence confidence."""
        text = "DOB: 22/07/1985"
        ev = make_evidence("22/07/1985", EvidenceLabel.SPACY_DATE, 5)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        if result:
            assert result[0].confidence > ev.raw_confidence

    def test_promoted_date_confidence_in_range(self, fusion):
        text = "Date of birth: 15 Aug 1990"
        ev = make_evidence("15 Aug 1990", EvidenceLabel.SPACY_DATE, 15)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        for d in result:
            assert 0.0 <= d.confidence <= 1.0


# ---------------------------------------------------------------------------
# GPE evidence → ADDRESS promotion
# ---------------------------------------------------------------------------

class TestGPEPromotion:

    def test_promotes_gpe_with_address_keyword(self, fusion):
        """Address keyword before the span should promote GPE to ADDRESS."""
        text = "Address: Green Park, Hyderabad"
        ev = make_evidence("Hyderabad", EvidenceLabel.SPACY_GPE, 21)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        addrs = [d for d in result if d.category == PIICategory.ADDRESS]
        assert len(addrs) == 1

    def test_promotes_gpe_with_pin_code_after(self, fusion):
        """A PIN code after the GPE span confirms it's an address."""
        text = "Flat 5 Green Park Hyderabad 500016"
        # "Hyderabad" starts at index 18 in this string
        char_start = text.index("Hyderabad")
        ev = make_evidence("Hyderabad", EvidenceLabel.SPACY_GPE, char_start)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        addrs = [d for d in result if d.category == PIICategory.ADDRESS]
        assert len(addrs) == 1

    def test_promotes_gpe_with_context_corroboration(self, fusion):
        """Context detector already confirmed ADDRESS at this span — promote."""
        text = "City: Hyderabad"
        ev = make_evidence("Hyderabad", EvidenceLabel.SPACY_GPE, 6)
        ctx_det = make_detection(
            "Hyderabad", PIICategory.ADDRESS, 6, DetectionSource.CONTEXT
        )
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [ctx_det])
        addrs = [d for d in result if d.category == PIICategory.ADDRESS]
        assert len(addrs) == 1

    def test_discards_gpe_without_any_signal(self, fusion):
        """A bare GPE with no address context, no PIN, no context det → discard."""
        text = "She was born in Hyderabad and moved away."
        ev = make_evidence("Hyderabad", EvidenceLabel.SPACY_GPE, 15)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        addrs = [d for d in result if d.category == PIICategory.ADDRESS]
        assert len(addrs) == 0

    def test_discards_gpe_with_low_ocr_confidence(self, fusion):
        """GPE evidence with very low OCR confidence must be discarded."""
        text = "Address: Hyderabad"
        ev = make_evidence("Hyderabad", EvidenceLabel.SPACY_GPE, 9)
        ocr = make_ocr_result(text, conf=0.35)
        result = fusion.fuse([ev], text, ocr, [], [])
        assert len(result) == 0

    def test_promoted_gpe_has_correct_category(self, fusion):
        text = "Address: Mumbai"
        ev = make_evidence("Mumbai", EvidenceLabel.SPACY_GPE, 9)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        assert all(d.category == PIICategory.ADDRESS for d in result)

    def test_promoted_gpe_confidence_in_range(self, fusion):
        text = "Locality: Bengaluru 560001"
        ev = make_evidence("Bengaluru", EvidenceLabel.SPACY_GPE, 10)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        for d in result:
            assert 0.0 <= d.confidence <= 1.0


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

class TestFusionEdgeCases:

    def test_empty_evidence_returns_empty(self, fusion):
        ocr = make_ocr_result("Some text here.")
        result = fusion.fuse([], "Some text here.", ocr, [], [])
        assert result == []

    def test_multiple_evidence_independent(self, fusion):
        """Multiple evidence items are processed independently."""
        text = "DOB: 10/10/1996 Address: Green Park Hyderabad 500001"
        date_ev = make_evidence("10/10/1996", EvidenceLabel.SPACY_DATE, 5)
        gpe_ev = make_evidence("Hyderabad", EvidenceLabel.SPACY_GPE, 33)
        ocr = make_ocr_result(text)
        result = fusion.fuse([date_ev, gpe_ev], text, ocr, [], [])
        categories = {d.category for d in result}
        assert PIICategory.DATE_OF_BIRTH in categories
        assert PIICategory.ADDRESS in categories

    def test_result_items_are_pii_detection(self, fusion):
        """All promoted items must be PIIDetection instances."""
        text = "Date of birth: 22/07/1985"
        ev = make_evidence("22/07/1985", EvidenceLabel.SPACY_DATE, 15)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        for item in result:
            assert isinstance(item, PIIDetection)

    def test_promoted_source_is_aggregated(self, fusion):
        """Promoted detections should use AGGREGATED source."""
        text = "DOB: 15/08/1990"
        ev = make_evidence("15/08/1990", EvidenceLabel.SPACY_DATE, 5)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        for d in result:
            assert d.source == DetectionSource.AGGREGATED

    def test_char_span_preserved(self, fusion):
        """Promoted detection must preserve the original char span."""
        text = "DOB: 22/07/1985"
        ev = make_evidence("22/07/1985", EvidenceLabel.SPACY_DATE, 5)
        ocr = make_ocr_result(text)
        result = fusion.fuse([ev], text, ocr, [], [])
        if result:
            assert result[0].char_start == 5
            assert result[0].char_end == 5 + len("22/07/1985")
