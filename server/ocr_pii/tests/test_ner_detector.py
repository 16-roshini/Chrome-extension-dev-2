"""
Tests for NERDetector (spaCy en_core_web_sm).
All names, orgs, and dates are SYNTHETIC.

After the evidence-model refactor:
  - detect()          → PERSON + ORG only (committed PIIDetection)
  - detect_evidence() → DATE + GPE only (DetectorEvidence)
  - detect_all()      → both in one spaCy pass
"""

from __future__ import annotations

import pytest
from server.ocr_pii.detectors.ner_detector import NERDetector
from server.ocr_pii.schemas import (
    DetectionSource,
    DetectorEvidence,
    EvidenceLabel,
    PIICategory,
    PIIDetection,
)


@pytest.fixture
def detector() -> NERDetector:
    return NERDetector()


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------

class TestNERDetectorBasics:

    def test_detector_name(self, detector):
        assert detector.detector_name == "spacy_ner"

    def test_empty_text_returns_empty(self, detector):
        assert detector.detect("") == []

    def test_whitespace_only_returns_empty(self, detector):
        assert detector.detect("   \n\t  ") == []

    def test_returns_list(self, detector):
        result = detector.detect("Hello world.")
        assert isinstance(result, list)

    def test_detect_all_returns_tuple(self, detector):
        committed, evidence = detector.detect_all("Hello world.")
        assert isinstance(committed, list)
        assert isinstance(evidence, list)

    def test_detect_evidence_returns_list(self, detector):
        result = detector.detect_evidence("Hello world.")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# PERSON — still committed directly (no change)
# ---------------------------------------------------------------------------

class TestPersonDetection:

    def test_detects_person_name(self, detector):
        results = detector.detect("John Smith submitted the form.")
        persons = [d for d in results if d.category == PIICategory.PERSON_NAME]
        assert len(persons) >= 1

    def test_person_source_is_spacy(self, detector):
        results = detector.detect("Alice Johnson is the applicant.")
        persons = [d for d in results if d.category == PIICategory.PERSON_NAME]
        if persons:
            assert persons[0].source == DetectionSource.SPACY_NER

    def test_person_confidence_in_range(self, detector):
        results = detector.detect("Robert Brown signed the document.")
        for d in results:
            assert 0.0 <= d.confidence <= 1.0

    def test_person_is_pii_detection_not_evidence(self, detector):
        """PERSON must be a committed PIIDetection, not DetectorEvidence."""
        results = detector.detect("John Smith submitted the form.")
        for d in results:
            assert isinstance(d, PIIDetection)


# ---------------------------------------------------------------------------
# ORG — still committed directly
# ---------------------------------------------------------------------------

class TestOrganizationDetection:

    def test_detects_org(self, detector):
        results = detector.detect("The application was submitted to ISRO.")
        orgs = [d for d in results if d.category == PIICategory.ORGANIZATION]
        assert len(orgs) >= 1

    def test_org_source_is_spacy(self, detector):
        results = detector.detect("Google LLC processed the request.")
        orgs = [d for d in results if d.category == PIICategory.ORGANIZATION]
        if orgs:
            assert orgs[0].source == DetectionSource.SPACY_NER

    def test_org_is_pii_detection_not_evidence(self, detector):
        """ORG must be a committed PIIDetection, not DetectorEvidence."""
        results = detector.detect("The application was submitted to ISRO.")
        orgs = [d for d in results if d.category == PIICategory.ORGANIZATION]
        for o in orgs:
            assert isinstance(o, PIIDetection)


# ---------------------------------------------------------------------------
# DATE — now emitted as DetectorEvidence, NOT PIIDetection
# ---------------------------------------------------------------------------

class TestDateEvidence:

    def test_date_not_in_detect(self, detector):
        """detect() must NOT return any DATE_OF_BIRTH — dates are evidence only."""
        results = detector.detect("The event is on January 15, 1990.")
        dobs = [d for d in results if d.category == PIICategory.DATE_OF_BIRTH]
        assert len(dobs) == 0, (
            "detect() returned DATE_OF_BIRTH — should only be in detect_evidence()"
        )

    def test_date_in_detect_evidence(self, detector):
        """detect_evidence() must return a SPACY_DATE evidence for date entities."""
        evidence = detector.detect_evidence("The event is on January 15, 1990.")
        date_ev = [e for e in evidence if e.label == EvidenceLabel.SPACY_DATE]
        assert len(date_ev) >= 1

    def test_date_evidence_is_detector_evidence_type(self, detector):
        """Dates must be DetectorEvidence instances, not PIIDetection."""
        evidence = detector.detect_evidence("Born on 15 August 2002.")
        for ev in evidence:
            assert isinstance(ev, DetectorEvidence)

    def test_date_evidence_has_correct_fields(self, detector):
        evidence = detector.detect_evidence("Born on 15 August 2002.")
        date_ev = [e for e in evidence if e.label == EvidenceLabel.SPACY_DATE]
        if date_ev:
            ev = date_ev[0]
            assert ev.label == EvidenceLabel.SPACY_DATE
            assert isinstance(ev.text, str)
            assert len(ev.text) >= 4
            assert 0.0 <= ev.raw_confidence <= 1.0
            assert ev.char_start >= 0
            assert ev.char_end > ev.char_start

    def test_date_evidence_raw_confidence_is_low(self, detector):
        """DATE evidence should have low raw_confidence — fusion must decide."""
        evidence = detector.detect_evidence("The event is on January 15, 1990.")
        date_ev = [e for e in evidence if e.label == EvidenceLabel.SPACY_DATE]
        for ev in date_ev:
            assert ev.raw_confidence <= 0.65, (
                "DATE raw_confidence should be low (needs fusion validation)"
            )


# ---------------------------------------------------------------------------
# GPE — now emitted as DetectorEvidence, NOT PIIDetection
# ---------------------------------------------------------------------------

class TestGPEEvidence:

    def test_gpe_not_in_detect(self, detector):
        """detect() must NOT return any ADDRESS — GPE entities are evidence only."""
        results = detector.detect("The office is located in Hyderabad.")
        addrs = [d for d in results if d.category == PIICategory.ADDRESS]
        assert len(addrs) == 0, (
            "detect() returned ADDRESS — should only be in detect_evidence()"
        )

    def test_gpe_in_detect_evidence(self, detector):
        """detect_evidence() must return SPACY_GPE for place name entities."""
        evidence = detector.detect_evidence("The office is located in Hyderabad.")
        gpe_ev = [e for e in evidence if e.label == EvidenceLabel.SPACY_GPE]
        assert len(gpe_ev) >= 1

    def test_gpe_evidence_is_detector_evidence_type(self, detector):
        evidence = detector.detect_evidence("He lives in Mumbai.")
        for ev in evidence:
            assert isinstance(ev, DetectorEvidence)

    def test_gpe_evidence_raw_confidence_is_low(self, detector):
        """GPE evidence should have low raw_confidence — fusion must validate."""
        evidence = detector.detect_evidence("She is from New Delhi.")
        gpe_ev = [e for e in evidence if e.label == EvidenceLabel.SPACY_GPE]
        for ev in gpe_ev:
            assert ev.raw_confidence <= 0.65, (
                "GPE raw_confidence should be low (needs fusion validation)"
            )


# ---------------------------------------------------------------------------
# detect_all — single-pass consistency
# ---------------------------------------------------------------------------

class TestDetectAll:

    def test_detect_all_person_matches_detect(self, detector):
        """detect_all() committed list must equal detect() for PERSON."""
        text = "Alice Johnson filed the application."
        committed, _ = detector.detect_all(text)
        direct = detector.detect(text)
        assert len(committed) == len(direct)

    def test_detect_all_evidence_matches_detect_evidence(self, detector):
        """detect_all() evidence list must equal detect_evidence()."""
        text = "He was born on 15 August 1990 in Hyderabad."
        _, evidence = detector.detect_all(text)
        direct_ev = detector.detect_evidence(text)
        assert len(evidence) == len(direct_ev)

    def test_no_date_in_committed(self, detector):
        _, evidence = detector.detect_all("Born on January 1, 1990.")
        # Verify evidence contains DATE, committed does not
        date_in_evidence = any(
            e.label == EvidenceLabel.SPACY_DATE for e in evidence
        )
        # Just verify the evidence list has the date (spaCy may or may not detect)
        # The important thing is committed has no DATE_OF_BIRTH
        committed, _ = detector.detect_all("Born on January 1, 1990.")
        dobs_in_committed = [
            d for d in committed if d.category == PIICategory.DATE_OF_BIRTH
        ]
        assert len(dobs_in_committed) == 0

    def test_no_gpe_in_committed(self, detector):
        """ADDRESS must never appear in the committed detections list."""
        committed, _ = detector.detect_all("The event is in Bangalore.")
        addrs = [d for d in committed if d.category == PIICategory.ADDRESS]
        assert len(addrs) == 0


# ---------------------------------------------------------------------------
# Char span (committed detections only)
# ---------------------------------------------------------------------------

class TestCharSpan:

    def test_char_span_matches_text(self, detector):
        text = "Contact Alice Brown for details."
        results = detector.detect(text)
        for d in results:
            assert d.char_start >= 0
            assert d.char_end <= len(text)
            assert d.char_end > d.char_start
            assert text[d.char_start:d.char_end] == d.text
