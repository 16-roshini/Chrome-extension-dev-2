"""
Tests for ContextDetector (keyword-based label detection).
All PII values are SYNTHETIC.
"""

from __future__ import annotations

import pytest
from server.ocr_pii.detectors.context_detector import ContextDetector
from server.ocr_pii.schemas import DetectionSource, PIICategory


@pytest.fixture
def detector() -> ContextDetector:
    return ContextDetector()


class TestContextDetectorBasics:

    def test_detector_name(self, detector):
        assert detector.detector_name == "context"

    def test_empty_text_returns_empty(self, detector):
        assert detector.detect("") == []

    def test_whitespace_only_returns_empty(self, detector):
        assert detector.detect("   ") == []

    def test_returns_list(self, detector):
        assert isinstance(detector.detect("Some random text."), list)


class TestPasswordDetection:

    def test_detects_password_label(self, detector):
        results = detector.detect("Password: MySecret@123")
        passwords = [d for d in results if d.category == PIICategory.PASSWORD]
        assert len(passwords) >= 1

    def test_detects_pwd_abbreviation(self, detector):
        results = detector.detect("pwd: secret99")
        passwords = [d for d in results if d.category == PIICategory.PASSWORD]
        assert len(passwords) >= 1

    def test_password_value_captured(self, detector):
        results = detector.detect("Password: MySecret@123")
        passwords = [d for d in results if d.category == PIICategory.PASSWORD]
        assert any("MySecret" in d.text for d in passwords)

    def test_password_source_is_context(self, detector):
        results = detector.detect("Password: abc123")
        passwords = [d for d in results if d.category == PIICategory.PASSWORD]
        if passwords:
            assert passwords[0].source == DetectionSource.CONTEXT


class TestDOBContextDetection:

    def test_detects_dob_label(self, detector):
        results = detector.detect("Date of Birth: 15/08/1990")
        dobs = [d for d in results if d.category == PIICategory.DATE_OF_BIRTH]
        assert len(dobs) >= 1

    def test_detects_dob_abbreviation(self, detector):
        results = detector.detect("DOB: 22-07-1985")
        dobs = [d for d in results if d.category == PIICategory.DATE_OF_BIRTH]
        assert len(dobs) >= 1


class TestAddressContextDetection:

    def test_detects_address_label(self, detector):
        results = detector.detect("Address: 42 Baker Street, Mumbai")
        addrs = [d for d in results if d.category == PIICategory.ADDRESS]
        assert len(addrs) >= 1

    def test_detects_street_label(self, detector):
        results = detector.detect("Street: 10 MG Road")
        addrs = [d for d in results if d.category == PIICategory.ADDRESS]
        assert len(addrs) >= 1


class TestBankAccountContextDetection:

    def test_detects_account_number_label(self, detector):
        results = detector.detect("Account No: 123456789012")
        accounts = [d for d in results if d.category == PIICategory.BANK_ACCOUNT]
        assert len(accounts) >= 1

    def test_detects_acc_no_abbreviation(self, detector):
        results = detector.detect("Acc No: 987654321098")
        accounts = [d for d in results if d.category == PIICategory.BANK_ACCOUNT]
        assert len(accounts) >= 1


class TestDeduplication:

    def test_no_duplicate_spans(self, detector):
        # Multiple keywords for same category should not produce duplicate spans
        text = "Password: secret123\npasswd: secret123"
        results = detector.detect(text)
        spans = [(d.char_start, d.char_end) for d in results]
        assert len(spans) == len(set(spans))


class TestConfidenceAndSpan:

    def test_confidence_in_range(self, detector):
        results = detector.detect("Password: Test@2024\nDOB: 01/01/2000")
        for d in results:
            assert 0.0 <= d.confidence <= 1.0

    def test_char_start_less_than_char_end(self, detector):
        results = detector.detect("Address: 5 Park Avenue, Delhi")
        for d in results:
            assert d.char_start < d.char_end
