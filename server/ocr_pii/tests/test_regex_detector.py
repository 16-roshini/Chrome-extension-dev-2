"""
Tests for RegexDetector.
All PII values are SYNTHETIC — no real personal data.
"""

from __future__ import annotations

import pytest
from server.ocr_pii.detectors.regex_detector import RegexDetector
from server.ocr_pii.detectors.patterns import luhn_check
from server.ocr_pii.schemas import DetectionSource, PIICategory


@pytest.fixture
def detector() -> RegexDetector:
    return RegexDetector()


# ---------------------------------------------------------------------------
# Luhn check
# ---------------------------------------------------------------------------

class TestLuhnCheck:

    def test_valid_visa(self):
        assert luhn_check("4111111111111111") is True

    def test_valid_mastercard(self):
        assert luhn_check("5500005555555559") is True

    def test_invalid_card(self):
        assert luhn_check("1234567890123456") is False

    def test_too_short(self):
        assert luhn_check("123456") is False

    def test_with_spaces(self):
        # luhn_check strips non-digits
        assert luhn_check("4111 1111 1111 1111") is True


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

class TestEmailDetection:

    def test_detects_basic_email(self, detector):
        results = detector.detect("Contact: johndoe@example.com")
        assert any(d.category == PIICategory.EMAIL for d in results)

    def test_detects_email_text(self, detector):
        results = detector.detect("Send to jane.smith@testmail.co.in")
        emails = [d for d in results if d.category == PIICategory.EMAIL]
        assert len(emails) == 1
        assert emails[0].text == "jane.smith@testmail.co.in"

    def test_no_false_positive_on_plain_word(self, detector):
        results = detector.detect("notanemail")
        emails = [d for d in results if d.category == PIICategory.EMAIL]
        assert len(emails) == 0

    def test_email_source_is_regex(self, detector):
        results = detector.detect("test@example.com")
        emails = [d for d in results if d.category == PIICategory.EMAIL]
        assert emails[0].source == DetectionSource.REGEX


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------

class TestPhoneDetection:

    def test_detects_10_digit_mobile(self, detector):
        results = detector.detect("Call me at 9876543210")
        phones = [d for d in results if d.category == PIICategory.PHONE]
        assert len(phones) == 1

    def test_detects_with_country_code(self, detector):
        results = detector.detect("+91 98765 43210")
        phones = [d for d in results if d.category == PIICategory.PHONE]
        assert len(phones) >= 1

    def test_rejects_landline_starting_with_1(self, detector):
        # Numbers starting with 1-5 are not valid Indian mobiles
        results = detector.detect("Call 1234567890")
        phones = [d for d in results if d.category == PIICategory.PHONE]
        assert len(phones) == 0


# ---------------------------------------------------------------------------
# Aadhaar
# ---------------------------------------------------------------------------

class TestAadhaarDetection:

    def test_detects_spaced_aadhaar(self, detector):
        results = detector.detect("Aadhaar: 2345 6789 0123")
        aadhaars = [d for d in results if d.category == PIICategory.AADHAAR]
        assert len(aadhaars) == 1

    def test_detects_hyphen_aadhaar(self, detector):
        results = detector.detect("UID: 2345-6789-0123")
        aadhaars = [d for d in results if d.category == PIICategory.AADHAAR]
        assert len(aadhaars) == 1

    def test_detects_continuous_aadhaar(self, detector):
        results = detector.detect("234567890123")
        aadhaars = [d for d in results if d.category == PIICategory.AADHAAR]
        assert len(aadhaars) == 1

    def test_rejects_11_digits(self, detector):
        results = detector.detect("23456789012")
        aadhaars = [d for d in results if d.category == PIICategory.AADHAAR]
        assert len(aadhaars) == 0


# ---------------------------------------------------------------------------
# PAN
# ---------------------------------------------------------------------------

class TestPANDetection:

    def test_detects_valid_pan(self, detector):
        results = detector.detect("PAN: ABCDE1234F")
        pans = [d for d in results if d.category == PIICategory.PAN]
        assert len(pans) == 1
        assert pans[0].text == "ABCDE1234F"

    def test_rejects_lowercase_pan(self, detector):
        results = detector.detect("abcde1234f")
        pans = [d for d in results if d.category == PIICategory.PAN]
        assert len(pans) == 0


# ---------------------------------------------------------------------------
# Credit Card
# ---------------------------------------------------------------------------

class TestCreditCardDetection:

    def test_detects_luhn_valid_card(self, detector):
        results = detector.detect("Card: 4111 1111 1111 1111")
        cards = [d for d in results if d.category == PIICategory.CREDIT_CARD]
        assert len(cards) == 1

    def test_rejects_luhn_invalid_card(self, detector):
        results = detector.detect("Card: 1234 5678 9012 3456")
        cards = [d for d in results if d.category == PIICategory.CREDIT_CARD]
        assert len(cards) == 0


# ---------------------------------------------------------------------------
# IP Address
# ---------------------------------------------------------------------------

class TestIPDetection:

    def test_detects_private_ip(self, detector):
        results = detector.detect("Server: 192.168.1.100")
        ips = [d for d in results if d.category == PIICategory.IP_ADDRESS]
        assert len(ips) == 1

    def test_detects_loopback(self, detector):
        results = detector.detect("Host: 127.0.0.1")
        ips = [d for d in results if d.category == PIICategory.IP_ADDRESS]
        assert len(ips) == 1


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------

class TestURLDetection:

    def test_detects_https_url(self, detector):
        results = detector.detect("Visit https://example.gov.in/portal")
        urls = [d for d in results if d.category == PIICategory.URL]
        assert len(urls) == 1

    def test_detects_http_url(self, detector):
        results = detector.detect("Go to http://test.com")
        urls = [d for d in results if d.category == PIICategory.URL]
        assert len(urls) == 1


# ---------------------------------------------------------------------------
# IFSC
# ---------------------------------------------------------------------------

class TestIFSCDetection:

    def test_detects_ifsc(self, detector):
        results = detector.detect("IFSC: SBIN0001234")
        ifsc = [d for d in results if d.category == PIICategory.IFSC]
        assert len(ifsc) == 1
        assert ifsc[0].text == "SBIN0001234"


# ---------------------------------------------------------------------------
# Date of Birth
# ---------------------------------------------------------------------------

class TestDOBDetection:

    def test_detects_dd_mm_yyyy(self, detector):
        results = detector.detect("DOB: 15/08/1990")
        dobs = [d for d in results if d.category == PIICategory.DATE_OF_BIRTH]
        assert len(dobs) >= 1

    def test_detects_yyyy_mm_dd(self, detector):
        results = detector.detect("Born: 1990-08-15")
        dobs = [d for d in results if d.category == PIICategory.DATE_OF_BIRTH]
        assert len(dobs) >= 1

    def test_detects_dd_mon_yyyy(self, detector):
        results = detector.detect("15 Aug 1990")
        dobs = [d for d in results if d.category == PIICategory.DATE_OF_BIRTH]
        assert len(dobs) >= 1


# ---------------------------------------------------------------------------
# Passport
# ---------------------------------------------------------------------------

class TestPassportDetection:

    def test_detects_passport(self, detector):
        results = detector.detect("Passport: A1234567")
        passports = [d for d in results if d.category == PIICategory.PASSPORT]
        assert len(passports) >= 1


# ---------------------------------------------------------------------------
# Bank Account
# ---------------------------------------------------------------------------

class TestBankAccountDetection:

    def test_detects_bank_account(self, detector):
        results = detector.detect("Account No: 123456789012")
        accounts = [d for d in results if d.category == PIICategory.BANK_ACCOUNT]
        assert len(accounts) >= 1


# ---------------------------------------------------------------------------
# Confidence and char span
# ---------------------------------------------------------------------------

class TestConfidenceAndSpan:

    def test_confidence_in_range(self, detector, sample_text_all):
        results = detector.detect(sample_text_all)
        for d in results:
            assert 0.0 <= d.confidence <= 1.0

    def test_char_span_valid(self, detector, sample_text_all):
        results = detector.detect(sample_text_all)
        for d in results:
            assert d.char_start >= 0
            assert d.char_end > d.char_start
            assert sample_text_all[d.char_start:d.char_end].strip() != ""

    def test_empty_text_returns_empty(self, detector):
        assert detector.detect("") == []
