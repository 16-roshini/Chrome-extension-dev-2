"""
Shared pytest fixtures for all OCR + PII detection tests.
All PII used here is SYNTHETIC / FAKE — no real personal data.
"""

from __future__ import annotations

import io
import pytest
from PIL import Image, ImageDraw, ImageFont

from server.ocr_pii.schemas import (
    BoundingBox,
    DetectionSource,
    OCRResult,
    OCRWord,
    PIICategory,
    PIIDetection,
)


# ---------------------------------------------------------------------------
# Synthetic text fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_text_basic() -> str:
    """Plain text with a mix of PII types — all synthetic."""
    return (
        "Name: John Doe\n"
        "Email: johndoe@example.com\n"
        "Phone: 9876543210\n"
        "DOB: 15/08/1990\n"
        "Address: 42 Baker Street, Mumbai, Maharashtra 400001\n"
    )


@pytest.fixture
def sample_text_aadhaar() -> str:
    return "Aadhaar Number: 2345 6789 0123"


@pytest.fixture
def sample_text_pan() -> str:
    return "PAN: ABCDE1234F"


@pytest.fixture
def sample_text_credit_card() -> str:
    # 4111111111111111 is the standard Luhn-valid test card number
    return "Card Number: 4111 1111 1111 1111"


@pytest.fixture
def sample_text_passport() -> str:
    return "Passport No: A1234567"


@pytest.fixture
def sample_text_ifsc() -> str:
    return "IFSC Code: SBIN0001234"


@pytest.fixture
def sample_text_ip() -> str:
    return "Server IP: 192.168.1.100"


@pytest.fixture
def sample_text_url() -> str:
    return "Visit us at https://example.gov.in/portal"


@pytest.fixture
def sample_text_bank_account() -> str:
    return "Account No: 123456789012"


@pytest.fixture
def sample_text_password() -> str:
    return "Password: MySecretPass@123"


@pytest.fixture
def sample_text_all() -> str:
    """One block containing every PII category — all synthetic."""
    return (
        "Name: Jane Smith\n"
        "Email: jane.smith@testmail.com\n"
        "Phone: +91 98765 43210\n"
        "Aadhaar: 2345 6789 0123\n"
        "PAN: ABCDE1234F\n"
        "Card Number: 4111 1111 1111 1111\n"
        "DOB: 22-07-1985\n"
        "Address: 10 MG Road, Bengaluru, Karnataka 560001\n"
        "Password: TestPass@2024\n"
        "IP: 10.0.0.1\n"
        "Website: https://test.example.com\n"
        "Organization: ISRO\n"
        "Passport No: B9876543\n"
        "Account No: 987654321098\n"
        "IFSC Code: HDFC0001234\n"
    )


# ---------------------------------------------------------------------------
# Synthetic OCR result fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ocr_word() -> OCRWord:
    return OCRWord(
        text="johndoe@example.com",
        bounding_box=BoundingBox(x=10, y=20, width=150, height=20),
        confidence=0.92,
    )


@pytest.fixture
def sample_ocr_result(sample_text_all) -> OCRResult:
    """Synthetic OCRResult built from sample_text_all."""
    words = [
        OCRWord(text=word, bounding_box=None, confidence=0.90)
        for word in sample_text_all.split()
        if word.strip()
    ]
    return OCRResult(
        words=words,
        full_text=sample_text_all,
        engine_used="tesseract",
    )


# ---------------------------------------------------------------------------
# Synthetic PIIDetection fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detection_email() -> PIIDetection:
    return PIIDetection(
        text="jane.smith@testmail.com",
        category=PIICategory.EMAIL,
        confidence=0.97,
        source=DetectionSource.REGEX,
        bounding_box=None,
        char_start=0,
        char_end=23,
    )


@pytest.fixture
def detection_phone() -> PIIDetection:
    return PIIDetection(
        text="9876543210",
        category=PIICategory.PHONE,
        confidence=0.90,
        source=DetectionSource.REGEX,
        bounding_box=None,
        char_start=30,
        char_end=40,
    )


@pytest.fixture
def detection_person_ner() -> PIIDetection:
    return PIIDetection(
        text="Jane Smith",
        category=PIICategory.PERSON_NAME,
        confidence=0.80,
        source=DetectionSource.SPACY_NER,
        bounding_box=None,
        char_start=6,
        char_end=16,
    )


@pytest.fixture
def detection_password_context() -> PIIDetection:
    return PIIDetection(
        text="TestPass@2024",
        category=PIICategory.PASSWORD,
        confidence=0.75,
        source=DetectionSource.CONTEXT,
        bounding_box=None,
        char_start=100,
        char_end=113,
    )


# ---------------------------------------------------------------------------
# Synthetic PIL image fixture (white background, black text)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image() -> Image.Image:
    """
    Creates a simple white 400x100 PIL image with synthetic PII text.
    Used for OCR engine tests.
    """
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Use default font (no file dependency)
    draw.text((10, 10), "Email: test@example.com", fill=(0, 0, 0))
    draw.text((10, 40), "Phone: 9876543210", fill=(0, 0, 0))
    return img


@pytest.fixture
def synthetic_image_bytes(synthetic_image) -> bytes:
    """Return synthetic_image as PNG bytes."""
    buf = io.BytesIO()
    synthetic_image.save(buf, format="PNG")
    return buf.getvalue()
