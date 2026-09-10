"""
Pydantic schemas for OCR + PII detection module.
These are the structured output types consumed by Dev 3.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# PII Category Enum
# ---------------------------------------------------------------------------

class PIICategory(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    AADHAAR = "aadhaar"
    PAN = "pan"
    CREDIT_CARD = "credit_card"
    PERSON_NAME = "person_name"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    PASSWORD = "password"
    IP_ADDRESS = "ip_address"
    URL = "url"
    ORGANIZATION = "organization"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    IFSC = "ifsc"


# ---------------------------------------------------------------------------
# Detection Source Enum
# ---------------------------------------------------------------------------

class DetectionSource(str, Enum):
    REGEX = "regex"
    SPACY_NER = "spacy_ner"
    CONTEXT = "context"
    AGGREGATED = "aggregated"


# ---------------------------------------------------------------------------
# Bounding Box
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Pixel coordinates of a detected region in the image."""
    x: int = Field(..., description="Left edge (pixels)")
    y: int = Field(..., description="Top edge (pixels)")
    width: int = Field(..., description="Width (pixels)")
    height: int = Field(..., description="Height (pixels)")


# ---------------------------------------------------------------------------
# OCR Word — single word result from OCR engine
# ---------------------------------------------------------------------------

class OCRWord(BaseModel):
    """A single word extracted by the OCR engine."""
    text: str = Field(..., description="Extracted word text")
    bounding_box: Optional[BoundingBox] = Field(None, description="Word bounding box in image")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence for this word (0–1)")


# ---------------------------------------------------------------------------
# OCR Result — full output of one OCR engine run
# ---------------------------------------------------------------------------

class OCRResult(BaseModel):
    """Full output of a single OCR engine run on one image."""
    words: List[OCRWord] = Field(default_factory=list, description="Individual word results")
    full_text: str = Field(..., description="Complete extracted text (words joined)")
    engine_used: str = Field(..., description="OCR engine name: 'tesseract' or 'easyocr'")


# ---------------------------------------------------------------------------
# PII Detection — single detected PII entity
# ---------------------------------------------------------------------------

class PIIDetection(BaseModel):
    """A single detected PII entity."""
    text: str = Field(..., description="The detected PII text")
    category: PIICategory = Field(..., description="PII category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0–1)")
    source: DetectionSource = Field(..., description="Detection method that found this entity")
    bounding_box: Optional[BoundingBox] = Field(None, description="Location in image (if available)")
    char_start: int = Field(..., ge=0, description="Character start offset in full_text")
    char_end: int = Field(..., ge=0, description="Character end offset in full_text")


# ---------------------------------------------------------------------------
# PII Detection Result — full pipeline output for one image
# ---------------------------------------------------------------------------

class PIIDetectionResult(BaseModel):
    """
    Complete output of the OCR + PII pipeline for one image.
    This is the structured output handed to Dev 3.
    """
    detections: List[PIIDetection] = Field(
        default_factory=list,
        description="All detected PII entities after aggregation and deduplication"
    )
    ocr_result: OCRResult = Field(..., description="Full OCR output used for detection")
    processing_time_ms: float = Field(..., description="Total pipeline processing time in milliseconds")
    image_hash: str = Field(..., description="MD5 hash of the input image for deduplication")


# ---------------------------------------------------------------------------
# API Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    """Response from POST /ocr-pii/analyze"""
    success: bool
    result: Optional[PIIDetectionResult] = None
    error: Optional[str] = None


class OCROnlyResponse(BaseModel):
    """Response from POST /ocr-pii/ocr-only"""
    success: bool
    result: Optional[OCRResult] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response from GET /ocr-pii/health"""
    status: str
    tesseract_available: bool
    easyocr_available: bool
    spacy_model_loaded: bool
    version: str = "1.0.0"


class SupportedPIIResponse(BaseModel):
    """Response from GET /ocr-pii/supported-pii"""
    categories: List[str]
    total: int


# ---------------------------------------------------------------------------
# Clean output schema — the ONLY thing returned to callers
# ---------------------------------------------------------------------------

class PIIEntity(BaseModel):
    """
    Single PII detection in the clean output format.
    This is what the API returns — no internal fields exposed.
    """
    type: str = Field(..., description="PII category in uppercase, e.g. EMAIL, PERSON_NAME")
    text: str = Field(..., description="The detected PII text")
    bbox: Optional[List[int]] = Field(
        None,
        description="Bounding box as [x, y, x2, y2] in pixels, or null if not available"
    )
    ocr_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Average OCR confidence of the words covering this entity (0-1)"
    )
    pii_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Final PII detection confidence score (0-1)"
    )


class DetectResponse(BaseModel):
    """Clean response from POST /ocr-pii/analyze"""
    detections: List[PIIEntity]


class GroupedDetectResponse(BaseModel):
    """
    Grouped response from POST /ocr-pii/analyze?grouped=true

    PII entities are grouped by category. Each field holds a list
    so multiple occurrences (e.g. two email addresses) are preserved.
    All field values are Optional — fields with no detection are null.

    This is a convenience format for UIs and Dev 3 integration.
    The underlying data is identical to DetectResponse.
    """
    name:          Optional[List[PIIEntity]] = None
    email:         Optional[List[PIIEntity]] = None
    phone:         Optional[List[PIIEntity]] = None
    date_of_birth: Optional[List[PIIEntity]] = None
    address:       Optional[List[PIIEntity]] = None
    aadhaar:       Optional[List[PIIEntity]] = None
    pan:           Optional[List[PIIEntity]] = None
    organization:  Optional[List[PIIEntity]] = None
    password:      Optional[List[PIIEntity]] = None
    credit_card:   Optional[List[PIIEntity]] = None
    bank_account:  Optional[List[PIIEntity]] = None
    ifsc:          Optional[List[PIIEntity]] = None
    passport:      Optional[List[PIIEntity]] = None
    ip_address:    Optional[List[PIIEntity]] = None
    url:           Optional[List[PIIEntity]] = None


# ---------------------------------------------------------------------------
# Evidence model — internal only, never returned to callers
#
# DetectorEvidence represents a candidate signal from a detector that is
# NOT yet a committed PII classification.  The FusionLayer decides whether
# to promote it to a PIIDetection based on corroborating signals.
#
# This is used for ambiguous NER labels:
#   - spaCy DATE entity  → could be a birth date or any other date
#   - spaCy GPE entity   → could be part of a postal address or just a
#                           place name mentioned in passing
#
# Detectors that produce fully-determined PII (REGEX, PERSON/ORG from NER,
# and CONTEXT) bypass this and emit PIIDetection directly.
# ---------------------------------------------------------------------------

class EvidenceLabel(str, Enum):
    """
    The raw detector label — what the detector actually observed.
    These are distinct from PIICategory because they represent observations,
    not final classifications.
    """
    SPACY_DATE = "spacy_date"     # spaCy DATE entity → may become DATE_OF_BIRTH
    SPACY_GPE  = "spacy_gpe"     # spaCy GPE entity  → may become ADDRESS


class DetectorEvidence(BaseModel):
    """
    A candidate signal from a detector that needs validation before
    being promoted to a confirmed PIIDetection.

    Only used internally — never serialised into the API response.
    """
    text: str = Field(..., description="The observed text")
    label: EvidenceLabel = Field(..., description="Raw detector label")
    raw_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Base confidence from the detector (before any boosting)"
    )
    char_start: int = Field(..., ge=0)
    char_end: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Browser analysis schemas
# ---------------------------------------------------------------------------

class BrowserTab(BaseModel):
    """A single browser tab extracted from the screenshot."""
    title: str = Field(..., description="Tab title text")
    active: bool = Field(..., description="True if this is the currently active tab")
    bbox: Optional[List[int]] = Field(None, description="[x1,y1,x2,y2] of the tab in the screenshot")


class OCRDetection(BaseModel):
    """A single OCR word with its bounding box and confidence."""
    text: str = Field(..., description="The word text")
    bbox: Optional[List[int]] = Field(None, description="[x1,y1,x2,y2] pixel coordinates")
    ocr_confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence for this word")


class BrowserAnalysisResult(BaseModel):
    """
    Complete result of browser screenshot analysis.
    Returned by POST /ocr-pii/browser-analyze.
    """
    tabs: List[BrowserTab] = Field(
        default_factory=list,
        description="All browser tabs detected in the tab bar region"
    )
    active_url: Optional[str] = Field(
        None,
        description="URL extracted from the address bar, or null if not found"
    )
    page_title: Optional[str] = Field(
        None,
        description="Main heading/title of the visible page content"
    )
    page_content: str = Field(
        default="",
        description="Full text extracted from the page body region"
    )
    ocr_detections: List[OCRDetection] = Field(
        default_factory=list,
        description="Every word OCR found in the page body with its bbox and confidence"
    )
    pii_detected: List[PIIEntity] = Field(
        default_factory=list,
        description="PII entities detected in the page content"
    )


# ---------------------------------------------------------------------------
# Browser analysis schemas — used by POST /ocr-pii/browser-analyze
# ---------------------------------------------------------------------------

class BrowserTab(BaseModel):
    """A single browser tab extracted from the tab bar region."""
    title: str = Field(..., description="Tab title text as read by OCR")
    active: bool = Field(
        ...,
        description="True if this is the currently visible/active tab"
    )


class OCRDetection(BaseModel):
    """
    A single OCR word or phrase from the page body.
    One entry per OCR token — not per PII entity.
    """
    text: str = Field(..., description="The word/phrase text")
    bbox: Optional[List[int]] = Field(
        None,
        description="Pixel coordinates [x1, y1, x2, y2]"
    )
    ocr_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="OCR confidence for this word (0–1)"
    )


class BrowserAnalysisResult(BaseModel):
    """
    Complete result of POST /ocr-pii/browser-analyze.

    Works for any input image:
    - Browser screenshots: extracts tabs, URL bar, page content
    - Plain documents: tabs/active_url will be empty/null,
      page_content contains all text
    - ID cards, forms, screenshots: page_content contains all OCR text,
      pii_detected contains all found PII
    """
    tabs: List[BrowserTab] = Field(
        default_factory=list,
        description="Browser tabs found in the tab bar region. "
                    "Empty list if no tab bar detected."
    )
    active_url: Optional[str] = Field(
        None,
        description="URL from the address bar. Null if not found."
    )
    page_title: Optional[str] = Field(
        None,
        description="First prominent heading in the page body. Null if none."
    )
    page_content: str = Field(
        default="",
        description="Full text from the page body as plain text."
    )
    ocr_detections: List[OCRDetection] = Field(
        default_factory=list,
        description="Every OCR word from the page body with bbox and confidence."
    )
    pii_detected: List[PIIEntity] = Field(
        default_factory=list,
        description="PII entities found in page content. "
                    "Same format as /analyze: type, text, bbox, ocr_confidence, pii_score."
    )
