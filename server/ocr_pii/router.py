"""
FastAPI router for OCR + PII detection endpoints.

Endpoints:
    POST /ocr-pii/analyze       — full OCR + PII pipeline
    POST /ocr-pii/ocr-only      — OCR only, no PII detection
    GET  /ocr-pii/health        — health check
    GET  /ocr-pii/supported-pii — list all supported PII categories
"""

from __future__ import annotations

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from server.ocr_pii.pipeline import OCRPIIPipeline
from server.ocr_pii.schemas import (
    DetectResponse,
    GroupedDetectResponse,
    HealthResponse,
    OCROnlyResponse,
    PIICategory,
    PIIEntity,
    SupportedPIIResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr-pii", tags=["OCR + PII"])

# Allowed image MIME types
_ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


# ---------------------------------------------------------------------------
# Pipeline singleton — one instance per worker process
# ---------------------------------------------------------------------------

# Pipeline singleton — one instance per worker process
# Using a module-level variable instead of lru_cache so that
# code changes (--reload) are always picked up on the next request.
_pipeline_instance: OCRPIIPipeline | None = None


def _get_pipeline() -> OCRPIIPipeline:
    """Return the shared OCRPIIPipeline instance (created once per process)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = OCRPIIPipeline()
    return _pipeline_instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _read_image_bytes(file: UploadFile) -> bytes:
    """
    Validate and read uploaded image bytes.

    Raises:
        HTTPException 415 if content type is not an image.
        HTTPException 400 if the file is empty.
    """
    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                f"Allowed types: {sorted(_ALLOWED_CONTENT_TYPES)}"
            ),
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    return image_bytes


# ---------------------------------------------------------------------------
# GET /ocr-pii/health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns availability status of Tesseract, EasyOCR, and spaCy.",
)
async def health() -> HealthResponse:
    """Check whether all required components are available."""
    # Check Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        tesseract_ok = True
    except Exception:
        tesseract_ok = False

    # Check EasyOCR
    try:
        import easyocr  # noqa: F401
        easyocr_ok = True
    except ImportError:
        easyocr_ok = False

    # Check spaCy model
    try:
        import spacy
        spacy.load("en_core_web_sm")
        spacy_ok = True
    except Exception:
        spacy_ok = False

    return HealthResponse(
        status="ok" if (tesseract_ok and spacy_ok) else "degraded",
        tesseract_available=tesseract_ok,
        easyocr_available=easyocr_ok,
        spacy_model_loaded=spacy_ok,
    )


# ---------------------------------------------------------------------------
# GET /ocr-pii/supported-pii
# ---------------------------------------------------------------------------

@router.get(
    "/supported-pii",
    response_model=SupportedPIIResponse,
    summary="List supported PII categories",
    description="Returns all PII category names this module can detect.",
)
async def supported_pii() -> SupportedPIIResponse:
    """Return all supported PII category names."""
    categories = [c.value for c in PIICategory]
    return SupportedPIIResponse(
        categories=categories,
        total=len(categories),
    )


# ---------------------------------------------------------------------------
# POST /ocr-pii/analyze
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# POST /ocr-pii/analyze
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    summary="Full OCR + PII detection",
    description=(
        "Upload an image. Runs Tesseract OCR (with EasyOCR fallback), "
        "then detects PII using regex, spaCy NER, and context detection. "
        "Add ?grouped=true to get results grouped by PII category "
        "(name, email, phone, etc.) instead of a flat list."
    ),
)
async def analyze(
    file: UploadFile = File(..., description="Image file (PNG, JPEG, BMP, TIFF, WebP)"),
    grouped: bool = False,
):
    """
    Run the full OCR + PII pipeline on an uploaded image.

    - grouped=false (default): returns {"detections": [...]}
    - grouped=true: returns {"name": [...], "email": [...], ...}
    """
    try:
        image_bytes = await _read_image_bytes(file)
        pipeline = _get_pipeline()
        result: DetectResponse = await pipeline.analyze(image_bytes)

        if grouped:
            return _group_detections(result.detections)
        return result

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Pipeline error in /analyze: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )


# ---------------------------------------------------------------------------
# POST /ocr-pii/ocr-only
# ---------------------------------------------------------------------------

@router.post(
    "/ocr-only",
    response_model=OCROnlyResponse,
    summary="OCR only (no PII detection)",
    description=(
        "Upload an image. Runs Tesseract OCR (with EasyOCR fallback). "
        "Returns extracted text and word-level bounding boxes. "
        "No PII detection is performed."
    ),
)
async def ocr_only(
    file: UploadFile = File(..., description="Image file (PNG, JPEG, BMP, TIFF, WebP)"),
) -> OCROnlyResponse:
    """Run OCR only on an uploaded image."""
    try:
        image_bytes = await _read_image_bytes(file)
        pipeline = _get_pipeline()
        result = await pipeline.ocr_only(image_bytes)
        return OCROnlyResponse(success=True, result=result)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Pipeline error in /ocr-only: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )


# ---------------------------------------------------------------------------
# Helper — group flat detections by PII category
# ---------------------------------------------------------------------------

# Maps PIIEntity.type (uppercase string) to GroupedDetectResponse field name
_TYPE_TO_FIELD: dict[str, str] = {
    "PERSON_NAME":    "name",
    "EMAIL":          "email",
    "PHONE":          "phone",
    "DATE_OF_BIRTH":  "date_of_birth",
    "ADDRESS":        "address",
    "AADHAAR":        "aadhaar",
    "PAN":            "pan",
    "ORGANIZATION":   "organization",
    "PASSWORD":       "password",
    "CREDIT_CARD":    "credit_card",
    "BANK_ACCOUNT":   "bank_account",
    "IFSC":           "ifsc",
    "PASSPORT":       "passport",
    "IP_ADDRESS":     "ip_address",
    "URL":            "url",
}


def _group_detections(detections: list[PIIEntity]) -> GroupedDetectResponse:
    """
    Group a flat list of PIIEntity objects by category into
    a GroupedDetectResponse.

    Each field in GroupedDetectResponse is a list of PIIEntity objects.
    Fields with no detections are null (omitted from JSON by default).
    """
    groups: dict[str, list[PIIEntity]] = {}
    for entity in detections:
        field = _TYPE_TO_FIELD.get(entity.type)
        if field is None:
            continue
        groups.setdefault(field, []).append(entity)

    return GroupedDetectResponse(**groups)


# ---------------------------------------------------------------------------
# POST /ocr-pii/browser-analyze
# ---------------------------------------------------------------------------

from server.ocr_pii.browser_analyzer import BrowserAnalyzer
from server.ocr_pii.schemas import BrowserAnalysisResult

# Browser analyzer singleton — reuses the same pipeline instance
_browser_analyzer: "BrowserAnalyzer | None" = None


def _get_browser_analyzer() -> "BrowserAnalyzer":
    global _browser_analyzer
    if _browser_analyzer is None:
        _browser_analyzer = BrowserAnalyzer(pipeline=_get_pipeline())
    return _browser_analyzer


@router.post(
    "/browser-analyze",
    response_model=BrowserAnalysisResult,
    summary="Browser screenshot analysis",
    description=(
        "Upload a browser screenshot or any image. "
        "Extracts: browser tabs, active URL, page title, page content, "
        "word-level OCR detections with bboxes, and PII found in the page. "
        "Works for any image — browser screenshots, forms, ID cards, documents."
    ),
)
async def browser_analyze(
    file: UploadFile = File(..., description="Image file (PNG, JPEG, BMP, TIFF, WebP)"),
) -> BrowserAnalysisResult:
    """
    Analyze a browser screenshot and return structured information:

    - tabs: list of browser tab titles and which one is active
    - active_url: URL from the address bar
    - page_title: main heading of the visible page
    - page_content: full text from the page body
    - ocr_detections: every OCR word with bbox and confidence
    - pii_detected: all PII found (type, text, bbox, ocr_confidence, pii_score)
    """
    try:
        image_bytes = await _read_image_bytes(file)
        analyzer = _get_browser_analyzer()
        return await analyzer.analyze(image_bytes)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Pipeline error in /browser-analyze: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Browser analysis error: {exc}",
        )
