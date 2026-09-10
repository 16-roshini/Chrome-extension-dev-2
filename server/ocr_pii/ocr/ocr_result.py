"""
Helper to construct OCRResult objects from raw engine output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from server.ocr_pii.schemas import BoundingBox, OCRResult, OCRWord


def build_ocr_result(
    words: List[OCRWord],
    engine_name: str,
) -> OCRResult:
    """
    Assemble an OCRResult from a list of OCRWord objects.

    Args:
        words:       List of OCRWord objects produced by an engine.
        engine_name: Name of the engine that produced the words.

    Returns:
        OCRResult with full_text joined from word texts.
    """
    full_text = " ".join(w.text for w in words if w.text.strip())
    return OCRResult(
        words=words,
        full_text=full_text,
        engine_used=engine_name,
    )


def make_bounding_box(
    x: int, y: int, width: int, height: int
) -> BoundingBox:
    """Convenience constructor for BoundingBox."""
    return BoundingBox(x=x, y=y, width=width, height=height)


def tesseract_bbox_to_schema(
    left: int, top: int, width: int, height: int
) -> BoundingBox:
    """Convert Tesseract's (left, top, width, height) to BoundingBox."""
    return BoundingBox(x=left, y=top, width=width, height=height)


def easyocr_bbox_to_schema(
    bbox: List[List[int]],
) -> Optional[BoundingBox]:
    """
    Convert EasyOCR's bounding box format to BoundingBox.

    EasyOCR returns [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (4 corner points).
    We convert to (x, y, width, height).
    """
    try:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x = int(min(xs))
        y = int(min(ys))
        width = int(max(xs) - min(xs))
        height = int(max(ys) - min(ys))
        return BoundingBox(x=x, y=y, width=width, height=height)
    except Exception:
        return None
