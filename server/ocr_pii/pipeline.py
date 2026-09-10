"""
OCR + PII Detection Pipeline.

Refactored orchestration order (evidence-based architecture):

  1. OCR  — Tesseract (primary) or EasyOCR (fallback)
            → OCRResult: words + full_text + per-word confidence + bboxes

  2. Detectors run on full_text:
       a. RegexDetector         → committed PIIDetection list
       b. NERDetector.detect_all() → committed PIIDetection list (PERSON/ORG)
                                  + DetectorEvidence list (DATE/GPE)
       c. ContextDetector       → committed PIIDetection list

  3. FusionLayer.fuse(evidence, full_text, ocr_result, regex, context)
       → promotes evidence to PIIDetection only when supporting signals exist
       → every DATE/GPE without context or format validation is discarded here

  4. Combine all committed detections + promoted evidence

  5. Aggregator.remove_duplicates()  — exact 4-tuple dedup
     Aggregator.aggregate()          — category-scoped sweep-line merge

  6. _build_entities()
       → find overlapping OCR words for each detection (char span matching)
       → compute per-entity OCR confidence from matched words
       → apply per-source OCR confidence gate (drop below threshold)
       → merge word bboxes → [x1, y1, x2, y2]
       → apply final confidence scoring using PER-ENTITY OCR confidence
         (not global image average — this was the previous bug)
       → emit PIIEntity

  API output unchanged: type, text, bbox, ocr_confidence, pii_score
"""

from __future__ import annotations

import hashlib
import io
from typing import List, Optional

from PIL import Image

from server.ocr_pii.schemas import (
    DetectResponse,
    OCRResult,
    OCRWord,
    PIIDetection,
    PIIEntity,
)
from server.ocr_pii.ocr.tesseract_engine import TesseractEngine
from server.ocr_pii.ocr.easyocr_engine import EasyOCREngine
from server.ocr_pii.detectors.regex_detector import RegexDetector
from server.ocr_pii.detectors.ner_detector import NERDetector
from server.ocr_pii.detectors.context_detector import ContextDetector
from server.ocr_pii.scoring.aggregator import Aggregator
from server.ocr_pii.scoring.fusion import FusionLayer
from server.ocr_pii.scoring.scorer import compute_confidence


class OCRPIIPipeline:
    """
    Main pipeline: image bytes → clean PIIEntity list.

    Evidence model:
      - Detectors produce either committed PIIDetection or DetectorEvidence.
      - FusionLayer decides whether evidence becomes a detection.
      - Final scoring uses per-entity OCR confidence, not image average.
    """

    def __init__(self) -> None:
        self._tesseract = TesseractEngine()
        self._easyocr = EasyOCREngine()
        self._regex = RegexDetector()
        self._ner = NERDetector()
        self._context = ContextDetector()
        self._aggregator = Aggregator()
        self._fusion = FusionLayer()

    async def analyze(self, image_bytes: bytes) -> DetectResponse:
        """
        Run the full OCR + PII pipeline.

        Returns DetectResponse with PIIEntity list.
        Each entity has: type, text, bbox, ocr_confidence, pii_score.
        """
        image = _bytes_to_image(image_bytes)
        ocr_result = self._run_ocr(image)

        if not ocr_result.full_text.strip():
            return DetectResponse(detections=[])

        text = ocr_result.full_text

        # --- Step 2: Detectors ---
        regex_dets = self._regex.detect(text)

        # NERDetector runs spaCy once, returns committed + evidence
        ner_committed, ner_evidence = self._ner.detect_all(text)

        context_dets = self._context.detect(text)

        # --- Step 3: FusionLayer promotes evidence ---
        promoted = self._fusion.fuse(
            evidence=ner_evidence,
            full_text=text,
            ocr_result=ocr_result,
            regex_detections=regex_dets,
            context_detections=context_dets,
        )

        # --- Step 4: Combine all committed + promoted ---
        all_detections: List[PIIDetection] = (
            regex_dets + ner_committed + context_dets + promoted
        )

        # --- Step 5: Dedup + aggregate ---
        deduped = self._aggregator.remove_duplicates(all_detections)
        aggregated = self._aggregator.aggregate(deduped)

        # --- Step 6: BBox mapping + per-entity rescore + OCR gate ---
        entities = _build_entities(aggregated, ocr_result)

        return DetectResponse(detections=entities)

    async def ocr_only(self, image_bytes: bytes) -> OCRResult:
        """Run OCR only — no PII detection."""
        image = _bytes_to_image(image_bytes)
        return self._run_ocr(image)

    def _run_ocr(self, image: Image.Image) -> OCRResult:
        if self._tesseract.is_available():
            result = self._tesseract.run(image)
            if result.full_text.strip():
                return result
            if self._easyocr.is_available():
                fallback = self._easyocr.run(image)
                if fallback.full_text.strip():
                    return fallback
            return result
        if self._easyocr.is_available():
            return self._easyocr.run(image)
        return OCRResult(words=[], full_text="", engine_used="none")


# ---------------------------------------------------------------------------
# BBox mapping + per-entity scoring
# ---------------------------------------------------------------------------

# Minimum per-entity OCR confidence required per detection source.
# Detections below their threshold are dropped before scoring.
_OCR_MIN_BY_SOURCE: dict[str, float] = {
    "regex":      0.50,   # deterministic patterns — tolerates moderate OCR
    "spacy_ner":  0.60,   # statistical model — lowered to catch valid NER on forms
    "context":    0.55,   # keyword heuristic — lowered to catch password/name on forms
    "aggregated": 0.55,   # combined evidence — moderate threshold
}


def _build_entities(
    detections: List[PIIDetection],
    ocr_result: OCRResult,
) -> List[PIIEntity]:
    """
    Convert internal PIIDetection objects to clean PIIEntity objects.

    For each detection:
      1. Find all OCR words overlapping the detection's char span.
      2. Compute per-entity OCR confidence from those words.
      3. Apply OCR precision gate — drop if below per-source minimum.
      4. Compute final pii_score using per-entity OCR confidence
         (NOT the global image average used previously).
      5. Merge word bboxes → [x1, y1, x2, y2].
      6. Emit PIIEntity.
    """
    full_text = ocr_result.full_text
    words = ocr_result.words
    word_offsets = _compute_word_offsets(full_text, words)

    entities: List[PIIEntity] = []

    for det in detections:
        # Find OCR words overlapping this detection's span
        matched_words = [
            words[i] for i, (ws, we) in enumerate(word_offsets)
            if ws < det.char_end and we > det.char_start
        ]

        # Per-entity OCR confidence
        ocr_conf = _words_avg_confidence(matched_words)

        # Precision gate — drop low-quality detections
        min_required = _OCR_MIN_BY_SOURCE.get(det.source.value, 0.60)
        if ocr_conf < min_required:
            continue

        # Final pii_score uses per-entity OCR confidence
        # (previously used global image average — this was incorrect)
        pii_score = compute_confidence(det, ocr_confidence=ocr_conf)

        bbox = _merge_bboxes(matched_words)

        entities.append(PIIEntity(
            type=det.category.value.upper(),
            text=det.text,
            bbox=bbox,
            ocr_confidence=round(ocr_conf, 4),
            pii_score=round(pii_score, 4),
        ))

    return entities


# ---------------------------------------------------------------------------
# Shared helpers (also imported by fusion.py for OCR word matching)
# ---------------------------------------------------------------------------

def _compute_word_offsets(
    full_text: str,
    words: List[OCRWord],
) -> List[tuple[int, int]]:
    """
    Forward cursor scan to find each OCR word's (start, end) in full_text.
    Handles repeated words correctly by always scanning forward.
    """
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


def _merge_bboxes(words: List[OCRWord]) -> Optional[List[int]]:
    """
    Merge word bounding boxes into one enclosing rectangle.
    Returns [x1, y1, x2, y2] or None if no bboxes available.
    """
    boxes = [w.bounding_box for w in words if w.bounding_box is not None]
    if not boxes:
        return None
    x1 = min(b.x for b in boxes)
    y1 = min(b.y for b in boxes)
    x2 = max(b.x + b.width for b in boxes)
    y2 = max(b.y + b.height for b in boxes)
    return [x1, y1, x2, y2]


def _words_avg_confidence(words: List[OCRWord]) -> float:
    """Average OCR confidence of matched words. Returns 1.0 if none matched."""
    if not words:
        return 1.0
    return sum(w.confidence for w in words) / len(words)


# ---------------------------------------------------------------------------
# Module-level helpers (used by tests)
# ---------------------------------------------------------------------------

def _bytes_to_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _md5_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()
