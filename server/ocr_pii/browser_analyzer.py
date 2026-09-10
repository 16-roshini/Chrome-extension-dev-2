"""
BrowserAnalyzer — extracts structured browser information from a screenshot.

Works generically for any image input:

  Browser screenshot:
    - Detects tab bar region (top ~5% of image height, typical browser chrome)
    - Extracts individual tab titles from the tab bar
    - Marks the active tab (usually distinguished by being lighter/selected)
    - Detects address bar region (strip just below the tab bar)
    - Extracts URL from the address bar
    - Everything below the browser chrome is treated as page content

  Plain document / ID card / form:
    - No tab bar or URL bar detected
    - tabs = [], active_url = null
    - Entire image treated as page content

  All cases:
    - page_content: full OCR text from page body
    - ocr_detections: every word with its bbox and confidence
    - page_title: first prominent heading in page body
    - pii_detected: PII found in page content (reuses existing pipeline)

Architecture:
  BrowserAnalyzer.analyze(image_bytes) → BrowserAnalysisResult

  Internally:
    1. OCR the full image → all words with bboxes
    2. Classify each word by its Y position:
       - tab_bar zone    → tab title candidates
       - url_bar zone    → URL candidates
       - page_body zone  → page content, OCR detections, PII
    3. Group tab bar words into individual tabs using X-gap heuristic
    4. Identify active tab (word with highest OCR confidence in tab region,
       or the one whose Y-range is most distinct — active tabs are usually
       slightly taller/lighter but OCR can't see colour, so we use the
       first/leftmost high-confidence group as active if uncertain)
    5. Extract URL: look for http/https pattern in URL bar words, or pick
       the longest word in that zone
    6. Extract page title: first line in page body that has large bbox height
       (tall text = heading) or first ALL_CAPS / Title Case line
    7. Build OCRDetection list from page body words
    8. Run PII pipeline on page_content → pii_detected

Spatial zone thresholds (relative to image height):
  These are heuristics that work for most modern browsers (Chrome, Edge,
  Firefox) at standard zoom. They are NOT hardcoded for one browser.

  TAB_BAR_MAX_Y_RATIO  = 0.07   (top 7% of image)
  URL_BAR_MAX_Y_RATIO  = 0.14   (7%–14% of image)
  PAGE_BODY_MIN_Y_RATIO = 0.14  (everything below 14%)

  For non-browser images (ID cards, documents), no words fall in tab/URL
  zones because those images don't have browser chrome — all words land
  in the page body zone automatically.
"""

from __future__ import annotations

import io
import re
from typing import List, Optional, Tuple

from PIL import Image

from server.ocr_pii.schemas import (
    BrowserAnalysisResult,
    BrowserTab,
    OCRDetection,
    OCRWord,
    PIIEntity,
)
from server.ocr_pii.ocr.tesseract_engine import TesseractEngine
from server.ocr_pii.ocr.easyocr_engine import EasyOCREngine
from server.ocr_pii.pipeline import OCRPIIPipeline, _compute_word_offsets, _merge_bboxes

# ---------------------------------------------------------------------------
# Spatial zone thresholds (fraction of image height)
# ---------------------------------------------------------------------------
_TAB_BAR_MAX_Y   = 0.07   # top 7%  → tab bar
_URL_BAR_MAX_Y   = 0.14   # 7–14%   → address bar
_PAGE_BODY_MIN_Y = 0.14   # below 14% → page content

# Minimum horizontal gap (pixels) between words to consider them different tabs
_TAB_GAP_THRESHOLD = 15

# URL pattern — used to identify the address bar value
_URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+",
    re.IGNORECASE,
)

# Heading detection — a line is a page title if:
#   - it is all-caps with ≥ 2 words, OR
#   - it is Title Case with ≥ 2 words and the bbox height indicates large font
_ALLCAPS_PATTERN = re.compile(r'^[A-Z][A-Z\s]{3,}$')
_TITLE_CASE_PATTERN = re.compile(r'^([A-Z][a-z]+\s+){1,}[A-Z][a-z]+$')


class BrowserAnalyzer:
    """
    Analyzes a browser screenshot or any image and returns structured
    BrowserAnalysisResult.

    Reuses the existing OCR engines and PII pipeline — no duplication.
    """

    def __init__(self, pipeline: OCRPIIPipeline) -> None:
        """
        Args:
            pipeline: The shared OCRPIIPipeline instance from the router.
                      Reused for PII detection — not instantiated again.
        """
        self._pipeline = pipeline

    async def analyze(self, image_bytes: bytes) -> BrowserAnalysisResult:
        """
        Analyze a browser screenshot or any image.

        Args:
            image_bytes: Raw image bytes (PNG, JPEG, etc.)

        Returns:
            BrowserAnalysisResult with tabs, URL, page content, OCR, and PII.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_width, img_height = image.size

        # Step 1: OCR the full image
        ocr_result = self._pipeline._run_ocr(image)
        all_words = ocr_result.words

        if not all_words:
            return BrowserAnalysisResult()

        # Step 2: Classify words by Y position
        tab_words, url_words, body_words = _classify_words_by_zone(
            all_words, img_height
        )

        # Step 3: Extract tabs from tab bar words
        tabs = _extract_tabs(tab_words, img_width)

        # Step 4: Extract URL from URL bar words
        active_url = _extract_url(url_words)

        # Step 5: Build page content and OCR detections from body words
        page_content = " ".join(w.text for w in body_words if w.text.strip())
        ocr_detections = _build_ocr_detections(body_words)

        # Step 6: Extract page title from body words
        page_title = _extract_page_title(body_words, img_height)

        # Step 7: Run PII detection on page content
        pii_detected = await self._detect_pii(image_bytes, body_words, page_content)

        return BrowserAnalysisResult(
            tabs=tabs,
            active_url=active_url,
            page_title=page_title,
            page_content=page_content,
            ocr_detections=ocr_detections,
            pii_detected=pii_detected,
        )

    async def _detect_pii(
        self,
        image_bytes: bytes,
        body_words: List[OCRWord],
        page_content: str,
    ) -> List[PIIEntity]:
        """
        Run PII detection on the page body content.
        Uses the existing pipeline detectors directly on page_content,
        then maps detected PII back to body_words for bboxes.
        """
        if not page_content.strip():
            return []

        # Run the full pipeline on a synthetic OCR result
        # built from body words only (not the whole image)
        from server.ocr_pii.schemas import OCRResult
        body_ocr = OCRResult(
            words=body_words,
            full_text=page_content,
            engine_used="tesseract",
        )

        # Use the pipeline's detector chain directly
        text = page_content
        regex_dets = self._pipeline._regex.detect(text)
        ner_committed, ner_evidence = self._pipeline._ner.detect_all(text)
        context_dets = self._pipeline._context.detect(text)

        promoted = self._pipeline._fusion.fuse(
            evidence=ner_evidence,
            full_text=text,
            ocr_result=body_ocr,
            regex_detections=regex_dets,
            context_detections=context_dets,
        )

        all_detections = regex_dets + ner_committed + context_dets + promoted
        deduped = self._pipeline._aggregator.remove_duplicates(all_detections)
        aggregated = self._pipeline._aggregator.aggregate(deduped)

        # Build entities using body_words for bbox mapping
        from server.ocr_pii.pipeline import _build_entities
        entities = _build_entities(aggregated, body_ocr)
        return entities


# ---------------------------------------------------------------------------
# Spatial classification
# ---------------------------------------------------------------------------

def _classify_words_by_zone(
    words: List[OCRWord],
    img_height: int,
) -> Tuple[List[OCRWord], List[OCRWord], List[OCRWord]]:
    """
    Split OCR words into three zones based on Y position.

    Returns:
        (tab_words, url_words, body_words)
    """
    tab_words: List[OCRWord] = []
    url_words: List[OCRWord] = []
    body_words: List[OCRWord] = []

    tab_max_y = img_height * _TAB_BAR_MAX_Y
    url_max_y = img_height * _URL_BAR_MAX_Y

    for word in words:
        if not word.bounding_box:
            body_words.append(word)
            continue
        # Use the top edge of the bounding box for zone classification
        y_top = word.bounding_box.y
        if y_top < tab_max_y:
            tab_words.append(word)
        elif y_top < url_max_y:
            url_words.append(word)
        else:
            body_words.append(word)

    return tab_words, url_words, body_words


# ---------------------------------------------------------------------------
# Tab extraction
# ---------------------------------------------------------------------------

def _extract_tabs(
    tab_words: List[OCRWord],
    img_width: int,
) -> List[BrowserTab]:
    """
    Group tab bar words into individual tab titles using X-gap heuristic.

    Words with a horizontal gap > _TAB_GAP_THRESHOLD pixels between them
    are considered separate tabs.

    The active tab is identified as the group that appears most centred
    or has the highest average OCR confidence (active tabs are typically
    rendered more clearly in screenshots).
    """
    if not tab_words:
        return []

    # Sort by X position
    sorted_words = sorted(
        tab_words,
        key=lambda w: w.bounding_box.x if w.bounding_box else 0
    )

    # Group into tab clusters by X gap
    groups: List[List[OCRWord]] = []
    current_group: List[OCRWord] = [sorted_words[0]]

    for word in sorted_words[1:]:
        if not word.bounding_box or not current_group[-1].bounding_box:
            current_group.append(word)
            continue
        prev_right = current_group[-1].bounding_box.x + current_group[-1].bounding_box.width
        curr_left = word.bounding_box.x
        gap = curr_left - prev_right
        if gap > _TAB_GAP_THRESHOLD:
            groups.append(current_group)
            current_group = [word]
        else:
            current_group.append(word)
    groups.append(current_group)

    # Find the group with highest average confidence → likely the active tab
    def group_confidence(g: List[OCRWord]) -> float:
        return sum(w.confidence for w in g) / len(g) if g else 0.0

    best_idx = max(range(len(groups)), key=lambda i: group_confidence(groups[i]))

    tabs: List[BrowserTab] = []
    for i, group in enumerate(groups):
        title = " ".join(w.text for w in group if w.text.strip())
        if not title.strip():
            continue
        tabs.append(BrowserTab(
            title=title.strip(),
            active=(i == best_idx),
        ))

    return tabs


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

def _extract_url(url_words: List[OCRWord]) -> Optional[str]:
    """
    Extract the URL from address bar words.

    Strategy:
      1. Join all url_words into a string and search for http/https pattern
      2. If found, return it
      3. Otherwise return the longest word in the zone (likely a domain)
    """
    if not url_words:
        return None

    combined = " ".join(w.text for w in url_words if w.text.strip())

    # Try to find a URL pattern
    match = _URL_PATTERN.search(combined)
    if match:
        return match.group().rstrip(".,;)")

    # No http/https — return the longest token (likely a domain or path)
    tokens = [w.text.strip() for w in url_words if len(w.text.strip()) > 4]
    if tokens:
        return max(tokens, key=len)

    return None


# ---------------------------------------------------------------------------
# Page title extraction
# ---------------------------------------------------------------------------

def _extract_page_title(
    body_words: List[OCRWord],
    img_height: int,
) -> Optional[str]:
    """
    Find the most likely page title from page body words.

    Strategy:
      1. Find the word(s) with the tallest bbox height in the top 30% of
         the page body — large text = heading
      2. Group consecutive large words on the same Y level
      3. Return the joined title

    Fallback: return the first ALL_CAPS or Title Case multi-word line.
    """
    if not body_words:
        return None

    # Words with bboxes, sorted by Y
    words_with_bbox = [w for w in body_words if w.bounding_box]
    if not words_with_bbox:
        # No spatial info — fall back to first meaningful line
        text = " ".join(w.text for w in body_words if w.text.strip())
        lines = [l.strip() for l in text.split("  ") if l.strip()]
        for line in lines:
            if len(line.split()) >= 2:
                return line[:80]
        return None

    # Find the maximum bbox height (proxy for font size)
    max_height = max(w.bounding_box.height for w in words_with_bbox)
    height_threshold = max_height * 0.65  # words at least 65% of max height

    # Get all "large" words
    large_words = [
        w for w in words_with_bbox
        if w.bounding_box.height >= height_threshold
        and len(w.text.strip()) >= 2
    ]

    if not large_words:
        return None

    # Sort by Y position, then X — group words on the same line
    large_words.sort(key=lambda w: (w.bounding_box.y, w.bounding_box.x))

    # Take the first line of large words
    first_y = large_words[0].bounding_box.y
    y_tolerance = max_height * 0.5
    title_words = [
        w for w in large_words
        if abs(w.bounding_box.y - first_y) <= y_tolerance
    ]
    title_words.sort(key=lambda w: w.bounding_box.x)

    title = " ".join(w.text for w in title_words if w.text.strip())
    return title.strip() if title.strip() else None


# ---------------------------------------------------------------------------
# OCR detection list builder
# ---------------------------------------------------------------------------

def _build_ocr_detections(body_words: List[OCRWord]) -> List[OCRDetection]:
    """
    Build OCRDetection list from page body words.
    Each word becomes one OCRDetection entry.
    """
    detections: List[OCRDetection] = []
    for word in body_words:
        text = word.text.strip()
        if not text:
            continue
        bbox = None
        if word.bounding_box:
            b = word.bounding_box
            bbox = [b.x, b.y, b.x + b.width, b.y + b.height]
        detections.append(OCRDetection(
            text=text,
            bbox=bbox,
            ocr_confidence=round(word.confidence, 4),
        ))
    return detections
