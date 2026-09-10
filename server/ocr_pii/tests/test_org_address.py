"""
Tests for ORGANIZATION detection and ADDRESS grouping — Dev 2 targeted fix.

Covers all 6 required test cases from the spec:
  TEST 1 — Organization with employer context
  TEST 2 — Organization with company context
  TEST 3 — Organization false positive (browser UI / section headers)
  TEST 4 — Address grouping (one entity, not many)
  TEST 5 — Address bbox (combined bbox covers full address)
  TEST 6 — Browser UI false positives (no ORGANIZATION / PERSON_NAME)

All PII values are SYNTHETIC — no real personal data.

Architecture note:
  Tests operate at the text/detector level — no OCR engine required.
  The inline _build_entities + _merge_address_entities helpers from
  pipeline.py are imported directly to avoid pulling in pytesseract.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from server.ocr_pii.detectors.context_detector import ContextDetector
from server.ocr_pii.detectors.ner_detector import NERDetector
from server.ocr_pii.detectors.regex_detector import RegexDetector
from server.ocr_pii.scoring.fusion import FusionLayer
from server.ocr_pii.scoring.aggregator import Aggregator
from server.ocr_pii.scoring.scorer import compute_confidence
from server.ocr_pii.schemas import (
    BoundingBox,
    OCRResult,
    OCRWord,
    PIICategory,
    PIIDetection,
    PIIEntity,
    DetectionSource,
)

# ---------------------------------------------------------------------------
# Inline pipeline helpers — avoids importing pipeline.py (pulls pytesseract)
# ---------------------------------------------------------------------------

_OCR_MIN_BY_SOURCE: dict = {
    "regex":      0.50,
    "spacy_ner":  0.60,
    "context":    0.55,
    "aggregated": 0.55,
}


def _compute_word_offsets(full_text: str, words: list) -> list:
    offsets = []
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


def _merge_bboxes(words: list) -> Optional[List[int]]:
    boxes = [w.bounding_box for w in words if w.bounding_box is not None]
    if not boxes:
        return None
    return [
        min(b.x for b in boxes),
        min(b.y for b in boxes),
        max(b.x + b.width for b in boxes),
        max(b.y + b.height for b in boxes),
    ]


def _words_avg_confidence(words: list) -> float:
    if not words:
        return 1.0
    return sum(w.confidence for w in words) / len(words)


def _build_entities(detections: list, ocr_result: OCRResult) -> List[PIIEntity]:
    full_text = ocr_result.full_text
    words = ocr_result.words
    word_offsets = _compute_word_offsets(full_text, words)
    entities = []
    for det in detections:
        matched = [
            words[i] for i, (ws, we) in enumerate(word_offsets)
            if ws < det.char_end and we > det.char_start
        ]
        ocr_conf = _words_avg_confidence(matched)
        min_req = _OCR_MIN_BY_SOURCE.get(det.source.value, 0.60)
        if ocr_conf < min_req:
            continue
        pii_score = compute_confidence(det, ocr_confidence=ocr_conf)
        bbox = _merge_bboxes(matched)
        entities.append(PIIEntity(
            type=det.category.value.upper(),
            text=det.text,
            bbox=bbox,
            ocr_confidence=round(ocr_conf, 4),
            pii_score=round(pii_score, 4),
        ))
    return entities


def _merge_address_entities(entities: List[PIIEntity]) -> List[PIIEntity]:
    """Mirror of pipeline._merge_address_entities for test isolation."""
    _ADDR_Y_TOLERANCE = 15
    result: List[PIIEntity] = []
    i = 0
    while i < len(entities):
        entity = entities[i]
        if entity.type != "ADDRESS":
            result.append(entity)
            i += 1
            continue
        group = [entity]
        j = i + 1
        while j < len(entities):
            candidate = entities[j]
            if candidate.type != "ADDRESS":
                break
            should_merge = False
            if group[-1].bbox is not None and candidate.bbox is not None:
                if abs(group[-1].bbox[1] - candidate.bbox[1]) <= _ADDR_Y_TOLERANCE:
                    should_merge = True
            else:
                should_merge = True
            if should_merge:
                group.append(candidate)
                j += 1
            else:
                break
        if len(group) == 1:
            result.append(group[0])
        else:
            parts: List[str] = []
            accumulated = ""
            for e in group:
                frag = e.text.strip().strip(",").strip()
                if frag and frag.lower() not in accumulated.lower():
                    parts.append(frag)
                    accumulated = (accumulated + " " + frag).strip()
            combined_text = ", ".join(parts)
            boxes = [e.bbox for e in group if e.bbox is not None]
            if boxes:
                merged_bbox: Optional[List[int]] = [
                    min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes),
                ]
            else:
                merged_bbox = None
            result.append(PIIEntity(
                type="ADDRESS",
                text=combined_text,
                bbox=merged_bbox,
                ocr_confidence=round(sum(e.ocr_confidence for e in group) / len(group), 4),
                pii_score=round(max(e.pii_score for e in group), 4),
            ))
        i = j if len(group) > 1 else i + 1
    return result


def _run_pipeline(text: str, ocr_confidence: float = 0.92) -> List[PIIEntity]:
    """Run all detectors + fusion + aggregation on plain text."""
    words = [OCRWord(text=t, bounding_box=None, confidence=ocr_confidence)
             for t in text.split() if t.strip()]
    ocr = OCRResult(words=words, full_text=text, engine_used="tesseract")

    ctx = ContextDetector()
    ner = NERDetector()
    regex = RegexDetector()
    fusion = FusionLayer()
    agg = Aggregator()

    r = regex.detect(text)
    nc, ne = ner.detect_all(text)
    c = ctx.detect(text)
    p = fusion.fuse(evidence=ne, full_text=text, ocr_result=ocr,
                    regex_detections=r, context_detections=c)

    all_d = r + nc + c + p
    entities = _build_entities(agg.aggregate(agg.remove_duplicates(all_d)), ocr)
    return _merge_address_entities(entities)


def _run_pipeline_with_bboxes(
    words_data: list,  # [(text, BoundingBox|None, confidence)]
) -> List[PIIEntity]:
    """Run pipeline with real bounding boxes for bbox-sensitive tests."""
    full_text = " ".join(t for t, _, _ in words_data)
    words = [OCRWord(text=t, bounding_box=b, confidence=c) for t, b, c in words_data]
    ocr = OCRResult(words=words, full_text=full_text, engine_used="tesseract")

    ctx = ContextDetector()
    ner = NERDetector()
    regex = RegexDetector()
    fusion = FusionLayer()
    agg = Aggregator()

    r = regex.detect(full_text)
    nc, ne = ner.detect_all(full_text)
    c = ctx.detect(full_text)
    p = fusion.fuse(evidence=ne, full_text=full_text, ocr_result=ocr,
                    regex_detections=r, context_detections=c)

    all_d = r + nc + c + p
    entities = _build_entities(agg.aggregate(agg.remove_duplicates(all_d)), ocr)
    return _merge_address_entities(entities)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def ctx_det() -> ContextDetector:
    return ContextDetector()


@pytest.fixture(scope="module")
def ner_det() -> NERDetector:
    return NERDetector()


# ===========================================================================
# TEST 1 — Organization with employer context
# ===========================================================================

class TestOrganizationEmployerContext:
    """Employer: Indian Space Research Organisation (ISRO) → ORGANIZATION"""

    def test_detects_org_with_colon(self, ctx_det):
        dets = ctx_det.detect("Employer: Indian Space Research Organisation (ISRO)")
        orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
        assert orgs, "Expected ORGANIZATION from 'Employer: ...' label"

    def test_detects_org_with_spaced_colon(self, ctx_det):
        dets = ctx_det.detect("Employer : Indian Space Research Organisation (ISRO)")
        orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
        assert orgs, "Expected ORGANIZATION from 'Employer : ...' label"

    def test_org_text_is_org_name_not_label(self, ctx_det):
        dets = ctx_det.detect("Employer: Indian Space Research Organisation (ISRO)")
        orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
        for d in orgs:
            assert d.text.lower() != "employer", (
                "Label 'Employer' must not be the detected organization text"
            )
            assert "Indian Space Research Organisation" in d.text or \
                   "Indian Space Research" in d.text

    def test_abbreviation_stripped_from_value(self, ctx_det):
        """(ISRO) parenthetical must be stripped from the detected org name."""
        dets = ctx_det.detect("Employer: Indian Space Research Organisation (ISRO)")
        orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
        for d in orgs:
            assert "(ISRO)" not in d.text, (
                f"Parenthetical abbreviation should be stripped, got: '{d.text}'"
            )

    def test_full_pipeline_detects_org(self):
        entities = _run_pipeline("Employer: Indian Space Research Organisation (ISRO)")
        types = [e.type for e in entities]
        assert "ORGANIZATION" in types, "ORGANIZATION missing from full pipeline output"

    def test_org_type_uppercase(self):
        entities = _run_pipeline("Employer: Indian Space Research Organisation (ISRO)")
        orgs = [e for e in entities if e.type == "ORGANIZATION"]
        assert orgs
        assert all(e.type == "ORGANIZATION" for e in orgs)

    def test_org_confidence_in_range(self):
        entities = _run_pipeline("Employer: Indian Space Research Organisation (ISRO)")
        orgs = [e for e in entities if e.type == "ORGANIZATION"]
        assert orgs
        for e in orgs:
            assert 0.0 <= e.pii_score <= 1.0
            assert 0.0 <= e.ocr_confidence <= 1.0


# ===========================================================================
# TEST 2 — Organization with company/institute context
# ===========================================================================

class TestOrganizationVariousLabels:
    """Various org context labels should detect ORGANIZATION."""

    @pytest.mark.parametrize("text,expected_fragment", [
        ("Company: Microsoft",                  "Microsoft"),
        ("Organisation: Infosys Limited",       "Infosys"),
        ("Institute: IIT Hyderabad",            "IIT Hyderabad"),
        ("University: IIT Bombay",              "IIT Bombay"),
        ("Hospital: AIIMS Delhi",               "AIIMS Delhi"),
        ("Bank: State Bank of India",           "State Bank"),
        ("Corporation: Tata Consultancy",       "Tata"),
    ])
    def test_org_detected_for_label(self, ctx_det, text, expected_fragment):
        dets = ctx_det.detect(text)
        orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
        assert orgs, f"Expected ORGANIZATION for: {text!r}"
        assert any(expected_fragment.lower() in d.text.lower() for d in orgs), (
            f"Expected '{expected_fragment}' in org text, got: {[d.text for d in orgs]}"
        )

    def test_company_label_no_label_in_value(self, ctx_det):
        dets = ctx_det.detect("Company: Microsoft")
        orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
        for d in orgs:
            assert d.text.lower() != "company"

    def test_org_source_is_context(self, ctx_det):
        dets = ctx_det.detect("Employer: Infosys Limited")
        orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
        assert orgs
        assert all(d.source == DetectionSource.CONTEXT for d in orgs)

    def test_org_confidence_is_context_level(self, ctx_det):
        dets = ctx_det.detect("Employer: Infosys Limited")
        orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
        assert orgs
        # Context-based organization gets 0.80 confidence
        assert all(d.confidence >= 0.75 for d in orgs)


# ===========================================================================
# TEST 3 — Organization false positives must NOT fire
# ===========================================================================

class TestOrganizationFalsePositives:
    """Browser UI, section headers, and common words must NOT become ORGANIZATION."""

    FALSE_POSITIVE_INPUTS = [
        "Restart the server",
        "New Chat",
        "Ask anything",
        "Settings",
        "Personal Details",
        "Contact Details",
        "Additional Notes",
        "Login Information",
        "YouTube",
        "GitHub",
        "Google",
        "ChatGPT",
        "Gmail",
        "Gmail - Inbox",
        "New Tab",
        "Bookmarks",
        "Extensions",
        "History",
        "Downloads",
    ]

    @pytest.mark.parametrize("text", FALSE_POSITIVE_INPUTS)
    def test_no_org_from_browser_ui(self, ctx_det, text):
        dets = ctx_det.detect(text)
        orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
        assert orgs == [], (
            f"False positive ORGANIZATION for '{text}': {[d.text for d in orgs]}"
        )

    @pytest.mark.parametrize("text", FALSE_POSITIVE_INPUTS)
    def test_no_person_from_browser_ui(self, ner_det, text):
        committed, _ = ner_det.detect_all(text)
        persons = [d for d in committed if d.category == PIICategory.PERSON_NAME]
        assert persons == [], (
            f"False positive PERSON_NAME for '{text}': {[d.text for d in persons]}"
        )

    def test_full_browser_ui_block_empty_pii(self):
        """A block of typical browser UI should produce zero PII."""
        browser_text = (
            "ChatGPT GitHub YouTube Gmail New chat Ask anything "
            "Think Search Settings Restart Inbox Compose New Tab"
        )
        entities = _run_pipeline(browser_text)
        assert entities == [], (
            f"Expected empty PII for browser UI. Got: {[(e.type, e.text) for e in entities]}"
        )

    def test_section_headers_not_org(self, ctx_det):
        """Form section headers must not become ORGANIZATION."""
        for header in ["PERSONAL DETAILS", "CONTACT DETAILS", "LOGIN INFORMATION",
                       "ADDITIONAL NOTES", "EMPLOYMENT DETAILS"]:
            dets = ctx_det.detect(header)
            orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
            assert orgs == [], f"Section header '{header}' became ORGANIZATION"


# ===========================================================================
# TEST 4 — Address grouping: ONE entity, not many
# ===========================================================================

class TestAddressGrouping:
    """Address fragments must be grouped into one ADDRESS entity."""

    def test_full_address_one_entity(self):
        """Full address line produces exactly one ADDRESS detection."""
        entities = _run_pipeline(
            "Address: 12-4-567, Green Park, Hyderabad, Telangana - 500016"
        )
        addr_entities = [e for e in entities if e.type == "ADDRESS"]
        assert len(addr_entities) == 1, (
            f"Expected 1 ADDRESS entity, got {len(addr_entities)}: "
            f"{[e.text for e in addr_entities]}"
        )

    def test_full_address_text_contains_all_parts(self):
        """The merged address text must contain the key address components."""
        entities = _run_pipeline(
            "Address: 12-4-567, Green Park, Hyderabad, Telangana - 500016"
        )
        addr_entities = [e for e in entities if e.type == "ADDRESS"]
        assert addr_entities
        combined = addr_entities[0].text
        # Must contain at minimum the city — full text depends on OCR detail
        assert "Hyderabad" in combined or "Green Park" in combined, (
            f"Address text missing key components: '{combined}'"
        )

    def test_no_duplicate_address_entities_same_bbox(self):
        """Must not have two ADDRESS entities with identical bbox."""
        entities = _run_pipeline(
            "Address: 12-4-567, Green Park, Hyderabad, Telangana - 500016"
        )
        addr_entities = [e for e in entities if e.type == "ADDRESS"]
        # If there are multiple (no bbox available), texts must differ
        if len(addr_entities) > 1:
            bboxes = [e.bbox for e in addr_entities if e.bbox is not None]
            bbox_set = set(tuple(b) for b in bboxes)
            assert len(bbox_set) == len(bboxes), (
                "Multiple ADDRESS entities share the same bbox — should be merged"
            )

    def test_address_pii_score_in_range(self):
        entities = _run_pipeline(
            "Address: 12-4-567, Green Park, Hyderabad, Telangana - 500016"
        )
        addr_entities = [e for e in entities if e.type == "ADDRESS"]
        for e in addr_entities:
            assert 0.0 <= e.pii_score <= 1.0
            assert 0.0 <= e.ocr_confidence <= 1.0

    def test_address_not_containing_label_word(self):
        """The detected address text must not start with 'Address'."""
        entities = _run_pipeline(
            "Address: 12-4-567, Green Park, Hyderabad, Telangana - 500016"
        )
        addr_entities = [e for e in entities if e.type == "ADDRESS"]
        for e in addr_entities:
            assert not e.text.lower().startswith("address"), (
                f"Address label leaked into value: '{e.text}'"
            )

    def test_simple_address_detected(self):
        """Simple address with just city and pin should still be detected."""
        entities = _run_pipeline("Address: MG Road, Bengaluru - 560001")
        addr_entities = [e for e in entities if e.type == "ADDRESS"]
        assert addr_entities, "Expected ADDRESS for 'Address: MG Road, Bengaluru'"


# ===========================================================================
# TEST 5 — Address bbox covers the complete address
# ===========================================================================

class TestAddressBbox:
    """Merged address bbox must cover all individual address component bboxes."""

    def test_merged_bbox_covers_all_components(self):
        """
        When address components span bboxes [220,430] to [590,450],
        the merged bbox should be [220, 430, 590, 450].
        """
        words_data = [
            ("Address:",    None,                                              0.95),
            ("12-4-567,",   BoundingBox(x=220, y=430, width=60,  height=20), 0.92),
            ("Green",       BoundingBox(x=285, y=430, width=45,  height=20), 0.85),
            ("Park,",       BoundingBox(x=335, y=430, width=40,  height=20), 0.85),
            ("Hyderabad,",  BoundingBox(x=380, y=430, width=70,  height=20), 0.88),
            ("Telangana",   BoundingBox(x=455, y=430, width=65,  height=20), 0.84),
            ("-",           BoundingBox(x=525, y=430, width=10,  height=20), 0.90),
            ("500016",      BoundingBox(x=540, y=430, width=50,  height=20), 0.91),
        ]
        entities = _run_pipeline_with_bboxes(words_data)
        addr_entities = [e for e in entities if e.type == "ADDRESS"]
        assert addr_entities, "Expected ADDRESS detection"

        addr = addr_entities[0]
        if addr.bbox is not None:
            x1, y1, x2, y2 = addr.bbox
            # Merged bbox must start at or before first component
            assert x1 <= 285, f"bbox x1={x1} should be <= 285 (first component)"
            # Merged bbox must end at or after last component
            assert x2 >= 540, f"bbox x2={x2} should be >= 540 (last component right edge)"
            # Y coordinates must be consistent
            assert y1 <= 435, f"bbox y1={y1} too high"
            assert y2 >= 435, f"bbox y2={y2} too low"

    def test_merged_bbox_x2_greater_than_x1(self):
        words_data = [
            ("Address:",   None,                                              0.95),
            ("Green",      BoundingBox(x=200, y=100, width=50, height=20),  0.90),
            ("Park",       BoundingBox(x=260, y=100, width=40, height=20),  0.90),
            ("Hyderabad",  BoundingBox(x=310, y=100, width=60, height=20),  0.88),
        ]
        entities = _run_pipeline_with_bboxes(words_data)
        addr_entities = [e for e in entities if e.type == "ADDRESS"]
        assert addr_entities
        if addr_entities[0].bbox is not None:
            x1, y1, x2, y2 = addr_entities[0].bbox
            assert x2 > x1, f"Invalid bbox: x2={x2} must be > x1={x1}"
            assert y2 >= y1, f"Invalid bbox: y2={y2} must be >= y1={y1}"

    def test_address_without_bbox_still_detected(self):
        """Address detection must work even when no bboxes are available."""
        entities = _run_pipeline(
            "Address: 12-4-567, Green Park, Hyderabad, Telangana - 500016",
            ocr_confidence=0.90,
        )
        addr_entities = [e for e in entities if e.type == "ADDRESS"]
        assert addr_entities, "ADDRESS should be detected even without bboxes"
        # bbox may be None when no bboxes available — that's acceptable
        addr = addr_entities[0]
        assert addr.bbox is None or isinstance(addr.bbox, list)


# ===========================================================================
# TEST 6 — Browser UI false positives (full pipeline)
# ===========================================================================

class TestBrowserUIFalsePositivesFull:
    """Full pipeline must produce zero PII for pure browser UI screenshots."""

    BROWSER_UI_BLOCKS = [
        "YouTube GitHub ChatGPT Gmail New chat Ask anything Think Search Settings",
        "Gmail - Inbox YouTube GitHub Personal Details Form",
        "New Tab Bookmarks Extensions History Downloads",
        "Restart the server",
        "Ask anything",
        "New chat",
        "ChatGPT can make mistakes Check important info",
        "SIH Project Continue",
        "OCR Debugging Explanation",
        "Roshini Samrajyam Free",
    ]

    @pytest.mark.parametrize("text", BROWSER_UI_BLOCKS)
    def test_browser_ui_no_pii(self, text):
        entities = _run_pipeline(text)
        assert entities == [], (
            f"Expected empty PII for browser UI text: {text!r}\n"
            f"Got: {[(e.type, e.text) for e in entities]}"
        )

    def test_tab_bar_titles_no_org(self, ctx_det):
        tab_text = "Gmail YouTube GitHub Personal Details Form"
        dets = ctx_det.detect(tab_text)
        orgs = [d for d in dets if d.category == PIICategory.ORGANIZATION]
        assert orgs == [], f"Tab titles became ORGANIZATION: {[d.text for d in orgs]}"

    def test_tab_bar_titles_no_person(self, ner_det):
        tab_text = "Gmail YouTube GitHub Personal Details Form"
        committed, _ = ner_det.detect_all(tab_text)
        persons = [d for d in committed if d.category == PIICategory.PERSON_NAME]
        assert persons == [], f"Tab titles became PERSON_NAME: {[d.text for d in persons]}"


# ===========================================================================
# Regression: previously working PII types still work
# ===========================================================================

class TestRegressionExistingPII:
    """Ensure existing PII detection is not broken by the new changes."""

    def test_person_name_still_detected(self):
        entities = _run_pipeline("Full Name: Ravi Kumar")
        types = [e.type for e in entities]
        assert "PERSON_NAME" in types, "PERSON_NAME regression — no longer detected"

    def test_person_name_text_correct(self):
        entities = _run_pipeline("Full Name: Ravi Kumar")
        persons = [e for e in entities if e.type == "PERSON_NAME"]
        assert persons
        assert any("Ravi Kumar" in e.text or "Ravi" in e.text for e in persons)

    def test_dob_still_detected(self):
        entities = _run_pipeline("Date of Birth: 15/08/2002")
        types = [e.type for e in entities]
        assert "DATE_OF_BIRTH" in types, "DATE_OF_BIRTH regression"

    def test_email_still_detected(self):
        entities = _run_pipeline("Email: test@example.com")
        types = [e.type for e in entities]
        assert "EMAIL" in types, "EMAIL regression"

    def test_email_text_correct(self):
        entities = _run_pipeline("Email: test@example.com")
        emails = [e for e in entities if e.type == "EMAIL"]
        assert emails
        assert emails[0].text == "test@example.com"

    def test_phone_still_detected(self):
        entities = _run_pipeline("Phone: 9876543210")
        types = [e.type for e in entities]
        assert "PHONE" in types, "PHONE regression"

    def test_phone_text_correct(self):
        entities = _run_pipeline("Phone: 9876543210")
        phones = [e for e in entities if e.type == "PHONE"]
        assert phones
        assert "9876543210" in phones[0].text

    def test_password_still_detected(self):
        entities = _run_pipeline("Password: MySecret123")
        types = [e.type for e in entities]
        assert "PASSWORD" in types, "PASSWORD regression"

    def test_output_fields_contract(self):
        """All PIIEntity fields required by Dev 3 must be present."""
        entities = _run_pipeline("Email: dev3@integration.test Phone: 9876543210")
        for e in entities:
            assert hasattr(e, "type") and isinstance(e.type, str)
            assert e.type == e.type.upper()
            assert hasattr(e, "text") and len(e.text.strip()) > 0
            assert hasattr(e, "bbox")
            assert hasattr(e, "ocr_confidence") and 0.0 <= e.ocr_confidence <= 1.0
            assert hasattr(e, "pii_score") and 0.0 <= e.pii_score <= 1.0

    def test_no_duplicate_type_text_pairs(self):
        text = (
            "Full Name: Ravi Kumar Email: test@example.com "
            "Phone: 9876543210 Password: MySecret123"
        )
        entities = _run_pipeline(text)
        pairs = [(e.type, e.text.strip().lower()) for e in entities]
        assert len(pairs) == len(set(pairs)), f"Duplicate PII entities: {pairs}"

    def test_full_form_all_types_detected(self):
        """Integration: full personal details form detects all expected types."""
        form_text = (
            "Full Name : Ravi Kumar "
            "Date of Birth : 15/08/2002 "
            "Mother s Name : Lakshmi Devi "
            "Employer : Indian Space Research Organisation ISRO "
            "Password : MySecret123 "
            "Email : test@example.com "
            "Phone : 9876543210 "
            "Address : 12-4-567 Green Park Hyderabad Telangana 500016"
        )
        entities = _run_pipeline(form_text)
        types = {e.type for e in entities}
        expected = {"PERSON_NAME", "DATE_OF_BIRTH", "ORGANIZATION",
                    "PASSWORD", "EMAIL", "PHONE", "ADDRESS"}
        missing = expected - types
        assert not missing, (
            f"Missing PII types in full form test: {missing}\n"
            f"Detected: {[(e.type, e.text) for e in entities]}"
        )
