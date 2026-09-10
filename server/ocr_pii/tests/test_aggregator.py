"""
Tests for Aggregator and scorer utilities.
All PII values are SYNTHETIC.
"""

from __future__ import annotations

import pytest
from server.ocr_pii.scoring.aggregator import Aggregator
from server.ocr_pii.scoring.scorer import combine_scores, compute_confidence
from server.ocr_pii.schemas import DetectionSource, PIICategory, PIIDetection


@pytest.fixture
def aggregator() -> Aggregator:
    return Aggregator()


def make_detection(
    text: str,
    category: PIICategory,
    confidence: float,
    source: DetectionSource,
    char_start: int,
    char_end: int,
) -> PIIDetection:
    """Helper to build a synthetic PIIDetection."""
    return PIIDetection(
        text=text,
        category=category,
        confidence=confidence,
        source=source,
        bounding_box=None,
        char_start=char_start,
        char_end=char_end,
    )


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------

class TestCombineScores:

    def test_single_score_unchanged(self):
        result = combine_scores([0.9])
        assert abs(result - 0.9) < 0.01

    def test_two_scores_higher_than_either(self):
        result = combine_scores([0.7, 0.8])
        assert result > 0.8

    def test_perfect_scores_give_one(self):
        result = combine_scores([1.0, 1.0])
        assert result == pytest.approx(1.0)

    def test_empty_gives_zero(self):
        assert combine_scores([]) == 0.0

    def test_result_clamped_to_one(self):
        result = combine_scores([0.99, 0.99, 0.99])
        assert result <= 1.0

    def test_result_never_negative(self):
        assert combine_scores([0.0]) >= 0.0


class TestComputeConfidence:

    def test_high_ocr_confidence_no_penalty(self, detection_email):
        score = compute_confidence(detection_email, ocr_confidence=1.0)
        assert score <= 1.0
        assert score > 0.0

    def test_low_ocr_confidence_reduces_score(self, detection_email):
        score_high = compute_confidence(detection_email, ocr_confidence=1.0)
        score_low = compute_confidence(detection_email, ocr_confidence=0.3)
        assert score_low < score_high

    def test_score_always_in_range(self, detection_email):
        for ocr_conf in [0.0, 0.3, 0.6, 0.9, 1.0]:
            score = compute_confidence(detection_email, ocr_conf)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Aggregator — empty and single inputs
# ---------------------------------------------------------------------------

class TestAggregatorEmpty:

    def test_empty_list_returns_empty(self, aggregator):
        assert aggregator.aggregate([]) == []

    def test_single_detection_returned_as_is(self, aggregator):
        d = make_detection("test@example.com", PIICategory.EMAIL, 0.97,
                           DetectionSource.REGEX, 0, 16)
        result = aggregator.aggregate([d])
        assert len(result) == 1
        assert result[0].text == "test@example.com"


# ---------------------------------------------------------------------------
# Aggregator — overlapping spans
# ---------------------------------------------------------------------------

class TestAggregatorOverlapping:

    def test_overlapping_spans_merged_to_one(self, aggregator):
        # Same span, two sources — should merge into one
        d1 = make_detection("9876543210", PIICategory.PHONE, 0.90,
                            DetectionSource.REGEX, 10, 20)
        d2 = make_detection("9876543210", PIICategory.PHONE, 0.75,
                            DetectionSource.CONTEXT, 10, 20)
        result = aggregator.aggregate([d1, d2])
        assert len(result) == 1

    def test_merged_confidence_higher_than_max_input(self, aggregator):
        d1 = make_detection("ABCDE1234F", PIICategory.PAN, 0.95,
                            DetectionSource.REGEX, 5, 15)
        d2 = make_detection("ABCDE1234F", PIICategory.PAN, 0.75,
                            DetectionSource.CONTEXT, 5, 15)
        result = aggregator.aggregate([d1, d2])
        assert result[0].confidence >= 0.95

    def test_multi_source_sets_aggregated(self, aggregator):
        d1 = make_detection("Jane Smith", PIICategory.PERSON_NAME, 0.80,
                            DetectionSource.SPACY_NER, 6, 16)
        d2 = make_detection("Jane Smith", PIICategory.PERSON_NAME, 0.75,
                            DetectionSource.CONTEXT, 6, 16)
        result = aggregator.aggregate([d1, d2])
        assert result[0].source == DetectionSource.AGGREGATED

    def test_single_source_preserves_source(self, aggregator):
        d = make_detection("test@example.com", PIICategory.EMAIL, 0.97,
                           DetectionSource.REGEX, 0, 16)
        result = aggregator.aggregate([d])
        assert result[0].source == DetectionSource.REGEX

    def test_regex_wins_category_over_context(self, aggregator):
        # Same span — regex should win for category
        d_regex = make_detection("123456789012", PIICategory.BANK_ACCOUNT, 0.70,
                                 DetectionSource.REGEX, 12, 24)
        d_ctx = make_detection("123456789012", PIICategory.BANK_ACCOUNT, 0.75,
                                DetectionSource.CONTEXT, 12, 24)
        result = aggregator.aggregate([d_regex, d_ctx])
        assert result[0].category == PIICategory.BANK_ACCOUNT


# ---------------------------------------------------------------------------
# Aggregator — non-overlapping spans
# ---------------------------------------------------------------------------

class TestAggregatorNonOverlapping:

    def test_non_overlapping_kept_separate(self, aggregator):
        d1 = make_detection("test@example.com", PIICategory.EMAIL, 0.97,
                            DetectionSource.REGEX, 0, 16)
        d2 = make_detection("9876543210", PIICategory.PHONE, 0.90,
                            DetectionSource.REGEX, 50, 60)
        result = aggregator.aggregate([d1, d2])
        assert len(result) == 2

    def test_output_sorted_by_char_start(self, aggregator):
        d1 = make_detection("9876543210", PIICategory.PHONE, 0.90,
                            DetectionSource.REGEX, 50, 60)
        d2 = make_detection("test@example.com", PIICategory.EMAIL, 0.97,
                            DetectionSource.REGEX, 0, 16)
        result = aggregator.aggregate([d1, d2])
        assert result[0].char_start < result[1].char_start


# ---------------------------------------------------------------------------
# Aggregator — remove_duplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:

    def test_removes_exact_duplicate(self, aggregator):
        d = make_detection("test@example.com", PIICategory.EMAIL, 0.97,
                           DetectionSource.REGEX, 0, 16)
        result = aggregator.remove_duplicates([d, d])
        assert len(result) == 1

    def test_keeps_different_spans(self, aggregator):
        d1 = make_detection("test@example.com", PIICategory.EMAIL, 0.97,
                            DetectionSource.REGEX, 0, 16)
        d2 = make_detection("test@example.com", PIICategory.EMAIL, 0.97,
                            DetectionSource.REGEX, 20, 36)
        result = aggregator.remove_duplicates([d1, d2])
        assert len(result) == 2
