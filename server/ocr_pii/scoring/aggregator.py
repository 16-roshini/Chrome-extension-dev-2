"""
Aggregator — merges detections from all three detectors into a
clean, deduplicated, confidence-scored final list.

Fix applied:
  - Sweep-line now only merges spans that overlap AND share the same category.
    Previously, any overlapping spans were merged regardless of category,
    causing an email at char 175 to be swallowed by a DOB span at 80-200.
  - Within a same-category group, the highest-priority source wins.
  - Confidence is combined via noisy-OR.
  - Source is set to AGGREGATED when multiple sources contributed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

from server.ocr_pii.schemas import DetectionSource, PIICategory, PIIDetection
from .scorer import combine_scores

# Source priority for resolving conflicts within a group
_SOURCE_PRIORITY: dict[DetectionSource, int] = {
    DetectionSource.REGEX:      3,
    DetectionSource.SPACY_NER:  2,
    DetectionSource.CONTEXT:    1,
    DetectionSource.AGGREGATED: 4,
}


class Aggregator:
    """
    Merges, deduplicates, and re-scores PII detections from all sources.
    """

    def aggregate(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """
        Aggregate a flat list of raw detections.

        Steps:
          1. Group by PIICategory.
          2. Within each category, sweep-line merge overlapping spans.
          3. Merge each overlapping group into one detection.
          4. Flatten all categories and sort by char_start.
        """
        if not detections:
            return []

        # Group by category first — never merge different categories together
        by_category: dict[PIICategory, List[PIIDetection]] = defaultdict(list)
        for det in detections:
            by_category[det.category].append(det)

        merged: List[PIIDetection] = []
        for category_dets in by_category.values():
            # Sort by char_start within category
            sorted_dets = sorted(category_dets, key=lambda d: (d.char_start, d.char_end))
            groups = self._group_overlapping(sorted_dets)
            for group in groups:
                merged.append(self._merge_group(group))

        return sorted(merged, key=lambda d: d.char_start)

    def remove_duplicates(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """
        Remove exact duplicate detections (same text + category + span).
        """
        seen: set[tuple[str, str, int, int]] = set()
        unique: List[PIIDetection] = []
        for d in detections:
            key = (d.text.strip().lower(), d.category.value, d.char_start, d.char_end)
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _group_overlapping(
        self, detections: List[PIIDetection]
    ) -> List[List[PIIDetection]]:
        """
        Group detections whose character spans overlap.
        Only called within a single category, so cross-category merging
        is impossible by design.
        """
        if not detections:
            return []

        groups: List[List[PIIDetection]] = []
        current_group: List[PIIDetection] = [detections[0]]
        current_end = detections[0].char_end

        for det in detections[1:]:
            if det.char_start < current_end:
                current_group.append(det)
                current_end = max(current_end, det.char_end)
            else:
                groups.append(current_group)
                current_group = [det]
                current_end = det.char_end

        groups.append(current_group)
        return groups

    def _merge_group(self, group: List[PIIDetection]) -> PIIDetection:
        """
        Merge a group of same-category overlapping detections into one.

        - Text + category: from highest-priority source.
        - Confidence: noisy-OR combination.
        - Source: AGGREGATED if >1 distinct sources, else original.
        - BBox: first non-None bbox in the group.
        - Char span: narrowest span (use best detection's span, not widest).
          Using the widest span was causing unrelated text to be included.
        """
        if len(group) == 1:
            return group[0]

        # Best = highest priority source
        best = max(group, key=lambda d: _SOURCE_PRIORITY.get(d.source, 0))

        combined_confidence = combine_scores([d.confidence for d in group])

        distinct_sources = {d.source for d in group}
        final_source = (
            DetectionSource.AGGREGATED
            if len(distinct_sources) > 1
            else best.source
        )

        bbox = next(
            (d.bounding_box for d in group if d.bounding_box is not None), None
        )

        # Use the best detection's span (not the widest) to avoid
        # span inflation that previously swallowed nearby detections
        return PIIDetection(
            text=best.text,
            category=best.category,
            confidence=combined_confidence,
            source=final_source,
            bounding_box=bbox,
            char_start=best.char_start,
            char_end=best.char_end,
        )
