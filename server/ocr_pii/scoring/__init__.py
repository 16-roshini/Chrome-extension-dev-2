"""
Scoring subpackage.
  - scorer.py    — per-detection confidence calculation
  - aggregator.py — merge overlapping detections, combine evidence,
                    deduplicate, produce final PIIDetection list
  - fusion.py    — promotes DetectorEvidence to PIIDetection
"""

from .scorer import compute_confidence
from .aggregator import Aggregator
from .fusion import FusionLayer

__all__ = ["compute_confidence", "Aggregator", "FusionLayer"]
