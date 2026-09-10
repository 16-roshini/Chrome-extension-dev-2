"""
Detectors subpackage.
Three detection strategies:
  1. RegexDetector  — deterministic pattern matching
  2. NERDetector    — spaCy en_core_web_sm named entity recognition
  3. ContextDetector — label/keyword context around sensitive fields
"""

from .regex_detector import RegexDetector
from .ner_detector import NERDetector
from .context_detector import ContextDetector

__all__ = ["RegexDetector", "NERDetector", "ContextDetector"]
