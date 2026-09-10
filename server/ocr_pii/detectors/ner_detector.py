"""
spaCy NER-based PII detector.
Uses en_core_web_sm to detect PERSON, ORG, GPE, DATE entities.

Architecture (evidence model):
  - PERSON, ORG  → committed PIIDetection directly (category unambiguous)
  - DATE, GPE    → DetectorEvidence only (ambiguous; FusionLayer decides)

The NERDetector exposes two methods:
  detect(text)          → List[PIIDetection]   (PERSON + ORG only)
  detect_evidence(text) → List[DetectorEvidence] (DATE + GPE)

The pipeline calls both and passes evidence to FusionLayer.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

import spacy
from spacy.language import Language

from server.ocr_pii.schemas import (
    DetectionSource,
    DetectorEvidence,
    EvidenceLabel,
    PIICategory,
    PIIDetection,
)
from .base_detector import BaseDetector

# ---------------------------------------------------------------------------
# spaCy label routing
# COMMITTED: label maps directly to a PIICategory with no ambiguity
# EVIDENCE:  label requires context validation before classification
# ---------------------------------------------------------------------------
_COMMITTED_LABEL_MAP: dict[str, PIICategory] = {
    "PERSON": PIICategory.PERSON_NAME,
    "ORG":    PIICategory.ORGANIZATION,
}

_EVIDENCE_LABEL_MAP: dict[str, EvidenceLabel] = {
    "DATE": EvidenceLabel.SPACY_DATE,
    "GPE":  EvidenceLabel.SPACY_GPE,
}

# Base confidence for evidence labels (before fusion boosting)
_EVIDENCE_CONFIDENCE: dict[str, float] = {
    "DATE": 0.55,   # low — many dates are not birth dates
    "GPE":  0.50,   # low — most GPE mentions are not postal addresses
}

# Base confidence for committed detections
_COMMITTED_CONFIDENCE: dict[str, float] = {
    "PERSON": 0.80,
    "ORG":    0.75,
}

_MODEL_NAME = "en_core_web_sm"
_nlp: Language | None = None

# ---------------------------------------------------------------------------
# ORG stopwords (unicode-normalised lowercase)
# ---------------------------------------------------------------------------
_ORG_STOPWORDS: frozenset[str] = frozenset({
    "pan", "aadhaar", "aadhar", "uid", "dob", "dob:", "pan:",
    "email", "phone", "mobile", "address", "name", "gender",
    "age", "nationality", "religion", "caste", "occupation",
    "ifsc", "cvv", "otp",
    "personal information", "personal details", "contact details",
    "personal information & name", "personal info",
    "applicant details", "applicant information",
    "bank details", "bank information",
    "declaration", "signature", "date",
    "form", "application", "certificate",
    "government of india", "govt of india",
    "aadhaar", "aadhar", "uidai", "adhikar",
    "aadhaar-aam admi", "aadhar-aam admi",
    "aam admi ka adhikar", "aam admi",
    "income tax department",
    "permanent account number",
    "permanent account number card",
})

# ---------------------------------------------------------------------------
# PERSON entity stopwords — whole-entity rejection
# ---------------------------------------------------------------------------
_PERSON_ENTITY_STOPWORDS: frozenset[str] = frozenset({
    "adhikar", "aadhaar", "aadhar", "india", "bharat",
    "government", "authority", "department", "ministry",
    "identification", "unique", "enrolment", "enrollment",
    "number", "card", "citizen",
})

# ---------------------------------------------------------------------------
# PERSON trailing stopwords — stripped from the END of a name span
# ---------------------------------------------------------------------------
_PERSON_TRAILING_STOPWORDS: frozenset[str] = frozenset({
    "email", "phone", "mobile", "address", "dob", "pan", "id",
    "no", "number", "rm", "ref", "sr", "jr", "esq",
    "mr", "mrs", "ms", "dr", "prof",
    "adhikar", "aadhaar", "aadhar",
    # OCR-appended field labels and section headings
    "employer", "employee", "name", "designation", "occupation",
    "password", "username", "contact", "details", "information",
    "father", "mother", "guardian", "spouse",
    "male", "female", "gender", "dob:", "age",
})

_MAX_PERSON_TOKENS = 4
_MIN_ENTITY_LENGTH = 4

# ---------------------------------------------------------------------------
# ORG keywords — required in multi-word all-caps entities to confirm they
# are a real organisation, not a person name in caps
# ---------------------------------------------------------------------------
_ORG_KEYWORDS: frozenset[str] = frozenset({
    "bank", "ministry", "department", "authority",
    "commission", "council", "corporation", "limited",
    "ltd", "university", "institute", "association",
    "organisation", "organization", "board", "bureau",
    "centre", "center", "society", "trust", "fund",
})

# ---------------------------------------------------------------------------
# OCR noise patterns
# ---------------------------------------------------------------------------
_OCR_NOISE_PATTERNS = [
    re.compile(r'[/\\|]'),
    re.compile(r'[!?@#$%^&*]$'),
    re.compile(r"^['\"\u2018\u2019]"),
    re.compile(r'[a-z][A-Z]'),
]


def _normalise(text: str) -> str:
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    result = text
    for src, dst in replacements.items():
        result = result.replace(src, dst)
    return unicodedata.normalize("NFKC", result).lower().strip()


def _is_ocr_noise(text: str) -> bool:
    stripped = text.strip()
    for pattern in _OCR_NOISE_PATTERNS:
        if pattern.search(stripped):
            return True
    for token in stripped.split():
        if re.match(r'^[A-Za-z0-9]+$', token):
            if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
                return True
        if all(ord(c) > 127 for c in token if not c.isspace()):
            return True
    return False


def _has_non_ascii_tokens(text: str) -> bool:
    for token in text.split():
        non_ascii = sum(1 for c in token if ord(c) > 127)
        if non_ascii > len(token) * 0.5:
            return True
    return False


def _clean_person_span(text: str) -> str:
    # Stop at UI boundary characters before trimming tokens
    text = re.split(r'[+@<>|•·©®]', text)[0].strip()
    tokens = text.split()[:_MAX_PERSON_TOKENS]
    while tokens and tokens[-1].lower().rstrip(".:,") in _PERSON_TRAILING_STOPWORDS:
        tokens.pop()
    return " ".join(tokens)


def _is_valid_org(text: str) -> bool:
    norm = _normalise(text)
    if norm in _ORG_STOPWORDS:
        return False
    stripped = text.strip()
    if " " not in stripped and _normalise(stripped) in _ORG_STOPWORDS:
        return False
    if _is_ocr_noise(text):
        return False
    if len(stripped) < _MIN_ENTITY_LENGTH:
        return False
    if " " not in stripped and "." in stripped:
        return False
    if " " not in stripped:
        if not re.match(r'^[A-Z]{4,8}$', stripped):
            return False
    if "@" in text:
        return False
    tokens = text.split()
    if len(tokens) >= 2:
        if _normalise(tokens[-1]) in _ORG_STOPWORDS:
            return False
    if len(tokens) >= 2:
        all_caps_alpha = all(re.match(r'^[A-Z]+$', t) for t in tokens)
        if all_caps_alpha:
            if not ({t.lower() for t in tokens} & _ORG_KEYWORDS):
                return False
    return True


def _get_nlp() -> Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(_MODEL_NAME)
    return _nlp


class NERDetector(BaseDetector):
    """
    NER-based PII detector using spaCy en_core_web_sm.

    Committed detections (PERSON, ORG):
        detect(text) → List[PIIDetection]

    Evidence candidates (DATE, GPE) — require FusionLayer validation:
        detect_evidence(text) → List[DetectorEvidence]

    The pipeline calls both methods and passes results to FusionLayer.
    """

    @property
    def detector_name(self) -> str:
        return "spacy_ner"

    def detect(self, text: str) -> List[PIIDetection]:
        """
        Return committed PII detections for PERSON and ORG entities.
        Does NOT return DATE or GPE — use detect_evidence() for those.
        """
        committed, _ = self._run_spacy(text)
        return committed

    def detect_evidence(self, text: str) -> List[DetectorEvidence]:
        """
        Return evidence candidates for DATE and GPE entities.
        These require FusionLayer validation before becoming PIIDetection.
        """
        _, evidence = self._run_spacy(text)
        return evidence

    def detect_all(
        self, text: str
    ) -> Tuple[List[PIIDetection], List[DetectorEvidence]]:
        """
        Run spaCy once, return both committed detections and evidence.
        Used by the pipeline to avoid running spaCy twice.
        """
        return self._run_spacy(text)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _run_spacy(
        self, text: str
    ) -> Tuple[List[PIIDetection], List[DetectorEvidence]]:
        if not text.strip():
            return [], []

        nlp = _get_nlp()
        doc = nlp(text)

        committed: List[PIIDetection] = []
        evidence: List[DetectorEvidence] = []

        for ent in doc.ents:
            entity_text = ent.text
            char_start = ent.start_char
            char_end = ent.end_char

            # Global noise rejection — applies to all labels
            if _is_ocr_noise(entity_text):
                continue
            if _has_non_ascii_tokens(entity_text):
                continue

            # --- Evidence labels (DATE, GPE) ---
            ev_label = _EVIDENCE_LABEL_MAP.get(ent.label_)
            if ev_label is not None:
                if len(entity_text.strip()) >= _MIN_ENTITY_LENGTH:
                    evidence.append(DetectorEvidence(
                        text=entity_text,
                        label=ev_label,
                        raw_confidence=_EVIDENCE_CONFIDENCE.get(ent.label_, 0.50),
                        char_start=char_start,
                        char_end=char_end,
                    ))
                continue

            # --- Committed labels (PERSON, ORG) ---
            category = _COMMITTED_LABEL_MAP.get(ent.label_)
            if category is None:
                continue

            # ORG validation
            if ent.label_ == "ORG":
                if not _is_valid_org(entity_text):
                    # Recover all-caps pure-alpha multi-word entities as
                    # PERSON_NAME — on Indian ID cards names are printed in
                    # ALL CAPS and spaCy misclassifies them as ORG.
                    # e.g. "CHAMPESWAR PATRA", "ABHIRAM PATRA"
                    tokens = entity_text.strip().split()
                    if (
                        2 <= len(tokens) <= 4
                        and all(re.match(r'^[A-Z]{2,}$', t) for t in tokens)
                        and not ({t.lower() for t in tokens} & _ORG_KEYWORDS)
                    ):
                        cleaned = _clean_person_span(entity_text)
                        if (
                            cleaned
                            and len(cleaned) >= _MIN_ENTITY_LENGTH
                            and not _is_ocr_noise(cleaned)
                            and not _has_non_ascii_tokens(cleaned)
                            and _normalise(cleaned) not in _PERSON_ENTITY_STOPWORDS
                        ):
                            committed.append(PIIDetection(
                                text=cleaned,
                                category=PIICategory.PERSON_NAME,
                                confidence=_COMMITTED_CONFIDENCE.get("PERSON", 0.75),
                                source=DetectionSource.SPACY_NER,
                                bounding_box=None,
                                char_start=char_start,
                                char_end=char_start + len(cleaned),
                            ))
                            
                    continue
                char_end = char_start + len(entity_text)

            # PERSON validation
            if ent.label_ == "PERSON":
                cleaned = _clean_person_span(entity_text)
                if not cleaned or len(cleaned) < _MIN_ENTITY_LENGTH:
                    continue
                if _is_ocr_noise(cleaned) or _has_non_ascii_tokens(cleaned):
                    continue
                if _normalise(cleaned) in _PERSON_ENTITY_STOPWORDS:
                    continue
                char_end = char_start + len(cleaned)
                entity_text = cleaned

            if len(entity_text.strip()) < _MIN_ENTITY_LENGTH:
                continue

            committed.append(PIIDetection(
                text=entity_text,
                category=category,
                confidence=_COMMITTED_CONFIDENCE.get(ent.label_, 0.60),
                source=DetectionSource.SPACY_NER,
                bounding_box=None,
                char_start=char_start,
                char_end=char_end,
            ))

        return committed, evidence
