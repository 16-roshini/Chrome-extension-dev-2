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
    # --- Browser / web-app product names ---
    # These are NOT organisations in the PII sense; they are product/service
    # names that commonly appear in browser tab bars and UI chrome.
    "chatgpt", "youtube", "github", "gmail", "google", "twitter",
    "facebook", "instagram", "linkedin", "reddit", "whatsapp",
    "netflix", "amazon", "flipkart", "snapchat", "telegram",
    "discord", "slack", "zoom", "teams", "outlook", "chrome",
    "firefox", "safari", "edge", "opera", "brave",
    "wikipedia", "stackoverflow", "notion", "figma", "canva",
    "dropbox", "drive", "docs", "sheets", "slides",
    # --- Generic browser UI labels ---
    "new tab", "settings", "history", "bookmarks", "extensions",
    "downloads", "search", "inbox", "compose",
    # --- Common form/document section headers ---
    # These appear as all-caps headings in forms and are often mis-tagged as ORG
    "details", "information", "notes", "summary", "overview",
    "section", "title", "description", "remarks", "declaration",
    "personal details", "contact details", "login information",
    "additional notes", "employment details", "bank details",
    "educational details", "family details", "other details",
})

# ---------------------------------------------------------------------------
# PERSON entity stopwords — whole-entity rejection
# ---------------------------------------------------------------------------
_PERSON_ENTITY_STOPWORDS: frozenset[str] = frozenset({
    "adhikar", "aadhaar", "aadhar", "india", "bharat",
    "government", "authority", "department", "ministry",
    "identification", "unique", "enrolment", "enrollment",
    "number", "card", "citizen",
    # --- Common single-word browser UI tokens misclassified as PERSON ---
    # Action words / navigation labels
    "restart", "search", "settings", "compose", "signin", "signout",
    "login", "logout", "register", "subscribe", "unsubscribe",
    "continue", "submit", "confirm", "cancel", "back", "next",
    "refresh", "reload", "close", "open", "start", "stop",
    "pause", "resume", "download", "upload", "share", "delete",
    "edit", "save", "send", "reply", "forward", "archive",
    "spam", "report", "block", "mute", "follow", "unfollow",
    # Browser tab / UI noun tokens
    "inbox", "outbox", "drafts", "trash", "starred", "important",
    "notifications", "messages", "chats", "groups", "channels",
    "trending", "explore", "discover", "library", "history",
    "bookmarks", "downloads", "extensions", "preferences",
    "dashboard", "profile", "account", "security", "privacy",
    "help", "feedback", "support", "about", "contact",
    # Product/service names that look like names but are not PII
    "chatgpt", "youtube", "github", "gmail", "google", "twitter",
    "facebook", "instagram", "linkedin", "reddit", "whatsapp",
    "netflix", "amazon", "flipkart", "snapchat", "telegram",
    "discord", "slack", "zoom", "teams", "outlook", "chrome",
    "firefox", "safari", "edge", "opera", "brave",
    "wikipedia", "stackoverflow", "notion", "figma", "canva",
    "copilot", "gemini", "claude", "bard", "perplexity",
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
# Browser UI single-word tokens — block as PERSON when they appear alone.
# A legitimate person name is almost never a single action/navigation word.
# Normalised to lowercase for comparison.
# ---------------------------------------------------------------------------
_BROWSER_UI_SINGLE_TOKENS: frozenset[str] = frozenset({
    # Navigation / actions
    "restart", "search", "settings", "compose", "signin", "signout",
    "login", "logout", "register", "subscribe", "unsubscribe",
    "continue", "submit", "confirm", "cancel", "back", "next",
    "refresh", "reload", "close", "open", "start", "stop",
    "pause", "resume", "download", "upload", "share", "delete",
    "edit", "save", "send", "reply", "forward", "archive",
    "spam", "report", "block", "mute", "follow", "unfollow",
    "like", "dislike", "comment", "post", "create", "read",
    "write", "view", "watch", "play", "explore", "discover",
    # Browser / app UI labels
    "inbox", "outbox", "drafts", "trash", "starred", "important",
    "notifications", "messages", "chats", "groups", "channels",
    "trending", "library", "history", "bookmarks", "downloads",
    "extensions", "preferences", "dashboard", "profile", "account",
    "security", "privacy", "help", "feedback", "support",
    "about", "contact", "home", "menu", "sidebar", "toolbar",
    "tab", "window", "panel", "widget", "icon",
    # Common English words spaCy may tag as PERSON/ORG in short UI snippets
    "free", "pro", "plus", "premium", "basic", "standard",
    "new", "old", "recent", "latest", "popular", "top",
    "all", "more", "less", "other", "general",
    "ask", "anything", "think", "type", "enter",
    # Product names that must not become PERSON_NAME
    "chatgpt", "youtube", "github", "gmail", "google", "twitter",
    "facebook", "instagram", "linkedin", "reddit", "whatsapp",
    "netflix", "amazon", "flipkart", "snapchat", "telegram",
    "discord", "slack", "zoom", "teams", "outlook", "chrome",
    "firefox", "safari", "edge", "opera", "brave",
    "wikipedia", "stackoverflow", "notion", "figma", "canva",
    "copilot", "gemini", "claude", "bard", "perplexity",
})

# ---------------------------------------------------------------------------
# Common English dictionary words that are not surnames/given-names.
# Used to reject PERSON spans where every token is a plain English word.
# Deliberately small — only words that clearly cannot be names.
# ---------------------------------------------------------------------------
_COMMON_ENGLISH_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "had", "her", "was", "one", "our", "out", "day",
    "get", "has", "him", "his", "how", "its", "may", "new",
    "now", "old", "see", "two", "way", "who", "boy", "did",
    "big", "end", "far", "few", "got", "let", "man", "put",
    "say", "she", "too", "use", "via", "yet", "ask", "act",
    "add", "age", "ago", "aid", "aim", "air", "any", "app",
    "apt", "art", "bar", "bay", "bed", "bit", "box", "bug",
    "bus", "buy", "bye", "car", "cut", "doc", "dog", "due",
    "ear", "eat", "egg", "end", "era", "eye", "fee", "fit",
    "fix", "fly", "fun", "gap", "gas", "hit", "hot", "hub",
    "ice", "job", "key", "kid", "lab", "law", "lay", "led",
    "leg", "log", "map", "mix", "mob", "mod", "net", "now",
    "oil", "opt", "own", "pay", "pin", "pop", "pot", "raw",
    "red", "ref", "run", "set", "sit", "six", "sky", "sub",
    "sum", "tag", "tap", "tax", "tip", "top", "try", "tab",
    "try", "url", "win", "yes",
    # longer common words
    "account", "action", "active", "after", "again", "alert",
    "allow", "also", "apply", "audio", "basic", "batch",
    "below", "board", "boost", "build", "cache", "check",
    "clean", "clear", "click", "cloud", "color", "count",
    "cover", "daily", "dark", "data", "debug", "deep",
    "demo", "deny", "detail", "done", "each", "empty", "enable",
    "enter", "event", "every", "extra", "field", "file",
    "filter", "find", "flag", "flash", "flow", "focus",
    "force", "fresh", "front", "full", "global", "grant",
    "group", "guest", "guide", "hover", "image", "index",
    "info", "input", "item", "join", "jump", "label", "large",
    "later", "layer", "level", "light", "limit", "line",
    "link", "list", "live", "load", "local", "lock", "loop",
    "main", "make", "mark", "match", "media", "model", "mode",
    "more", "most", "move", "name", "node", "none", "note",
    "null", "only", "open", "order", "other", "output", "page",
    "part", "path", "plan", "play", "plug", "poll", "pool",
    "port", "post", "press", "print", "push", "queue", "quick",
    "range", "rate", "real", "reply", "reset", "retry", "role",
    "root", "rule", "safe", "same", "scan", "send", "size",
    "skip", "slow", "sort", "span", "spec", "stat", "step",
    "stop", "store", "sync", "text", "then", "time", "title",
    "toggle", "token", "tool", "type", "undo", "unit", "user",
    "valid", "value", "video", "view", "void", "warn", "when",
    "with", "word", "work", "wrap", "zone",
})

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
    # Stop at UI boundary characters — includes hyphen/dash so that
    # "Gmail - Inbox" → "Gmail" (and "Gmail" is then caught by stopwords)
    text = re.split(r'[+@<>|•·©®\-–—]', text)[0].strip()
    tokens = text.split()[:_MAX_PERSON_TOKENS]
    # Trim from the end for trailing stopwords
    while tokens and tokens[-1].lower().rstrip(".:,") in _PERSON_TRAILING_STOPWORDS:
        tokens.pop()
    # Also stop at the FIRST token that is a field label anywhere in the span
    # e.g. "Devi Employer Indian Space" → stop at "Employer" → "Devi"
    cut_at = len(tokens)
    for i, tok in enumerate(tokens):
        if tok.lower().rstrip(".:,") in _PERSON_TRAILING_STOPWORDS:
            cut_at = i
            break
    tokens = tokens[:cut_at]
    return " ".join(tokens)


def _clean_org_span(text: str) -> str:
    """
    Strip trailing label/noise tokens from an ORG entity span.
    spaCy sometimes absorbs adjacent form-field labels into the ORG span,
    e.g. "Indian Space Research Organisation ISRO Job Title :"
    → "Indian Space Research Organisation"
    """
    _ORG_TRAILING_TOKENS = frozenset({
        ":", "–", "-", "|", "•",
        "job", "title", "position", "designation",
        "dept", "department", "division", "section",
        "employer", "employee", "name", "contact",
    })
    # Known abbreviations that appear after the org name
    _ORG_ABBREV = re.compile(r'^\([A-Z]{2,8}\)$')
    _ALLCAPS_ABBREV = re.compile(r'^[A-Z]{2,8}$')

    tokens = text.strip().split()

    # Find the first trailing noise token and truncate there
    cut_at = len(tokens)
    for i, tok in enumerate(tokens):
        tok_clean = tok.lower().rstrip(".:,()")
        if tok_clean in _ORG_TRAILING_TOKENS:
            cut_at = i
            break
        # Stop at a bare all-caps abbreviation that follows the org name
        # e.g. "Organisation ISRO" → stop at "ISRO" only if preceded by
        # a known org-structure word
        if i > 0 and _ALLCAPS_ABBREV.match(tok):
            prev = tokens[i-1].lower().rstrip(".:,")
            if prev in {"organisation", "organization", "institute",
                        "university", "corporation", "limited", "ltd",
                        "research", "centre", "center", "council"}:
                cut_at = i
                break

    tokens = tokens[:cut_at]

    # Strip bare punctuation at the end
    while tokens and re.match(r'^[^A-Za-z0-9]+$', tokens[-1]):
        tokens.pop()
    # Strip parenthetical abbreviations: (ISRO), (NASA)
    if tokens and _ORG_ABBREV.match(tokens[-1]):
        tokens.pop()

    return " ".join(tokens).strip()


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
    # Single-word entities must pass the all-caps 4–8 char gate
    if " " not in stripped:
        if not re.match(r'^[A-Z]{4,8}$', stripped):
            return False
        # Additionally reject if the lowercased word is a browser UI token
        if stripped.lower() in _BROWSER_UI_SINGLE_TOKENS:
            return False
    if "@" in text:
        return False
    tokens = text.split()
    # Hard cap — real org names rarely exceed 8 tokens; longer spans are
    # almost certainly spaCy over-capturing into adjacent label text
    if len(tokens) > 8:
        return False
    # Strip trailing label/punctuation tokens before further checks
    _ORG_TRAILING = frozenset({
        ":", "–", "-", "|", "•", "job", "title", "position",
        "designation", "dept", "department", "division",
        "isro", "nasa", "ltd", "pvt", "inc", "corp",
    })
    while tokens and tokens[-1].lower().rstrip(".:,()") in _ORG_TRAILING:
        tokens.pop()
    # Also strip parenthetical abbreviations at the end: "(ISRO)", "(NASA)"
    if tokens and re.match(r'^\([A-Z]{2,8}\)$', tokens[-1]):
        tokens.pop()
    if not tokens:
        return False
    text = " ".join(tokens)
    if len(tokens) >= 2:
        if _normalise(tokens[-1]) in _ORG_STOPWORDS:
            return False
    if len(tokens) >= 2:
        all_caps_alpha = all(re.match(r'^[A-Z]+$', t) for t in tokens)
        if all_caps_alpha:
            if not ({t.lower() for t in tokens} & _ORG_KEYWORDS):
                return False
    # Reject multi-word mixed-case spans where more than half the tokens
    # are common English words — these are OCR noise / UI sentences, not
    # real organisation names.
    if len(tokens) >= 3:
        common_count = sum(
            1 for t in tokens
            if t.lower() in _COMMON_ENGLISH_WORDS
            or t.lower() in _BROWSER_UI_SINGLE_TOKENS
        )
        if common_count >= len(tokens) // 2:
            return False
    # Hard cap: any ORG span of 4+ tokens that has NO org-specific keyword
    # and is all Title-Case or mixed-case is almost certainly a sentence
    # fragment, not a real organisation name.
    if len(tokens) >= 4:
        has_org_keyword = bool({t.lower() for t in tokens} & _ORG_KEYWORDS)
        if not has_org_keyword:
            all_alpha = all(re.match(r'^[A-Za-z]+$', t) for t in tokens)
            if all_alpha:
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
                    # Capped at 3 tokens (not 4) to reduce false positives
                    # like "OCR DEBUGGING EXPLANATION" being labelled a name.
                    tokens = entity_text.strip().split()
                    if (
                        2 <= len(tokens) <= 3
                        and all(re.match(r'^[A-Z]{2,}$', t) for t in tokens)
                        and not ({t.lower() for t in tokens} & _ORG_KEYWORDS)
                        # Reject if any token is a common English / UI word
                        and not any(
                            t.lower() in _COMMON_ENGLISH_WORDS
                            or t.lower() in _BROWSER_UI_SINGLE_TOKENS
                            for t in tokens
                        )
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
                # Clean the ORG span — strip trailing label tokens that
                # spaCy may have absorbed (e.g. "ISRO Job Title :")
                entity_text = _clean_org_span(entity_text)
                if not entity_text or len(entity_text) < _MIN_ENTITY_LENGTH:
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

                tokens = cleaned.split()

                # Require at least 2 tokens OR the single token must look
                # like an actual name (not a UI action/navigation word).
                # Single-word spans like "Restart", "Search", "GitHub" are
                # almost never real person names in a PII context.
                if len(tokens) == 1:
                    tok_lower = tokens[0].lower()
                    if tok_lower in _BROWSER_UI_SINGLE_TOKENS:
                        continue
                    if tok_lower in _COMMON_ENGLISH_WORDS:
                        continue
                    # Single-token names must start with a capital letter
                    # (spaCy sometimes returns lowercased noise tokens)
                    if not tokens[0][0].isupper():
                        continue

                # Reject if ANY token is a common English / UI word when
                # it occupies the LAST position — this catches spans like
                # "Samrajyam Free" where the trailing token is a UI word.
                if len(tokens) >= 2:
                    last_lower = tokens[-1].lower()
                    if (last_lower in _BROWSER_UI_SINGLE_TOKENS
                            or last_lower in _COMMON_ENGLISH_WORDS):
                        continue

                # Reject if ALL tokens are common English words
                # (catches "Ask Anything", "New Chat", "Think Deep", etc.)
                if len(tokens) >= 2:
                    if all(t.lower() in _COMMON_ENGLISH_WORDS
                           or t.lower() in _BROWSER_UI_SINGLE_TOKENS
                           for t in tokens):
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
