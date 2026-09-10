"""
Regex patterns for deterministic PII detection.

Coverage:
  - Email
  - Indian phone number (mobile + landline)
  - Aadhaar number (12-digit, space/hyphen separated)
  - PAN card
  - Credit card (with Luhn validation helper)
  - IP address (IPv4)
  - URL
  - IFSC code
  - Date of birth (common formats)
  - Indian passport number
  - Bank account number (generic Indian format)
"""

import re
from typing import Pattern

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
EMAIL: Pattern = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# ---------------------------------------------------------------------------
# Indian phone number
# Handles: +91-XXXXXXXXXX, 91-XXXXXXXXXX, 0XXXXXXXXXX, XXXXXXXXXX
# Also handles spaces/hyphens between groups, e.g. +91 98765 43210
# ---------------------------------------------------------------------------
PHONE: Pattern = re.compile(
    r"(?<!\d)"
    r"(?:\+?91[\s\-]?)?"          # optional country code (+91, 91, +91-, +91 )
    r"(?:0)?"                      # optional leading 0
    r"[6-9]\d{4}"                  # first 5 digits starting with 6-9
    r"[\s\-]?"                     # optional separator mid-number (space or hyphen)
    r"\d{5}"                       # last 5 digits
    r"(?!\d)"
)

# ---------------------------------------------------------------------------
# Aadhaar number — 12 digits, optionally in groups of 4 separated by space/hyphen
# ---------------------------------------------------------------------------
AADHAAR: Pattern = re.compile(
    r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b"
)

# ---------------------------------------------------------------------------
# PAN card — format: AAAAA9999A (5 letters, 4 digits, 1 letter)
# ---------------------------------------------------------------------------
PAN: Pattern = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
)

# ---------------------------------------------------------------------------
# Credit card — 13–19 digits, optionally space/hyphen separated in groups of 4
# Luhn validation is done separately in luhn_check()
# ---------------------------------------------------------------------------
CREDIT_CARD: Pattern = re.compile(
    r"\b(?:\d{4}[\s\-]?){3}\d{1,7}\b"
)

# ---------------------------------------------------------------------------
# IPv4 address
# ---------------------------------------------------------------------------
IP_ADDRESS: Pattern = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# ---------------------------------------------------------------------------
# URL — http/https/ftp with optional path
# ---------------------------------------------------------------------------
URL: Pattern = re.compile(
    r"\b(?:https?|ftp)://"
    r"(?:[\w\-]+\.)+[A-Za-z]{2,}"
    r"(?:/[^\s]*)?\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# IFSC code — format: AAAA0XXXXXX (4 letters, 0, 6 alphanumeric)
# ---------------------------------------------------------------------------
IFSC: Pattern = re.compile(
    r"\b[A-Z]{4}0[A-Z0-9]{6}\b"
)

# ---------------------------------------------------------------------------
# Date of birth — common formats:
#   DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
#   YYYY/MM/DD, YYYY-MM-DD
#   DD Mon YYYY (e.g. 15 Jan 1990)
#
# The year is restricted to 1900–2099 to avoid Aadhaar number fragments
# (e.g. "6976") matching as DOB.
# Requires both a separator AND a valid month component — bare 4-digit
# numbers like "6976" will NOT match because they have no separator context.
# ---------------------------------------------------------------------------
DOB: Pattern = re.compile(
    r"\b(?:"
    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    r"(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.]((?:19|20)\d{2})"
    r"|"
    # YYYY-MM-DD or YYYY/MM/DD  (year first, must be 19xx or 20xx)
    r"((?:19|20)\d{2})[/\-](?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])"
    r"|"
    # DD Mon YYYY
    r"(?:0?[1-9]|[12]\d|3[01])\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"\s+((?:19|20)\d{2})"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Indian passport number — format: A9999999 (1 letter + 7 digits)
# ---------------------------------------------------------------------------
PASSPORT: Pattern = re.compile(
    r"\b[A-PR-WY][1-9]\d{6}\b"
)

# ---------------------------------------------------------------------------
# Indian bank account number — 9 to 18 digits (generic)
# ---------------------------------------------------------------------------
BANK_ACCOUNT: Pattern = re.compile(
    r"\b\d{9,18}\b"
)


# ---------------------------------------------------------------------------
# Luhn algorithm — credit card validation
# ---------------------------------------------------------------------------

def luhn_check(card_number: str) -> bool:
    """
    Validate a credit card number using the Luhn algorithm.

    Args:
        card_number: Digits-only string (no spaces or hyphens).

    Returns:
        True if the number passes Luhn check, False otherwise.
    """
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13:
        return False

    # Double every second digit from the right
    total = 0
    reverse = digits[::-1]
    for i, digit in enumerate(reverse):
        if i % 2 == 1:
            doubled = digit * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += digit

    return total % 10 == 0


# ---------------------------------------------------------------------------
# Context keywords used by ContextDetector
# Keys are PIICategory values; values are trigger label words
# ---------------------------------------------------------------------------
CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "person_name": [
        "full name", "name", "applicant name", "candidate name",
        "student name", "employee name", "customer name",
        "father name", "father's name", "mother name", "mother's name",
        "mother s name", "father s name",
        "guardian name", "spouse name", "nominee name",
        "first name", "last name", "surname",
    ],
    "password": [
        "password", "passwd", "pass", "pwd",
        "confirm password", "retype password", "new password",
        "old password", "current password",
    ],
    "date_of_birth": [
        "date of birth", "dob", "birth date", "birthdate",
        "born on", "born", "d.o.b",
    ],
    "address": [
        "address", "addr", "street", "locality", "city",
        "state", "pin code", "pincode", "zip", "residence",
        "permanent address", "current address",
    ],
    "organization": [
        "employer", "organization", "organisation", "company",
        "firm", "institution", "office", "workplace",
        "employer name", "company name", "organization name",
    ],
    "bank_account": [
        "account number", "account no", "acc no", "acct",
        "bank account", "savings account", "current account",
    ],
    "aadhaar": [
        "aadhaar number", "aadhar number",
        "aadhaar no", "aadhar no",
        "uidai number", "uidai no",
        "enrolment number", "enrollment number",
    ],
    "pan": [
        "pan", "pan card", "pan number", "permanent account number",
    ],
    "ifsc": [
        "ifsc", "ifsc code", "bank code",
    ],
    "credit_card": [
        "card number", "credit card", "debit card",
        "card no", "cvv", "expiry",
    ],
}
