"""Input and output guardrails for the HR Policy Copilot API.

Two layers of protection:
  1. Input guardrails  — run BEFORE the RAG pipeline (reject bad questions early,
                         save LLM cost + block prompt injection / PII leaks).
  2. Output guardrails — run AFTER the LLM answer (catch hallucinations, PII
                         leaks, toxic content before it reaches the user).

Each guardrail returns None when the content is clean, or a dict with
{"blocked": True, "reason": "...", "detail": "..."} when it must be stopped.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Shared patterns
# ---------------------------------------------------------------------------

# Indian PII patterns — Aadhaar, PAN, phone, email, bank account, UAN
_PII_PATTERNS = {
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "phone_in": re.compile(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "bank_account": re.compile(r"\b\d{9,18}\b"),
    "uan": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
}

# Prompt-injection markers — classic jailbreak / role-override attempts
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.IGNORECASE),
    re.compile(r"new\s+persona", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"<\s*/\s*instruction\s*>", re.IGNORECASE),
    re.compile(r"<\s*instruction\s*>", re.IGNORECASE),
    re.compile(r"DAN\s*mode", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+(are|have)", re.IGNORECASE),
]

# Toxic / abusive keywords (lightweight blocklist — not exhaustive)
_TOXIC_PATTERNS = [
    re.compile(r"\b(kill|murder|suicide|terrorist|bomb)\w*\b", re.IGNORECASE),
    re.compile(r"\b(hate|racist|sexist|slur)\w*\b", re.IGNORECASE),
]

# HR-policy relevance — reject clearly off-topic questions
_HR_KEYWORDS = re.compile(
    r"\b(leave|salary|pay|bonus|attendance|holiday|probation|resign|notice|"
    r"benefit|insurance|pf|gratuity|loan|travel|expense|reimburse|policy|"
    r"harassment|posh|grievance|promotion|transfer|work\s*hour|overtime|"
    r"maternity|paternity|sick|casual|earned|compensatory|"
    r"interview|hire|recruit|onboard|offer|referral|bgv|background|"
    r"laptop|asset|it|email|password|security|confidential|nda|"
    r"code\s*of\s*conduct|ethics|disciplinary|pip|performance|"
    r"ctc|basic|hra|da|allowance|deduction|tds|form\s*16|"
    r"manager|hr|employee|staff|colleague|team|company|organisation)\b",
    re.IGNORECASE,
)

# Known refusal phrases the LLM sometimes emits when it should have answered
_REFUSAL_PHRASES = [
    re.compile(r"i\s+(do\s+not|don't|cannot|can't)\s+have\s+that\s+information", re.IGNORECASE),
    re.compile(r"i\s+am\s+(not\s+)?(able|allowed)\s+to\s+(answer|provide|assist)", re.IGNORECASE),
    re.compile(r"this\s+is\s+not\s+covered\s+in\s+the\s*policy\s+documents?", re.IGNORECASE),
]
# ---------------------------------------------------------------------------
# Input guardrails — run before the pipeline
# ---------------------------------------------------------------------------

def check_input_guardrails(question: str) -> dict | None:
    """Return a block dict if the input is unsafe/invalid, else None."""
    """Return a block dict if the input is unsafe / invalid, else None."""

    # 1. Empty / whitespace-only
    if not question or not question.strip():
        return {"blocked": True, "reason": "empty_input", "detail": "Question cannot be empty."}

    cleaned = question.strip()

    # 2. Prompt injection detection
    for pat in _INJECTION_PATTERNS:
        if pat.search(cleaned):
            return {
                "blocked": True,
                "reason": "prompt_injection",
                "detail": "Your question contains instructions that conflict with this service. Please rephrase.",
            }

    # 3. PII detection — don't let users paste Aadhaar / PAN / phone into a chat
    for pii_name, pat in _PII_PATTERNS.items():
        if pat.search(cleaned):
            return {
                "blocked": True,
                "reason": "pii_detected",
                "detail": f"Your question appears to contain personal information ({pii_name}). Please remove it and try again.",
            }

    # 4. Toxic content
    for pat in _TOXIC_PATTERNS:
        if pat.search(cleaned):
            return {
                "blocked": True,
                "reason": "toxic_content",
                "detail": "Your question contains inappropriate content and cannot be processed.",
            }

    # 5. Off-topic relevance check — only if question is long enough to judge
    if len(cleaned) > 25 and not _HR_KEYWORDS.search(cleaned):
        return {
            "blocked": True,
            "reason": "off_topic",
            "detail": "Your question doesn't appear to relate to HR policies. Please ask about leave, payroll, conduct, or other workplace policies.",
        }

    return None  # clean


# ---------------------------------------------------------------------------
# Output guardrails — run after the LLM answer
# ---------------------------------------------------------------------------

def check_output_guardrails(answer: str, context_chunks: list[str]) -> dict | None:
    """Return a block dict if the output is unsafe / a hallucination, else None."""

    if not answer:
        return {"blocked": True, "reason": "empty_output", "detail": "The system returned an empty answer."}

    # 1. PII leak detection — LLM should never echo back Aadhaar / PAN / phone
    for pii_name, pat in _PII_PATTERNS.items():
        if pat.search(answer):
            return {
                "blocked": True,
                "reason": "pii_leak",
                "detail": f"The generated answer contains what looks like {pii_name} data and has been blocked for safety.",
            }

    # 2. Toxic content in output
    for pat in _TOXIC_PATTERNS:
        if pat.search(answer):
            return {
                "blocked": True,
                "reason": "toxic_output",
                "detail": "The generated answer was flagged for inappropriate content.",
            }

    # 3. Hallucination heuristic — if the answer contains a number/figure that
    #    does not appear in ANY retrieved context chunk, flag it as likely hallucinated.
    #    This is a lightweight check, not a full NLI model.
    if context_chunks:
        answer_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", answer))
        context_text = " ".join(context_chunks)
        context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", context_text))
        # allow small numbers (1-12 months, etc.) — only flag figures >= 1000
        novel_big_numbers = {n for n in answer_numbers if int(n.replace(".", "")) >= 1000} - context_numbers
        if novel_big_numbers:
                        return {
                "blocked": True,
                "reason": "suspected_hallucination",
                                "detail": f"The answer contains figures ({', '.join(sorted(novel_big_numbers)[:3])}) not found in the policy documents. Please verify with HR.",
            }

    return None  # clean
