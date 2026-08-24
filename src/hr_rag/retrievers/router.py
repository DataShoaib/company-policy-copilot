import re

from hr_rag.config import CATEGORIES

_KEYWORDS = {
    "leave": {"leave", "vacation", "maternity", "paternity", "sick", "probation"},
    "compensation": {"salary", "payroll", "bonus", "gratuity", "provident", "insurance", "pf"},
    "conduct": {"conduct", "harassment", "grievance", "disciplinary", "misconduct"},
    "performance": {"performance", "pip", "promotion", "rating", "increment"},
    "recruitment": {"recruitment", "interview", "referral", "candidate", "onboarding", "hiring"},
    "finance": {"finance", "expense", "expenses", "reimbursement", "invoice", "invoices", "budget", "travel", "procurement"},
    "it": {"laptop", "vpn", "password", "phishing", "security", "software", "mfa"},
    "legal": {"legal", "contract", "contracts", "vendor", "compliance", "privacy", "regulatory", "litigation"},
    "operations": {"operations", "office", "badge", "visitor", "continuity", "incident", "supplies"},
}


def route_question(question: str, allowed_categories: list[str], requested_category: str | None = None) -> list[str]:
    allowed = set(allowed_categories)
    if requested_category:
        return [requested_category] if requested_category in allowed else []

    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    matches = [category for category in CATEGORIES if category in allowed and words & _KEYWORDS.get(category, set())]
    return matches or list(allowed_categories)
