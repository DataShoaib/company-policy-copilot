import re
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from hr_rag.config import POLICIES_DIR

# filename -> category, also used for RBAC filtering in the API layer
_CATEGORY_MAP = {
    "01_leave_policy": "leave",
    "02_compensation_payroll_policy": "compensation",
    "03_code_of_conduct_policy": "conduct",
    "04_performance_management_policy": "performance",
    "05_recruitment_onboarding_policy": "recruitment",
    "06_finance_expense_policy": "finance",
    "07_it_security_policy": "it",
    "08_legal_compliance_policy": "legal",
    "09_operations_workplace_policy": "operations",
}

_DOC_ID_RE = re.compile(r"Document ID:\s*([A-Z0-9\-]+)")
_VERSION_RE = re.compile(r"Version:\s*([\d.]+)")
_EFFECTIVE_RE = re.compile(r"Effective:\s*([A-Za-z0-9 ,]+?)(?:\r?\n|$|\|)")
_TITLE_RE = re.compile(r"^#\s*(.+?)\r?$", re.MULTILINE)


def _enrich_metadata(doc: Document) -> Document:
    stem = Path(doc.metadata["source"]).stem
    text = doc.page_content

    doc_id = _DOC_ID_RE.search(text)
    version = _VERSION_RE.search(text)
    effective = _EFFECTIVE_RE.search(text)
    title = _TITLE_RE.search(text)

    doc.metadata.update({
        "category": _CATEGORY_MAP.get(stem, "unknown"),
        "policy_doc_id": doc_id.group(1) if doc_id else "unknown",
        "policy_version": version.group(1) if version else "unknown",
        "effective_date": effective.group(1).strip() if effective else "unknown",
        "title": title.group(1).strip() if title else stem,
    })
    return doc


def load_policy_documents(policies_dir: Path = POLICIES_DIR) -> list[Document]:
    loader = DirectoryLoader(str(policies_dir), glob="**/*.md", loader_cls=TextLoader)
    return [_enrich_metadata(d) for d in loader.load()]
