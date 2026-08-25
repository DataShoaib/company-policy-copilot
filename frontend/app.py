"""Policy Desk — grounded assistant over company policy documents.

Streamlit front-end for the Company Policy Copilot API:
JWT login/signup, silent refresh-token rotation, department-scoped questions,
cited answers with cache/latency chips and a chat-style history.
"""
from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = (5, 90)  # (connect, read)

CATEGORY_LABELS = {
    None: "🌐 All allowed policies",
    "leave": "🏖️ Leave",
    "compensation": "💰 Compensation",
    "conduct": "⚖️ Conduct",
    "performance": "📈 Performance",
    "recruitment": "🧭 Recruitment",
    "finance": "💳 Finance & Expense",
    "it": "🔐 IT Security",
    "legal": "📜 Legal & Compliance",
    "operations": "🏢 Operations",
}
LABEL_TO_CATEGORY = {label: cat for cat, label in CATEGORY_LABELS.items()}

ROLE_ALLOWED = {
    "employee": ["leave", "conduct", "recruitment", "it", "operations"],
    "manager": ["leave", "conduct", "recruitment", "performance", "it", "operations"],
    "finance_user": ["leave", "conduct", "recruitment", "finance", "it", "operations"],
    "hr_admin": ["leave", "compensation", "conduct", "performance",
                 "recruitment", "finance", "it", "legal", "operations"],
}
ROLE_COLORS = {"hr_admin": "#f4c95d", "manager": "#8ab6ff",
               "finance_user": "#c9a0ff", "employee": "#7fd6a0"}

EXAMPLES = {
    "leave": ["How many days of casual leave do I get per year?",
              "Maternity leave for a third child?",
              "Can I carry forward unused casual leave?"],
    "compensation": ["What is the employer PF contribution?",
                     "How is the annual bonus calculated?"],
    "it": ["What do I do after clicking a phishing link?",
           "What are the password rules?"],
    "finance": ["What are the travel reimbursement limits?",
                "How late can I submit an expense claim?"],
    "default": ["Who approves offers above the standard band?",
                "I lost my office badge — what now?"],
}

for _key, _default in {
    "access_token": None,
    "refresh_token": None,
    "role": None,
    "username": None,
    "history": [],   # [{kind: user|assistant|error, ...}]
    "pending": None,  # example-button handoff across reruns
}.items():
    st.session_state.setdefault(_key, _default)


def is_logged_in() -> bool:
    return bool(st.session_state.access_token)

# --------------------------------------------------------------------------- #
# HTTP helpers (login / signup / refresh-with-retry / ask)
# --------------------------------------------------------------------------- #
def jwt_claims(token: str | None) -> dict:
    """Decode JWT payload WITHOUT verification — UI display only."""
    if not token or token.count(".") != 2:
        return {}
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:  # noqa: BLE001 - display-only decode; never trust token shape
        return {}


def _post(path: str, payload: dict, token: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return requests.post(f"{API_URL}{path}", json=payload,
                         headers=headers, timeout=REQUEST_TIMEOUT)


def _detail(resp, fallback: str) -> str:
    try:
        return resp.json().get("detail") or fallback
    except ValueError:
        return fallback


def try_refresh() -> bool:
    """Exchange refresh token for a new pair (backend rotates single-use)."""
    rt = st.session_state.refresh_token
    if not rt:
        return False
    try:
        resp = _post("/auth/refresh", {"refresh_token": rt})
    except requests.RequestException:
        return False
    if resp.status_code == 200:
        data = resp.json()
        st.session_state.access_token = data["access_token"]
        st.session_state.refresh_token = data["refresh_token"]
        st.session_state.role = data["role"]
        return True
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    return False


def api_login(username: str, password: str) -> tuple[bool, str]:
    try:
        resp = _post("/auth/login", {"username": username.strip(),
                                     "password": password})
    except requests.RequestException as exc:
        return False, (f"API unreachable ({exc.__class__.__name__}) — "
                       f"is the backend running on {API_URL}?")
    if resp.status_code == 200:
        data = resp.json()
        st.session_state.update(access_token=data["access_token"],
                                refresh_token=data["refresh_token"],
                                role=data["role"],
                                username=username.strip())
        return True, ""
    if resp.status_code == 429:
        return False, ("Too many login attempts — "
                       f"retry in {resp.headers.get('Retry-After', '?')}s.")
    return False, _detail(resp, "Login failed.")


def api_signup(username: str, password: str, full_name: str) -> tuple[bool, str]:
    try:
        resp = _post("/auth/signup", {"username": username.strip(),
                                      "password": password,
                                      "full_name": full_name.strip()})
    except requests.RequestException as exc:
        return False, (f"API unreachable ({exc.__class__.__name__}) — "
                       f"is the backend running on {API_URL}?")
    if resp.status_code == 200:
        data = resp.json()
        st.session_state.update(access_token=data["access_token"],
                                refresh_token=data["refresh_token"],
                                role=data["role"],
                                username=username.strip())
        return True, ""
    return False, _detail(resp, "Account creation failed.")


def api_ask(question: str, category: str | None) -> dict:
    """/query with one automatic retry after silent token refresh."""
    payload: dict = {"question": question}
    if category:
        payload["category"] = category
    resp = None
    for attempt in (1, 2):
        try:
            resp = _post("/query", payload, token=st.session_state.access_token)
        except requests.RequestException:
            return {"ok": False,
                    "message": ("Backend unreachable — start it with "
                                "`uvicorn hr_rag.api.main:app --port 8000`.")}
        if resp.status_code == 200:
            return {"ok": True, **resp.json()}
        if resp.status_code == 401 and attempt == 1 and try_refresh():
            continue
        break
    if resp.status_code == 401:
        message = "Session expired — please sign in again."
    elif resp.status_code == 429:
        message = (f"Slow down 🐢 — rate limit hit. "
                   f"Retry in {resp.headers.get('Retry-After', '?')}s.")
    elif resp.status_code == 503:
        message = "Backend dependency is down — start Redis/Qdrant and retry."
    else:
        message = _detail(resp, f"Request failed ({resp.status_code}).")
    return {"ok": False, "message": message}

st.set_page_config(page_title="Company Policy Copilot", page_icon="🧭",
                   layout="wide", initial_sidebar_state="expanded")

THEME_CSS = """
<style>
html, body, [class*="css"] { font-family: "Segoe UI", system-ui, sans-serif; }
.stApp { background: #f6f3ec; }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0e2b21 0%, #14402f 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] * { color: #e9f1ea !important; }
.side-title { font-size: 1.15rem; font-weight: 800; margin: 4px 0 2px; }
.user-card { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14);
             border-radius: 14px; padding: 12px 14px; margin: 10px 0; }
.u-name { font-weight: 700; font-size: .98rem; margin-bottom: 6px; }
.role-chip { display: inline-block; border-radius: 999px; padding: 2px 12px;
             font-size: .72rem; font-weight: 700; }
.u-meta { opacity: .75; font-size: .74rem; margin-top: 8px; }
.hero { background: linear-gradient(120deg, #123a2c, #1e5c44);
        border-radius: 18px; padding: 24px 30px; color: #fff;
        margin-bottom: 16px; box-shadow: 0 12px 30px rgba(18,58,44,.22); }
.hero h1 { margin: 4px 0 6px; font-size: 2rem; font-weight: 800; color: #fff; }
.hero p { margin: 0; opacity: .88; font-size: .95rem; }
.hero-badge { display: inline-block; background: rgba(255,255,255,.16);
              padding: 4px 12px; border-radius: 999px;
              font-size: .76rem; letter-spacing: .5px; }
.chip { display: inline-block; background: #efece2; border: 1px solid #ddd7c6;
        color: #4a463a; border-radius: 999px; padding: 2px 10px;
        font-size: .72rem; margin: 2px 6px 2px 0; font-weight: 600; }
.chip.light { background: rgba(255,255,255,.10); border-color: rgba(255,255,255,.22);
              color: #dfe9df; margin-top: 4px; }
.chip.ok { background: #e4f3e6; border-color: #bfe0c4; color: #1e6b34; }
.src-card { background: #fffdf7; border: 1px solid #e6e1d2;
            border-left: 4px solid #d36b3d; border-radius: 12px;
            padding: 10px 14px; margin: 8px 0; }
.docid { font-family: ui-monospace, monospace; background: #f0ede3;
         padding: 1px 8px; border-radius: 6px; font-size: .72rem;
         color: #6b5d3f; margin-left: 8px; }
.stButton > button { border-radius: 10px; font-weight: 600; }
[data-testid="stChatInput"] textarea { border-radius: 14px; }
#MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


def _esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def chips_html(entry: dict) -> str:
    cache_chip = ('<span class="chip ok">⚡ Cached</span>' if entry.get("cached")
                  else '<span class="chip">🔎 Fresh</span>')
    return (f'{cache_chip}<span class="chip">⏱ {entry.get("latency", 0)} ms</span>'
            f'<span class="chip">🗂 {_esc(entry.get("scope", ""))}</span>')


def source_card(src: dict) -> str:
    title = _esc(src.get("title", "Untitled"))
    doc_id = _esc(src.get("policy_doc_id", ""))
    snippet = _esc(src.get("snippet", ""))
    cat = _esc(src.get("category", ""))
    return (f'<div class="src-card"><b>{title}</b>'
            f'<span class="docid">{doc_id}</span> '
            f'<span class="chip">{cat}</span><br>{snippet}</div>')

# --------------------------------------------------------------------------- #
# Sidebar — auth / user card
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="side-title">🧭 Policy Desk</div>',
                unsafe_allow_html=True)
    st.caption("Grounded answers from official company policies")

    if is_logged_in():
        claims = jwt_claims(st.session_state.access_token)
        name = claims.get("sub") or st.session_state.username or "user"
        role = st.session_state.role or "employee"
        color = ROLE_COLORS.get(role, "#cccccc")
        allowed = ROLE_ALLOWED.get(role, [])
        scope_chips = "".join(
            f'<span class="chip light">{CATEGORY_LABELS.get(c, c)}</span>'
            for c in allowed)
        exp = claims.get("exp")
        valid_until = (datetime.fromtimestamp(exp, tz=timezone.utc)
               .strftime("%H:%M UTC")) if exp else "?"

        st.markdown(
            f'<div class="user-card"><div class="u-name">👤 {_esc(name)}</div>'
            f'<span class="role-chip" style="background:{color};color:#10241b">'
            f'{_esc(role)}</span>'
            f'<div class="u-meta">token valid till {valid_until} · '
            f'{len(allowed)} scopes</div></div>{scope_chips}',
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        if col_a.button("🧹 Clear chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        if col_b.button("🚪 Sign out", use_container_width=True):
            for k in ("access_token", "refresh_token", "role", "username"):
                st.session_state[k] = None
            st.session_state.history = []
            st.rerun()
    else:
        tab_login, tab_signup = st.tabs(["🔑 Sign in", "✨ Create account"])
        with tab_login:
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("Sign in", type="primary", use_container_width=True):
                ok, msg = api_login(li_user, li_pass)
                if ok:
                    st.toast(f"Welcome back, {st.session_state.username}! 👋")
                    st.rerun()
                st.error(msg)
        with tab_signup:
            su_user = st.text_input("Choose username", key="su_user")
            su_name = st.text_input("Full name", key="su_name")
            su_pass = st.text_input("Choose password", type="password",
                                    key="su_pass")
            st.caption("New accounts start with the **employee** role.")
            if st.button("Create account", use_container_width=True):
                ok, msg = api_signup(su_user, su_pass, su_name)
                if ok:
                    st.toast("Account created 🎉")
                    st.rerun()
                st.error(msg)

    st.divider()
    st.caption(f"API · `{API_URL}`\n\nEvery answer cites its source document.")

# --------------------------------------------------------------------------- #
# Main area — hero, scope picker, examples, chat
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">COMPANY POLICY COPILOT</div>
      <h1>Ask once. Get the official answer.</h1>
      <p>Grounded strictly in company policy documents · role-scoped · every answer cites its source</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not is_logged_in():
    st.info("👋 **Sign in from the sidebar** to start asking. "
            "New here? Create an account in one click.")
    st.stop()

role = st.session_state.role or "employee"
allowed = ROLE_ALLOWED.get(role, [])
# Document-type picker rendered as always-visible pills: every allowed scope
# is a clickable chip, so there is no dropdown that can ever come up empty.
scope_map = {label: cat for cat, label in CATEGORY_LABELS.items()
             if cat is None or cat in allowed}
if not scope_map:  # unknown role -> still offer the all-scopes default
    scope_map = {next(iter(CATEGORY_LABELS.values())): None}
default_label = next(iter(scope_map))
sel_label = st.pills(
    "🔎 Search scope (document type)",
    list(scope_map),
    default=default_label,
    key="scope_pills",
    help="Questions are answered only from the selected department's "
         "policy collection.",
)
# Stale widget state can hand back None or a removed label — normalize it.
if sel_label not in scope_map:
    sel_label = default_label
category = scope_map[sel_label]

pool = EXAMPLES.get(category) or EXAMPLES["default"]
example_cols = st.columns(len(pool))
for col, qtext in zip(example_cols, pool):
    if col.button(qtext, key=f"ex::{qtext}", use_container_width=True):
        st.session_state.pending = qtext

for msg in st.session_state.history:
    with st.chat_message("user" if msg["kind"] == "user" else "assistant",
                         avatar="🧑‍💼" if msg["kind"] == "user" else
                                ("⚠️" if msg["kind"] == "error" else "🧭")):
        if msg["kind"] == "error":
            st.error(msg["content"])
        elif msg["kind"] == "user":
            st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])
            st.markdown(chips_html(msg), unsafe_allow_html=True)
            sources = msg.get("sources") or []
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for src in sources:
                        st.markdown(source_card(src), unsafe_allow_html=True)

question = (st.chat_input("Ask anything about company policies…")
            or st.session_state.pending)
st.session_state.pending = None

if question and question.strip():
    question = question.strip()
    st.session_state.history.append({"kind": "user", "content": question})
    with st.chat_message("assistant", avatar="🧭"):
        with st.spinner("Searching policy collections…"):
            result = api_ask(question, category)
        if result["ok"]:
            entry = {"kind": "assistant",
                     "content": result["answer"],
                     "sources": result.get("sources", []),
                     "cached": result.get("cached", False),
                     "latency": result.get("latency_ms", 0),
                     "scope": sel_label}
        else:
            entry = {"kind": "error", "content": result["message"]}
        st.session_state.history.append(entry)
    st.rerun()