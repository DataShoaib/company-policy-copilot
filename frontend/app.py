import base64
import json
import os
from datetime import datetime, timezone

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = (5, 90)

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

# Provisionable roles, in the same order as the API's VALID_ROLES (rbac.py).
ROLE_LABELS = {"employee": "Employee", "manager": "Manager",
               "finance_user": "Finance user", "hr_admin": "HR admin"}

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
    "history": [],
    "pending": None,
    "provision_result": None,
}.items():
    st.session_state.setdefault(_key, _default)


def is_logged_in() -> bool:
    return bool(st.session_state.access_token)


def jwt_claims(token: str | None) -> dict:
    if not token or token.count(".") != 2:
        return {}
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:  # noqa: BLE001 - a malformed token just reads as logged out
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
    if resp.status_code == 401:
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


def api_provision(username: str, password: str, full_name: str,
                  role: str) -> tuple[bool, str]:
    """/auth/provision — HR-admin only, creates an account with any role."""
    payload = {"username": username.strip(), "password": password,
               "full_name": full_name.strip(), "role": role}
    resp = None
    for attempt in (1, 2):
        try:
            resp = _post("/auth/provision", payload,
                         token=st.session_state.access_token)
        except requests.RequestException as exc:
            return False, (f"API unreachable ({exc.__class__.__name__}) — "
                           f"is the backend running on {API_URL}?")
        if resp.status_code == 401 and attempt == 1 and try_refresh():
            continue
        break
    if resp.status_code == 200:
        # The endpoint answers with a token pair for the *new* user. Deliberately
        # discarded: stashing it would silently switch the admin's own session
        # over to the account they just created.
        return True, (f"Provisioned “{username.strip()}” as "
                      f"{ROLE_LABELS.get(role, role)}. They can sign in now.")
    if resp.status_code == 401:
        return False, "Session expired — sign in again."
    if resp.status_code == 403:
        return False, "Only HR admins can provision users."
    return False, _detail(resp, "Provisioning failed.")


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

/* ── 0. app background ──────────────────────────────────────────────── */
.stApp { background: #f6f3ec; }
[data-testid="stHeader"] { background: transparent; }

/* ── 1. BLANKET: har text dark. .stApp har Streamlit version me hota hai ── */
.stApp, .stApp * { color: #1f2937 !important; }

/* ── 2. SIDEBAR wapas light-on-dark (blanket se zyada specific) ──────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0e2b21 0%, #14402f 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {
  color: #e9f1ea !important;
}
section[data-testid="stSidebar"] .role-chip { color: #10241b !important; }
/* Username / Password / provision text inputs in the sidebar: their own
   background is a light cream box (not the dark sidebar background), so
   they need dark text — this specific rule was still assigning them the
   sidebar's light text color from the blanket rule above, which is why
   typed text stayed faint/invisible even after the other fixes. */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
  color: #1f2937 !important;
  -webkit-text-fill-color: #1f2937 !important;
  background-color: #fffdf7 !important;
}
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {
  color: #8a8474 !important;
  -webkit-text-fill-color: #8a8474 !important;
}

/* Sign in / Create account tab labels: keep them light against the dark
   sidebar. Streamlit dims INACTIVE tab text by default (a low-opacity gray)
   which disappears on a dark background even after the color is overridden,
   and different Streamlit/BaseWeb builds expose the tab as stTabs, role="tab"
   or data-baseweb="tab" — so all three are covered, plus opacity/background
   are pinned explicitly since -webkit-text-fill-color alone wasn't enough. */
section[data-testid="stSidebar"] [data-testid="stTabs"] button,
section[data-testid="stSidebar"] [data-testid="stTabs"] button *,
section[data-testid="stSidebar"] [role="tab"],
section[data-testid="stSidebar"] [role="tab"] *,
section[data-testid="stSidebar"] [data-baseweb="tab"],
section[data-testid="stSidebar"] [data-baseweb="tab"] * {
  color: #e9f1ea !important;
  -webkit-text-fill-color: #e9f1ea !important;
  opacity: 1 !important;
  background: transparent !important;
}
/* Sign in / Create account buttons: red only. */
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button[kind="primary"] {
  background: #dc2626 !important;
  border-color: #dc2626 !important;
}

/* Password eye button only: black. */
section[data-testid="stSidebar"] input[type="password"] ~ button,
section[data-testid="stSidebar"] [data-testid="stTextInput"] button {
  color: #000000 !important;
  background: transparent !important;
}

section[data-testid="stSidebar"] [data-testid="stTextInput"] button svg {
  color: #000000 !important;
  fill: #000000 !important;
  stroke: #000000 !important;
}

section[data-testid="stSidebar"] [aria-selected="true"],
section[data-testid="stSidebar"] [data-testid="stTabs"] [aria-selected="true"] * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

/* Select boxes (e.g. Provision → Role) sit on a light control background,
   not the dark sidebar background, so they need dark text — both the
   closed control and the open dropdown list. BaseWeb renders the open list
   in a portal outside the sidebar's own DOM, hence the separate rule below
   that isn't scoped to section[data-testid="stSidebar"]. */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background-color: #fffdf7 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] * {
  color: #1f2937 !important;
  -webkit-text-fill-color: #1f2937 !important;
}
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul[role="listbox"] {
  background-color: #fffdf7 !important;
}
[data-baseweb="popover"] [data-baseweb="menu"] *,
[data-baseweb="popover"] ul[role="listbox"] * {
  color: #1f2937 !important;
  -webkit-text-fill-color: #1f2937 !important;
}

/* Sidebar auth/action buttons: Sign in, Create account, Clear chat,
   and Sign out use a red background with white text. */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stButton > button * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

section[data-testid="stSidebar"] .stButton > button {
  background: #dc2626 !important;
  border-color: #dc2626 !important;
}
/* Password visibility eye button only */
section[data-testid="stSidebar"] input[type="password"] + button,
section[data-testid="stSidebar"] input[type="password"] ~ button {
  color: #000000 !important;
  -webkit-text-fill-color: #000000 !important;
}
section[data-testid="stSidebar"] input[type="password"] + button svg,
section[data-testid="stSidebar"] input[type="password"] ~ button svg {
  color: #000000 !important;
  fill: #000000 !important;
  stroke: #000000 !important;
}

/* ── 3. DARK surfaces in main pane -> white text ────────────────────── */
.stApp .hero, .stApp .hero *, .stApp .hero h1,
.stApp .hero p, .stApp .hero-badge { color: #ffffff !important; }

/* Only PRIMARY buttons (Sign in, Provision user, etc.) get forced white text —
   they have a dark/colored background from Streamlit's own theming.
   Secondary buttons (Clear chat, Sign out, example-question pills) keep the
   default dark text from the blanket rule, since their background is light. */
.stApp .stButton > button[kind="primary"], .stApp .stButton > button[kind="primary"] *,
.stApp [data-testid="stFormSubmitButton"] button,
.stApp [data-testid="stFormSubmitButton"] button * { color: #ffffff !important; }

.stApp [data-testid="stPills"] button,
.stApp [data-testid="stPills"] button * { color: #ffffff !important; }

/* Chat input box: its own background is light (not the dark sidebar), so it
   needs dark typed text — the earlier light-text rule was invisible on it.
   The send button is a separate small control with an SVG arrow icon, which
   `color` alone doesn't reach — fill/stroke are set explicitly so the icon
   itself is visible against its dark button background. */
.stApp [data-testid="stChatInput"] {
  background: #fffdf7 !important;
  border: 1px solid #e6e1d2 !important;
  border-radius: 14px;
}
.stApp [data-testid="stChatInput"] textarea {
  color: #1f2937 !important;
  -webkit-text-fill-color: #1f2937 !important;
  border-radius: 14px;
}
.stApp [data-testid="stChatInput"] textarea::placeholder {
  color: #8a8474 !important;
  -webkit-text-fill-color: #8a8474 !important;
}
.stApp [data-testid="stChatInput"] button {
  background: #123a2c !important;
  border-radius: 10px !important;
}
.stApp [data-testid="stChatInput"] button svg {
  fill: #ffffff !important;
  stroke: #ffffff !important;
  color: #ffffff !important;
  opacity: 1 !important;
}

/* ── 4. chat bubbles + expanders light so dark text reads ────────────── */
.stApp [data-testid="stChatMessage"] {
  background: #fffdf7 !important;
  border: 1px solid #e6e1d2;
  border-radius: 14px;
}
.stApp [data-testid="stChatMessage"] * { color: #1f2937 !important; }
.stApp [data-testid="stExpander"], .stApp details {
  background: #fffdf7 !important; border: 1px solid #e6e1d2; border-radius: 12px;
}
.stApp [data-testid="stExpander"] *, .stApp details summary,
.stApp details summary * { color: #1f2937 !important; }
.stApp [data-testid="stAlert"] * { color: #1f2937 !important; }

/* ── SEARCH SCOPE PILLS ONLY ─────────────────────────────────────────── */
/* Force only the document-type pills to stay readable from the start. */
.stApp [data-testid="stPills"] button,
.stApp [data-testid="stPills"] button[role="radio"],
.stApp [data-testid="stPills"] button[data-baseweb] {
  background-color: #fffdf7 !important;
  background: #fffdf7 !important;
  color: #1f2937 !important;
  -webkit-text-fill-color: #1f2937 !important;
  opacity: 1 !important;
  border-color: #d8d2c4 !important;
}

.stApp [data-testid="stPills"] button span,
.stApp [data-testid="stPills"] button div,
.stApp [data-testid="stPills"] button p,
.stApp [data-testid="stPills"] button label,
.stApp [data-testid="stPills"] button strong,
.stApp [data-testid="stPills"] button small {
  color: #1f2937 !important;
  -webkit-text-fill-color: #1f2937 !important;
  opacity: 1 !important;
  background: transparent !important;
}

.stApp [data-testid="stPills"] button:hover,
.stApp [data-testid="stPills"] button:hover span,
.stApp [data-testid="stPills"] button:hover div,
.stApp [data-testid="stPills"] button:hover p,
.stApp [data-testid="stPills"] button:hover label {
  background-color: #fffdf7 !important;
  background: #fffdf7 !important;
  color: #1f2937 !important;
  -webkit-text-fill-color: #1f2937 !important;
}

.stApp [data-testid="stPills"] button[aria-checked="true"],
.stApp [data-testid="stPills"] button[aria-checked="true"] span,
.stApp [data-testid="stPills"] button[aria-checked="true"] div,
.stApp [data-testid="stPills"] button[aria-checked="true"] p,
.stApp [data-testid="stPills"] button[aria-checked="true"] label {
  background-color: #fffdf7 !important;
  background: #fffdf7 !important;
  color: #1d4ed8 !important;
  -webkit-text-fill-color: #1d4ed8 !important;
  border-color: #f28b82 !important;
}

/* Search scope / category section */
.stApp [data-testid="stPills"] {
  background: linear-gradient(135deg, #fffaf0 0%, #fff5f5 100%) !important;
  border: 1px solid #e6d8c8 !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
  box-shadow: 0 5px 16px rgba(31, 41, 55, 0.07) !important;
}

.stApp [data-testid="stPills"] > label {
  color: #7f1d1d !important;
  font-weight: 700 !important;
}

.stApp [data-testid="stPills"] button {
  border-radius: 999px !important;
  border: 1px solid #d8d2c4 !important;
  box-shadow: none !important;
  transition: all .15s ease-in-out !important;
}

.stApp [data-testid="stPills"] button:hover {
  border-color: #ef4444 !important;
  transform: translateY(-1px);
}

.stApp [data-testid="stPills"] button[aria-checked="true"] {
  border-color: #dc2626 !important;
  box-shadow: 0 2px 8px rgba(220, 38, 38, 0.14) !important;
}

/* Example search-question buttons */
.stApp .stButton > button {
  border: 1px solid #e6e1d2 !important;
  box-shadow: 0 2px 8px rgba(31, 41, 55, 0.05) !important;
}

.stApp .stButton > button:hover {
  border-color: #f28b82 !important;
  box-shadow: 0 4px 12px rgba(31, 41, 55, 0.09) !important;
}

/* Source cards */
.stApp .src-card {
  background: linear-gradient(135deg, #fffdf7 0%, #fff7ed 100%) !important;
  border: 1px solid #eadfd0 !important;
  border-left: 4px solid #dc2626 !important;
  border-radius: 13px !important;
  padding: 12px 15px !important;
  margin: 9px 0 !important;
  box-shadow: 0 4px 12px rgba(31, 41, 55, 0.06) !important;
}

.stApp .src-card .docid {
  background: #f3eee5 !important;
  border: 1px solid #e2d8c8 !important;
}

.stApp .src-card .chip {
  background: #fee2e2 !important;
  border-color: #fecaca !important;
  color: #991b1b !important;
}

/* Sources expander */
.stApp [data-testid="stExpander"] {
  box-shadow: 0 3px 12px rgba(31, 41, 55, 0.05) !important;
}


/* ── 5. custom components ───────────────────────────────────────────── */
.side-title { font-size: 1.15rem; font-weight: 800; margin: 4px 0 2px; }
.user-card { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14);
             border-radius: 14px; padding: 12px 14px; margin: 10px 0; }
.u-name { font-weight: 700; font-size: .98rem; margin-bottom: 6px; }
.role-chip { display: inline-block; border-radius: 999px; padding: 2px 12px;
             font-size: .72rem; font-weight: 700; }
.u-meta { opacity: .75; font-size: .74rem; margin-top: 8px; }
.hero { background: linear-gradient(120deg, #123a2c, #1e5c44);
        border-radius: 18px; padding: 24px 30px; margin-bottom: 16px;
        box-shadow: 0 12px 30px rgba(18,58,44,.22); }
.hero h1 { margin: 4px 0 6px; font-size: 2rem; font-weight: 800; }
.hero p { margin: 0; font-size: 1.05rem; font-weight: 500; }
.hero-badge { display: inline-block; background: rgba(255,255,255,.16);
              padding: 4px 12px; border-radius: 999px;
              font-size: .76rem; letter-spacing: .5px; }

.stApp .chip {
  display: inline-block; background: #efece2; border: 1px solid #ddd7c6;
  color: #4a463a !important; border-radius: 999px; padding: 2px 10px;
  font-size: .72rem; margin: 2px 6px 2px 0; font-weight: 600;
}

section[data-testid="stSidebar"] .chip.light {
  background: rgba(255,255,255,.10); border-color: rgba(255,255,255,.22);
  color: #dfe9df !important; margin-top: 4px;
}

.stApp .chip.ok {
  background: #e4f3e6; border-color: #bfe0c4; color: #1e6b34 !important;
}

.stApp .src-card {
  background: #fffdf7; border: 1px solid #e6e1d2;
  border-left: 4px solid #d36b3d; border-radius: 12px;
  padding: 10px 14px; margin: 8px 0;
}

.stApp .src-card, .stApp .src-card * { color: #1f2937 !important; }

.stApp .docid {
  font-family: ui-monospace, monospace; background: #f0ede3;
  padding: 1px 8px; border-radius: 6px; font-size: .72rem;
  color: #6b5d3f !important; margin-left: 8px;
}

.stButton > button { border-radius: 10px; font-weight: 600; }
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

        if role == "hr_admin":
            with st.expander("👥 Provision a user", expanded=False):
                st.caption("HR admins only — the API returns 403 for every "
                           "other role. The teammate signs in themselves "
                           "with the temporary password below.")

                if st.session_state.provision_result:
                    st.success(st.session_state.provision_result)

                with st.form("provision_form", clear_on_submit=True):
                    pv_user = st.text_input("Username", max_chars=100)
                    pv_name = st.text_input("Full name", max_chars=200)
                    pv_pass = st.text_input("Temporary password",
                                            type="password",
                                            help="8–72 characters.")

                    pv_role = st.selectbox(
                        "Role", list(ROLE_ALLOWED),
                        format_func=lambda r: ROLE_LABELS.get(r, r))

                    pv_submit = st.form_submit_button(
                        "Provision user", type="primary",
                        use_container_width=True)

                st.caption(f"{len(ROLE_ALLOWED.get(pv_role, []))} of "
                           f"{len(CATEGORY_LABELS) - 1} policy scopes for "
                           f"**{ROLE_LABELS.get(pv_role, pv_role)}**.")

                if pv_submit:
                    ok, msg = api_provision(pv_user, pv_pass, pv_name, pv_role)
                    if ok:
                        st.session_state.provision_result = msg
                        st.rerun()
                    st.error(msg)

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
            su_pass = st.text_input("Choose password", type="password", key="su_pass")

            st.caption("New accounts start with the **employee** role.")

            if st.button("Create account", use_container_width=True):
                ok, msg = api_signup(su_user, su_pass, su_name)
                if ok:
                    st.toast("Account created 🎉")
                    st.rerun()
                st.error(msg)

    st.divider()
    st.caption(f"API · `{API_URL}`\n\nEvery answer cites its source document.")


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

scope_map = {label: cat for cat, label in CATEGORY_LABELS.items()
             if cat is None or cat in allowed}

if not scope_map:
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

if sel_label not in scope_map:
    sel_label = default_label

category = scope_map[sel_label]

pool = EXAMPLES.get(category) or EXAMPLES["default"]

example_cols = st.columns(len(pool))

for col, qtext in zip(example_cols, pool, strict=True):
    if col.button(qtext, key=f"ex::{qtext}", use_container_width=True):
        st.session_state.pending = qtext


for msg in st.session_state.history:
    with st.chat_message(
        "user" if msg["kind"] == "user" else "assistant",
        avatar="🧑‍💼" if msg["kind"] == "user"
        else ("⚠️" if msg["kind"] == "error" else "🧭")
    ):
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
                        st.markdown(
                            source_card(src),
                            unsafe_allow_html=True,
                        )


question = (
    st.chat_input("Ask anything about company policies…")
    or st.session_state.pending
)

st.session_state.pending = None

if question and question.strip():
    question = question.strip()

    st.session_state.history.append({
        "kind": "user",
        "content": question,
    })

    with st.chat_message("assistant", avatar="🧭"):
        with st.spinner("Searching policy collections…"):
            result = api_ask(question, category)

        if result["ok"]:
            entry = {
                "kind": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "cached": result.get("cached", False),
                "latency": result.get("latency_ms", 0),
                "scope": sel_label,
            }
        else:
            entry = {
                "kind": "error",
                "content": result["message"],
            }

        st.session_state.history.append(entry)

    st.rerun()