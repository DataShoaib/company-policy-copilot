import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CATEGORIES = {
    "All allowed policies": None,
    "HR - Leave": "leave",
    "HR - Compensation": "compensation",
    "HR - Conduct": "conduct",
    "HR - Performance": "performance",
    "HR - Recruitment": "recruitment",
    "Finance": "finance",
    "IT and Security": "it",
    "Legal and Compliance": "legal",
    "Operations and Workplace": "operations",
}
ROLE_CATEGORIES = {
    "employee": {"leave", "conduct", "recruitment", "it", "operations"},
    "manager": {"leave", "conduct", "recruitment", "performance", "it", "operations"},
    "finance_user": {"leave", "conduct", "recruitment", "finance", "it", "operations"},
    "hr_admin": set(CATEGORIES.values()) - {None},
}


def _error_detail(response, fallback):
    """Extract the API's error message without crashing on non-JSON bodies
    (e.g. an HTML 502 page from a reverse proxy)."""
    try:
        return response.json().get("detail", fallback)
    except ValueError:
        return fallback

st.set_page_config(page_title="Policy Desk", page_icon="📚", layout="wide")
st.markdown("""
<style>
    .stApp { background: #f4f1ea; color: #17211b; }
    [data-testid="stSidebar"] { background: #18352b; }
    [data-testid="stSidebar"] * { color: #f4f1ea; }
    .hero { padding: 2rem 0 1rem; border-bottom: 1px solid #c9c5b9; }
    .hero h1 { color: #18352b; font-size: 3rem; margin: 0; }
    .hero p { color: #556057; font-size: 1.05rem; }
    .answer { background: #fffdf8; border-left: 5px solid #d36b3d; padding: 1.2rem 1.4rem; }
    .source { background: #e8e4d9; padding: .7rem 1rem; margin: .4rem 0; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "role" not in st.session_state:
    st.session_state.role = None

st.sidebar.title("Policy Desk")
st.sidebar.caption("Grounded answers from company policy")

if not st.session_state.access_token:
    login_tab, signup_tab = st.sidebar.tabs(["Sign in", "Create account"])
    with login_tab:
        username = st.text_input("Username", value="employee1", key="login_username")
        password = st.text_input("Password", type="password", value="employee123", key="login_password")
        if st.button("Sign in", type="primary", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/auth/login", json={"username": username, "password": password}, timeout=10)
                if response.ok:
                    data = response.json()
                    st.session_state.access_token = data["access_token"]
                    st.session_state.role = data["role"]
                    st.rerun()
                else:
                    st.error(_error_detail(response, "Sign in failed"))
            except requests.RequestException as exc:
                st.error(f"API unavailable: {exc}")
    with signup_tab:
        new_username = st.text_input("Username", key="signup_username")
        new_full_name = st.text_input("Full name", key="signup_full_name")
        new_password = st.text_input("Password", type="password", key="signup_password")
        st.caption("New accounts are created as employees. An administrator must grant elevated access.")
        if st.button("Create employee account", use_container_width=True):
            try:
                response = requests.post(
                    f"{API_URL}/auth/signup",
                    json={"username": new_username, "full_name": new_full_name, "password": new_password, "role": "employee"},
                    timeout=10,
                )
                if response.ok:
                    data = response.json()
                    st.session_state.access_token = data["access_token"]
                    st.session_state.role = data["role"]
                    st.rerun()
                else:
                    st.error(_error_detail(response, "Account creation failed"))
            except requests.RequestException as exc:
                st.error(f"API unavailable: {exc}")
else:
    st.sidebar.success(f"Signed in as {st.session_state.role}")
    if st.sidebar.button("Sign out", use_container_width=True):
        st.session_state.access_token = None
        st.session_state.role = None
        st.rerun()

st.markdown('<div class="hero"><h1>Ask the right policy.</h1><p>Select a department, then get an answer grounded only in that policy collection.</p></div>', unsafe_allow_html=True)

if st.session_state.access_token:
    allowed = ROLE_CATEGORIES.get(st.session_state.role, set())
    available_categories = {
        label: category for label, category in CATEGORIES.items()
        if category is None or category in allowed
    }
    selected_label = st.selectbox("Search department", list(available_categories))
    question = st.text_area("Your question", placeholder="Example: What is the domestic meal allowance?", height=110)
    ask = st.button("Ask policy assistant", type="primary", disabled=not question.strip())

    if ask:
        category = available_categories[selected_label]
        try:
            response = requests.post(
                f"{API_URL}/query",
                headers={"Authorization": f"Bearer {st.session_state.access_token}"},
                json={"question": question.strip(), "category": category},
                timeout=90,
            )
            if response.status_code == 401:
                st.session_state.access_token = None
                st.error("Your session expired. Please sign in again.")
                st.rerun()
            response.raise_for_status()
            result = response.json()
            st.markdown('<div class="answer">', unsafe_allow_html=True)
            st.write(result["answer"])
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(f"Collection scope: {selected_label} | {'Cached' if result['cached'] else 'Fresh'} | {result['latency_ms']} ms")
            st.subheader("Sources")
            for source in result["sources"]:
                st.markdown(f'<div class="source"><b>{source["title"]}</b> · {source["policy_doc_id"]}<br>{source["snippet"]}</div>', unsafe_allow_html=True)
        except requests.RequestException as exc:
            if getattr(exc.response, "status_code", None) == 503:
                st.error("Backend dependency unavailable. Start Redis and Qdrant, then try again.")
            else:
                st.error(f"Request failed: {exc}")
else:
    st.info("Sign in from the sidebar to search your permitted policy collections.")
