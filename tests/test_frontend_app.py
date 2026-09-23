"""Frontend smoke tests.

Guards the regression that made the UI unusable: the backend URL was baked in at
import time and silently pointed at a port with nothing behind it, so every
sign-in failed with a generic error and there was no way to correct it without
editing source. These tests boot the real Streamlit app via AppTest and assert
the URL is resolvable from the environment, visible in the sidebar, and quoted
back in the failure message.
"""
import os
import sys

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1 import element_tree as element_tree_module

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "frontend", "app.py")

# Port 9 is the discard port: nothing listens, so requests fail immediately
# instead of hanging on the 90s read timeout.
DEAD_URL = "http://127.0.0.1:9"


@pytest.fixture(autouse=True)
def _tolerate_pills_labels():
    """Streamlit 1.49's AppTest cannot serialise st.pills' emoji labels.

    ButtonGroup.indices() raises on a label like "🌐 All allowed policies", which
    aborts the AppTest run. The browser UI is unaffected; this only unblocks the
    test driver, so the patch is scoped to the harness.
    """
    original = element_tree_module.ButtonGroup.indices.fget

    def tolerant_indices(self):
        try:
            return original(self)
        except (TypeError, ValueError):
            return []

    element_tree_module.ButtonGroup.indices = property(tolerant_indices)
    yield
    element_tree_module.ButtonGroup.indices = property(original)


def _app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=120)


def test_backend_url_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_URL", DEAD_URL)
    app = _app()
    app.run()

    assert not app.exception
    assert app.text_input(key="api_url").value == DEAD_URL


def test_backend_url_can_be_overridden_in_the_sidebar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_URL", DEAD_URL)
    app = _app()
    app.run()

    app.text_input(key="api_url").set_value("http://127.0.0.1:8123")
    app.run()

    sign_in = [button for button in app.button if button.label == "Sign in"]
    assert len(sign_in) == 1

    for widget in app.text_input:
        if widget.key == "li_user":
            widget.set_value("employee1")
        elif widget.key == "li_pass":
            widget.set_value("employee123")

    sign_in[0].click()
    app.run()

    assert not app.exception
    assert not app.session_state.access_token
    # The failure text must name the URL the user actually configured, otherwise
    # there is no way to tell which backend was tried.
    assert any("127.0.0.1:8123" in message.value for message in app.error)


def test_sign_in_failure_names_the_configured_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_URL", DEAD_URL)
    app = _app()
    app.run()

    for widget in app.text_input:
        if widget.key == "li_user":
            widget.set_value("employee1")
        elif widget.key == "li_pass":
            widget.set_value("employee123")

    [button for button in app.button if button.label == "Sign in"][0].click()
    app.run()

    assert not app.exception
    assert any(DEAD_URL in message.value for message in app.error)


def test_app_boots_without_a_configured_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["streamlit", "run", "frontend/app.py"])

    app = _app()
    app.run()

    assert not app.exception
    assert app.text_input(key="api_url").value == "http://localhost:8000"
