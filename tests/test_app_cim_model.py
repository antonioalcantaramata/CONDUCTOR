"""
The System tab with a network loaded as CIM, driven headlessly (Streamlit's AppTest) against the real backend
started in-process with its storage in a temporary folder: the map and the CIM layer pane render, the pane shows the
model (headline, picture, tabs, the model's columns) and the solved-state check. Skipped where Streamlit is not
installed (CI).
"""
import io
import pathlib
import socket
import sys
import threading
import time

import pytest

pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("uvicorn")
pytest.importorskip("lxml")

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = str(ROOT / "llm_agent" / "app.py")
FIXTURE = ROOT / "tests" / "fixtures" / "cgmes" / "pandapower_example_multivoltage.zip"


@pytest.fixture(scope="module")
def backend_url(tmp_path_factory):
    import uvicorn

    import main_backend as mb
    store = tmp_path_factory.mktemp("app_cim")
    patch = pytest.MonkeyPatch()
    patch.setenv("GRID_PROFILE", "pglib_case30")
    patch.setattr(mb, "SYSTEMS_DIR", str(store / "systems"))
    patch.setattr(mb, "DATA_FILES_DIR", str(store / "data_files"))
    (store / "systems").mkdir()
    (store / "data_files").mkdir()
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(store / f"{name}.json"))
    saved = dict(mb.app_data)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(mb.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import urllib.request
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    url = f"http://127.0.0.1:{port}"
    import httpx
    r = httpx.post(f"{url}/api/network/upload", timeout=300,
                   files={"file": ("example.zip", io.BytesIO(FIXTURE.read_bytes()), "application/zip")})
    assert r.status_code == 200, r.text
    yield url
    server.should_exit = True
    thread.join(timeout=10)
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


@pytest.fixture
def app(backend_url, monkeypatch):
    monkeypatch.setenv("DT_BACKEND_URL", backend_url)
    sys.path.insert(0, str(ROOT / "llm_agent"))
    # The address is read once, at import: modules an earlier test imported still hold its backend's.
    import agent.config as config
    import agent.tools as tools
    for module in (config, tools):
        monkeypatch.setattr(module, "BASE_URL", backend_url, raising=False)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["_launched"] = True
    at.session_state["history"] = []
    at.session_state["auto_suggest"] = False
    at.session_state["messages"] = []
    yield at
    if str(ROOT / "llm_agent") in sys.path:
        sys.path.remove(str(ROOT / "llm_agent"))


def test_the_map_and_the_cim_layer_pane_show_the_model(app):
    app.run()
    assert not app.exception
    app.session_state["system_pane"] = "semantics"
    app.run()
    assert not app.exception
    assert "CIM model" in " ".join(m.value for m in app.markdown)
    picture = [s for s in app.get("segmented_control") if s.label == "Picture"][0]
    assert "The CIM model" in picture.options
    labels = [t.label for t in app.tabs]
    assert any(t.startswith("CIM model · ") for t in labels) and any(t.startswith("Other equipment · ") for t in labels)
    columns = [list(d.value.columns) for d in app.dataframe]
    assert any("Voltage level" in c for c in columns) and any("In the layer" in c for c in columns)
    # pandapower's example does not reproduce its own solution (its equivalents): said, not hidden.
    assert any("own solution" in w.value for w in app.warning)
    picture.set_value("model").run()
    assert not app.exception
