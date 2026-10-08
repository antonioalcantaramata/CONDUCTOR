import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")

for _path in (REPO_ROOT, BACKEND_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandapower.networks as pn  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def case14_net():
    """A small, real pandapower network — fast to build, no solve required."""
    return pn.case14()


@pytest.fixture(scope="module")
def backend_client(tmp_path_factory):
    """The real backend, in-process, on the bundled IEEE 14-bus profile.

    About 1.5 s to start and no IPOPT needed for assessment endpoints. Saved
    uploads on this machine are ignored, and the module-level `app_data` is
    restored afterwards so later tests do not see this network.
    """
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient

    scratch = tmp_path_factory.mktemp("uploads")
    patch = pytest.MonkeyPatch()
    patch.setenv("GRID_PROFILE", "pglib_case14")
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL"):
        patch.setattr(mb, name, str(scratch / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as client:
        yield client
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()
