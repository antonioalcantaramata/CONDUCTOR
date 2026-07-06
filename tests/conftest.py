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
