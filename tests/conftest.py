import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plvalue import config as cfg  # noqa: E402
from synthetic import write_synthetic  # noqa: E402


@pytest.fixture()
def workspace(tmp_path):
    """Point the package at a temp folder holding synthetic data; restore afterwards."""
    saved = (cfg.DATA_DIR, cfg.MODELS_DIR, cfg.REPORTS_DIR)
    cfg.configure(data_dir=tmp_path / "data", models_dir=tmp_path / "models",
                  reports_dir=tmp_path / "reports")
    write_synthetic(cfg.DATA_DIR)
    yield tmp_path
    cfg.configure(data_dir=saved[0], models_dir=saved[1], reports_dir=saved[2])
