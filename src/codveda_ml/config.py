"""Project configuration, path resolution, checksum verification, and runtime metadata."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import yaml


def get_project_root() -> Path:
    """Resolve the repository root directory deterministically."""
    return Path(__file__).resolve().parents[2]


def load_config(root: Path | None = None) -> dict[str, Any]:
    """Load `config/project_config.yaml`."""
    project_root = root or get_project_root()
    cfg_path = project_root / "config" / "project_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_directories(root: Path | None = None) -> dict[str, Path]:
    """Ensure all standard output directories exist and return a mapping of paths."""
    project_root = root or get_project_root()
    cfg = load_config(project_root)
    resolved: dict[str, Path] = {"root": project_root}
    for key, rel in cfg["paths"].items():
        path = project_root / rel
        path.mkdir(parents=True, exist_ok=True)
        resolved[key] = path
    return resolved


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_runtime_metadata() -> dict[str, Any]:
    """Record exact Python and scientific library versions for reproducibility."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
            "joblib": joblib.__version__,
        },
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON deterministically with clean formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
