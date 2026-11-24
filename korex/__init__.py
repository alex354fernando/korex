"""KORE-X · Clasificación ordinal del riesgo renal KDIGO con stacking ordinal,
protocolo anti-circularidad y explicabilidad dual SHAP+LIME."""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

__version__ = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=4)
def _read_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Carga config.yaml (ruta por argumento, env KOREX_CONFIG o raíz del proyecto)."""
    path = path or os.environ.get("KOREX_CONFIG") or ROOT / "config.yaml"
    return _read_config(str(Path(path).resolve()))


def resolve_path(relative: str) -> Path:
    """Rutas de config.yaml relativas a la raíz del proyecto."""
    p = Path(relative)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    """Propaga la semilla a random y numpy (los modelos la reciben explícitamente)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
