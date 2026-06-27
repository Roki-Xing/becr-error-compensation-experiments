from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not math.isfinite(value):
            return None
        return value
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(value) for value in obj]
    return obj


def dumps_strict(obj: Any, *, indent: int | None = 2, sort_keys: bool = False) -> str:
    return json.dumps(
        sanitize_for_json(obj),
        indent=indent,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def write_json_strict(path: Path, obj: Any, *, indent: int | None = 2, sort_keys: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_strict(obj, indent=indent, sort_keys=sort_keys), encoding="utf-8")


def write_jsonl_strict(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [dumps_strict(row, indent=None, sort_keys=True) for row in rows]
    path.write_text("\n".join(serialized) + "\n", encoding="utf-8")
