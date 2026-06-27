from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from .json_utils import sanitize_for_json


def canonical_config_json(config: dict[str, Any]) -> str:
    return json.dumps(
        sanitize_for_json(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def _slug(text: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return lowered or "run"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_run_id(
    *,
    task: str,
    experiment: str,
    method: str,
    seed: int,
    short_commit: str,
    config_hash_value: str,
    timestamp: str | None = None,
) -> str:
    stamp = timestamp or utc_timestamp()
    return (
        f"{_slug(task)}_{_slug(experiment)}_{_slug(method)}_seed{int(seed):03d}_"
        f"{stamp}_{short_commit[:7]}_{config_hash_value[:8]}"
    )
