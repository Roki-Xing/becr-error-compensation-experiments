#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


CITE_PATTERN = re.compile(
    r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}",
    re.MULTILINE,
)
BIB_PATTERN = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)


def _collect_tex_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.tex")
        if "build" not in path.parts
    )


def _extract_citation_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for raw_group in CITE_PATTERN.findall(text):
            for key in raw_group.split(","):
                normalized = key.strip()
                if normalized:
                    keys.add(normalized)
    return keys


def _extract_bib_keys(bib_path: Path) -> list[str]:
    text = bib_path.read_text(encoding="utf-8")
    return BIB_PATTERN.findall(text)


def main() -> int:
    root = Path(__file__).resolve().parent
    bib_path = root / "references.bib"
    tex_files = _collect_tex_files(root)
    cited_keys = _extract_citation_keys(tex_files)
    bib_keys = _extract_bib_keys(bib_path)

    failures: list[str] = []
    warnings: list[str] = []

    duplicates = sorted({key for key in bib_keys if bib_keys.count(key) > 1})
    if duplicates:
        failures.append(f"duplicate bibliography keys: {', '.join(duplicates)}")

    non_lowercase = sorted(key for key in bib_keys if key != key.lower())
    if non_lowercase:
        failures.append(
            "bibliography keys must be lowercase: " + ", ".join(non_lowercase)
        )

    bib_key_set = set(bib_keys)
    missing = sorted(cited_keys - bib_key_set)
    if missing:
        failures.append("missing bibliography keys: " + ", ".join(missing))

    unused = sorted(bib_key_set - cited_keys)
    if unused:
        warnings.append("unused bibliography keys: " + ", ".join(unused))

    print("citation keys:", ", ".join(sorted(cited_keys)) or "(none)")
    print("bibliography keys:", ", ".join(sorted(bib_key_set)) or "(none)")
    for warning in warnings:
        print("warning:", warning)
    if failures:
        for failure in failures:
            print("error:", failure)
        return 1

    print("bibliography check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
