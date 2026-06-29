#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


FORBIDDEN_PHRASES = [
    "Fira fails",
    "Original Fira generally does not converge",
    "Exact Fira generally converges",
    "BECR is SOTA",
    "BECR beats AdamW",
    "BECR beats Fira",
    "BECR beats LDAdam",
    "BECR validates neural performance",
    "BECR proves Adam convergence",
    "BECR solves LDAdam-style state compression",
    "BECR is memory-neutral",
    "Old CIFAR results",
    "Old MNIST results",
    "Old WikiText results",
    "All official Fira code paths are parity verified",
    "Anisotropic synthetic proves broad projected-baseline superiority",
    "CPU synthetic memory accounting proves GPU savings",
    "BECR is stronger than clipping in the fixed-stale regime",
    "High-dimensional synthetic proves full stationarity",
    "LDAdam is weaker than BECR",
]


def _collect_texts(root: Path) -> dict[Path, str]:
    texts: dict[Path, str] = {}
    for path in root.rglob("*"):
        if path.suffix in {".tex", ".md", ".bib"}:
            texts[path] = path.read_text(encoding="utf-8")
    return texts


def main() -> int:
    root = Path(__file__).resolve().parent
    texts = _collect_texts(root)
    failures: list[str] = []

    for phrase in FORBIDDEN_PHRASES:
        for path, text in texts.items():
            if phrase in text:
                failures.append(f"forbidden phrase found: {phrase!r} in {path.relative_to(root)}")

    theorem3_path = root / "sections" / "05_bounded_recovery_becr.tex"
    theorem3_text = theorem3_path.read_text(encoding="utf-8")
    theorem3_required = [
        r"h_t = b y_t + e_t",
        r"z_t = \rho h_t",
        r"e_{t+1} = h_t - z_t",
        r"y_{t+1} = y_t - \eta s_t z_t",
        r"s_t = \operatorname{clip}",
        r"0 < \rho < \frac{4}{2+\eta b s_{\min}}",
    ]
    for needle in theorem3_required:
        if needle not in theorem3_text:
            failures.append(f"Theorem 3 missing required text: {needle}")

    related_text = (root / "sections" / "08_related_work.tex").read_text(encoding="utf-8")
    if "LDAdam is the closest related method and the largest novelty risk." not in related_text:
        failures.append("LDAdam boundary paragraph missing exact opening sentence")

    limitations_text = (root / "sections" / "09_limitations_conclusion.tex").read_text(encoding="utf-8")
    limitation_needles = [
        "do not prove that\noriginal \\Fira generally fails",
        "not an Adam convergence theorem",
        "do not\ninclude neural diagnostics",
        "memory-neutrality",
        "do not offer\nan empirical LDAdam comparison",
    ]
    for needle in limitation_needles:
        if needle not in limitations_text:
            failures.append(f"Limitations missing required boundary text: {needle}")

    if failures:
        for item in failures:
            print(item)
        return 1

    print("claim check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
