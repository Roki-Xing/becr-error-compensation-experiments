#!/usr/bin/env python3
from __future__ import annotations

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
    "BECR has same memory as Fira",
    "BECR keeps same memory as Fira",
    "gradient-buffer reuse proves memory savings",
    "compressed residual has same theorem",
    "residual is free",
    "full residual is free",
]

PROOF_FILES = [
    "appendix/app_theorem1_proof.tex",
    "appendix/app_theorem2_proof.tex",
    "appendix/app_theorem3_proof.tex",
    "appendix/app_prop4_proof.tex",
    "appendix/app_theorem5_proof.tex",
]

PROOF_PLACEHOLDER_FRAGMENTS = [
    "TODO",
    "placeholder",
    "Status.",
    "proof slot",
    "fill proof",
    "insert accepted proof",
]


def _collect_texts(root: Path) -> dict[Path, str]:
    texts: dict[Path, str] = {}
    for path in root.rglob("*"):
        if path.suffix in {".tex", ".md", ".bib"}:
            texts[path] = path.read_text(encoding="utf-8")
    return texts


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _require_contains(
    failures: list[str],
    text: str,
    needles: list[str],
    label: str,
) -> None:
    haystack = _normalize(text)
    for needle in needles:
        if _normalize(needle) not in haystack:
            failures.append(f"{label} missing required text: {needle}")


def _require_absent(
    failures: list[str],
    text: str,
    needles: list[str],
    label: str,
) -> None:
    haystack = _normalize(text)
    for needle in needles:
        if _normalize(needle) in haystack:
            failures.append(f"{label} contains forbidden text: {needle}")


def main() -> int:
    root = Path(__file__).resolve().parent
    texts = _collect_texts(root)
    failures: list[str] = []

    for phrase in FORBIDDEN_PHRASES:
        for path, text in texts.items():
            if phrase.lower() in text.lower():
                failures.append(
                    f"forbidden phrase found: {phrase!r} in {path.relative_to(root)}"
                )

    theorem1_text = (root / "sections" / "04_positive_epsilon_failure.tex").read_text(
        encoding="utf-8"
    )
    _require_contains(
        failures,
        theorem1_text,
        [
            r"x_0\neq 0",
            r"y_0\neq 0",
            r"M_0",
            r"\eta b M_0",
            r"\sum_{t=0}^\infty \phi(a x_t) < \infty",
            r"y_t \to y_\infty \neq 0",
            "geometrically",
        ],
        "Theorem 1 main section",
    )
    _require_absent(
        failures,
        theorem1_text,
        ["there exists a nonempty set of initial conditions"],
        "Theorem 1 main section",
    )

    theorem23_prop4_text = (
        root / "sections" / "05_bounded_recovery_becr.tex"
    ).read_text(encoding="utf-8")
    _require_contains(
        failures,
        theorem23_prop4_text,
        [
            r"s_{\max}<\frac{2}{\eta b}",
            r"h_t = b y_t + e_t",
            r"z_t = \rho h_t",
            r"e_{t+1} = h_t - z_t",
            r"y_{t+1} = y_t - \eta s_t z_t",
            r"\rho_{\min}",
            r"\rho_{\max}",
            r"\subset (0,2)",
            r"\delta=",
            r"max\{|1-\rho_{\min}|,\;|1-\rho_{\max}|\}",
            "identity transmission",
        ],
        "Theorem 2 / Theorem 3 / Proposition 4 main section",
    )

    theorem5_text = (root / "sections" / "06_becr_sgd_abstraction.tex").read_text(
        encoding="utf-8"
    )
    _require_contains(
        failures,
        theorem5_text,
        [
            r"p_t=g_t+e_t",
            r"q_t=C_t^{\mathrm{BECR}}(p_t)",
            r"e_{t+1}=p_t-q_t",
            r"x_{t+1}=x_t-\eta q_t",
            r"\mathcal G_t",
            r"\sigma(\mathcal F_t,g_t)",
            "measurable",
            r"K_\delta=\frac{2(1-\delta)(2-\delta)}{\delta^2}",
            r"\delta\in(0,1)",
            r"\eta\le \frac{1}{2L}",
            r"\eta\le \frac{1}{2L\sqrt{K_\delta}}",
            r"\delta=1",
            "standard SGD",
            "not compression",
        ],
        "Theorem 5 main section",
    )
    _require_absent(
        failures,
        theorem5_text,
        [
            r"\eta_t=\Theta",
            r"x_{t+1}=x_t-\eta_t",
            "recovery-scaled residual theorem",
            r"\delta\in(0,1]",
        ],
        "Theorem 5 main section",
    )
    if r"\eta_t" in theorem5_text and "not covered" not in theorem5_text:
        failures.append(
            "Theorem 5 main section mentions variable step sizes without an explicit not-covered boundary"
        )

    related_text = (root / "sections" / "08_related_work.tex").read_text(
        encoding="utf-8"
    )
    if "LDAdam is the closest related method and the largest novelty risk." not in related_text:
        failures.append("LDAdam boundary paragraph missing exact opening sentence")

    limitations_text = (
        root / "sections" / "09_limitations_conclusion.tex"
    ).read_text(encoding="utf-8")
    _require_contains(
        failures,
        limitations_text,
        [
            "do not prove that original \\Fira generally fails",
            "not an Adam convergence theorem",
            "do not include neural diagnostics",
            "memory-neutrality",
            "do not offer an empirical LDAdam comparison",
        ],
        "Limitations section",
    )

    memory_appendix_text = (
        root / "appendix" / "app_ablations_memory.tex"
    ).read_text(encoding="utf-8")
    memory_table_text = (
        root / "tables" / "memory_runtime_accounting.tex"
    ).read_text(encoding="utf-8")
    claim_matrix_text = (
        root / "tables" / "claim_evidence_matrix.tex"
    ).read_text(encoding="utf-8")
    memory_text = "\n".join(
        [limitations_text, memory_appendix_text, memory_table_text, claim_matrix_text]
    )
    _require_contains(
        failures,
        memory_text,
        [
            "full raw-gradient residual",
            "error-feedback buffer",
            "gradient-buffer reuse",
            "peak-memory measurements",
            "separate theory",
            "approximate EF",
        ],
        "Memory limitations / appendix note",
    )
    if r"O(d)" not in memory_text and r"\Omega(d)" not in memory_text:
        failures.append(
            "Memory limitations / appendix note missing O(d) or \\Omega(d) lower-bound wording"
        )
    _require_contains(
        failures,
        memory_appendix_text,
        [
            "Memory lower bound for exact residual conservation",
            r"q_t + e_{t+1} = g_t + e_t",
            r"p_{t+1}=g_{t+1}+e_{t+1}",
            "Proof sketch",
            "no-free-lunch observation",
        ],
        "Appendix memory lower-bound note",
    )

    for relative_path in PROOF_FILES:
        path = root / relative_path
        text = path.read_text(encoding="utf-8")
        for fragment in PROOF_PLACEHOLDER_FRAGMENTS:
            if fragment in text:
                failures.append(
                    f"proof placeholder fragment found in {relative_path}: {fragment}"
                )

    proof_checks = {
        "appendix/app_theorem1_proof.tex": [
            r"z_t=|x_t|",
            r"h(z)",
            r"q=",
            "induction",
            r"z_t\le z_0",
            r"z_{t+1}=z_t h(z_t)\le q z_t",
            r"q^t z_0",
            r"\sum_{t=0}^\infty \phi(a x_t)<\infty",
            r"y_0\prod",
            r"y_\infty\neq 0",
        ],
        "appendix/app_theorem2_proof.tex": [
            r"q_y",
            r"s_{\max}<\frac{2}{\eta b}",
            r"|y_{t+1}|\le q_y |y_t|",
        ],
        "appendix/app_theorem3_proof.tex": [
            r"s_t=s_{\min}",
            r"\rho<\frac{4}{2+\eta b s_{\min}}",
            "Jury conditions",
        ],
        "appendix/app_prop4_proof.tex": [
            r"\mathcal C_t^{\mathrm{BECR}}(v)",
            r"(1-\delta)\norm{v}^2",
            "identity transmission",
        ],
        "appendix/app_theorem5_proof.tex": [
            r"\mathcal G_t=\sigma(\mathcal F_t,g_t)",
            r"\delta\in(0,1)",
            r"p_t=g_t+e_t",
            "measurable",
            r"\beta=\frac{\delta}{2(1-\delta)}",
            r"\tilde x_t=x_t-\eta e_t",
            r"K_\delta=\frac{2(1-\delta)(2-\delta)}{\delta^2}",
            r"\delta=1",
            "standard SGD",
            r"e_t=0",
            r"\tilde x_{t+1} = \tilde x_t-\eta_t g_t+(\eta_t-\eta_{t+1})e_{t+1}",
        ],
    }
    for relative_path, needles in proof_checks.items():
        text = (root / relative_path).read_text(encoding="utf-8")
        _require_contains(failures, text, needles, f"Proof file {relative_path}")

    if failures:
        for item in failures:
            print(item)
        return 1

    print("claim check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
