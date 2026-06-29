# Paper Draft V3 Integration Source

This file is the local integration source used for `P1-LATEX-INTEGRATION`.
The repository did not contain a standalone `Paper Draft V3` manuscript file, so
the LaTeX package was reconstructed from:

1. the accepted task brief for `P1-LATEX-INTEGRATION`;
2. the merged corrected synthetic review snapshot under
   `experiments/tier1-synthetic/corrected_synthetic_artifacts/review_snapshot/`;
3. the theorem/mechanism notes in `docs/source-notes/Error compensation.md`;
4. the accepted project scope decisions from PRs #1--#4.

Integrated constraints:

- Title:
  `When Norm Recovery Is Not Error Feedback: Diagnosing and Repairing Fira-style Recovery`
- Main-paper structure:
  Abstract; Introduction; Background and taxonomy; Norm recovery is not Error
  Feedback; Positive-epsilon stale-subspace failure; Bounded recovery and BECR;
  BECR-SGD abstraction; Corrected synthetic diagnostics; Related work and
  novelty boundary; Limitations; Conclusion.
- Exact numbering:
  Theorem 1, Theorem 2, Theorem 3, Proposition 4, Theorem 5.
- Theorem 3 must use separated `rho/s` dynamics and raw-gradient residual
  units only.
- Corrected synthetic evidence must come only from the accepted PR #4 review
  snapshot, never from legacy `ALL_RUNS.json` or old neural results.
- Neural diagnostics remain blocked and are not included.

The integrated LaTeX text deliberately uses the weakest claims supported by the
merged artifacts and accepted scope notes. It does not introduce optimizer
superiority, Adam convergence, memory-neutrality, or broad official-Fira
behavior claims.
