Repository: https://github.com/xichen-fy/Fira
Pinned commit: 5af6a9860b633138f12837ad25e528b7d54217eb
Pinned files:
- optimizer_torch/__init__.py
- optimizer_torch/fira_adamw.py
- optimizer_torch/gradient_projection.py
- LICENSE

License: Apache License 2.0
Fetch date: 2026-06-26

Notes:
- The optimizer core files above are copied verbatim from the pinned upstream commit.
- This snapshot is read-only for parity testing.
- Local compatibility shims under `transformers/utils/versions.py` are not part of the
  upstream optimizer logic; they only satisfy the import used by the upstream file.
