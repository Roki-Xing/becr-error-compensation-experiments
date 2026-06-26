"""Minimal compatibility shim for the vendored official Fira optimizer."""


def require_version(*_args, **_kwargs):
    """Mirror the upstream import contract without adding a transformers dependency."""

