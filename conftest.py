# conftest.py — loaded before any test collection.
# Import torch here so its Windows DLL initialization happens before pytest's
# assertion-rewriting import mechanism can interfere.
try:
    import torch  # noqa: F401
except Exception:
    pass
