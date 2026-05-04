"""
wandb_compat.py
===============
Provides a safe wandb interface even when the real package is unavailable
or shadowed by a local directory named "wandb".
"""

from typing import Any, Dict, List


class _NoOpTable:
    """Minimal wandb.Table stand-in used when wandb is unavailable."""
    def __init__(self, columns: List[str] | None = None):
        self.columns = columns or []
        self.rows: List[List[Any]] = []

    def add_data(self, *args):
        self.rows.append(list(args))


class _NoOpPlot:
    """Subset of wandb.plot API used by this repository."""
    @staticmethod
    def confusion_matrix(**kwargs):
        return {"type": "confusion_matrix", "kwargs": kwargs}


class _NoOpWandb:
    """No-op wandb replacement preserving call compatibility."""
    def __init__(self):
        self.summary: Dict[str, Any] = {}
        self.plot = _NoOpPlot()
        self.Table = _NoOpTable

    @staticmethod
    def init(*args, **kwargs):
        return None

    @staticmethod
    def log(*args, **kwargs):
        return None

    @staticmethod
    def define_metric(*args, **kwargs):
        return None

    @staticmethod
    def finish(*args, **kwargs):
        return None

    @staticmethod
    def Image(path: str):
        return path


def get_wandb():
    """Return real wandb if importable, otherwise return no-op fallback."""
    try:
        import wandb as real_wandb  # type: ignore

        if hasattr(real_wandb, "init") and callable(real_wandb.init):
            return real_wandb
    except Exception:
        pass
    return _NoOpWandb()


wandb = get_wandb()
