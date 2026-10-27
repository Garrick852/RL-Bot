"""Read-only IBKR active-config adapter; order routing is intentionally absent."""

from __future__ import annotations

from pathlib import Path

from .config_watcher import ActiveConfigWatcher


class IBKRBridge:
    """Expose only validated configuration until a separately reviewed executor exists."""

    def __init__(self, config_path: str | Path) -> None:
        self.watcher = ActiveConfigWatcher(config_path)

    def current_config(self) -> dict:
        return self.watcher.read_validated()

    def submit_order(self, *_args, **_kwargs):
        raise RuntimeError("IBKR order execution is disabled in this scaffold")
