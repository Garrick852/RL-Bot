"""Read-only cTrader active-config adapter; order routing is intentionally absent."""

from __future__ import annotations

from pathlib import Path

from .config_watcher import ActiveConfigWatcher


class CTraderBridge:
    """Expose only validated configuration until a separately reviewed executor exists."""

    def __init__(self, config_path: str | Path) -> None:
        self.watcher = ActiveConfigWatcher(config_path)

    def current_config(self) -> dict:
        return self.watcher.read_validated()

    def submit_order(self, *_args, **_kwargs):
        raise RuntimeError("cTrader order execution is disabled in this scaffold")
