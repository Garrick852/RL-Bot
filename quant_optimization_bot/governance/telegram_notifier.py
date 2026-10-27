"""Explicit Telegram Bot API notifications; no background sending is performed."""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen


def send_message(text: str, *, token: str | None = None, chat_id: str | None = None) -> dict:
    """Send one Markdown message; credentials are read only when called."""
    bot_token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    destination = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not destination:
        raise RuntimeError("Telegram requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    request = Request(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data=json.dumps(
            {"chat_id": destination, "text": text, "parse_mode": "Markdown"}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        payload = json.load(response)
    if not payload.get("ok"):
        raise RuntimeError(f"Telegram rejected notification: {payload!r}")
    return payload
