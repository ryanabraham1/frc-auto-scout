"""Paths and settings. Secrets come from env vars or `.secrets/.env` (git-ignored)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AUTOSCOUT_DATA", ROOT / "data"))
CONFIG_DIR = ROOT / "configs"
FIELDS_DIR = CONFIG_DIR / "fields"
EVENTS_DIR = CONFIG_DIR / "events"


def _load_env_file() -> None:
    for path in (ROOT / ".secrets" / ".env", ROOT / ".env"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env_file()


def tba_key() -> str:
    key = os.environ.get("TBA_API_KEY", "")
    if not key:
        raise RuntimeError("TBA_API_KEY is not set (env var or .secrets/.env)")
    return key


def yt_cookies() -> Path | None:
    """Path to a Netscape cookies.txt for yt-dlp, from YT_COOKIES or base64 YT_COOKIES_B64."""
    p = os.environ.get("YT_COOKIES")
    if p:
        path = Path(p)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path
    b64 = os.environ.get("YT_COOKIES_B64")
    if b64:
        import base64

        path = DATA_DIR / "cookies.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(b64))
        return path
    return None


def device() -> str:
    """'cuda' when a GPU is available (e.g. RTX 3060), else 'cpu'."""
    forced = os.environ.get("AUTOSCOUT_DEVICE")
    if forced:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
