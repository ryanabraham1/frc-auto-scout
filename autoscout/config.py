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


def yt_cookies_browser() -> str | None:
    """Optional yt-dlp browser/profile spec, e.g. firefox:C:/path/to/profile.

    A configured cookies file takes precedence. Browser credentials remain in
    their existing profile; we do not export them to a project file.
    """
    return os.environ.get("YT_COOKIES_BROWSER", "").strip() or None


def device() -> str:
    """'cuda' (NVIDIA, e.g. RTX 3060), 'mps' (Apple silicon GPU, e.g. M3), else 'cpu'.

    Override with AUTOSCOUT_DEVICE=cpu|cuda|mps.
    """
    forced = os.environ.get("AUTOSCOUT_DEVICE")
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def has_gpu() -> bool:
    return device() in ("cuda", "mps")


def detector_imgsz() -> int:
    """1600 improved robot precision/recall on this PC's broadcast benchmark."""
    size = int(os.environ.get("AUTOSCOUT_IMGSZ", "1600" if device() == "cuda" else "1280"))
    if size < 320 or size % 32:
        raise ValueError("AUTOSCOUT_IMGSZ must be a multiple of 32, at least 320")
    return size
