"""Render a demo video of a processed match: tracked robots labelled with team numbers, airborne
game pieces, and a live scoreboard panel of what each robot has been credited with so far.

Usage: python scripts/render_demo.py MATCH_KEY VIDEO OUT.mp4 [--width 1400]

The live panel counts events as they happen: in 2026 "Scored" is the live broadcast score credited to
the robot shooting just before each increase; the other columns count *detected* events (each
robot's coral by level, algae, shots, feeds...). The final column ("official split") shows the end-of-match numbers after the
official FMS totals were split among robots, i.e. what the dashboard reports.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from autoscout.video import Video  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("match_key")
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--width", type=int, default=1400, help="width of the broadcast frame (sidebar is extra)")
ap.add_argument("--fps", type=float, default=30)
ap.add_argument("--start", type=float, default=None, help="seconds into the match to start (preview)")
ap.add_argument("--duration", type=float, default=None, help="seconds to render (preview)")
a = ap.parse_args()

r = json.loads(Path(f"data/results/{a.match_key}.json").read_text())
year = r["year"]
clock = r["clock"]
t0 = clock["auto_end_vt"] - clock["auto_len"] - 1.0
t1 = clock["teleop_end_vt"] + 4.0
if a.start is not None:
    t0 += a.start
if a.duration is not None:
    t1 = min(t1, t0 + a.duration)
view = r["calibration"]["view"]
tracks = r["raw_tracks"]
tl = r.get("identity_timeline") or {}
ident = {k: int(v) for k, v in r["identity"].items()}


def team_at(slot, t):
    for s, e, team in tl.get(slot, []):
        if s <= t < e:
            return team
    return ident.get(slot)


# events -> per team live counters
if year == 2025:
    COLS = ["L4", "L3", "L2", "L1", "net", "processor"]
    HEAD = ["L4", "L3", "L2", "L1", "Net", "Proc"]
    FINAL = lambda m: f"{m['coral_total']}c {m['algae_net']}n {m['algae_processor']}p"  # noqa: E731
    KEEP = set(COLS)
else:
    COLS = ["scored", "shot", "feed"]
    HEAD = ["Scored", "Shots", "Fed"]
    FINAL = lambda m: f"{m['fuel_scored']} scored {m['fuel_fed']} fed"  # noqa: E731
    KEEP = {"scored", "shot", "feed"}
events = sorted([e for e in r.get("events", []) if e.get("team") and e.get("kind") in KEEP], key=lambda e: e["t"])

# launches with their points for drawing flights
flights = []
for name, ls in r.get("launches", {}).items():
    for L in ls:
        if L.get("pts"):
            flights.append((L["t0"], L["pts"][-1][0], np.array([(p[1], p[2]) for p in L["pts"]]), L.get("target")))

teams = {"red": r["red"], "blue": r["blue"]}
video = Video(a.video)
# Layout: the whole broadcast frame (its scoreboard and close-up camera views stay visible) with
# robots and flights drawn in place, and a sidebar with the clock, the live score and every robot's
# running credit.
W = a.width
scale = W / video.width
vh = int(round(video.height * scale))
vh += vh % 2
SIDE = 520
H = max(vh, 760)
tmp = Path(a.out).with_suffix(".tmp.mp4")
writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W + SIDE, H))
step = max(1, int(round(video.fps / a.fps)))

counts = defaultdict(lambda: defaultdict(float))
ei = 0
flash = {}  # team -> (until_t, text)
feed = []  # recent events for the sidebar: (t, team, text)
FONT = cv2.FONT_HERSHEY_SIMPLEX
RED, BLUE, WHITE, GREY, GREEN = (70, 70, 235), (235, 150, 50), (240, 240, 240), (150, 150, 150), (90, 220, 90)
BG, CARD = (24, 22, 20), (38, 35, 32)
LIGHT = {"red": (40, 40, 120), "blue": (110, 70, 25)}
track_arr = {k: np.array([row[0] for row in v]) for k, v in tracks.items()}
tl_t = np.array(r["timeline"]["t"])


def box_at(slot, t):
    ts = track_arr.get(slot)
    if ts is None or not len(ts):
        return None, False
    i = int(np.searchsorted(ts, t))
    best = None
    for j in range(max(0, i - 3), min(len(ts), i + 2)):
        row = tracks[slot][j]
        if row[3] and row[4] and abs(row[0] - t) < 0.6:
            if best is None or abs(row[0] - t) < abs(best[0] - t):
                best = row
    return (best[4], True) if best else (None, False)


def clk(t):
    """Match clock shown on the broadcast at video time t, e.g. '1:07' (A for auto)."""
    if t < clock["auto_end_vt"]:
        rem = max(clock["auto_end_vt"] - t, 0)
        return f"A {int(rem)}"
    rem = max(clock["teleop_end_vt"] - t, 0)
    return f"{int(rem) // 60}:{int(rem) % 60:02d}"


def text(img, s, org, size=0.55, col=WHITE, thick=1, right=False):
    if right:
        (tw, _), _ = cv2.getTextSize(s, FONT, size, thick)
        org = (org[0] - tw, org[1])
    cv2.putText(img, s, org, FONT, size, col, thick, cv2.LINE_AA)


def sidebar(t):
    sb = np.full((H, SIDE, 3), BG, np.uint8)
    phase = "AUTO" if t < clock["auto_end_vt"] else "TELEOP" if t < clock["teleop_end_vt"] else "FINAL"
    i = max(int(np.searchsorted(tl_t, t)) - 1, 0)
    rs, bs = int(r["timeline"]["red"][i]), int(r["timeline"]["blue"][i])
    text(sb, r.get("label") or a.match_key, (20, 36), 0.75, WHITE, 2)
    text(sb, f"{phase}  {clk(t)}" if phase != "FINAL" else phase, (SIDE - 20, 36), 0.65, GREY, 1, right=True)
    # live score (read from the broadcast) and the official final from FMS
    for k, (al, sc) in enumerate((("red", rs), ("blue", bs))):
        x0 = 20 + k * (SIDE - 40) // 2
        cv2.rectangle(sb, (x0, 52), (x0 + (SIDE - 50) // 2, 112), LIGHT[al], -1)
        text(sb, al.upper(), (x0 + 12, 74), 0.45, WHITE, 1)
        text(sb, str(sc), (x0 + 12, 104), 0.95, WHITE, 2)
        off = r.get(f"{al}_score")
        if off is not None:
            text(sb, f"official {off}", (x0 + (SIDE - 50) // 2 - 10, 104), 0.45, WHITE, 1, right=True)
    # per-robot credit table
    y = 140
    colx = [130 + k * 56 for k in range(len(HEAD))]
    for al in ("red", "blue"):
        cv2.rectangle(sb, (14, y), (SIDE - 14, y + 30 + 3 * 34), CARD, -1)
        cv2.rectangle(sb, (14, y), (18, y + 30 + 3 * 34), RED if al == "red" else BLUE, -1)
        text(sb, "Team", (28, y + 20), 0.45, GREY)
        for cx, hname in zip(colx, HEAD):
            text(sb, hname, (cx + 30, y + 20), 0.45, GREY, right=True)
        text(sb, "final", (SIDE - 24, y + 20), 0.45, GREY, right=True)
        for ri, team in enumerate(teams[al]):
            yy = y + 52 + ri * 34
            hot = team in flash and flash[team][0] > t
            text(sb, str(team), (28, yy), 0.62, GREEN if hot else WHITE, 2)
            for cx, c in zip(colx, COLS):
                text(sb, f"{counts[team][c]:.0f}", (cx + 30, yy), 0.6, WHITE, 1, right=True)
            if phase == "FINAL":
                text(sb, FINAL(r["teams"][str(team)]), (SIDE - 24, yy), 0.45, GREEN, 1, right=True)
        y += 30 + 3 * 34 + 14
    # recent events
    text(sb, "Recent", (20, y + 18), 0.5, GREY)
    for k, (et, team, msg) in enumerate(reversed(feed[-8:])):
        yy = y + 44 + k * 26
        if yy > H - 10:
            break
        al = "red" if team in teams["red"] else "blue"
        text(sb, clk(et), (20, yy), 0.45, GREY)
        text(sb, str(team), (80, yy), 0.5, RED if al == "red" else BLUE, 2)
        text(sb, msg, (150, yy), 0.5, WHITE)
    return sb


for t, frame in video.iter_frames(t0, t1, step=step):
    img = cv2.resize(frame, (W, vh), interpolation=cv2.INTER_AREA)
    # flights in progress (draw the trail up to now)
    for ft0, ft1, P, tgt in flights:
        if ft0 <= t <= ft1 + 0.3:
            n = max(2, int(len(P) * min(1.0, (t - ft0) / max(ft1 - ft0, 1e-3))))
            pts = (P[:n] * scale).astype(np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 255) if not tgt else (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[-1]), 3, (0, 255, 255), -1)
    # robots
    for slot in tracks:
        b, obs = box_at(slot, t)
        if b is None:
            continue
        team = team_at(slot, t)
        col = RED if slot.startswith("red") else BLUE
        p0 = (int(b[0] * scale), int(b[1] * scale))
        p1 = (int(b[2] * scale), int(b[3] * scale))
        cv2.rectangle(img, p0, p1, col, 2, cv2.LINE_AA)
        label = str(team)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(img, (p0[0], p0[1] - th - 8), (p0[0] + tw + 8, p0[1]), col, -1)
        cv2.putText(img, label, (p0[0] + 4, p0[1] - 5), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        if team in flash and flash[team][0] > t:
            cv2.putText(img, flash[team][1], (p0[0], p1[1] + 18), FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    # events up to now
    while ei < len(events) and events[ei]["t"] <= t:
        e = events[ei]
        team = int(e["team"])
        k = e["kind"]
        n = e.get("n", 1) or 1
        counts[team][k] += n if year == 2026 else 1
        if k == "scored":  # the live score arrives ball by ball: one feed line per increase is enough
            if not feed or feed[-1][1] != team or feed[-1][2] != "scored" or e["t"] - feed[-1][0] > 2:
                feed.append((e["t"], team, "scored"))
        else:
            msg = ("+" + k.upper()) if year == 2025 else ("+shot" if k == "shot" else "+feed")
            flash[team] = (t + 1.2, msg)
            feed.append((e["t"], team, msg[1:] if year == 2026 else k))
        ei += 1
    out = np.full((H, W + SIDE, 3), BG, np.uint8)
    out[: vh, : W] = img
    out[:, W:] = sidebar(t)
    writer.write(out)
writer.release()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
tmp.unlink()
print("wrote", a.out)
