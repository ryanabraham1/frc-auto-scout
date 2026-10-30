"""Render a demo video of a processed match: tracked robots labelled with team numbers, airborne
game pieces, and a live scoreboard panel of what each robot has been credited with so far.

Usage: python scripts/render_demo.py MATCH_KEY VIDEO OUT.mp4 [--width 1280]

The live panel counts the *detected* events as they happen (each robot's coral by level, algae,
shots, feeds...). The final column ("official split") shows the end-of-match numbers after the
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
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--fps", type=float, default=30)
a = ap.parse_args()

r = json.loads(Path(f"data/results/{a.match_key}.json").read_text())
year = r["year"]
clock = r["clock"]
t0 = clock["auto_end_vt"] - clock["auto_len"] - 1.0
t1 = clock["teleop_end_vt"] + 4.0
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
    COLS = ["shot", "made", "feed"]
    HEAD = ["Shots", "Seen in", "Fed"]
    FINAL = lambda m: f"{m['fuel_scored']} scored {m['fuel_fed']} fed"  # noqa: E731
    KEEP = {"shot", "feed"}
events = sorted([e for e in r.get("events", []) if e.get("team") and e.get("kind") in KEEP], key=lambda e: e["t"])

# launches with their points for drawing flights
flights = []
for name, ls in r.get("launches", {}).items():
    for L in ls:
        if L.get("pts"):
            flights.append((L["t0"], L["pts"][-1][0], np.array([(p[1], p[2]) for p in L["pts"]]), L.get("target")))

teams = {"red": r["red"], "blue": r["blue"]}
video = Video(a.video)
W = a.width
scale = W / view[2]
vh = int(view[3] * scale)
panel_h = 230
H = vh + panel_h
H += H % 2
tmp = Path(a.out).with_suffix(".tmp.mp4")
writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
step = max(1, int(round(video.fps / a.fps)))

counts = defaultdict(lambda: defaultdict(float))
ei = 0
flash = {}  # team -> (until_t, text)
FONT = cv2.FONT_HERSHEY_SIMPLEX
RED, BLUE, WHITE, GREY = (60, 60, 230), (230, 140, 40), (240, 240, 240), (150, 150, 150)

track_arr = {k: np.array([row[0] for row in v]) for k, v in tracks.items()}


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


for t, frame in video.iter_frames(t0, t1, step=step):
    x, y, w, h = view
    img = cv2.resize(frame[y: y + h, x: x + w], (W, vh))
    # flights in progress (draw the trail up to now)
    for ft0, ft1, P, tgt in flights:
        if ft0 <= t <= ft1 + 0.3:
            n = max(2, int(len(P) * min(1.0, (t - ft0) / max(ft1 - ft0, 1e-3))))
            pts = ((P[:n] - [x, y]) * scale).astype(np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 255) if not tgt else (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[-1]), 4, (0, 255, 255), -1)
    # robots
    for slot in tracks:
        b, obs = box_at(slot, t)
        if b is None:
            continue
        team = team_at(slot, t)
        col = RED if slot.startswith("red") else BLUE
        p0 = (int((b[0] - x) * scale), int((b[1] - y) * scale))
        p1 = (int((b[2] - x) * scale), int((b[3] - y) * scale))
        cv2.rectangle(img, p0, p1, col, 2)
        label = str(team)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
        cv2.rectangle(img, (p0[0], p0[1] - th - 8), (p0[0] + tw + 8, p0[1]), col, -1)
        cv2.putText(img, label, (p0[0] + 4, p0[1] - 5), FONT, 0.55, WHITE, 2, cv2.LINE_AA)
        if team in flash and flash[team][0] > t:
            cv2.putText(img, flash[team][1], (p0[0], p1[1] + 22), FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    # events up to now
    while ei < len(events) and events[ei]["t"] <= t:
        e = events[ei]
        team = int(e["team"])
        k = e["kind"]
        n = e.get("n", 1) or 1
        counts[team][k] += n if year == 2026 else 1
        if year == 2026 and e.get("made"):
            counts[team]["made"] += n
        flash[team] = (t + 1.2, ("+" + k.upper()) if year == 2025 else ("+shot" if k == "shot" else "+feed"))
        ei += 1
    # panel
    panel = np.full((H - vh, W, 3), 25, np.uint8)
    phase = "AUTO" if t < clock["auto_end_vt"] else "TELEOP"
    rem = clock["auto_end_vt"] - t if phase == "AUTO" else clock["teleop_end_vt"] - t
    i = int(np.searchsorted(np.array(r["timeline"]["t"]), t)) - 1
    rs = r["timeline"]["red"][max(i, 0)]
    bs = r["timeline"]["blue"][max(i, 0)]
    cv2.putText(panel, f"{a.match_key}  {phase} {max(rem, 0):5.1f}s", (12, 26), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
    cv2.putText(panel, f"RED {int(rs)}", (W - 260, 26), FONT, 0.7, RED, 2, cv2.LINE_AA)
    cv2.putText(panel, f"BLUE {int(bs)}", (W - 140, 26), FONT, 0.7, BLUE, 2, cv2.LINE_AA)
    colw = (W - 24) // 2
    for ai, al in enumerate(("red", "blue")):
        ox = 12 + ai * colw
        col = RED if al == "red" else BLUE
        cv2.putText(panel, "Team", (ox, 56), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        for ci, hname in enumerate(HEAD):
            cv2.putText(panel, hname, (ox + 70 + ci * 52, 56), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        cv2.putText(panel, "final (official split)", (ox + 70 + len(HEAD) * 52, 56), FONT, 0.45, GREY, 1, cv2.LINE_AA)
        for ri, team in enumerate(teams[al]):
            yy = 86 + ri * 40
            cv2.putText(panel, str(team), (ox, yy), FONT, 0.65, col, 2, cv2.LINE_AA)
            for ci, c in enumerate(COLS):
                v = counts[team][c]
                cv2.putText(panel, f"{v:.0f}", (ox + 78 + ci * 52, yy), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
            if t > clock["teleop_end_vt"]:
                m = r["teams"][str(team)]
                cv2.putText(panel, FINAL(m), (ox + 70 + len(HEAD) * 52, yy), FONT, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    writer.write(np.vstack([img, panel]))
writer.release()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "26", "-pix_fmt", "yuv420p", a.out], check=True)
tmp.unlink()
print("wrote", a.out)
