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
ap.add_argument("--results-dir", type=Path, default=Path("data/results"), help="saved analysis snapshot to render")
ap.add_argument("--flight-labels", action="store_true", help="label each ball and its attributed shooter")
ap.add_argument("--observations-only", action="store_true", help="replay saved observations on a diagram, without match footage (use '-' for VIDEO)")
a = ap.parse_args()

r = json.loads((a.results_dir / f"{a.match_key}.json").read_text())
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
            flights.append((len(flights) + 1, L, np.asarray(L["pts"], float)))

# 2025 debug layers: coral seen placed on the reefs, moving algae, reef close-up reads, decoded jumps
placements = r.get("element_changes", [])
place_by_t = {}
for e in r.get("events", []):
    if e.get("placed"):
        place_by_t[(e["placed"]["elem"], e["placed"]["t"])] = e
blobs = None
bp = a.results_dir / f"{a.match_key}_blobs.npz"
if bp.exists():
    z = np.load(bp)
    off = z["view"][:2] if "view" in z.files else (0, 0)
    blobs = {k: (z[k][:, 0], z[k][:, 1:3] + off) for k in z.files if k != "view"}
aux = r.get("aux_reads", [])
aux_boxes = sorted(r.get("aux_boxes", []), key=lambda b: b[0])
aux_bt = np.array([b[0] for b in aux_boxes])
# close-up camera fusion: where each close-up sighting lands on the field, drawn in the wide shot
cams_p = Path(f"configs/events/{r['event_key']}_cams.json")
cams = json.loads(cams_p.read_text()) if cams_p.exists() else {}
fused_at = []  # (t, wide-shot px, team or None, alliance, fills a gap the wide shot missed)
if cams and aux_boxes:
    from autoscout.field import Calibration
    from autoscout.vision.fusion import to_field

    c = r["calibration"]
    wcal = Calibration(event_key=r["event_key"], year=r["year"], view=tuple(c["view"]),
                       points={k: tuple(v) for k, v in c["points"].items()})
    # every close-up sighting where the camera puts it (rings) ...
    for bt, view, x0, y0, x1, y1, al, team, conf in aux_boxes:
        if view not in cams:
            continue
        fx, fy = to_field(cams[view]["H"], (x0 + x1) / 2, y1)
        if -0.3 <= fx <= 17.85 and -0.3 <= fy <= 8.35:
            px = wcal.to_image([(fx, fy)])[0]
            fused_at.append((bt, (float(px[0]), float(px[1])), team, al, False))
    # ... and the positions the analysis actually added to tracks where the wide shot had lost the robot
    for ft, slot, fx, fy in r.get("fused_rows", []):
        px = wcal.to_image([(fx, fy)])[0]
        fused_at.append((ft, (float(px[0]), float(px[1])), team_at(slot, ft), slot[:-1], True))
fused_at.sort(key=lambda f: f[0])
fused_t = np.array([f[0] for f in fused_at])
n_fills = sum(1 for f in fused_at if f[4])
n_seen = len(fused_at) - n_fills
aux_views = {}
try:
    import yaml
    aux_views = (yaml.safe_load(open(f"configs/events/{r['event_key']}.yaml")) or {}).get("aux_views") or {}
except Exception:
    pass
jumps = sorted([e for e in r.get("events", []) if e.get("alliance") and e.get("kind")], key=lambda e: e["t"])
trail = defaultdict(list)  # slot -> recent box bottoms (image px)

teams = {"red": r["red"], "blue": r["blue"]}
if a.observations_only:
    from autoscout.field import Calibration, FieldModel

    class ObservationReplay:
        width, height, fps = 1920, 1080, a.fps

        def iter_frames(self, start, end, step=1):
            c = r["calibration"]
            cal = Calibration(r["event_key"], year, tuple(c["view"]), c["points"])
            field = FieldModel.load(year)
            canvas = np.full((self.height, self.width, 3), (28, 28, 28), np.uint8)
            for x in np.linspace(0, field.length, 9):
                line = cal.to_image([(x, 0), (x, field.width)]).astype(np.int32)
                cv2.polylines(canvas, [line], False, (65, 65, 65), 1, cv2.LINE_AA)
            for y in np.linspace(0, field.width, 5):
                line = cal.to_image([(0, y), (field.length, y)]).astype(np.int32)
                cv2.polylines(canvas, [line], False, (65, 65, 65), 1, cv2.LINE_AA)
            cv2.putText(canvas, "SAVED OBSERVATION REPLAY - NO MATCH FOOTAGE", (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1., (220, 220, 220), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Robot boxes, ball paths and credit come from the saved match analysis.",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, .75, (180, 180, 180), 1, cv2.LINE_AA)
            for t in np.arange(start, end, step / self.fps):
                yield float(t), canvas.copy()

    video = ObservationReplay()
else:
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
tmp.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W + SIDE, H))
if not writer.isOpened():
    raise RuntimeError(f"cannot write demo video: {tmp}")
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
    if fused_at:
        text(sb, f"close-up cams: {n_seen} sightings placed on the field, {n_fills} filled where the wide shot lost the robot",
             (20, y + 4), 0.38, (0, 220, 220))
        y += 16
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


REEF_PT = {a: r["calibration"]["points"].get(f"{a}_reef_center") for a in ("red", "blue")}
for e in jumps:
    pl = e.get("placed")
    if pl and "@" not in pl["elem"]:  # a close-up placement is drawn in its own view, not linked to the wide shot
        e["_target"] = (pl["x"], pl["y"])
    elif str(e.get("kind", "")).startswith("L") and REEF_PT.get(e["alliance"]):
        e["_target"] = tuple(REEF_PT[e["alliance"]])


def recent_credit(t):
    return [e for e in jumps if e.get("shares") and 0 <= t - e["t"] <= 1.5]


LEVEL = lambda h: "L4" if h > 0.8 else "L3" if h > 0.6 else "L2" if h > 0.4 else "L1"  # noqa: E731


def draw_2025(img, t):
    """Debug layers: coral placements on the reefs (ring when it appears, then a dot labelled with the
    team credited), moving algae seen in the wide shot, close-up reads on the reef cams, and a banner
    decoding each live score increase."""
    for p in placements:
        if p["t"] > t:
            continue
        c = (int(p["x"] * scale), int(p["y"] * scale))
        e = place_by_t.get((p["elem"], p["t"]))
        age = t - p["t"]
        col = (0, 255, 0) if e else (200, 200, 200)
        if age < 1.5:
            cv2.circle(img, c, int(8 + 14 * age), col, 2, cv2.LINE_AA)
            cv2.putText(img, f"coral {LEVEL(p['h'])}?" + (f" {e['team']}" if e else " (unmatched)"),
                        (c[0] + 10, c[1] - 10), FONT, 0.45, col, 1, cv2.LINE_AA)
        cv2.circle(img, c, 3, col, -1, cv2.LINE_AA)
    if blobs:
        for name, (bt, xy) in blobs.items():
            m = (bt > t - 0.4) & (bt <= t)
            for (x_, y_), tt in zip(xy[m], bt[m]):
                cv2.circle(img, (int(x_ * scale), int(y_ * scale)), 2, (255, 255, 0), -1)
    # robots the close-up cameras see (boxes come every 0.5 s: show the latest within 0.6 s), labelled with
    # the team read on the bumper, or "?" when it couldn't be read
    if len(aux_bt):
        i0, i1 = np.searchsorted(aux_bt, [t - 0.6, t + 1e-3])
        latest = {}
        for b in aux_boxes[i0:i1]:
            latest.setdefault(b[1], b[0])
            latest[b[1]] = max(latest[b[1]], b[0])
        for bt, view, x0, y0, x1, y1, al, team, conf in aux_boxes[i0:i1]:
            if bt != latest[view]:
                continue
            col = RED if al == "red" else BLUE if al == "blue" else GREY
            p0, p1 = (int(x0 * scale), int(y0 * scale)), (int(x1 * scale), int(y1 * scale))
            cv2.rectangle(img, p0, p1, col, 2, cv2.LINE_AA)
            label = f"{team} {conf:.0%}" if team else "?"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
            cv2.rectangle(img, (p0[0], p0[1] - th - 6), (p0[0] + tw + 6, p0[1]), col, -1)
            cv2.putText(img, label, (p0[0] + 3, p0[1] - 4), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
    # fused close-up sightings in the wide shot: a ring where the close-up puts the robot; a filled,
    # labelled marker when it is a robot the wide shot had lost at that moment
    if len(fused_t):
        i0, i1 = np.searchsorted(fused_t, [t - 0.5, t + 1e-3])
        for bt, (fx, fy), team, al, gap in fused_at[i0:i1]:
            col = RED if al == "red" else BLUE if al == "blue" else GREY
            c = (int(fx * scale), int(fy * scale))
            if gap:
                cv2.circle(img, c, 9, col, -1, cv2.LINE_AA)
                cv2.circle(img, c, 11, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f"{team} via close-up", (c[0] + 12, c[1] + 4), FONT, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(img, c, 7, col, 2, cv2.LINE_AA)
    # banner: the last score increase of each alliance and what it was decoded as
    for al, x_ in (("blue", 0.30), ("red", 0.56)):
        last = [e for e in jumps if e["alliance"] == al and 0 <= t - e["t"] < 2.5]
        if not last:
            continue
        parts = []
        for e in last[-2:]:
            k = e["kind"]
            who = f" {e['team']}" if e.get("team") else ""
            parts.append(f"{k}{who}" + (" (seen placed)" if e.get("placed") else ""))
        txt = f"+{last[-1].get('jump', last[-1].get('points', ''))}: " + ", ".join(parts)
        org = (int(x_ * W), int(0.17 * vh))
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.55, 2)
        cv2.rectangle(img, (org[0] - 6, org[1] - th - 8), (org[0] + tw + 6, org[1] + 6), (0, 0, 0), -1)
        cv2.putText(img, txt, org, FONT, 0.55, RED if al == "red" else BLUE, 2, cv2.LINE_AA)


for t, frame in video.iter_frames(t0, t1, step=step):
    img = cv2.resize(frame, (W, vh), interpolation=cv2.INTER_AREA)
    # flights in progress (draw the trail up to now)
    for flight_id, flight, P in flights:
        ft0, ft1, tgt = flight["t0"], P[-1, 0], flight.get("target")
        if ft0 <= t <= ft1 + 0.3:
            # Use actual observation times: occlusion leaves irregular gaps.
            # Indexing by fraction of track duration can draw future ball positions.
            visible = P[P[:, 0] <= t]
            if not len(visible):
                continue
            pts = (visible[:, 1:3] * scale).astype(np.int32)
            col = (0, 255, 255) if not tgt else (0, 255, 0)
            cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[-1]), 3, col, -1)
            if a.flight_labels:
                label = f"#{flight_id} " + (str(flight["robot"]) if flight.get("robot") else "unassigned")
                cv2.putText(img, label, tuple(pts[-1] + [5, -5]), FONT, .4, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, label, tuple(pts[-1] + [5, -5]), FONT, .4, col, 1, cv2.LINE_AA)
    if year == 2025:
        draw_2025(img, t)
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
        if year == 2025:
            foot = (int((b[0] + b[2]) / 2 * scale), int(b[3] * scale))
            tr = trail[slot]
            tr.append((t, foot))
            while tr and t - tr[0][0] > 2.5:
                tr.pop(0)
            if len(tr) > 1:
                cv2.polylines(img, [np.array([q for _, q in tr], np.int32)], False, col, 1, cv2.LINE_AA)
            # who got credit for a score in the last 1.5 s: a line to where it was scored
            for e in recent_credit(t):
                if str(team) in (e.get("shares") or {}) and e["shares"][str(team)] >= 0.2:
                    tgt = e.get("_target")
                    if tgt is not None:
                        cv2.line(img, (int((b[0] + b[2]) / 2 * scale), int((b[1] + b[3]) / 2 * scale)),
                                 (int(tgt[0] * scale), int(tgt[1] * scale)), (0, 255, 0),
                                 max(1, int(4 * e["shares"][str(team)])), cv2.LINE_AA)
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
            if year == 2025 and e.get("shares"):
                sh = sorted(e["shares"].items(), key=lambda kv: -kv[1])
                desc = f"{k} {int(100 * sh[0][1])}%" + (f" ({sh[1][0]} {int(100 * sh[1][1])}%)" if len(sh) > 1 and sh[1][1] >= 0.1 else "")
                feed.append((e["t"], team, desc + (" *" if e.get("placed") else "")))
            else:
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
