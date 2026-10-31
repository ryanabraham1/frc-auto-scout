"""Build the robot-detector dataset from many events' broadcasts (no hand labelling).

For each event, a few qualification matches spread over the schedule are downloaded and two clips
of each are pseudo-labelled (see autoscout/train/pseudo_label.py). Broadcast variety (camera
height, zoom, lighting, overlays) matters more than frame count, so many events beat many matches.

usage: python scripts/build_robot_dataset.py [--per-event 3] [--clips 2] EVENT [EVENT ...]
"""
import argparse
import json

from autoscout import games
from autoscout.field import Calibration
from autoscout.tba import TBA
from autoscout.train.pseudo_label import label_clip
from autoscout.train.sam_label import DATASET
from autoscout.video import Video, download
from autoscout.vision.detect import default_detector

ap = argparse.ArgumentParser()
ap.add_argument("events", nargs="+")
ap.add_argument("--per-event", type=int, default=3)
ap.add_argument("--clips", type=int, default=2)
ap.add_argument("--clip-s", type=float, default=30.0)
ap.add_argument("--hz", type=float, default=10.0)
ap.add_argument("--every", type=int, default=10, help="write every n-th processed frame")
a = ap.parse_args()

tba = TBA()
teacher = default_detector(conf=0.12, ensemble=True)  # recall matters here: tracks filter the noise
done_file = DATASET / "clips_done.json"
done = set(json.loads(done_file.read_text())) if done_file.exists() else set()
for ev in a.events:
    base = Calibration.load(ev)
    if base is not None and len(base.points) < 4:
        base = None  # a wide-shot crop but no homography: no field check
    fm = games.get(int(ev[:4])).field
    ms = [m for m in tba.get(f"/event/{ev}/matches") if m.get("videos") and m["comp_level"] == "qm"]
    ms.sort(key=lambda m: m["match_number"])
    if not ms:
        print(ev, "no videos", flush=True)
        continue
    pick = [ms[int(i * (len(ms) - 1) / max(a.per_event - 1, 1))] for i in range(a.per_event)]
    for m in pick:
        yt = m["videos"][0]["key"]
        try:
            v = Video(download(yt, height=1080))
        except Exception as e:  # private / removed videos
            print(m["key"], "download failed:", str(e)[:120], flush=True)
            continue
        # a match video starts ~5-15 s before AUTO; clips spread over auto + teleop
        starts = [12 + i * (min(v.duration, 175) - 12 - a.clip_s) / max(a.clips - 1, 1) for i in range(a.clips)]
        for s in starts:
            stem = f"{m['key']}_{int(s)}"
            if stem in done:
                continue
            cal = base.scaled((v.width, v.height)) if base is not None else None
            n = label_clip(v, teacher, s, a.clip_s, a.hz, a.every, stem, cal=cal, fm=fm)
            done.add(stem)
            done_file.write_text(json.dumps(sorted(done)))
            print(stem, n, "frames", flush=True)
print("dataset:", DATASET, flush=True)
