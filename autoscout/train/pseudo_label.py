"""Data engine: robot-detector training labels from broadcast video, with no hand labelling.

A teacher detector (the current model, ensembled with Spy Glass) is run at a low confidence on
every frame of short clips. Its detections are linked over time in image space, and only tracks
that behave like robots are kept as labels:

  * long enough (a robot is on screen for seconds; flickers are pieces, people, glare),
  * confident at least sometimes (a track that never scores well is probably not a robot),
  * not parked on one spot for the whole clip (alliance-coloured field elements),
  * one alliance colour (majority vote over the track).

Then each track's short gaps are filled by interpolating its box: those are exactly the frames
where the teacher missed a robot it saw a moment before and after (motion blur, pieces piled on
top, a partial occlusion), so the student learns the hard cases the teacher misses. Frames are
written as a YOLO dataset under data/datasets/robots (class 0 = red_robot, 1 = blue_robot).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..video import Video
from ..vision.detect import Det, alliance_from_pixels, iou, is_person_box
from .sam_label import write_label

MIN_TRACK_S = 1.5  # s a track must last
MIN_PEAK_CONF = 0.5  # some detection of the track must be this confident
MAX_FILL_S = 0.8  # s of missed frames filled by interpolation
STATIC_PX = 6.0  # px: a track whose box centre moves less than this (std) all clip is a field element


@dataclass
class _T:
    rows: list = field(default_factory=list)  # (frame index, x0, y0, x1, y1, conf, alliance)
    miss: int = 0

    def box(self):
        r = self.rows[-1]
        return Det(r[1], r[2], r[3], r[4], r[5], r[6])


def link(per_frame: list[list[Det]], max_miss: int) -> list[_T]:
    """Greedy IoU linking of per-frame detections into tracks (image space)."""
    live: list[_T] = []
    done: list[_T] = []
    for k, dets in enumerate(per_frame):
        dets = sorted(dets, key=lambda d: -d.conf)
        used = set()
        pairs = []
        for i, tr in enumerate(live):
            b = tr.box()
            for j, d in enumerate(dets):
                v = iou(b, d)
                # allow a looser match for small, fast robots: centre distance within half a box
                cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
                dx, dy = abs(cx - (d.x0 + d.x1) / 2), abs(cy - (d.y0 + d.y1) / 2)
                if v > 0.2 or (dx < 0.5 * b.w and dy < 0.5 * b.h):
                    pairs.append((v + 0.3 * (d.alliance == b.alliance), i, j))
        pairs.sort(reverse=True)
        took = set()
        for _, i, j in pairs:
            if i in took or j in used:
                continue
            d = dets[j]
            live[i].rows.append((k, d.x0, d.y0, d.x1, d.y1, d.conf, d.alliance))
            live[i].miss = 0
            took.add(i)
            used.add(j)
        nxt = []
        for i, tr in enumerate(live):
            if i not in took:
                tr.miss += 1
            (done if tr.miss > max_miss else nxt).append(tr)
        live = nxt
        for j, d in enumerate(dets):
            if j not in used:
                live.append(_T([(k, d.x0, d.y0, d.x1, d.y1, d.conf, d.alliance)]))
    return done + live


def on_field(rows, cal, fm, off=(0, 0)) -> bool:
    """With a calibration: the track's ground point is on the field and its box is robot-sized
    there (a full robot is ~0.7-1.3 m wide; refs, crowd and overlay graphics fail one or both)."""
    if cal is None:
        return True
    feet = np.array([((r[1] + r[3]) / 2 + off[0], r[4] + off[1]) for r in rows])
    f = cal.to_field(feet)
    fx, fy = np.median(f, 0)
    if not fm.in_field((fx, fy), 0.4):
        return False
    widths = np.array([(r[3] - r[1]) / max(cal.px_per_meter_x(tuple(p)), 1e-6) for r, p in zip(rows, f)])
    return 0.35 <= float(np.median(widths)) <= 2.0


def is_person(rows, people: dict[int, list]) -> bool:
    """Most of the track's boxes are a person's (a referee, a field worker, someone at the rail)."""
    hits = sum(any(is_person_box(Det(r[1], r[2], r[3], r[4], 0.0, ""), p) for p in people.get(r[0], []))
               for r in rows)
    return hits > 0.3 * len(rows)


def robot_tracks(tracks: list[_T], fps: float, cal=None, fm=None, off=(0, 0),
                 people: dict | None = None) -> list[tuple[str, dict[int, tuple]]]:
    """Keep robot-like tracks; returns [(alliance, {frame: (x0, y0, x1, y1)})] with gaps filled."""
    out = []
    for tr in tracks:
        R = tr.rows
        if (R[-1][0] - R[0][0]) / fps < MIN_TRACK_S or max(r[5] for r in R) < MIN_PEAK_CONF:
            continue
        if not on_field(R, cal, fm, off):
            continue
        if people and is_person(R, people):
            continue
        c = np.array([((r[1] + r[3]) / 2, (r[2] + r[4]) / 2) for r in R])
        if c.std(0).max() < STATIC_PX:
            continue
        votes = [r[6] for r in R if r[6]]
        if not votes:
            continue
        a = max(set(votes), key=votes.count)
        if votes.count(a) < 0.7 * len(votes):
            continue  # flips between alliances: two robots merged, or not a robot
        boxes = {r[0]: r[1:5] for r in R}
        ks = sorted(boxes)
        for k0, k1 in zip(ks, ks[1:]):
            if 1 < k1 - k0 <= MAX_FILL_S * fps:
                b0, b1 = np.array(boxes[k0]), np.array(boxes[k1])
                for k in range(k0 + 1, k1):
                    f = (k - k0) / (k1 - k0)
                    boxes[k] = tuple(b0 + f * (b1 - b0))
        out.append((a, boxes))
    return out


@functools.lru_cache(maxsize=1)
def _veto():
    from ..vision.detect import PersonVeto

    return PersonVeto()


def person_boxes(img: np.ndarray) -> list[Det]:
    """People in a frame, to veto person-shaped 'robots'."""
    return _veto().people(img)


def label_clip(video: Video, teacher, t0: float, dur: float, hz: float, every: int, stem: str,
               conf: float = 0.12, cal=None, fm=None) -> int:
    """Run the teacher on [t0, t0+dur) at `hz`, keep robot tracks, write every `every`-th frame.

    With a calibration (scaled to the video) frames are cropped to its wide shot, as the pipeline's
    detector sees them, and tracks are checked against the field."""
    step = max(1, int(round(video.fps / hz)))
    x, y, w, h = cal.view if cal is not None else (0, 0, video.width, video.height)
    frames, dets, people = {}, [], {}  # only frames that will be written are kept (memory)
    for k, (t, f) in enumerate(video.iter_frames(t0, t0 + dur, step=step)):
        f = np.ascontiguousarray(f[y: y + h, x: x + w])
        people[k] = person_boxes(f)
        ds = teacher(f, conf=conf)
        for d in ds:
            if not d.alliance:
                d.alliance, _ = alliance_from_pixels(f, d)
        if k % every == 0:
            frames[k] = f
        dets.append(ds)
    if not frames:
        return 0
    fps = video.fps / step
    tracks = robot_tracks(link(dets, max_miss=int(MAX_FILL_S * fps)), fps, cal, fm, (x, y), people)
    n = 0
    for k, f in frames.items():
        boxes = [(a, *b[k]) for a, b in tracks if k in b]
        write_label(f, boxes, f"{stem}_{k:04d}")
        n += 1
    return n
