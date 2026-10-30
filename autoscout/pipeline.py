"""End-to-end processing of one match video into per-team stats."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import config, games, overlay
from .field import Calibration, align_to_reference, view_similarity
from .tba import TBA, MatchInfo
from .video import Video, download
from .vision.detect import RobotDetector, StaticSuppressor, alliance_from_pixels, default_detector
from .vision.identity import IdentityEvidence
from .vision.tracker import SlotTracker
from .vision.trajectories import LaunchDetector

RESULTS_DIR = config.DATA_DIR / "results"


class NeedsCalibration(RuntimeError):
    pass


@dataclass
class Options:
    robot_hz: float = 0.0  # 0 = auto (10 Hz on GPU, 5 Hz on CPU)
    launch_fps: float = 30.0
    ocr_identity: bool = True
    debug_video: bool = False
    weights: str | None = None
    video_height: int = 1080
    log: bool = True


def overrides_for(match_key: str) -> dict[str, int]:
    p = config.EVENTS_DIR / f"{match_key.split('_')[0]}_overrides.yaml"
    if not p.exists():
        return {}
    d = yaml.safe_load(p.read_text()) or {}
    return {k: int(v) for k, v in (d.get(match_key) or {}).items()}


def reference_frame(cal: Calibration) -> np.ndarray | None:
    if cal.reference:
        p = Path(cal.reference)
        if not p.is_absolute():
            p = config.ROOT / p
        if p.exists():
            return cv2.imread(str(p))
    return None


def process_match(match_key: str, opts: Options | None = None, tba: TBA | None = None,
                  video_path: str | None = None, detector: RobotDetector | None = None) -> dict:
    opts = opts or Options()
    tba = tba or TBA()
    log = (lambda *a: print(f"[{match_key}]", *a, flush=True)) if opts.log else (lambda *a: None)
    t_start = time.time()
    m = MatchInfo.from_tba(tba.match(match_key))
    plugin = games.get(m.year)
    fm = plugin.field

    # ---- video ------------------------------------------------------------------------
    if video_path is None:
        if not m.videos:
            raise RuntimeError(f"{match_key}: no video on TBA")
        video_path = str(download(m.videos[0], height=opts.video_height))
    video = Video(video_path)
    log(f"video {Path(video_path).name} {video.width}x{video.height} {video.fps:.0f}fps {video.duration:.0f}s")

    # ---- calibration ------------------------------------------------------------------
    base_cal = Calibration.load(m.event_key)
    if base_cal is None:
        raise NeedsCalibration(f"event {m.event_key} is not calibrated: run `autoscout calibrate {match_key}` "
                               "or use the dashboard's Calibrate page")
    cal = base_cal.scaled((video.width, video.height))
    ref = reference_frame(base_cal)
    if ref is not None and ref.shape[1] != video.width:
        ref = cv2.resize(ref, (video.width, video.height))

    # ---- overlay: score timeline + match clock ----------------------------------------
    layout = None
    if base_cal.overlay and tuple(base_cal.overlay.get("frame_size", ())) == (video.width, video.height):
        layout = overlay.OverlayLayout.from_dict(base_cal.overlay)
    if layout is None:
        layout = overlay.detect_layout(video)
        if tuple(base_cal.frame_size) == (video.width, video.height):
            base_cal.overlay = layout.to_dict()
            base_cal.save()
    tl = overlay.read_timeline(video, layout, fm.auto_len, fm.teleop_len)
    clock = tl.clock
    if clock is None:
        raise RuntimeError("could not read the match clock from the overlay")
    log(f"match clock: auto {clock.start_vt:.1f}-{clock.auto_end_vt:.1f}s, teleop ends {clock.teleop_end_vt:.1f}s; "
        f"overlay final blue {int(tl.blue[-1])} red {int(tl.red[-1])}")

    # ---- per-video alignment ----------------------------------------------------------
    mid = video.frame_at(clock.teleop_start_vt + 20)
    if ref is not None and mid is not None:
        cal = align_to_reference(cal, mid, ref)
    if ref is None:
        ref = mid

    # ---- frame loop ---------------------------------------------------------------------
    det = detector or default_detector(opts.weights)
    static = StaticSuppressor()
    gpu = config.device() == "cuda"
    robot_hz = opts.robot_hz or (10.0 if gpu else 4.0)
    x, y, w, h = cal.view
    fps = video.fps
    launch_step = max(1, int(round(fps / opts.launch_fps)))
    robot_every = max(1, int(round(fps / robot_hz)))
    ppm_fn = lambda X, Y: np.pi * (0.5 * 0.15 * cal.px_per_meter((X, Y + 30)) / 2) ** 2  # noqa: E731
    launchers = {name: LaunchDetector(lo, hi, cal.view, piece_area_fn=ppm_fn, cal=cal, field_model=fm)
                 for name, (lo, hi) in plugin.launch_pieces.items()}
    tracker = SlotTracker(cal, fm)
    evidence = IdentityEvidence()
    ocr_budget = 400 if gpu else 150
    last_ocr = {}
    t0, t1 = clock.start_vt - 0.5, clock.teleop_end_vt + 3.5
    writer = None
    if opts.debug_video:
        out = RESULTS_DIR / f"{match_key}_debug.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps / launch_step, (w, h))
    cut = False
    n_frames = n_cut = 0
    for i, (t, frame) in enumerate(video.iter_frames(t0, t1, step=launch_step)):
        n_frames += 1
        if i % max(1, int(round(0.5 * fps / launch_step))) == 0:
            cut = view_similarity(frame, ref, cal.view) < 0.45
        if cut:
            n_cut += 1
            for L in launchers.values():
                L.prev = None
            continue
        view = frame[y: y + h, x: x + w]
        for L in launchers.values():
            L.step(t, view)
        if (i * launch_step) % robot_every < launch_step:
            dets = det(view)
            for d in dets:
                d.x0 += x; d.x1 += x; d.y0 += y; d.y1 += y
                if not d.alliance:
                    d.alliance, _ = alliance_from_pixels(frame, d)
            tracker.update(t, frame, static([d for d in dets if d.alliance]))
            if opts.ocr_identity and ocr_budget > 0:
                for s in tracker.all_slots():
                    if s.box is not None and s.last_t == t and t - last_ocr.get(s.key, -9) > 3.0 and s.box.w > 40:
                        evidence.ocr_bumper(frame, s.key, (s.box.x0, s.box.y0, s.box.x1, s.box.y1), m.teams(s.alliance))
                        last_ocr[s.key] = t
                        ocr_budget -= 1
        if writer is not None:
            vis = view.copy()
            for s in tracker.all_slots():
                if s.box is not None and t - s.last_t < 0.5:
                    b = s.box
                    col = (0, 0, 255) if s.alliance == "red" else (255, 0, 0)
                    cv2.rectangle(vis, (int(b.x0 - x), int(b.y0 - y)), (int(b.x1 - x), int(b.y1 - y)), col, 2)
                    cv2.putText(vis, s.key, (int(b.x0 - x), int(b.y0 - y) - 4), 0, 0.6, col, 2)
            for L in launchers.values():
                for tr in L.tracks:
                    if len(tr.pts) > 1:
                        p = np.array([(q[1], q[2]) for q in tr.pts[-20:]], np.int32)
                        cv2.polylines(vis, [p], False, (0, 255, 0), 2)
            ph, rem = clock.phase(t)
            cv2.putText(vis, f"{ph} {rem:5.1f}", (10, 30), 0, 1.0, (255, 255, 255), 2)
            writer.write(vis)
    if writer is not None:
        writer.release()
    launches = {name: L.finish() for name, L in launchers.items()}
    log(f"frames {n_frames} (camera cuts skipped {n_cut}); launches " +
        ", ".join(f"{k}={len(v)} ({sum(l.n_pieces for l in v):.0f} pieces)" for k, v in launches.items()) +
        "; rejected ground-rolling " + ", ".join(f"{k}={L.rejected_ground}" for k, L in launchers.items()))

    # ---- identity ------------------------------------------------------------------------
    tracks = {s.key: s.track for s in tracker.all_slots()}
    ctx = games.MatchContext(match=m, field=fm, cal=cal, clock=clock, timeline=tl, tracks=tracks,
                             identity={}, identity_conf={}, launches=launches)
    if hasattr(plugin, "identity_evidence"):
        plugin.identity_evidence(ctx, evidence)
    ov = overrides_for(match_key)
    ov.update(_db_overrides(match_key))
    for a in ("red", "blue"):
        keys = [f"{a}{i}" for i in range(3)]
        mapping, conf = evidence.solve(keys, m.teams(a), ov)
        ctx.identity.update(mapping)
        ctx.identity_conf.update(conf)
    log("identity " + ", ".join(f"{k}={v}" for k, v in ctx.identity.items()) + f" conf {ctx.identity_conf}")

    # ---- game analysis -------------------------------------------------------------------
    res = plugin.analyze(ctx)
    result = {
        "match_key": match_key, "event_key": m.event_key, "year": m.year, "label": m.label,
        "comp_level": m.comp_level, "match_number": m.match_number, "set_number": m.set_number,
        "red": m.red, "blue": m.blue, "red_score": m.red_score, "blue_score": m.blue_score,
        "video": m.videos[0] if m.videos else Path(video_path).stem,
        "video_offset": round(clock.start_vt, 2), "clock": clock.to_dict(),
        "timeline": tl.to_dict(), "identity": {k: v for k, v in ctx.identity.items()},
        "identity_conf": ctx.identity_conf, "identity_notes": evidence.notes[-30:],
        "tracks": {k: [(round(r[0], 2), round(r[1], 2), round(r[2], 2), bool(r[3])) for r in tr[::2]]
                   for k, tr in tracks.items()},
        "launches": {k: [dict(l.to_dict(), pts=[[round(a, 2), round(b), round(c)] for a, b, c in l.pts[::3]])
                         for l in v] for k, v in launches.items()},
        "raw_tracks": {k: [(round(r[0], 3), round(r[1], 3), round(r[2], 3), bool(r[3]), r[4]) for r in tr]
                       for k, tr in tracks.items()},
        "identity_scores": {f"{k[0]}:{k[1]}": round(v, 3) for k, v in evidence.score.items()},
        "calibration": {"view": list(cal.view), "points": {k: list(map(float, v)) for k, v in cal.points.items()}},
        "detector": str(det.weights.name),
        "processing_s": round(time.time() - t_start, 1),
        **res,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{match_key}.json").write_text(json.dumps(result, default=_json_default))
    log(f"done in {result['processing_s']}s")
    return result


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _db_overrides(match_key: str) -> dict[str, int]:
    from . import db

    try:
        return db.get_overrides(db.connect(), match_key)
    except Exception:
        return {}


def reanalyze(match_key: str, tba: TBA | None = None) -> dict:
    """Re-run identity + game analysis from stored observations (no video work)."""
    from .field import Calibration as Cal
    from .overlay import MatchClock, ScoreSample, ScoreTimeline
    from .vision.trajectories import Launch

    path = RESULTS_DIR / f"{match_key}.json"
    old = json.loads(path.read_text())
    tba = tba or TBA()
    m = MatchInfo.from_tba(tba.match(match_key))
    plugin = games.get(m.year)
    c = old["calibration"]
    cal = Cal(event_key=m.event_key, year=m.year, view=tuple(c["view"]),
              points={k: tuple(v) for k, v in c["points"].items()})
    clock = MatchClock(**old["clock"])
    tl_d = old["timeline"]
    samples = [ScoreSample(t, tm, b, r) for t, tm, b, r in zip(tl_d["t"], tl_d["timer"], tl_d["blue"], tl_d["red"])]
    tl = ScoreTimeline(samples=samples, clock=clock, blue=np.array(tl_d["blue"], float), red=np.array(tl_d["red"], float))
    tracks = {k: [tuple(r) for r in v] for k, v in old["raw_tracks"].items()}
    launches = {}
    for k, v in old["launches"].items():
        launches[k] = [Launch(t0=d["t0"], t1=d["t1"], start=tuple(d["start"]), end=tuple(d["end"]),
                              apex=tuple(d["apex"]), n_pieces=d["n"], rises=0.0,
                              pts=[tuple(p) for p in d["pts"]]) for d in v]
    ctx = games.MatchContext(match=m, field=plugin.field, cal=cal, clock=clock, timeline=tl, tracks=tracks,
                             identity={}, identity_conf={}, launches=launches)
    evidence = IdentityEvidence()
    for key, val in old.get("identity_scores", {}).items():
        slot, team = key.split(":")
        evidence.score[(slot, int(team))] = val
    ov = overrides_for(match_key)
    ov.update(_db_overrides(match_key))
    for a in ("red", "blue"):
        keys = [f"{a}{i}" for i in range(3)]
        mapping, conf = evidence.solve(keys, m.teams(a), ov)
        ctx.identity.update(mapping)
        ctx.identity_conf.update(conf)
    res = plugin.analyze(ctx)
    old.update(res)
    old.update(identity=ctx.identity, identity_conf=ctx.identity_conf, red_score=m.red_score, blue_score=m.blue_score)
    path.write_text(json.dumps(old, default=_json_default))
    return old
