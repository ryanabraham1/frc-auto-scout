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
from .vision.identity import IdentityEvidence, identity_timeline
from .vision.tracker import SlotTracker
from .vision.launches import LaunchExtractor
from .vision.trajectories import Launch, LaunchDetector

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


def launch_view(cal: Calibration) -> tuple:
    """Region searched for airborne pieces: the wide shot extended up to the top of the frame.

    High arcs leave the wide shot's top edge; the broadcast graphics above it are static, so
    they never produce *moving* piece-colored blobs and are safe to include.
    """
    x, y, w, h = cal.view
    return (x, 0, w, y + h)


def collect_blobs(match_key: str, video_path: str, fps: float = 30.0) -> None:
    """Re-run only the airborne-piece blob collection for a processed match (fast; no robots)."""
    old = json.loads((RESULTS_DIR / f"{match_key}.json").read_text())
    plugin = games.get(old["year"])
    c = old["calibration"]
    cal = Calibration(event_key=old["event_key"], year=old["year"], view=tuple(c["view"]),
                      points={k: tuple(v) for k, v in c["points"].items()})
    clock = old["clock"]
    video = Video(video_path)
    lview = launch_view(cal)
    step = max(1, int(round(video.fps / fps)))
    lag = max(1, int(round(video.fps / step / 30.0)))
    dets = {n: LaunchDetector(lo, hi, lview, lag=lag) for n, (lo, hi) in plugin.launch_pieces.items()}
    t0 = clock["auto_end_vt"] - clock["auto_len"] - 0.5
    t1 = clock["teleop_end_vt"] + 3.5
    X, Y, W, H = lview
    for t, frame in video.iter_frames(t0, t1, step=step):
        for d in dets.values():
            d.step(t, frame[Y: Y + H, X: X + W])
    np.savez_compressed(RESULTS_DIR / f"{match_key}_blobs.npz",
                        **{k: np.array(d.blobs, dtype=np.float32).reshape(-1, 6) for k, d in dets.items()},
                        view=np.array(lview))


def launch_targets(cal: Calibration, fm) -> dict:
    """Image locations of scoring openings, from optional '<element>_top' calibration points
    (e.g. blue_hub_top). Radius ~ half the opening width at that depth."""
    out = {}
    for name, p in cal.points.items():
        if not name.endswith("_top"):
            continue
        elem = name[:-4]
        e = fm.elements.get(elem)
        if e is None:
            continue
        ppm = cal.px_per_meter_x(e["center"])
        width = max(e.get("size", [1.0])[0], 2 * e.get("radius", 0.5))
        out[elem] = (float(p[0]), float(p[1]), 0.6 * width * ppm)
    return out


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
    # camera-cut detection compares against this video's own wide shot (framing varies per match)
    own_ref = mid if mid is not None else ref

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
    lview = launch_view(cal)
    lag = max(1, int(round(fps / launch_step / 30.0)))
    launchers = {name: LaunchDetector(lo, hi, lview, piece_area_fn=ppm_fn, cal=cal, field_model=fm, lag=lag)
                 for name, (lo, hi) in plugin.launch_pieces.items()}
    tracker = SlotTracker(cal, fm)
    evidence = IdentityEvidence()
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
            cut = view_similarity(frame, own_ref, cal.view) < 0.45
        if cut:
            n_cut += 1
            for L in launchers.values():
                L.prev = None
            continue
        view = frame[y: y + h, x: x + w]
        lv = frame[lview[1]: lview[1] + lview[3], lview[0]: lview[0] + lview[2]]
        for L in launchers.values():
            L.step(t, lv)
        if (i * launch_step) % robot_every < launch_step:
            dets = det(view)
            for d in dets:
                d.x0 += x; d.x1 += x; d.y0 += y; d.y1 += y
                if not d.alliance:
                    d.alliance, _ = alliance_from_pixels(frame, d)
            tracker.update(t, frame, static([d for d in dets if d.alliance]))
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    blobs = {k: np.array(L.blobs, dtype=np.float32).reshape(-1, 6) for k, L in launchers.items()}
    np.savez_compressed(RESULTS_DIR / f"{match_key}_blobs.npz", **blobs, view=np.array(lview))
    extractor = LaunchExtractor(lview, cal, fm, targets=launch_targets(cal, fm))
    launches = {}
    for name, B in blobs.items():
        launches[name] = extractor.extract(B)
        launchers[name].rejected_ground = extractor.stats.get("not_ballistic", 0)
    log(f"frames {n_frames} (camera cuts skipped {n_cut}); launches " +
        ", ".join(f"{k}={len(v)} ({sum(l.n_pieces for l in v):.0f} pieces)" for k, v in launches.items()) +
        "; rejected ground-rolling " + ", ".join(f"{k}={L.rejected_ground}" for k, L in launchers.items()))

    # ---- identity ------------------------------------------------------------------------
    tracks = {s.key: s.track for s in tracker.all_slots()}
    if opts.ocr_identity:
        collect_identity(video, tracks, {"red": m.red, "blue": m.blue}, evidence,
                         budget=600 if gpu else 250)
        log(f"identity OCR: {len(evidence.timed)} matched reads")
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
        ctx.identity_tl.update(identity_timeline(evidence.timed, keys, m.teams(a), mapping,
                                                 clock.start_vt, clock.teleop_end_vt + 4, ov))
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
        "identity_timed": evidence.timed,
        "identity_timeline": ctx.identity_tl,
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
                              pts=[tuple(p) for p in d["pts"]], target=d.get("target"),
                              rising=d.get("rising", True)) for d in v]
    ctx = games.MatchContext(match=m, field=plugin.field, cal=cal, clock=clock, timeline=tl, tracks=tracks,
                             identity={}, identity_conf={}, launches=launches)
    evidence = IdentityEvidence()
    for key, val in old.get("identity_scores", {}).items():
        slot, team = key.split(":")
        evidence.score[(slot, int(team))] = val
    evidence.timed = [tuple(x) for x in old.get("identity_timed", [])]
    ov = overrides_for(match_key)
    ov.update(_db_overrides(match_key))
    for a in ("red", "blue"):
        keys = [f"{a}{i}" for i in range(3)]
        mapping, conf = evidence.solve(keys, m.teams(a), ov)
        ctx.identity.update(mapping)
        ctx.identity_conf.update(conf)
        ctx.identity_tl.update(identity_timeline(evidence.timed, keys, m.teams(a), mapping,
                                                 clock.start_vt, clock.teleop_end_vt + 4, ov))
    res = plugin.analyze(ctx)
    old["identity_timeline"] = ctx.identity_tl
    old.update(res)
    old.update(identity=ctx.identity, identity_conf=ctx.identity_conf, red_score=m.red_score, blue_score=m.blue_score)
    path.write_text(json.dumps(old, default=_json_default))
    return old


def relaunch(match_key: str, tba: TBA | None = None) -> dict:
    """Re-extract launches from the saved per-frame blobs (no video decode), then reanalyze."""
    from .field import Calibration as Cal

    path = RESULTS_DIR / f"{match_key}.json"
    old = json.loads(path.read_text())
    blobs = np.load(RESULTS_DIR / f"{match_key}_blobs.npz")
    m_year = old["year"]
    fm = games.get(m_year).field
    base = Cal.load(old["event_key"])
    c = old["calibration"]
    cal = Cal(event_key=old["event_key"], year=m_year, view=tuple(c["view"]),
              points={**{k: tuple(v) for k, v in c["points"].items()},
                      **{k: tuple(v) for k, v in (base.points if base else {}).items() if k.endswith("_top")}})
    lview = tuple(int(v) for v in blobs["view"]) if "view" in blobs.files else cal.view
    ex = LaunchExtractor(lview, cal, fm, targets=launch_targets(cal, fm))
    out = {}
    for name in blobs.files:
        if name == "view":
            continue
        L = ex.extract(blobs[name])
        out[name] = [dict(l.to_dict(), pts=[[round(a, 2), round(b), round(c)] for a, b, c in l.pts[::3]]) for l in L]
    old["launches"] = out
    old["launch_stats"] = ex.stats
    path.write_text(json.dumps(old, default=_json_default))
    return reanalyze(match_key, tba)


def collect_identity(video: Video, tracks: dict, teams: dict, evidence: IdentityEvidence,
                     every: float = 1.5, budget: int = 400) -> None:
    """Bumper-number OCR on robot boxes sampled across the whole match (time-stamped evidence)."""
    jobs = []
    for slot, tr in tracks.items():
        last = -9.0
        for row in tr:
            t, obs, box = row[0], row[3], row[4]
            if obs and box and t - last >= every and (box[2] - box[0]) > 35:
                jobs.append((t, slot, box))
                last = t
    jobs.sort()
    step = max(1, len(jobs) // max(budget, 1))
    all_teams = teams["red"] + teams["blue"]
    for t, slot, box in jobs[::step]:
        frame = video.frame_at(t)
        if frame is not None:
            evidence.t = t
            evidence.ocr_bumper(frame, slot, box, teams[slot[:-1]], all_teams)
    evidence.t = None


def identity_pass(match_key: str, video_path: str, every: float = 1.5, budget: int = 400) -> dict:
    """Re-run the bumper-OCR identity pass for a processed match using its stored robot boxes."""
    old = json.loads((RESULTS_DIR / f"{match_key}.json").read_text())
    ev = IdentityEvidence()
    collect_identity(Video(video_path), old["raw_tracks"], {"red": old["red"], "blue": old["blue"]}, ev,
                     every, budget)
    old["identity_scores"] = {f"{k[0]}:{k[1]}": round(v, 3) for k, v in ev.score.items()}
    old["identity_timed"] = ev.timed
    old["identity_notes"] = ev.notes[-40:]
    (RESULTS_DIR / f"{match_key}.json").write_text(json.dumps(old, default=_json_default))
    return reanalyze(match_key)
