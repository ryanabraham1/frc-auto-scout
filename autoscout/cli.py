"""Command line: `autoscout <command>` (see README)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback

from . import config


def cmd_run_match(a):
    from . import db
    from .pipeline import Options, RESULTS_DIR, process_match

    r = process_match(a.match_key, Options(debug_video=a.debug_video, weights=a.weights), video_path=a.video)
    db.save_result(db.connect(), r, str(RESULTS_DIR / f"{a.match_key}.json"))
    print(json.dumps({t: {k: v for k, v in m.items() if not isinstance(v, dict)} for t, m in r["teams"].items()}, indent=1))


_WORKER = {}


def _init_worker(weights, year):
    """Each worker process loads its own detector once and reuses it for every match it gets."""
    from . import games
    from .tba import TBA
    from .vision.detect import default_detector

    _WORKER.update(tba=TBA(max_age_s=300),
                   det=default_detector(weights, ensemble=games.get(year).detector_ensemble))


def _work(match_key, video_path, opts):
    """Process one match in a worker; the parent saves results so only one process writes the database."""
    from .pipeline import NeedsCalibration, process_match

    try:
        return match_key, process_match(match_key, opts, tba=_WORKER["tba"], video_path=video_path,
                                        detector=_WORKER["det"]), None
    except NeedsCalibration as e:
        return match_key, None, ("calibration", str(e))
    except Exception as e:
        traceback.print_exc()
        return match_key, None, ("error", str(e))


def cmd_run_event(a):
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait

    from . import db
    from .pipeline import Options, RESULTS_DIR
    from .tba import TBA, MatchInfo, sort_key
    from .video import download

    tba = TBA(max_age_s=300)
    con = db.connect()
    ms = sorted((MatchInfo.from_tba(m) for m in tba.event_matches(a.event_key)), key=sort_key)
    if a.matches:  # an explicit list is processed whether or not it was done before
        want = {k if "_" in k else f"{a.event_key}_{k}" for k in a.matches.split(",") if k}
        todo = [m for m in ms if m.key in want and m.videos]
        missing = want - {m.key for m in todo}
        if missing:
            print("no video on TBA (skipped):", ", ".join(sorted(missing)))
    else:
        levels = set(a.levels.split(","))
        ms = [m for m in ms if m.comp_level in levels and m.videos]
        done = {r["match_key"] for r in con.execute("SELECT match_key FROM matches WHERE status='done'")}
        todo = [m for m in ms if a.force or m.key not in done][: a.limit or None]
    workers = max(1, min(a.workers, len(todo) or 1))
    print(f"{a.event_key}: {len(todo)} to process, {workers} at a time"
          f"{', with debug videos' if a.debug_video else ''}", flush=True)
    opts = Options(weights=a.weights, debug_video=a.debug_video)
    info = {m.key: m for m in todo}
    # Downloads run ahead of processing (network-bound), and up to `workers` matches process at once
    # (each frame loop is sequential, but separate matches share nothing until they are saved).
    with ThreadPoolExecutor(2) as dl, ProcessPoolExecutor(workers, initializer=_init_worker,
                                                          initargs=(a.weights, int(a.event_key[:4]))) as pool:
        fetch = {dl.submit(download, m.videos[0], height=opts.video_height): m for m in todo}
        pending = set(fetch)
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                if f in fetch:  # a download finished: queue the match
                    m = fetch[f]
                    try:
                        nxt = pool.submit(_work, m.key, str(f.result()), opts)
                        pending.add(nxt)
                    except Exception as e:
                        print(f"[{m.key}] download failed: {e}", flush=True)
                        db.save_error(con, m.key, m.event_key, m.year, f"download failed: {e}")
                    continue
                key, r, err = f.result()
                m = info[key]
                if err is None:
                    db.save_result(con, r, str(RESULTS_DIR / f"{key}.json"))
                elif err[0] == "calibration":
                    print(err[1])
                    dl.shutdown(cancel_futures=True)
                    pool.shutdown(cancel_futures=True)
                    return 2
                else:  # keep going; record the error for the dashboard
                    db.save_error(con, key, m.event_key, m.year, err[1])


def cmd_reanalyze(a):
    from . import db
    from .pipeline import RESULTS_DIR, reanalyze

    con = db.connect()
    keys = [a.key] if "_" in a.key else [r["match_key"] for r in con.execute(
        "SELECT match_key FROM matches WHERE event_key=? AND status='done'", (a.key,))]
    for k in keys:
        r = reanalyze(k)
        db.save_result(con, r, str(RESULTS_DIR / f"{k}.json"))
        print("reanalyzed", k)


def cmd_relaunch(a):
    from . import db
    from .pipeline import RESULTS_DIR, relaunch

    con = db.connect()
    keys = [a.key] if "_" in a.key else [r["match_key"] for r in con.execute(
        "SELECT match_key FROM matches WHERE event_key=? AND status='done'", (a.key,))]
    for k in keys:
        r = relaunch(k)
        db.save_result(con, r, str(RESULTS_DIR / f"{k}.json"))
        print("relaunched", k, r.get("launch_stats"))


def cmd_relabel(a):
    from . import db
    from .pipeline import RESULTS_DIR, relabel

    con = db.connect()
    keys = [a.key] if "_" in a.key else [r["match_key"] for r in con.execute(
        "SELECT match_key FROM matches WHERE event_key=? AND status='done'", (a.key,))]
    for k in keys:
        r = relabel(k)
        db.save_result(con, r, str(RESULTS_DIR / f"{k}.json"))
        print("relabelled", k, r.get("tracker"))


def cmd_rescore(a):
    from . import db
    from .pipeline import RESULTS_DIR, rescore

    con = db.connect()
    keys = [a.key] if "_" in a.key else [r["match_key"] for r in con.execute(
        "SELECT match_key FROM matches WHERE event_key=? AND status='done'", (a.key,))]
    for k in keys:
        r = rescore(k)
        db.save_result(con, r, str(RESULTS_DIR / f"{k}.json"))
        print("rescored", k)


def cmd_recollect(a):
    from .pipeline import collect_blobs

    collect_blobs(a.match_key, a.video)


def cmd_identify(a):
    from . import db
    from .pipeline import RESULTS_DIR, identity_pass

    r = identity_pass(a.match_key, a.video)
    db.save_result(db.connect(), r, str(RESULTS_DIR / f"{a.match_key}.json"))
    print(r["identity"], r["identity_conf"])


def cmd_aux(a):
    from . import db
    from .pipeline import RESULTS_DIR, aux_pass

    r = aux_pass(a.match_key, a.video)
    db.save_result(db.connect(), r, str(RESULTS_DIR / f"{a.match_key}.json"))
    print(len(r.get("aux_reads", [])), "close-up reads")


def cmd_calibrate(a):
    """Save a reference frame for an event and print the dashboard link for clicking landmarks."""
    import cv2

    from .field import Calibration
    from .tba import TBA, MatchInfo
    from .video import Video, download

    tba = TBA()
    m = MatchInfo.from_tba(tba.match(a.match_key))
    v = Video(a.video or download(m.videos[0]))
    t = a.t if a.t is not None else v.duration * 0.45
    img = v.frame_at(t)
    ref = config.EVENTS_DIR / f"{m.event_key}_ref.jpg"
    ref.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ref), img)
    cal = Calibration.load(m.event_key)
    if cal is None:
        cal = Calibration(event_key=m.event_key, year=m.year, view=(0, 0, v.width, v.height), points={},
                          frame_size=(v.width, v.height))
    cal.reference = str(ref.relative_to(config.ROOT))
    cal.frame_size = (v.width, v.height)
    cal.save()
    print(f"reference frame saved: {ref}\nOpen the dashboard (autoscout serve) -> Calibrate -> {m.event_key}")


def cmd_serve(a):
    import uvicorn

    uvicorn.run("autoscout.web.app:app", host=a.host, port=a.port, reload=False)


def cmd_label_seeds(a):
    from .train.sam_label import main

    main(a.seed_file)


def cmd_train(a):
    from .train.train_robots import train

    print("weights:", train(base=a.base, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                            hand_only=a.hand_only, freeze=a.freeze, lr0=a.lr0))


def cmd_report(a):
    from . import db
    from .validate import event_report

    print(event_report(db.connect(), a.event_key))


def cmd_export(a):
    from . import db
    from .aggregate import team_table

    rows = team_table(db.connect(), a.event_key)
    keys = sorted({k for r in rows for k in r if not k.endswith(".counts")}, key=lambda k: (k != "team", k))
    w = csv.DictWriter(open(a.out, "w", newline=""), fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    print("wrote", a.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="autoscout", description="Automated FRC scouting from match video")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run-match"); s.add_argument("match_key"); s.add_argument("--video")
    s.add_argument("--debug-video", action="store_true"); s.add_argument("--weights"); s.set_defaults(f=cmd_run_match)
    s = sub.add_parser("run-event"); s.add_argument("event_key"); s.add_argument("--levels", default="qm,sf,f")
    s.add_argument("--force", action="store_true"); s.add_argument("--limit", type=int, default=0)
    s.add_argument("--matches", help="comma-separated match keys (or qm12-style suffixes) to process, even if done")
    s.add_argument("--debug-video", action="store_true", help="also write each match's debug video")
    s.add_argument("--workers", type=int, default=1, help="matches processed at the same time")
    s.add_argument("--weights"); s.set_defaults(f=cmd_run_event)
    s = sub.add_parser("reanalyze", help="recompute stats from saved observations (match or event key)")
    s.add_argument("key"); s.set_defaults(f=cmd_reanalyze)
    s = sub.add_parser("relaunch", help="re-extract launches from saved blobs, then reanalyze")
    s.add_argument("key"); s.set_defaults(f=cmd_relaunch)
    s = sub.add_parser("relabel", help="re-label robots from saved tracklets + dashboard labels, then reanalyze")
    s.add_argument("key"); s.set_defaults(f=cmd_relabel)
    s = sub.add_parser("rescore", help="re-read the broadcast score timeline from video, then reanalyze")
    s.add_argument("key"); s.set_defaults(f=cmd_rescore)
    s = sub.add_parser("recollect", help="re-collect airborne-piece blobs from video (no robot detection)")
    s.add_argument("match_key"); s.add_argument("video"); s.set_defaults(f=cmd_recollect)
    s = sub.add_parser("identify", help="dedicated bumper-OCR identity pass over a processed match")
    s.add_argument("match_key"); s.add_argument("video"); s.set_defaults(f=cmd_identify)
    s = sub.add_parser("aux", help="read close-up camera views (e.g. reef cams) for a processed match")
    s.add_argument("match_key"); s.add_argument("video"); s.set_defaults(f=cmd_aux)
    s = sub.add_parser("calibrate"); s.add_argument("match_key"); s.add_argument("--video")
    s.add_argument("--t", type=float); s.set_defaults(f=cmd_calibrate)
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.set_defaults(f=cmd_serve)
    s = sub.add_parser("label-seeds"); s.add_argument("seed_file"); s.set_defaults(f=cmd_label_seeds)
    s = sub.add_parser("train-robots"); s.add_argument("--base"); s.add_argument("--epochs", type=int, default=60)
    s.add_argument("--hand-only", action="store_true", help="fine-tune using corrected frames only")
    s.add_argument("--freeze", type=int, default=0, help="number of backbone layers to freeze")
    s.add_argument("--lr0", type=float, default=0.0003)
    s.add_argument("--imgsz", type=int, default=1280); s.add_argument("--batch", type=int, default=8); s.set_defaults(f=cmd_train)
    s = sub.add_parser("report"); s.add_argument("event_key"); s.set_defaults(f=cmd_report)
    s = sub.add_parser("export"); s.add_argument("event_key"); s.add_argument("--out", default="scouting.csv")
    s.set_defaults(f=cmd_export)
    a = p.parse_args(argv)
    sys.exit(a.f(a) or 0)


if __name__ == "__main__":
    main()
