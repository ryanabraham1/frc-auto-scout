"""Known-path synthetic tracking benchmark; this is not match-video ground truth.

python -m scripts.benchmark_ball_tracking --out data/ball_benchmark.json
"""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from autoscout.vision.launches import LaunchExtractor
from autoscout.vision.trajectories import LaunchDetector


def scene(case):
    rng = np.random.default_rng(4)
    fps, duration = 60, 4.0
    starts = np.arange(.2, 2.4, .25 if case == "dense_volley" else .5)
    truth = []
    for i, start in enumerate(starts):
        t = np.arange(int(round(start * fps)), int(round((start + 1.25) * fps))) / fps
        dt = t - start
        x = 100 + 350 * dt
        y = 540 - 640 * dt + 400 * dt ** 2
        if case == "crossing" and i % 2:
            x = 750 - 350 * dt
        truth.append(np.column_stack([t, x, y]))
    frames = []
    for k in range(int(duration * fps)):
        t = k / fps
        frame = np.zeros((640, 960, 3), np.uint8)
        for ball in truth:
            row = ball[np.abs(ball[:, 0] - t) < .001]
            if not len(row):
                continue
            _, x, y = row[0]
            age = t - ball[0, 0]
            if case == "short_fragments" and not (age < .06 or .24 < age < .31):
                continue
            if case == "occlusion" and 270 < x < 325:
                continue
            if case == "centroid_noise":
                x, y = np.array([x, y]) + rng.normal(0, 2.0, 2)
            cv2.circle(frame, (round(x), round(y)), 7, (0, 220, 255), -1)
        # Static carpet pieces and a rolling piece must not become flights.
        for x in range(50, 900, 75):
            cv2.circle(frame, (x, 610), 7, (0, 220, 255), -1)
        cv2.circle(frame, (round(50 + 300 * t), 580), 7, (0, 220, 255), -1)
        frames.append((t, frame))
    return truth, frames


def evaluate(truth, launches):
    # A match requires most of the *observed* track to follow the same known
    # ball in space and time. One predicted track can recover only one ball.
    costs = np.full((len(truth), len(launches)), 1e6)
    for i, ball in enumerate(truth):
        for j, launch in enumerate(launches):
            P = np.asarray(launch.pts)
            inside = (P[:, 0] >= ball[0, 0] - .001) & (P[:, 0] <= ball[-1, 0] + .001)
            if inside.mean() < .8 or inside.sum() < 5:
                continue
            Q = P[inside]
            xy = np.column_stack([np.interp(Q[:, 0], ball[:, 0], ball[:, axis]) for axis in (1, 2)])
            err = np.linalg.norm(Q[:, 1:3] - xy, axis=1)
            if np.mean(err < 12) >= .8:
                costs[i, j] = np.median(err)
    rows, cols = linear_sum_assignment(costs)
    matched = sum(costs[i, j] < 1e6 for i, j in zip(rows, cols))
    return {"balls": len(truth), "tracks": len(launches), "matched_balls": int(matched),
            "recall": matched / len(truth), "precision": matched / len(launches) if launches else 0.0}


def run(case, background=0, details=False):
    truth, frames = scene(case)
    detector = LaunchDetector((18, 90, 110), (38, 255, 255), (0, 0, 960, 640), lag=2)
    detector.online, detector.bg_frames = False, background
    for t, frame in frames:
        detector.step(t, frame)
    ex = LaunchExtractor((0, 0, 960, 640))
    launches = ex.extract(np.asarray(detector.blobs))
    result = evaluate(truth, launches)
    return (result, launches) if details else result


def render(path):
    """A labeled synthetic diagnostic, never presented as broadcast footage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 720))
    for case in CASES:
        result, launches = run(case, details=True)
        _, frames = scene(case)
        for i, (t, frame) in enumerate(frames):
            if i % 2:
                continue
            canvas = np.zeros((720, 960, 3), np.uint8)
            canvas[80:] = frame
            for j, launch in enumerate(launches):
                if not launch.t0 <= t <= launch.t1 + .15:
                    continue
                P = np.asarray(launch.pts)
                xy = P[P[:, 0] <= t, 1:3] + [0, 80]
                if not len(xy):
                    continue
                pts = xy.astype(np.int32)
                cv2.polylines(canvas, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"#{j + 1}", tuple(pts[-1] + [8, 0]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            cv2.putText(canvas, "SYNTHETIC BALL TRACKING TEST - NOT MATCH FOOTAGE", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1, cv2.LINE_AA)
            label = f"{case}: {result['matched_balls']}/{result['balls']} balls recovered | green = extracted path"
            cv2.putText(canvas, label, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (190, 220, 190), 1, cv2.LINE_AA)
            writer.write(canvas)
    writer.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(temporary), "-c:v", "libx264",
                    "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
    temporary.unlink()


CASES = ("clean", "centroid_noise", "occlusion", "short_fragments", "dense_volley", "crossing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--video", type=Path, help="write a labeled synthetic diagnostic MP4")
    args = parser.parse_args()
    report = {"scope": "synthetic known paths; no claim of real-match scoring accuracy",
              "cases": {case: run(case) for case in CASES}}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.video:
        render(args.video)


if __name__ == "__main__":
    main()
