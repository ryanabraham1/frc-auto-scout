"""Prepare 100 blinded per-flight labels, then serve the review page on localhost:8001.

python scripts/review_2026_shots.py --prepare
python scripts/review_2026_shots.py
Predictions are frozen when preparing; labels never modify scouting results.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, FileResponse
from filelock import FileLock
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autoscout import config

DIRECTORY = config.DATA_DIR / 'shot_review_2026cc'
MANIFEST = DIRECTORY / 'manifest.json'
LABELS = DIRECTORY / 'labels.json'
app = FastAPI()


def prepare():
    from scripts.shot_truth import load
    from autoscout.games.rebuilt2026 import Rebuilt2026
    from autoscout.vision.trajectories import Launch
    from autoscout.tba import TBA, MatchInfo
    if MANIFEST.exists():
        raise SystemExit('Review set already exists; preserving frozen predictions and labels.')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, count in [('2026cc_qm62', 34), ('2026cc_qm10', 33), ('2026cc_qm28', 33)]:
        d, ctx = load(key)
        ctx.match = MatchInfo.from_tba(TBA(max_age_s=1e12).match(key))
        flights = [Launch(t0=l['t0'], t1=l['t1'], start=l['start'], end=l['end'], apex=l['apex'],
                          n_pieces=l['n'], rises=l['start'][1]-l['apex'][1], pts=l.get('pts', []),
                          target=l.get('target'), rising=l.get('rising', True)) for l in d['launches']['fuel']]
        plugin = Rebuilt2026()
        hz = plugin.hub_top_zone(ctx)
        links = plugin.link_hub_flights(ctx, flights, plugin.history_shooters(ctx, flights, hz), hz)
        candidates = [(i,l) for i,l in enumerate(flights) if l.target in ('red_hub', 'blue_hub') and l.n_pieces == 1]
        random.Random(2026).shuffle(candidates)
        for i, l in sorted(candidates[:count]):
            slot = links.get(id(l))
            boxes = []
            for s in ctx.tracks:
                rb = plugin.robot_box(ctx, s, l.t0)
                if rb:
                    boxes.append(dict(slot=s, box=list(rb[0]), observed=bool(rb[1])))
            rows.append(dict(id=f'{key}:{i}', match=key, t=l.t0, t1=l.t1, video=d['video'],
                             teams=d['red'] + d['blue'], boxes=boxes, pts=l.pts,
                             start=l.start, prediction_slot=slot,
                             prediction_team=ctx.team_at(slot, l.t0) if slot else None,
                             source_sha256=hashlib.sha256((config.DATA_DIR/'results'/f'{key}.json').read_bytes()).hexdigest()))
    MANIFEST.write_text(json.dumps(rows, indent=2))
    print(f'Prepared {len(rows)} single-piece tracked hub flights in {DIRECTORY}')


def items():
    if not MANIFEST.exists():
        raise HTTPException(404, 'Run with --prepare first')
    return json.loads(MANIFEST.read_text())


@app.get('/api/items')
def listing():
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    return dict(items=[{k:v for k,v in r.items() if not k.startswith('prediction') and k not in ('source_sha256','pts')}
                       for r in items()], labels=labels)


class Label(BaseModel):
    id: str
    team: int | None = None
    slot: str | None = None
    status: str = 'labeled'


@app.post('/api/label')
def save(label: Label):
    row = next((r for r in items() if r['id'] == label.id), None)
    if row is None or label.status not in ('labeled','unknown','not_make'):
        raise HTTPException(400, 'Invalid shot or status')
    if label.status == 'labeled' and label.team not in row['teams']:
        raise HTTPException(400, 'Choose a team from this match')
    if label.slot is not None and label.slot not in {b['slot'] for b in row['boxes']}:
        raise HTTPException(400, 'Invalid robot box')
    with FileLock(str(LABELS)+'.lock'):
        labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
        labels[label.id] = label.model_dump()
        temp = LABELS.with_suffix('.tmp')
        temp.write_text(json.dumps(labels, indent=2))
        temp.replace(LABELS)
    return {'ok': True}


@app.get('/api/summary')
def summary():
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    judged = [(r, labels[r['id']]) for r in items() if labels.get(r['id'], {}).get('status') == 'labeled']
    physical = [(r,l) for r,l in judged if l.get('slot')]
    return dict(comparison='Frozen predictions from review-set preparation',
                total=len(items()), reviewed=len(labels), team_labels=len(judged),
                team_correct=sum(r['prediction_team']==l['team'] for r,l in judged),
                box_labels=len(physical), box_correct=sum(r['prediction_slot']==l['slot'] for r,l in physical),
                scope='Single-piece tracked hub flights only; does not measure missed balls or detection recall.')


@app.get('/api/frame/{index}')
def frame(index: int, dt: float = 0):
    rows = items()
    if index < 0 or index >= len(rows) or not -3 <= dt <= 3:
        raise HTTPException(400, 'Invalid frame')
    r = rows[index]
    path = next((p for ext in ('mp4','mkv','webm') if (p := config.DATA_DIR/'videos'/f"{r['video']}.{ext}").exists()), None)
    if path is None:
        raise HTTPException(404, 'Cached video unavailable')
    from autoscout.video import Video
    f = Video(str(path)).frame_at(max(0, r['t'] + dt))
    if f is None:
        raise HTTPException(404, 'Frame unavailable')
    # Slot labels intentionally hide automatic team identity to avoid priming the reviewer.
    if dt == 0:
        for b in r['boxes']:
            x0,y0,x1,y1 = map(int,b['box'])
            color = (50,80,255) if b['slot'].startswith('red') else (255,160,30)
            cv2.rectangle(f,(x0,y0),(x1,y1),color,2)
            cv2.putText(f,b['slot'],(x0,y0-5),0,.6,color,2)
        for _,x,y in r['pts']:
            cv2.circle(f,(int(x),int(y)),3,(0,255,255),-1)
        cv2.circle(f,tuple(map(int,r['start'])),10,(0,255,255),2)
    ok, encoded = cv2.imencode('.jpg', f)
    if not ok:
        raise HTTPException(500, 'Could not encode frame')
    return Response(encoded.tobytes(), media_type='image/jpeg')


@app.get('/video/{index}')
def video(index: int):
    rows = items()
    if not 0 <= index < len(rows):
        raise HTTPException(400, 'Invalid shot')
    r = rows[index]
    path = config.DATA_DIR/'videos'/f"{r['video']}.mp4"
    if not path.exists():
        raise HTTPException(404, 'Video unavailable')
    return FileResponse(path, media_type='video/mp4')


@app.get('/', response_class=HTMLResponse)
def page():
    return (ROOT/'autoscout/web/static/shot_review.html').read_text(encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    args = ap.parse_args()
    if args.prepare:
        prepare()
    else:
        import uvicorn
        uvicorn.run(app, host='127.0.0.1', port=8001)
