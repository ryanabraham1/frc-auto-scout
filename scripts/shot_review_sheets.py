"""Render blinded evidence sheets for visual review (never displays predictions)."""
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/shot_review_2026cc'


def main(manifest=BASE/'manifest.json', output=BASE):
    rows = json.loads(Path(manifest).read_text())
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    caps = {}
    for batch in range(0, len(rows), 4):
        tiles = []
        for index in range(batch, min(batch+4, len(rows))):
            r = rows[index]
            if r['video'] not in caps:
                caps[r['video']]=cv2.VideoCapture(str(ROOT/'data/videos'/f"{r['video']}.mp4"))
            cap=caps[r['video']]
            panels = []
            # The hub's broadcast pane gives readable bumpers; wide view gives the release.
            # Select hub from trajectory location, never from automatic shooter/team predictions.
            hub_left = r['start'][0] < 960
            for dt in (-.6, -.25, 0):
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0,r['t']+dt)*1000)
                ok,f = cap.read()
                if not ok:
                    f=np.zeros((1080,1920,3),np.uint8)
                if dt == 0:
                    for b in r['boxes']:
                        x0,y0,x1,y1=map(int,b['box'])
                        cv2.rectangle(f,(x0,y0),(x1,y1),(255,255,255),2)
                        cv2.putText(f,b['slot'],(x0,y0-4),0,.6,(255,255,255),2)
                    for _,x,y in r['pts']:
                        cv2.circle(f,(int(x),int(y)),4,(0,255,255),-1)
                    cv2.circle(f,tuple(map(int,r['start'])),11,(0,255,255),2)
                wide=cv2.resize(f[290:642],(768,141))
                close=cv2.resize(f[645:1080,0:955] if hub_left else f[645:1080,960:1920],(768,348))
                panel=np.vstack([wide,close])
                cv2.rectangle(panel,(0,0),(768,28),(0,0,0),-1)
                cv2.putText(panel,f"#{index} {r['match']} {r['t']+dt:.2f}s ({dt:+.2f})",(5,21),0,.6,(255,255,255),1)
                panels.append(panel)
            tiles.append(np.hstack(panels))
        cv2.imwrite(str(output/f'sheet_{batch:03}.jpg'),np.vstack(tiles))
    for cap in caps.values():
        cap.release()


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,default=BASE/'manifest.json')
    ap.add_argument('--out',type=Path,default=BASE)
    args=ap.parse_args()
    main(args.manifest,args.out)
