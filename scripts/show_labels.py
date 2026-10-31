"""Grid of dataset images with their label boxes (red/blue), for checking labels by eye.

usage: python scripts/show_labels.py OUT.jpg [glob=*] [n=12] [seed=0]
"""
import random
import sys

import cv2
import numpy as np

from autoscout.train.sam_label import DATASET

out = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else "*"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
imgs = sorted((DATASET / "images").glob(f"{pat}.jpg"))
random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 0).shuffle(imgs)
tiles = []
for p in imgs[:n]:
    im = cv2.imread(str(p))
    H, W = im.shape[:2]
    lab = DATASET / "labels" / f"{p.stem}.txt"
    for ln in (lab.read_text().splitlines() if lab.exists() else []):
        c, x, y, w, h = map(float, ln.split())
        col = (0, 0, 255) if int(c) == 0 else (255, 0, 0)
        cv2.rectangle(im, (int((x - w / 2) * W), int((y - h / 2) * H)), (int((x + w / 2) * W), int((y + h / 2) * H)), col, 3)
    cv2.putText(im, p.stem, (10, 40), 0, 1.2, (0, 255, 255), 3)
    tiles.append(cv2.resize(im, (640, 360)))
while len(tiles) % 3:
    tiles.append(np.zeros((360, 640, 3), np.uint8))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
