"""usage: python scripts/fuel_link_view.py LINKS.json OUT.jpg KIND(ok|none|wrong_alliance) HOW(rising|at_hub|any) N
Renders N sample makes from fuel_link_diag output with robot boxes (qm62 paths are hard-coded)."""
import json, sys, cv2, numpy as np
sys.path.insert(0, ".")
rows = json.load(open(sys.argv[1])); kind, how, n = sys.argv[3], sys.argv[4], int(sys.argv[5])
r = json.load(open("data/results/2026cc_qm62.json"))
c = cv2.VideoCapture("data/videos/2ana-RAceoE.mp4"); fps = c.get(5)
sel = [x for x in rows if x["kind"] == kind and (how == "any" or (x["at_hub"] if how == "at_hub" else (x["rising"] and not x["at_hub"])))]
sel = sel[:: max(1, len(sel) // n)][:n]
tiles = []
for x in sel:
    t = x["t"]; c.set(1, int(round(t * fps))); ok, im = c.read()
    for slot, (bx, ob) in x.get("boxes", {}).items():
        x0, y0, x1, y1 = map(int, bx); col = (0, 0, 255) if slot.startswith("red") else (255, 0, 0)
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 2 if ob else 1)
        cv2.putText(im, slot + ("" if ob else "?"), (x0, y0 - 3), 0, 0.5, col, 1)
    p = np.array([(q[1], q[2]) for q in x["pts"]], np.int32)
    cv2.polylines(im, [p], False, (0, 255, 0), 2); cv2.circle(im, tuple(p[0]), 6, (0, 255, 255), 2)
    cx, cy = p[0]
    crop = im[max(0, cy - 230): cy + 160, max(0, cx - 300): cx + 300]
    crop = cv2.copyMakeBorder(crop, 0, 390 - crop.shape[0], 0, 600 - crop.shape[1], cv2.BORDER_CONSTANT)
    cv2.putText(crop, f"t={t:.1f} {x['a']} hub near={x['near'][:2]}", (5, 380), 0, 0.42, (255, 255, 255), 1)
    tiles.append(crop)
while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2], np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
