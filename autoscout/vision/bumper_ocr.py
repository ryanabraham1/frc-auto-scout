"""Bumper-number identification by *scoring the match's team numbers* against a robot crop.

Free-form OCR on a 20 px bumper must get every digit right and has no idea which strings are
possible. In a match only six team numbers are, so instead every candidate is scored with a
scene-text recognizer (PARSeq, teacher-forced): log P(text = "6696" | window) for windows slid
along the bumper strip. The best candidate's margin over the recognizer's own unconstrained read
and over the runner-up candidate says how sure the read is. This works on partial / blurred
digits that free OCR returns as "66" or "G696", and almost never invents a team.

The recognizer is downloaded once through torch.hub (baudm/parseq, ~90 MB) and runs on the GPU.
"""
from __future__ import annotations

import functools
import os

import cv2
import numpy as np

H_IN, W_IN = 32, 128  # PARSeq input size


def _ssl_certs():
    """The Windows-store Python ships without CA roots for urllib: use certifi's for downloads."""
    if not os.environ.get("SSL_CERT_FILE"):
        try:
            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass


@functools.lru_cache(maxsize=1)
def model():
    import torch

    from .. import config

    _ssl_certs()
    m = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True, verbose=False).eval()
    dev = "cuda" if config.device() == "cuda" else "cpu"
    return m.to(dev), dev


def _prep(img: np.ndarray):
    import torch

    img = cv2.cvtColor(cv2.resize(img, (W_IN, H_IN), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return (x - 0.5) / 0.5


def bumper_windows(frame: np.ndarray, box, min_h: int = 7) -> list[np.ndarray]:
    """Crops of a robot box's bumper strip where a number can be: the whole strip and windows
    about as wide as a 3-5 digit number at the strip's height, slid across it."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    w, h = x1 - x0, y1 - y0
    if w < 12 or h < 8:
        return []
    H, W = frame.shape[:2]
    ya, yb = max(0, y0 + int(0.45 * h)), min(H, y1 + int(0.08 * h))
    xa, xb = max(0, x0 - int(0.06 * w)), min(W, x1 + int(0.06 * w))
    strip = frame[ya:yb, xa:xb]
    if strip.shape[0] < min_h or strip.shape[1] < 2 * min_h:
        return []
    out = [strip]
    sh, sw = strip.shape[:2]
    # digits are ~0.55 of the strip height wide; a number with margins ~ 3.5 strip heights
    for k in (2.6, 3.8):
        ww = int(min(sw, k * sh))
        if ww >= sw * 0.9:
            continue
        n = max(2, int(np.ceil((sw - ww) / (0.35 * ww))) + 1)
        for s in np.linspace(0, sw - ww, n):
            out.append(strip[:, int(s): int(s) + ww])
    # the number often sits in the upper or lower half of a tall strip (box includes the floor)
    if sh > 26:
        out += [strip[: int(0.7 * sh)], strip[int(0.3 * sh):]]
    return out


class BumperReader:
    def __init__(self, batch: int = 256):
        self.batch = batch

    def score(self, crops: list[np.ndarray], candidates: list[str]):
        """(n_crops, n_cand) teacher-forced log-probabilities of each candidate string in each crop,
        and (n_crops,) log-probability of the recognizer's own greedy read."""
        import torch

        m, dev = model()
        tok = m.tokenizer
        if not crops:
            return np.zeros((0, len(candidates))), np.zeros(0)
        tgt = tok.encode(candidates, dev)  # (K, L): bos, chars, eos, pad
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        L = tgt_in.shape[1]
        mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=dev), 1)
        pad = (tgt_in == tok.pad_id) | (tgt_in == tok.eos_id)
        K = len(candidates)
        S, G = [], []
        with torch.no_grad():
            for i in range(0, len(crops), self.batch):
                x = torch.stack([_prep(c) for c in crops[i: i + self.batch]]).to(dev)
                greedy = m(x).log_softmax(-1)  # (B, T, C)
                g = greedy.max(-1)
                ids = g.indices
                # greedy read's log-prob up to and including its EOS
                lp = []
                for b in range(len(x)):
                    e = (ids[b] == tok.eos_id).nonzero()
                    n = int(e[0]) + 1 if len(e) else ids.shape[1]
                    lp.append(float(g.values[b, :n].sum()))
                G += lp
                mem = m.model.encode(x)  # (B, N, D)
                B = mem.shape[0]
                memk = mem.repeat_interleave(K, 0)
                tin = tgt_in.repeat(B, 1)
                tout = tgt_out.repeat(B, 1)
                out = m.model.decode(tin, memk, mask, pad.repeat(B, 1), tgt_query_mask=mask)
                logp = m.model.head(out).log_softmax(-1)
                is_pad = tout == tok.pad_id  # the head has no BOS/PAD outputs
                ll = logp.gather(-1, tout.masked_fill(is_pad, 0).unsqueeze(-1)).squeeze(-1)
                ll = ll.masked_fill(is_pad, 0.0).sum(-1)
                S.append(ll.view(B, K).float().cpu().numpy())
        return np.vstack(S), np.array(G)

    def read(self, frame: np.ndarray, box, teams: list[int]) -> dict[int, float]:
        """Evidence {team: weight} that the robot in `box` is one of `teams` (empty if unsure)."""
        return self.read_many([(frame, box)], teams)[0]

    def read_many(self, items: list[tuple[np.ndarray, tuple]], teams: list[int]) -> list[dict[int, float]]:
        cands = [str(t) for t in teams]
        wins, owner = [], []
        for k, (frame, box) in enumerate(items):
            ws = bumper_windows(frame, box)
            wins += ws
            owner += [k] * len(ws)
        out: list[dict[int, float]] = [{} for _ in items]
        if not wins:
            return out
        S, G = self.score(wins, cands)
        owner = np.array(owner)
        for k in range(len(items)):
            sel = owner == k
            if not sel.any():
                continue
            s = S[sel]  # (n_win, K)
            g = G[sel]
            best_w = s.max(0)  # per candidate, its best window
            order = np.argsort(-best_w)
            top, second = order[0], order[1] if len(order) > 1 else None
            # the window where the top candidate scores best: how far is it from the free read?
            w = int(np.argmax(s[:, top]))
            vs_free = s[w, top] - g[w]  # 0 when the free read is exactly the candidate
            vs_next = best_w[top] - (best_w[second] if second is not None else -30.0)
            p = float(np.exp(best_w[top] / max(len(cands[top]), 1)))  # per-character confidence
            if vs_next < 2.0 or vs_free < -4.0 or p < 0.25:
                continue
            out[k][teams[top]] = float(min(1.0, vs_next / 6.0) * min(1.0, p / 0.6))
        return out


    def read_text_regions(self, img: np.ndarray, teams: list[int]) -> list[tuple[int, float]]:
        """Team numbers in a larger image (e.g. a close-up camera pane): EasyOCR finds where text
        is, then each text region (padded) is scored against the teams. Returns [(team, weight)]."""
        from .. import ocr

        boxes = []
        H, W = img.shape[:2]
        for (x, y, w, h), _txt, _conf in ocr.read_text(img):
            if h < 6 or w < 6:
                continue
            px, py = int(0.15 * w) + 2, int(0.2 * h) + 2
            boxes.append(img[max(0, y - py): min(H, y + h + py), max(0, x - px): min(W, x + w + px)])
        if not boxes:
            return []
        cands = [str(t) for t in teams]
        S, G = self.score(boxes, cands)
        out = []
        for s, g in zip(S, G):
            k = int(np.argmax(s))
            nxt = np.partition(s, -2)[-2] if len(s) > 1 else -30.0
            p = float(np.exp(s[k] / len(cands[k])))
            if s[k] - nxt >= 2.0 and s[k] - g > -4.0 and p >= 0.3:
                out.append((teams[k], float(min(1.0, (s[k] - nxt) / 6.0) * min(1.0, p / 0.6))))
        return out


@functools.lru_cache(maxsize=1)
def reader() -> BumperReader:
    return BumperReader()
