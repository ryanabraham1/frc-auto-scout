"""Team-labelled robot tracker.

An FRC match has exactly three robots per alliance, but a greedy online tracker swaps two robots
whenever they meet (same alliance robots often look alike to a colour histogram), and one swap
smears a robot's stats onto a teammate for the rest of the match. Since a whole match is
processed at once, tracking is done offline in two stages:

  1. while the video plays: link detections into short, *pure* tracklets in field coordinates,
     cutting them whenever two robots get close enough to be confused, and read bumper numbers
     on each tracklet (every READ_EVERY s),
  2. at the end (`finalize`): attach close-up camera reads to tracklets, then label every tracklet
     with a team (or none, for false detections) by minimising motion discontinuity per team,
     alliance-colour mismatch and bumper-read disagreement, with no team in two places at once.

The output keeps the old shape: one track per station slot ("red0" = red's first team).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Det

MAX_SPEED = 5.5  # m/s, generous upper bound for FRC robots


@dataclass
class Slot:
    """One station's robot: its track over the match as (t, x, y, observed, box) rows."""
    alliance: str
    index: int
    last_t: float = -1.0
    track: list = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.alliance}{self.index}"


READ_EVERY = 0.5  # s between bumper reads (and appearance crops) of one tracklet
APP_SIZE = 96  # px, appearance crop size
APP_MARGIN = 0.05  # similarity margin over the next team that counts as a confident match
APP_WEIGHT = 4.0  # cost scale of appearance evidence for a long tracklet
NOT_ROBOT = 0  # manual label: the tracklet is not a robot (a person, a cart...)
NEAR = 0.8  # m: two robots this close can be confused, so their tracklets are cut
HIDE_NEAR = 1.0  # m from a tall element's edge: a robot that vanishes this close is behind it
HIDE_HEIGHT = 1.2  # m: elements at least this tall can hide a robot from the wide shot
HIDE_MIN_GAP = 0.6  # s: shorter gaps are detector misses, not occlusion
TRAVEL_SPEED = 2.5  # m/s, typical robot speed when crossing a gap
_KEEP = object()


@dataclass
class Tracklet:
    id: int
    obs: list = field(default_factory=list)  # (t, x, y, box, alliance)
    reads: dict = field(default_factory=dict)  # team -> bumper evidence
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_read: float = -9.0
    team: int | None = None
    manual: int | None = None  # team a person labelled this robot as (NOT_ROBOT for a false detection)
    crops: list = field(default_factory=list, repr=False)  # robot crops for appearance (in memory only)
    thumb: np.ndarray | None = field(default=None, repr=False)  # largest view, for the labelling page
    thumb_area: int = 0
    emb: np.ndarray | None = field(default=None, repr=False)  # appearance feature (unit length)
    app: dict = field(default_factory=dict)  # team -> appearance evidence from the event bank

    @property
    def t0(self):
        return self.obs[0][0]

    @property
    def t1(self):
        return self.obs[-1][0]

    @property
    def p0(self):
        return np.array(self.obs[0][1:3])

    @property
    def p1(self):
        return np.array(self.obs[-1][1:3])

    def predicted(self, t):
        return self.p1 + self.vel * min(max(t - self.t1, 0.0), 0.5)

    def _vel(self, obs) -> np.ndarray:
        if len(obs) < 2 or obs[-1][0] - obs[0][0] < 1e-3:
            return np.zeros(2)
        v = (np.array(obs[-1][1:3]) - np.array(obs[0][1:3])) / (obs[-1][0] - obs[0][0])
        sp = np.linalg.norm(v)
        return v / sp * MAX_SPEED if sp > MAX_SPEED else v

    def end_vel(self) -> np.ndarray:
        """Velocity over the last ~1 s (m/s)."""
        return self._vel([o for o in self.obs if o[0] >= self.t1 - 1.0])

    def start_vel(self) -> np.ndarray:
        return self._vel([o for o in self.obs if o[0] <= self.t0 + 1.0])

    def alliance_frac(self, a: str) -> float:
        votes = [o[4] for o in self.obs if o[4]]
        return sum(v == a for v in votes) / len(votes) if votes else 0.5


class RobotTracker:
    def __init__(self, cal, field_model, teams: dict[str, list[int]]):
        self.cal = cal
        self.fm = field_model
        self.teams = teams
        self.all_teams = teams["red"] + teams["blue"]
        self.alliance_of = {t: a for a in ("red", "blue") for t in teams[a]}
        self.active: list[Tracklet] = []
        self.done: list[Tracklet] = []
        self.times: list[float] = []
        self._next = 0
        self.slots: dict[str, Slot] = {f"{a}{i}": Slot(a, i) for a in ("red", "blue") for i in range(3)}
        self.stats: dict = {}

    # ---- stage 1: tracklets ----------------------------------------------------------------
    def update(self, t: float, frame: np.ndarray, dets: list[Det]) -> None:
        from .identity import read_team_numbers

        self.times.append(t)
        foot = self.cal.to_field([d.foot for d in dets]) if dets else np.zeros((0, 2))
        keep = [i for i in range(len(dets)) if self.fm.in_field(foot[i], 0.6)]
        dets, foot = [dets[i] for i in keep], foot[keep] if keep else np.zeros((0, 2))
        live = [tr for tr in self.active if t - tr.t1 <= 0.6]
        for tr in self.active:
            if tr not in live:
                self.done.append(tr)
        used = set()
        if live and dets:
            cost = np.full((len(live), len(dets)), 1e6)
            for i, tr in enumerate(live):
                p = tr.predicted(t)
                gate = 0.5 + MAX_SPEED * (t - tr.t1)
                for j, d in enumerate(dets):
                    dist = float(np.linalg.norm(foot[j] - p))
                    if dist <= gate:
                        cost[i, j] = dist + (0.4 if d.alliance and tr.alliance_frac(d.alliance) < 0.3 else 0.0)
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e5:
                    self._add(live[i], t, dets[j], foot[j])
                    used.add(j)
        for j in range(len(dets)):
            if j not in used:
                tr = Tracklet(self._next)
                self._next += 1
                self._add(tr, t, dets[j], foot[j])
                live.append(tr)
        # cut tracklets of robots close enough to be confused: each side starts fresh
        cur = [tr for tr in live if tr.t1 == t]
        cut = set()
        for a, b in itertools.combinations(cur, 2):
            if np.linalg.norm(a.p1 - b.p1) < NEAR:
                cut.update((a.id, b.id))
        self.active = []
        for tr in live:
            if tr.id in cut and len(tr.obs) > 1:
                last = tr.obs.pop()
                self.done.append(tr)
                nt = Tracklet(self._next)
                self._next += 1
                nt.obs.append(last)
                self.active.append(nt)
            else:
                self.active.append(tr)
        # bumper reads
        for tr in self.active:
            o = tr.obs[-1]
            if o[0] == t and t - tr.last_read >= READ_EVERY and o[3][2] - o[3][0] >= 24:
                tr.last_read = t
                for team, v in read_team_numbers(frame, o[3], self.all_teams).items():
                    tr.reads[team] = tr.reads.get(team, 0.0) + v
                x0, y0, x1, y1 = [int(v) for v in o[3]]
                crop = frame[max(0, y0): y1, max(0, x0): x1]
                if crop.size:
                    tr.crops.append(cv2.resize(crop, (APP_SIZE, APP_SIZE)))
                    if (x1 - x0) * (y1 - y0) > tr.thumb_area:
                        m = int(0.25 * (x1 - x0))
                        th = frame[max(0, y0 - m): y1 + m, max(0, x0 - m): x1 + m]
                        tr.thumb = cv2.resize(th, (max(1, int(th.shape[1] * 160 / th.shape[0])), 160))
                        tr.thumb_area = (x1 - x0) * (y1 - y0)

    def _add(self, tr: Tracklet, t, d: Det, p):
        if tr.obs:
            dt = max(t - tr.t1, 1e-3)
            v = (np.asarray(p) - tr.p1) / dt
            sp = np.linalg.norm(v)
            if sp > MAX_SPEED:
                v = v / sp * MAX_SPEED
            tr.vel = 0.6 * tr.vel + 0.4 * v
        tr.obs.append((t, float(p[0]), float(p[1]),
                       (round(d.x0, 1), round(d.y0, 1), round(d.x1, 1), round(d.y1, 1)), d.alliance))

    def live_boxes(self, t: float):
        """(tracklet id, alliance, box) seen at t, for the debug video."""
        out = []
        for tr in self.active:
            o = tr.obs[-1]
            if t - o[0] < 0.5:
                out.append((tr.id, o[4] or "", o[3]))
        return out

    # ---- stage 2: labelling -------------------------------------------------------------------
    def add_closeup_reads(self, aux_reads, dist_to_element, max_dist: float = 3.0) -> None:
        """A team read in a close-up of a field element is the robot of its alliance nearest that
        element at that moment (a defender is read in the opponent's close-up the same way)."""
        trs = self.done + self.active
        n = 0
        for t, view, team, conf in aux_reads:
            a = self.alliance_of.get(team)
            if a is None:
                continue
            cand = []
            for tr in trs:
                if tr.t0 - 0.3 <= t <= tr.t1 + 0.3 and tr.alliance_frac(a) >= 0.5:
                    o = min(tr.obs, key=lambda o: abs(o[0] - t))
                    d = dist_to_element((o[1], o[2]), view)
                    if d is not None:
                        cand.append((d, tr))
            cand.sort(key=lambda x: x[0])
            if not cand or cand[0][0] > max_dist:
                continue
            margin = cand[1][0] - cand[0][0] if len(cand) > 1 else max_dist
            if margin < 0.5:
                continue
            cand[0][1].reads[team] = cand[0][1].reads.get(team, 0.0) + 0.8 * conf * min(1.0, margin / 2)
            n += 1
        self.stats["closeup_reads_used"] = n

    def _unary(self, tr: Tracklet, team: int | None) -> float:
        n = len(tr.obs)
        if tr.manual is not None:  # a person said who this is
            return 0.0 if team == (None if tr.manual == NOT_ROBOT else tr.manual) else 1000.0
        if team is None:
            return 0.6 * n + 2.0 * sum(tr.reads.values())  # nearly every in-field detection is a robot
        a = self.alliance_of[team]
        c = 1.5 * n * max(0.0, 0.5 - tr.alliance_frac(a)) * 2  # wrong bumper colour
        c -= 3.0 * tr.reads.get(team, 0.0)
        c += 1.5 * sum(v for k, v in tr.reads.items() if k != team)  # it read as someone else
        c -= APP_WEIGHT * min(n, 30) / 30 * tr.app.get(team, 0.0)  # looks like that team's robot
        return c

    @staticmethod
    def _link(a: Tracklet, b: Tracklet) -> float:
        """Cost of b being the same robot as a, later: both are extrapolated with their own velocity
        toward the middle of the gap (so two robots crossing are not swapped), and the mismatch is
        compared with how much a robot can change course in that time."""
        gap = max(b.t0 - a.t1, 0.0)
        dist = float(np.linalg.norm(b.p0 - a.p1))
        reach = 0.6 + MAX_SPEED * min(gap, 8.0)
        if dist > reach:
            return 6.0 + 0.5 * dist  # impossible jump: tracklet ends can be impure, so evidence can still win
        h = min(gap / 2, 1.0)
        va, vb = a.end_vel(), b.start_vel()
        d = float(np.linalg.norm((a.p1 + va * h) - (b.p0 - vb * h)))
        dv = float(np.linalg.norm(va - vb))  # an abrupt reversal is less likely than carrying on
        return (min(3.0 * (d / (0.4 + 1.5 * gap)) ** 2, 25.0) + min((dv / (1.0 + 2.0 * gap)) ** 2, 5.0)
                + 0.05 * min(gap, 20.0))

    def _birth(self, tr: Tracklet) -> float:
        """Robots are on the field from the start: a team first seen late is unlikely."""
        t_start = self.times[0] if self.times else 0.0
        return 0.3 * min(max(tr.t0 - t_start, 0.0), 20.0)

    def _chain_cost(self, chain: list[Tracklet]) -> float:
        if not chain:
            return 0.0
        return self._birth(chain[0]) + sum(self._link(a, b) for a, b in zip(chain, chain[1:]))

    def _fits(self, chain: list[Tracklet], tr: Tracklet) -> bool:
        return all(o is tr or not _overlap(o, tr) for o in chain)

    def label(self, iters: int = 6) -> None:
        trs = sorted([t for t in self.done + self.active if len(t.obs) >= 2], key=lambda t: t.t0)
        chains: dict[int, list[Tracklet]] = {team: [] for team in self.all_teams}
        last: dict[int, Tracklet | None] = {team: None for team in self.all_teams}
        # greedy forward pass: each tracklet to the team that continues it best
        for tr in trs:
            best, best_c = None, self._unary(tr, None)
            for team in self.all_teams:
                if not self._fits(chains[team][-3:], tr):
                    continue
                c = self._unary(tr, team) + (self._link(last[team], tr) if last[team] else self._birth(tr))
                if c < best_c:
                    best, best_c = team, c
            tr.team = best
            if best is not None:
                chains[best].append(tr)
                last[best] = tr
        # local search: move a tracklet to another team (or none), or swap it with the tracklets
        # that team has at the same time, whenever that lowers the total cost
        for _ in range(iters):
            changed = 0
            for tr in trs:
                cur = tr.team
                base = self._unary(tr, cur) + (self._chain_cost(chains[cur]) if cur is not None else 0.0)
                best_move, best_gain = _KEEP, 1e-6
                for team in [None] + self.all_teams:
                    if team == cur:
                        continue
                    if team is None:
                        new = self._unary(tr, None) + (self._chain_cost([x for x in chains[cur] if x is not tr])
                                                         if cur is not None else 0.0)
                        old = base
                    else:
                        clash = [o for o in chains[team] if _overlap(o, tr)]
                        # tr -> team; the clashing tracklets -> cur (or none)
                        if cur is not None and any(not self._fits([x for x in chains[cur] if x is not tr], o)
                                                   for o in clash):
                            continue
                        new_team = sorted([x for x in chains[team] if x not in clash] + [tr], key=lambda x: x.t0)
                        new_cur = (sorted([x for x in chains[cur] if x is not tr] + clash, key=lambda x: x.t0)
                                   if cur is not None else None)
                        old = base + self._chain_cost(chains[team]) + sum(self._unary(o, team) for o in clash)
                        new = (self._unary(tr, team) + self._chain_cost(new_team)
                               + sum(self._unary(o, cur) for o in clash)
                               + (self._chain_cost(new_cur) if new_cur is not None else 0.0))
                    if old - new > best_gain:
                        best_move, best_gain = team, old - new
                if best_move is _KEEP:
                    continue
                team = best_move
                if cur is not None:
                    chains[cur] = [x for x in chains[cur] if x is not tr]
                if team is not None:
                    clash = [o for o in chains[team] if _overlap(o, tr)]
                    chains[team] = sorted([x for x in chains[team] if x not in clash] + [tr], key=lambda x: x.t0)
                    for o in clash:
                        o.team = cur
                        if cur is not None:
                            chains[cur].append(o)
                    if cur is not None:
                        chains[cur].sort(key=lambda x: x.t0)
                tr.team = team
                changed += 1
            if not changed:
                break
        self._swap_suffixes(chains)
        self.chains = chains
        n_read = {team: sum(1 for x in chains[team] if x.reads.get(team, 0) > 0) for team in chains}
        self.stats.update(tracklets=len(trs), dropped=sum(t.team is None for t in trs), anchored=n_read)

    def embed(self, model=None) -> None:
        """Appearance feature of every tracklet (mean over its crops); frees the crops."""
        from .appearance import features, mean_unit

        trs = [t for t in self.done + self.active if t.crops]
        crops = [c for t in trs for c in t.crops]
        if not crops:
            return
        F = features(crops, model)
        k = 0
        for t in trs:
            t.emb = mean_unit(F[k: k + len(t.crops)])
            k += len(t.crops)
            t.crops = []

    def add_appearance(self, bank: dict) -> None:
        """Evidence from what each team looked like in this event's other labelled matches: how much
        closer a tracklet is to one team's examples than to the best other team of its alliance."""
        from .appearance import similarity

        n = 0
        for tr in self.done + self.active:
            tr.app = {}
            if tr.emb is None:
                continue
            sims = similarity(tr.emb, bank, self.all_teams)
            for a in ("red", "blue"):
                cand = {t: v for t, v in sims.items() if t in self.teams[a]}
                if len(cand) < 2:
                    continue
                for t, v in cand.items():
                    other = max(u for k, u in cand.items() if k != t)
                    tr.app[t] = float(np.clip((v - other) / APP_MARGIN, -1.0, 1.0))
                n += 1
        self.stats["appearance_scored"] = n

    def _swap_suffixes(self, chains: dict, max_rounds: int = 30, max_span: float = 45.0) -> None:
        """Repair identity switches: the tracker sometimes hands robot A's track to robot B from
        some moment on. Moving one tracklet back would break motion continuity in both teams'
        chains, so try exchanging the two teams' tracklets over a stretch of time (or their whole
        futures) instead, and keep the exchange that lowers the total cost most (repeat until none
        does)."""
        def cost(team, chain):
            return self._chain_cost(chain) + sum(self._unary(x, team) for x in chain)

        for _ in range(max_rounds):
            best, best_gain = None, 1e-6
            for ta, tb in itertools.combinations(self.all_teams, 2):
                A, B = chains[ta], chains[tb]
                base = cost(ta, A) + cost(tb, B)
                ts = sorted({x.t0 for x in A + B}) + [float("inf")]
                # swap the stretch [T1, T2): T2 = inf is a swap of the whole future
                for i, T1 in enumerate(ts[:-1]):
                    for T2 in ts[i + 1:]:
                        if T2 - T1 > max_span and T2 != float("inf"):
                            continue
                        na = [x for x in A if not T1 <= x.t0 < T2] + [x for x in B if T1 <= x.t0 < T2]
                        nb = [x for x in B if not T1 <= x.t0 < T2] + [x for x in A if T1 <= x.t0 < T2]
                        na.sort(key=lambda x: x.t0)
                        nb.sort(key=lambda x: x.t0)
                        if not (self._disjoint(na) and self._disjoint(nb)):
                            continue
                        gain = base - cost(ta, na) - cost(tb, nb)
                        if gain > best_gain:
                            best, best_gain = (ta, tb, na, nb), gain
            if best is None:
                break
            ta, tb, na, nb = best
            chains[ta], chains[tb] = na, nb
            for x in na:
                x.team = ta
            for x in nb:
                x.team = tb
        self.stats["identity_swaps"] = _ if best is None else max_rounds

    @staticmethod
    def _disjoint(chain: list) -> bool:
        return all(not _overlap(a, b) for a, b in zip(chain, chain[1:]))

    def finalize(self, aux_reads=(), dist_to_element=None, bank=None, manual=None) -> None:
        """Label tracklets with teams and write per-station slot tracks.

        bank: per-event appearance examples {team: features}; manual: {tracklet id: team or NOT_ROBOT}.
        """
        if dist_to_element is not None:
            self.add_closeup_reads(aux_reads, dist_to_element)
        self.relabel(bank, manual)

    def relabel(self, bank=None, manual=None) -> None:
        for tr in self.done + self.active:
            tr.manual = (manual or {}).get(tr.id)
        if bank:
            self.add_appearance(bank)
        self.label()
        for a in ("red", "blue"):
            for i, team in enumerate(self.teams[a]):
                s = self.slots[f"{a}{i}"]
                obs = sorted(o for tr in self.chains.get(team, []) for o in tr.obs)
                s.track = _fill(obs, self.times)
                if obs:
                    s.last_t = obs[-1][0]

    def _hidden_at(self, p) -> bool:
        """Can a robot at field point p be hidden behind a tall field element?"""
        if self.fm is None:
            return False
        return any(e.get("height", 0.0) >= HIDE_HEIGHT and self.fm.dist_to_element(p, name) <= HIDE_NEAR
                   for name, e in self.fm.elements.items())

    # ---- persistence: relabelling after a person labels robots needs no video --------------------
    def to_dict(self) -> dict:
        return {"teams": self.teams, "times": [round(t, 3) for t in self.times],
                "tracklets": [{"id": t.id, "obs": [(round(o[0], 3), round(o[1], 3), round(o[2], 3), o[3], o[4])
                                                   for o in t.obs],
                               "reads": {str(k): round(v, 3) for k, v in t.reads.items()}, "team": t.team,
                               "emb": None if t.emb is None else [round(float(x), 4) for x in t.emb]}
                              for t in self.done + self.active]}

    @classmethod
    def from_dict(cls, d: dict, cal=None, field_model=None) -> "RobotTracker":
        tr = cls(cal, field_model, {a: [int(t) for t in d["teams"][a]] for a in ("red", "blue")})
        tr.times = list(d["times"])
        for x in d["tracklets"]:
            t = Tracklet(x["id"], obs=[tuple(o[:3]) + (tuple(o[3]), o[4]) for o in x["obs"]],
                         reads={int(k): v for k, v in x["reads"].items()}, team=x.get("team"))
            t.emb = None if x["emb"] is None else np.array(x["emb"], np.float32)
            tr.done.append(t)
        return tr

    def save_thumbs(self, out_dir) -> None:
        from pathlib import Path

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for t in self.done + self.active:
            if t.thumb is not None and len(t.obs) >= 4:
                cv2.imwrite(str(out_dir / f"{t.id}.jpg"), t.thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def to_ask(self, n: int = 16) -> list[int]:
        """Tracklets most worth a person's label: long ones with no bumper read, preferring robots
        and times the labels so far cover least."""
        cand = [t for t in self.done + self.active if len(t.obs) >= 8 and t.manual is None]
        cand.sort(key=lambda t: -len(t.obs) / (1.0 + 2.0 * sum(t.reads.values())))
        # take turns between the robots as currently guessed, so every team gets checked
        queues: dict = {}
        for t in cand:
            queues.setdefault(t.team, []).append(t)
        picked: list[Tracklet] = []
        while len(picked) < n and any(queues.values()):
            for q in queues.values():
                while q and len(picked) < n:
                    t = q.pop(0)
                    # skip a tracklet that is the same robot as one already picked moments earlier
                    if not any(abs(t.t0 - p.t1) < 2.0 and np.linalg.norm(t.p0 - p.p1) < 1.0 for p in picked):
                        picked.append(t)
                        break
        return [t.id for t in sorted(picked, key=lambda t: t.t0)]

    def confidence(self, team: int) -> float:
        """Share of the robot's observed time that sits in tracklets confirmed by a bumper read."""
        chain = self.chains.get(team, [])
        tot = sum(len(x.obs) for x in chain)
        ok = sum(len(x.obs) for x in chain if x.reads.get(team, 0) > 0)
        return round(ok / tot, 2) if tot else 0.0

    def all_slots(self) -> list[Slot]:
        return list(self.slots.values())


def _overlap(a: Tracklet, b: Tracklet, tol: float = 0.15) -> bool:
    """Seen at the same time (beyond a shared boundary frame, which is a detection artifact)."""
    return a.t1 > b.t0 + tol and b.t1 > a.t0 + tol


def _fill(obs: list, times: list[float], hidden_at=None) -> list:
    """Track rows (t, x, y, observed, box) at every update time: observations where the robot was
    seen, interpolation across gaps, and held positions before/after.

    hidden_at(p) says whether a robot at field point p can be hidden from the wide shot by a tall
    element (a reef, a hub). A robot that vanishes there is almost always parked behind it scoring,
    so it is held where it vanished and only travels (at a typical speed) in time to reappear, and
    one that reappears there is moved there first and held; other gaps are interpolated linearly.
    """
    if not obs:
        return []
    ot = np.array([o[0] for o in obs])
    out = []
    k = 0
    for t in times:
        while k + 1 < len(obs) and ot[k + 1] <= t:
            k += 1
        o = obs[k]
        if o[0] == t:
            out.append((t, o[1], o[2], True, o[3]))
        elif t < ot[0]:
            out.append((t, obs[0][1], obs[0][2], False, None))
        elif k + 1 < len(obs):
            n = obs[k + 1]
            gap = max(n[0] - o[0], 1e-6)
            f = (t - o[0]) / gap
            if hidden_at is not None and gap > HIDE_MIN_GAP:
                travel = min(np.hypot(n[1] - o[1], n[2] - o[2]) / TRAVEL_SPEED, gap)
                if hidden_at(o[1:3]):  # stay, then leave just in time
                    f = max(0.0, (t - (n[0] - travel)) / max(travel, 1e-6))
                elif hidden_at(n[1:3]):  # arrive early, then stay
                    f = min(1.0, (t - o[0]) / max(travel, 1e-6))
            out.append((t, o[1] + f * (n[1] - o[1]), o[2] + f * (n[2] - o[2]), False, None))
        else:
            out.append((t, o[1], o[2], False, None))
    return out


def slot_position_at(track, t, ts: np.ndarray | None = None):
    """(x, y, observed, box) of a slot track nearest in time to t (None if no data).
    `ts`: the track's times as an array, when the caller has them (much faster in loops)."""
    if not track:
        return None
    if ts is None:
        ts = np.array([r[0] for r in track])
    i = int(np.argmin(np.abs(ts - t)))
    if abs(ts[i] - t) > 1.0:
        return None
    r = track[i]
    # Use the most recent observed box within 1 s for image-space association.
    box = None
    for k in range(i, max(-1, i - 15), -1):
        if track[k][3] and t - track[k][0] < 1.0:
            box = track[k][4]
            break
    return r[1], r[2], r[3], box
