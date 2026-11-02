"""Exact-equivalence checks for processing optimizations."""
import bisect
import copy
import itertools
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest
import torch

from autoscout.vision.bumper_ocr import _greedy_logprobs, _recognize_with_memory, _prep, _prep_many
from autoscout.vision.tracker import RobotTracker, Tracklet


def legacy_swap(self, chains, max_rounds=30, max_span=45.0):
    """Original exhaustive interval search, retained as an independent oracle."""
    def cost(team, chain):
        return self._chain_cost(chain) + sum(self._unary(x, team) for x in chain)
    for _ in range(max_rounds):
        best, best_gain = None, 1e-6
        for ta, tb in itertools.combinations(self.all_teams, 2):
            A, B = chains[ta], chains[tb]
            base = cost(ta, A) + cost(tb, B)
            ts = sorted({x.t0 for x in A + B}) + [float('inf')]
            for i, T1 in enumerate(ts[:-1]):
                j = bisect.bisect_right(ts, T1 + max_span, i + 1)
                for T2 in ts[i + 1:j] + ([ts[-1]] if j < len(ts) else []):
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
    self.stats['identity_swaps'] = _ if best is None else max_rounds


@pytest.mark.parametrize('seed', range(16))
def test_interval_search_preserves_exhaustive_result(seed):
    rng = np.random.default_rng(seed)
    tracker = RobotTracker(None, None, {'red': [1, 2, 3], 'blue': [4, 5, 6]})
    chains = {t: [] for t in tracker.all_teams}
    for tid in range(24):
        team = int(rng.integers(1, 7))
        start = float(rng.integers(0, 12))  # shared starts and empty chains
        duration = float(rng.choice([0.1, 0.15, 0.5, 2.0]))
        tr = Tracklet(tid, team=team, obs=[(start, float(rng.uniform(0, 16)), 2., (0, 0, 1, 1), 'red'),
                                         (start + duration, float(rng.uniform(0, 16)), 2., (0, 0, 1, 1), 'red')])
        tr.reads = {int(rng.integers(1, 7)): float(rng.uniform(0, 3))}
        chains[team].append(tr)
    for ch in chains.values():
        ch.sort(key=lambda x: x.t0)
    outputs = []
    for method in (legacy_swap, RobotTracker._swap_suffixes):
        candidate = copy.deepcopy(chains)
        tracker._uc, tracker._lc = {}, {}
        method(tracker, candidate, max_rounds=3, max_span=4.)
        outputs.append(({k: [t.id for t in v] for k, v in candidate.items()}, dict(tracker.stats)))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_greedy_sums_are_bit_exact(dtype):
    gen = torch.Generator().manual_seed(42)
    logits = torch.randn(12, 9, 8, generator=gen, dtype=dtype)
    logits[0, 0, 0] = 100  # immediate EOS
    logits[1, :, 0] = -100  # no EOS
    logits[2, -1, 0] = 100  # final EOS
    logits[3, :, 0] = 100  # repeated EOS
    greedy = logits.log_softmax(-1)
    g = greedy.max(-1)
    expected = []
    for b in range(len(logits)):
        e = (g.indices[b] == 0).nonzero()
        n = int(e[0]) + 1 if len(e) else g.indices.shape[1]
        expected.append(float(g.values[b, :n].sum()))
    assert _greedy_logprobs(greedy, 0) == expected


@pytest.mark.parametrize('count', [1, 8, 64, 256])
def test_batched_preprocessing_is_bit_exact(count):
    rng = np.random.default_rng(42)
    # Include narrow crops and noncontiguous views, as produced by bumper_windows.
    images = [rng.integers(0, 256, (13 + i % 30, 42 + i % 60, 3), dtype=np.uint8)[:, ::2]
              for i in range(count)]
    expected = torch.stack([_prep(c) for c in images])
    actual = _prep_many(images)
    assert actual.is_contiguous()
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


class Recognizer(torch.nn.Module):
    def __init__(self, barrier=None, fail=False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.encoder = torch.nn.Linear(3, 4)
        self.model.encode = self.model.encoder
        self.barrier, self.fail = barrier, fail

    def forward(self, x):
        if self.barrier:
            self.barrier.wait(timeout=5)
        mem = self.model.encode(x)
        if self.fail:
            raise ValueError('failed')
        return mem.square()


def test_encoder_reused_without_changing_logits():
    model = Recognizer().eval()
    x = torch.ones(2, 3)
    calls = []
    hook = model.model.encoder.register_forward_hook(lambda *args: calls.append(1))
    expected = model(x)
    calls.clear()
    logits, mem = _recognize_with_memory(model, x)
    assert len(calls) == 1
    assert torch.equal(logits, expected)
    assert torch.equal(mem, model.model.encode(x))
    hook.remove()
    assert not model.model.encoder._forward_hooks


def test_encoder_capture_isolated_between_threads():
    model = Recognizer(threading.Barrier(2)).eval()
    inputs = [torch.zeros(2, 3), torch.ones(2, 3)]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda x: _recognize_with_memory(model, x), inputs))
    for x, (logits, mem) in zip(inputs, results):
        assert torch.equal(mem, model.model.encode(x))
        assert torch.equal(logits, mem.square())
    assert not model.model.encoder._forward_hooks


def test_encoder_hook_removed_on_error():
    model = Recognizer(fail=True)
    with pytest.raises(ValueError, match='failed'):
        _recognize_with_memory(model, torch.ones(2, 3))
    assert not model.model.encoder._forward_hooks
