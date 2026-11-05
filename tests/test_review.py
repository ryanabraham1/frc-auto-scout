"""Blind scorer review: candidate extraction, label storage and accuracy against labels."""
import json

import pytest
from fastapi import HTTPException

from autoscout import review
from autoscout.web import app as web


def result_2025():
    return {"match_key": "2025test_qm1", "event_key": "2025test", "year": 2025, "video": "vid",
            "red": [1, 2, 3], "blue": [4, 5, 6], "events": [
                {"t": 10.0, "alliance": "red", "kind": "L4", "points": 7, "shares": {"1": .7, "2": .3}},
                {"t": 10.0, "alliance": "red", "kind": "coral_seen", "shares": {"2": 1.0}},
                {"t": 20.0, "alliance": "blue", "kind": "net", "points": 4, "shares": {"4": .5, "5": .5}},
                {"t": 30.0, "alliance": "blue", "kind": "unexplained", "points": 2},
                {"t": 40.0, "alliance": "red", "kind": "net_shot_seen", "team": 3}]}


def result_2026():
    return {"match_key": "2026test_qm1", "event_key": "2026test", "year": 2026, "video": "vid",
            "red": [1, 2, 3], "blue": [4, 5, 6], "events": [
                {"t": 5.0, "alliance": "red", "kind": "shot", "team": 2, "n": 1, "start": [100, 200], "end": [300, 50]},
                {"t": 5.0, "alliance": "red", "kind": "shot", "team": 3, "n": 1, "start": [400, 200], "end": [300, 50]},
                {"t": 6.0, "alliance": "blue", "kind": "human_player", "n": 1, "start": [1, 2], "end": [3, 4]},
                {"t": 7.5, "alliance": "red", "kind": "scored", "team": 2, "n": 0.75, "period": "auto"},
                {"t": 7.5, "alliance": "red", "kind": "scored", "team": 3, "n": 0.25, "period": "auto"}]}


def test_2025_candidates_are_score_posts_only():
    cands = review.candidates(result_2025())
    assert sorted((c["alliance"], c["t"], c["kind"]) for c in cands) == [("blue", 20.0, "net"), ("red", 10.0, "L4")]
    by = {c["kind"]: c for c in cands}
    assert by["L4"]["predicted"] == 1
    assert by["net"]["predicted"] is None  # tied
    assert "predicted" not in review.blind(by["L4"]) and "shares" not in review.blind(by["L4"])


def test_2026_launches_and_grouped_score_increases():
    cands = review.candidates(result_2026())
    launches = [c for c in cands if c["action"] == "launch"]
    assert len(launches) == 3 and len({c["id"] for c in cands}) == len(cands)  # same-time shots get distinct ids
    assert next(c for c in launches if c["kind"] == "human_player")["predicted"] == "HP"
    score = next(c for c in cands if c["action"] == "score")
    assert score["shares"] == {"2": .75, "3": .25} and score["predicted"] == 2


def test_review_order_is_stable_and_not_chronological():
    r = result_2026()
    assert [c["id"] for c in review.candidates(r)] == [c["id"] for c in review.candidates(json.loads(json.dumps(r)))]


def test_save_label_validates_and_replaces(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "LABEL_DIR", tmp_path)
    r = result_2026()
    cands = {c["id"]: c for c in review.candidates(r)}
    launch = next(c for c in cands.values() if c["action"] == "launch" and c["alliance"] == "red")
    with pytest.raises(ValueError):
        review.save_label(r, launch, 4, "accepted")  # blue team on a red action
    review.save_label(r, launch, 1, "accepted")
    review.save_label(r, launch, "HP", "accepted")  # changing an answer replaces it
    score = next(c for c in cands.values() if c["action"] == "score")
    with pytest.raises(ValueError):
        review.save_label(r, score, "HP", "accepted")  # human player only for launches
    review.save_label(r, score, 2, "uncertain")
    doc = review.load_labels(r["match_key"])
    assert len(doc["labels"]) == 2
    assert {x["id"]: x["team"] for x in doc["labels"]} == {launch["id"]: "HP", score["id"]: None}
    review.delete_label(r["match_key"], score["id"])
    assert [x["id"] for x in review.load_labels(r["match_key"])["labels"]] == [launch["id"]]


def test_evaluate_counts_ties_and_missing_as_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "LABEL_DIR", tmp_path)
    r = result_2025()
    l4, net = sorted(review.candidates(r), key=lambda c: c["t"])
    review.save_label(r, l4, 1, "accepted")
    review.save_label(r, net, 4, "accepted")
    truth = review.load_labels(r["match_key"])
    truth["labels"].append({"id": "x", "alliance": "red", "post_t": 99.0, "team": 2, "status": "accepted"})
    truth["labels"].append({"id": "y", "alliance": "red", "post_t": 50.0, "team": None, "status": "rejected"})
    rep = review.evaluate(truth, r)
    assert rep["primary"]["accepted"] == 3 and rep["primary"]["correct"] == 1
    assert sorted(x["reason"] for x in rep["rows"]) == ["correct", "missing_prediction", "tied_top_credit"]


def test_matches_existing_pilot_evaluator():
    from scripts.manual_attribution_eval import evaluate as pilot

    truth = json.loads(open("benchmarks/attribution/2025casj_qm10.json").read())
    pred = json.loads(open("benchmarks/attribution/2025casj_qm10.predictions.json").read())
    assert review.evaluate(truth, pred)["primary"]["correct"] == pilot(truth, pred)["primary"]["correct"] == 5


def test_evaluate_all_pools_matches(tmp_path, monkeypatch):
    labels, results = tmp_path / "labels", tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(review, "LABEL_DIR", labels)
    for r in (result_2025(), result_2026()):
        (results / f"{r['match_key']}.json").write_text(json.dumps(r))
        for c in review.candidates(r):
            review.save_label(r, c, c["predicted"] if c["predicted"] is not None else None,
                              "accepted" if c["predicted"] is not None else "rejected")
    rep = review.evaluate_all(results)
    assert rep["primary"]["accuracy"] == 1.0 and rep["rejected"] == 1
    assert set(rep["by_action"]) == {"score", "launch"}
    assert review.evaluate_all(results, "2026test")["primary"]["accepted"] == 4


def test_review_api_hides_answers_until_labelled(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    r = result_2026()
    (results / f"{r['match_key']}.json").write_text(json.dumps(r))
    monkeypatch.setattr(web, "_results", lambda: results)
    monkeypatch.setattr(review, "LABEL_DIR", tmp_path / "labels")
    monkeypatch.setattr(web, "_review_video", lambda r: None)
    d = web.review_match(r["match_key"])
    assert d["candidates"] and all(c["reveal"] is None and "shares" not in c for c in d["candidates"])
    cid = d["candidates"][0]["id"]
    saved = web.review_save(r["match_key"], {"id": cid, "status": "uncertain"})
    assert saved["reveal"] is not None
    d = web.review_match(r["match_key"])
    assert d["labelled"] == 1 and next(c for c in d["candidates"] if c["id"] == cid)["reveal"] is not None
    with pytest.raises(HTTPException):
        web.review_save(r["match_key"], {"id": "nope", "status": "accepted"})
    with pytest.raises(HTTPException):
        web.review_match("../etc")
    assert web.review_index()[0]["labelled"] == 1
