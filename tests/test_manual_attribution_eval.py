from scripts.manual_attribution_eval import evaluate


def test_missing_and_tied_predictions_are_errors_not_silently_skipped():
    truth = {"match_key": "test", "reviewer_type": "ai_visual_review", "teams": {"red": [1, 2]},
             "labels": [{"id": str(i), "status": "accepted", "alliance": "red", "team": 1, "post_t": i}
                        for i in range(3)]}
    result = {"match_key": "test", "events": [
        {"t": 0, "alliance": "red", "shares": {"1": .8, "2": .2}},
        {"t": 1, "alliance": "red", "shares": {"1": .5, "2": .5}}]}
    scored = evaluate(truth, result)
    assert scored["primary"]["accepted"] == 3
    assert scored["primary"]["correct"] == 1
    assert [r["reason"] for r in scored["rows"]] == ["correct", "tied_top_credit", "missing_prediction"]


def test_uncertain_and_provisional_labels_reduce_coverage_not_accuracy():
    truth = {"match_key": "test", "reviewer_type": "ai_visual_review", "teams": {"red": [1]},
             "labels": [{"id": str(i), "status": status, "alliance": "red", "team": None, "post_t": i}
                        for i, status in enumerate(["uncertain", "provisional"])]}
    scored = evaluate(truth, {"match_key": "test", "events": []})
    assert scored["candidate_actions_reviewed"] == 2
    assert scored["coverage"] == 0
    assert scored["primary"]["accuracy"] is None
