"""Score frozen visual-review labels, retaining missing predictions in the denominator.

python -m scripts.manual_attribution_eval --result-file data/results/2025casj_qm10.json
Labels in the initial pilot are AI-reviewed, not independently human-verified ground truth.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def evaluate(truth, result):
    if truth["match_key"] != result["match_key"]:
        raise ValueError("Label and result match keys differ")
    labels = truth["labels"]
    if len({label["id"] for label in labels}) != len(labels):
        raise ValueError("Duplicate label identifiers")
    rows = []
    for label in labels:
        if label["status"] != "accepted":
            continue
        alliance, team = label["alliance"], label["team"]
        if team is None or team not in truth["teams"][alliance]:
            raise ValueError(f"Accepted label has invalid team: {label['id']}")
        matches = [e for e in result.get("events", []) if e.get("shares")
                   and e.get("alliance") == alliance and abs(e["t"]-label["post_t"]) < .05]
        top, mass, reason = None, 0., "missing_prediction"
        if len(matches) > 1:
            reason = "ambiguous_prediction_alignment"
        elif matches:
            shares = {int(k): float(v) for k, v in matches[0]["shares"].items()}
            mass = shares.get(team, 0.)
            highest = max(shares.values())
            winners = [k for k, v in shares.items() if abs(v-highest) < 1e-9]
            if len(winners) == 1:
                top = winners[0]
                reason = "correct" if top == team else "wrong_team"
            else:
                reason = "tied_top_credit"
        rows.append({"id": label["id"], "alliance": alliance, "reviewed_team": team,
                     "predicted_team": top, "correct": reason == "correct",
                     "credit_to_reviewed_team": mass, "reason": reason})
    def summary(xs):
        n = len(xs)
        correct = sum(x["correct"] for x in xs)
        return {"accepted": n, "correct": correct, "accuracy": correct/n if n else None,
                "mean_credit_to_reviewed_team": sum(x["credit_to_reviewed_team"] for x in xs)/n if n else None}
    return {"match_key": truth["match_key"], "reviewer_type": truth["reviewer_type"],
            "scope": "Candidate-action scorer identity; not full-match scoring precision or recall",
            "candidate_actions_reviewed": len(labels), "label_statuses": dict(Counter(x["status"] for x in labels)),
            "coverage": len(rows)/len(labels) if labels else None, "primary": summary(rows),
            "alliances": {a: summary([x for x in rows if x["alliance"] == a]) for a in ("red", "blue")},
            "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, default=Path("benchmarks/attribution/2025casj_qm10.json"))
    ap.add_argument("--result-file", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    truth_bytes, result_bytes = a.truth.read_bytes(), a.result_file.read_bytes()
    report = evaluate(json.loads(truth_bytes), json.loads(result_bytes))
    report.update(labels_sha256=hashlib.sha256(truth_bytes).hexdigest(),
                  result_sha256=hashlib.sha256(result_bytes).hexdigest())
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_bytes((json.dumps(report, indent=2)+"\n").encode("utf-8"))
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
