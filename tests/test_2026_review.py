import json

import pytest
from fastapi import HTTPException

from scripts import review_2026_shots as review
from scripts.audit_2026_identity import audit


def test_review_hides_predictions_and_saves_independent_truth(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    labels = tmp_path / 'labels.json'
    manifest.write_text(json.dumps([dict(id='match:1', teams=[1,2,3,4,5,6],
                                        boxes=[dict(slot='red0')], prediction_team=2,
                                        prediction_slot='red0', pts=[], source_sha256='hash')]))
    monkeypatch.setattr(review, 'MANIFEST', manifest)
    monkeypatch.setattr(review, 'LABELS', labels)
    assert 'prediction_team' not in review.listing()['items'][0]
    assert 'prediction_slot' not in review.listing()['items'][0]
    review.save(review.Label(id='match:1', team=3, slot='red0'))
    result = review.summary()
    assert (result['team_correct'], result['box_correct']) == (0, 1)
    review.save(review.Label(id='match:1', status='unknown'))
    assert review.summary()['team_labels'] == 0
    assert review.summary()['reviewed'] == 1
    with pytest.raises(HTTPException):
        review.save(review.Label(id='match:1', team=999))
    assert json.loads(manifest.read_text())[0]['prediction_team'] == 2


def test_identity_audit_checks_time_specific_mapping(monkeypatch):
    import scripts.audit_2026_identity as module
    from types import SimpleNamespace
    monkeypatch.setattr(module, 'slot_positions_between', lambda ctx,slot,a,b: [[1 if slot=='red0' else 4,0]])
    ctx = SimpleNamespace(field=SimpleNamespace(elements={'red_hub': {}},
                                               dist_to_element=lambda p,v: p[0]),
                          tracks={'red0': [], 'red1': []}, team_at=lambda s,t: 2 if t<10 else 1)
    d = dict(red=[1,2,3],blue=[4,5,6],aux_reads=[[5,'red_hub',1,.9],[15,'red_hub',1,.9]])
    assert [r['agrees'] for r in audit(d,ctx)] == [False,True]
