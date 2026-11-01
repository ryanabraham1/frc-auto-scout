"""Dashboard workflow contracts without running video processing."""
import csv
import io
import json

import pytest
from fastapi import HTTPException

from autoscout import db
from autoscout.web import app as web


def test_export_preserves_nested_metrics_and_quotes(tmp_path, monkeypatch):
    path = tmp_path / 'test.db'
    c = db.connect(path)
    c.execute('INSERT INTO team_matches VALUES (?,?,?,?,?,?,?)',
              ('2026test_qm1', 254, '2026test', 2026, 'red', 1,
               json.dumps({'fuel_scored': 42, 'periods': {'auto': 12}, 'note': 'a,b'})))
    c.commit()
    c.close()
    monkeypatch.setattr(web, 'con', lambda: db.connect(path))
    response = web.dashboard_export('2026test')
    rows = list(csv.DictReader(io.StringIO(response.body.decode())))
    assert rows[0]['team'] == '254'
    assert rows[0]['fuel_scored'] == '42'
    assert json.loads(rows[0]['periods']) == {'auto': 12}
    assert rows[0]['note'] == 'a,b'


def test_prepare_only_accepts_match_from_selected_event():
    with pytest.raises(HTTPException) as error:
        web.prepare_calibration('2026test', {'match_key': '2026other_qm1'})
    assert error.value.status_code == 400


def test_duplicate_job_does_not_start_process(monkeypatch):
    monkeypatch.setattr(web, '_job_status', lambda key: {'running': True})
    with pytest.raises(HTTPException) as error:
        web._start_job('2026test', ['reanalyze', '2026test'])
    assert error.value.status_code == 409


def test_invalid_event_is_rejected():
    with pytest.raises(HTTPException):
        web.event('../invalid')


def test_run_passes_selected_matches_debug_video_and_workers(monkeypatch, tmp_path):
    started = {}

    class Proc:
        def poll(self):
            return None

    monkeypatch.setattr(web.Calibration, 'load', staticmethod(lambda ek: type('C', (), {'points': {i: 0 for i in range(4)}})()))
    monkeypatch.setattr(web, '_event_job', lambda ek: None)
    monkeypatch.setattr(web.config, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(web.subprocess, 'Popen', lambda cmd, **kw: started.setdefault('cmd', cmd) and Proc())
    monkeypatch.setattr(web, 'JOBS', {})
    assert web.run_event('2026test', body={'matches': ['2026test_qm3', '2026test_sf1m1'], 'debug_video': True, 'workers': 9})['started']
    cmd = started['cmd']
    assert cmd[cmd.index('--matches') + 1] == '2026test_qm3,2026test_sf1m1'
    assert '--debug-video' in cmd
    assert cmd[cmd.index('--workers') + 1] == '4'


def test_run_rejects_match_from_another_event(monkeypatch):
    monkeypatch.setattr(web.Calibration, 'load', staticmethod(lambda ek: type('C', (), {'points': {i: 0 for i in range(4)}})()))
    monkeypatch.setattr(web, '_event_job', lambda ek: None)
    with pytest.raises(HTTPException) as error:
        web.run_event('2026test', body={'matches': ['2026other_qm1']})
    assert error.value.status_code == 400


def test_cancel_with_nothing_running(monkeypatch):
    monkeypatch.setattr(web, '_orphan_runs', lambda ek: [])
    monkeypatch.setattr(web, 'JOBS', {})
    assert web.cancel_run('2026test') == {'cancelled': False, 'reason': 'nothing is running for this event'}
