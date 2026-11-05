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


def test_secondary_camera_settings_round_trip_and_invalidate_moved_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(web.config, 'EVENTS_DIR', tmp_path)
    cal = web.Calibration(event_key='2025test', year=2025, view=(0, 0, 1920, 600), points={},
                          aux_views={'blue_reef': [0, 650, 960, 370]},
                          overlay={'counters': {'blue_algae': [20, 20, 40, 30]}})
    cal.save()
    cams = tmp_path / '2025test_cams.json'
    cams.write_text(json.dumps({'blue_reef': {'inliers': 30}}))
    body = {'view': [0, 0, 1920, 600], 'points': {},
            'aux_views': {'blue_reef': [0, 650, 960, 370]},
            'aux_watch': {'blue_reef': [100, 700, 500, 1000]}, 'overlay': {}}
    web.save_cal('2025test', body)
    assert json.loads(cams.read_text())['blue_reef']['inliers'] == 30
    result = web.get_cal('2025test')
    assert result['aux_views'] == body['aux_views']
    assert result['aux_watch'] == body['aux_watch']
    assert result['overlay']['counters']['blue_algae'] == [20, 20, 40, 30]
    body['aux_views']['blue_reef'] = [10, 650, 950, 370]
    web.save_cal('2025test', body)
    assert json.loads(cams.read_text()) == {}


@pytest.mark.parametrize('rect', [[0, 650, 2000, 370], [0, 0, -1, 10], [0, 0, float('nan'), 10]])
def test_secondary_camera_rejects_invalid_rectangle(tmp_path, monkeypatch, rect):
    monkeypatch.setattr(web.config, 'EVENTS_DIR', tmp_path)
    with pytest.raises(HTTPException) as error:
        web.save_cal('2025test', {'view': [0, 0, 1920, 600], 'points': {},
                                  'aux_views': {'blue_reef': rect}})
    assert error.value.status_code == 400
    assert not (tmp_path / '2025test.yaml').exists()


def test_fit_secondary_cameras_uses_event_job(tmp_path, monkeypatch):
    monkeypatch.setattr(web.config, 'EVENTS_DIR', tmp_path)
    with pytest.raises(HTTPException):
        web.fit_cameras('2025test')
    web.Calibration(event_key='2025test', year=2025, view=(0, 0, 1920, 600), points={},
                    aux_views={'blue_reef': [0, 650, 960, 370]}).save()
    monkeypatch.setattr(web, '_start_job', lambda key, args: (key, args))
    assert web.fit_cameras('2025test') == ('2025test', ['calibrate-cams', '2025test'])


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


def test_cancel_clears_stale_progress_without_a_running_process(tmp_path, monkeypatch):
    from autoscout import progress

    monkeypatch.setattr(web, '_orphan_runs', lambda ek: [])
    monkeypatch.setattr(web, 'JOBS', {})
    monkeypatch.setattr(progress, 'DIR', tmp_path)
    monkeypatch.setattr(web.config, 'DATA_DIR', tmp_path)
    for key in ('2025test_qm8', '2026other_qm1'):
        job = progress.Progress(key)
        job.update('frames', .33, force=True)
    assert web.cancel_run('2025test') == {'cancelled': True}
    cancelled = json.loads((tmp_path / '2025test_qm8.json').read_text())
    assert cancelled['stage'] == 'error' and cancelled['error'] == 'cancelled'
    assert cancelled['eta_s'] is None
    assert json.loads((tmp_path / '2026other_qm1.json').read_text())['stage'] == 'frames'
    assert web.cancel_run('2025test')['cancelled'] is False


def test_run_rejects_bad_workers(monkeypatch):
    monkeypatch.setattr(web.Calibration, 'load', staticmethod(lambda ek: type('C', (), {'points': {i: 0 for i in range(4)}})()))
    monkeypatch.setattr(web, '_event_job', lambda ek: None)
    with pytest.raises(HTTPException) as error:
        web.run_event('2026test', body={'workers': 'many'})
    assert error.value.status_code == 400


def test_annotate_accepts_quarterfinal_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(web, '_results', lambda: tmp_path)
    with pytest.raises(HTTPException) as error:
        web.annotate('2026test_qf1m1')
    assert error.value.status_code == 404  # passed key validation, no result yet
