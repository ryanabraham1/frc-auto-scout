"""Remote-job contracts without downloading video or loading vision models."""
import json

import pytest
from fastapi.testclient import TestClient

from autoscout.integration import create_app

TOKEN = 'test-secret-' * 4
HEADERS = {'Authorization': f'Bearer {TOKEN}', 'Idempotency-Key': 'request-1'}


@pytest.fixture
def client(tmp_path):
    app = create_app(root=tmp_path, token=TOKEN, start_worker=False)
    with TestClient(app) as c:
        c.headers.update(HEADERS)
        yield c, app.state.queue


def submit(c, keys=None, **options):
    return c.post('/v1/jobs', json={'match_keys': keys or ['2026txhou_qm11'], **options})


def test_requires_secret_at_startup(tmp_path):
    with pytest.raises(RuntimeError):
        create_app(root=tmp_path, token='')


def test_all_routes_require_authentication(client):
    c, _ = client
    for path in ['/v1/health', '/v1/jobs/bad', '/v1/jobs/bad/results']:
        assert c.get(path, headers={'Authorization': 'Bearer wrong'}).status_code == 401
    assert c.post('/v1/jobs', json={}, headers={'Authorization': ''}).status_code == 401
    assert c.get('/v1/health').json()['status'] == 'online'


@pytest.mark.parametrize('keys', [[], ['../foo'], ['2026txhou_qm0'], ['2027test_qm1'],
                                 ['2026txhou_qm1;echo bad'], ['2026txhou_qm1'] * 101])
def test_rejects_invalid_input(client, keys):
    c, _ = client
    assert c.post('/v1/jobs', json={'match_keys': keys}).status_code == 422


def test_idempotency_and_conflict(client):
    c, _ = client
    first = submit(c).json()
    assert submit(c).json()['id'] == first['id']
    assert submit(c, force=True).status_code == 409
    assert submit(c, ['2026txhou_qm12']).status_code == 409
    assert submit(c, arbitrary_command='whoami').status_code == 422
    assert c.get(f"/v1/jobs/{first['id']}/results").json()['results'] == []


def test_missing_request_id_and_bounded_queue(client):
    c, _ = client
    del c.headers['Idempotency-Key']
    assert submit(c).status_code == 422
    for i in range(20):
        c.headers['Idempotency-Key'] = str(i)
        assert submit(c).status_code == 202
    # A retry is allowed even when the queue is full.
    assert submit(c).status_code == 202
    c.headers['Idempotency-Key'] = 'one-too-many'
    assert submit(c).status_code == 429


def test_wrong_match_result_is_rejected(client):
    c, q = client
    job = submit(c).json()
    q.run_next(lambda *_: {'match_key': '2026other_qm1', 'teams': {}})
    assert q.get(job['id'])['status'] == 'failed'
    assert q.get(job['id'])['completed_matches'] == []


def test_processing_progress_evidence_and_restart(client, tmp_path):
    c, q = client
    job = submit(c, ['2026txhou_qm11', '2026txhou_sf1m1']).json()
    seen = []
    def runner(mk, force, log):
        assert q.get(job['id'])['status'] == 'running'
        seen.append(mk)
        return {'match_key': mk, 'teams': {'254': {'fuel_scored': 12}},
                'identity_conf': {'red1': 0.4}, 'alliances': {'red': {'unattributed_fuel': 8}},
                'raw_tracks': [1, 2, 3]}
    assert q.run_next(runner)
    response = c.get(f"/v1/jobs/{job['id']}/results").json()
    assert response['job']['status'] == 'succeeded'
    assert len(response['results']) == 2
    result = response['results'][0]['match']
    assert result['identity_conf'] == {'red1': 0.4}
    assert result['alliances']['red']['unattributed_fuel'] == 8
    assert 'raw_tracks' not in result
    # An interrupted force job resumes snapshots without repeating completed matches.
    q.update(job['id'], 'running', [seen[0]])
    q.recover()
    assert q.get(job['id'])['status'] == 'queued'
    assert q.run_next(runner)
    assert seen == ['2026txhou_qm11', '2026txhou_sf1m1', '2026txhou_sf1m1']
    assert not q.run_next(runner)


def test_failure_preserves_partial_results(client):
    c, q = client
    job = submit(c, ['2026txhou_qm11', '2026txhou_qm12']).json()
    def runner(mk, *args):
        if mk.endswith('qm12'):
            raise RuntimeError('secret/private/path')
        return {'match_key': mk, 'teams': {}}
    q.run_next(runner)
    response = c.get(f"/v1/jobs/{job['id']}/results").json()
    assert response['job']['status'] == 'failed'
    assert len(response['results']) == 1
    assert 'secret/private/path' not in json.dumps(response)


def test_worker_consumes_persisted_queue(tmp_path):
    import time
    root = tmp_path / 'queue'
    app = create_app(root=root, token=TOKEN, start_worker=False)
    with TestClient(app) as c:
        c.headers.update(HEADERS)
        job = submit(c).json()
    app = create_app(root=root, token=TOKEN,
                     runner=lambda mk, *_: {'match_key': mk, 'teams': {}})
    with TestClient(app) as c:
        c.headers.update(HEADERS)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = c.get(f"/v1/jobs/{job['id']}").json()['status']
            if status == 'succeeded':
                break
            time.sleep(0.02)
        assert status == 'succeeded'


def test_processor_uses_cache_or_safe_subprocess(tmp_path, monkeypatch):
    from autoscout import db, integration
    path = tmp_path / 'results.json'
    result = {'match_key': '2026txhou_qm11', 'teams': {}}
    path.write_text(json.dumps(result))
    database = tmp_path / 'results.db'
    c = db.connect(database)
    c.execute('INSERT INTO matches (match_key,status,result_path) VALUES (?,?,?)',
              (result['match_key'], 'done', str(path)))
    c.commit()
    c.close()
    connect = db.connect
    monkeypatch.setattr(db, 'connect', lambda: connect(database))
    calls = []
    monkeypatch.setattr(integration.subprocess, 'run', lambda args, **kw: calls.append((args, kw)))
    log = tmp_path / 'log.txt'
    assert integration.process_match(result['match_key'], False, log) == result
    assert not calls
    assert integration.process_match(result['match_key'], True, log) == result
    args, kw = calls[0]
    assert args[-2:] == ['run-match', result['match_key']]
    assert kw['check'] and kw['timeout'] == 7200
    assert 'shell' not in kw
    # A failed forced rerun must never publish the pre-existing cached output.
    def fail(*args, **kwargs):
        raise integration.subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(integration.subprocess, 'run', fail)
    with pytest.raises(integration.subprocess.CalledProcessError):
        integration.process_match(result['match_key'], True, log)
