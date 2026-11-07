"""Bridge and stream contracts, without network, video downloads or vision models."""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest

from autoscout import bridge as B
from autoscout import streams
from autoscout.integration import MatchSkipped, Queue

EVENT = '2026cc'
MK = '2026cc_qm40'


@pytest.mark.parametrize('url,want', [
    ('https://www.youtube.com/watch?v=zQ8Oibbg9Ro', 'https://www.youtube.com/watch?v=zQ8Oibbg9Ro'),
    ('https://youtube.com/watch?v=zQ8Oibbg9Ro&t=120&list=abc', 'https://www.youtube.com/watch?v=zQ8Oibbg9Ro'),
    ('https://youtu.be/zQ8Oibbg9Ro?si=x', 'https://www.youtube.com/watch?v=zQ8Oibbg9Ro'),
    ('https://www.youtube.com/live/zQ8Oibbg9Ro', 'https://www.youtube.com/watch?v=zQ8Oibbg9Ro'),
    ('https://www.youtube.com/@FIRSTinspires/live', 'https://www.youtube.com/@FIRSTinspires/live'),
    ('https://www.twitch.tv/FIRSTinspires', 'https://www.twitch.tv/firstinspires'),
    ('https://www.twitch.tv/videos/2345678901', 'https://www.twitch.tv/videos/2345678901'),
])
def test_allowed_sources_are_rebuilt(url, want):
    assert streams.parse_source(url) == want


@pytest.mark.parametrize('url', [
    'http://www.youtube.com/watch?v=zQ8Oibbg9Ro', 'https://evil.example/watch?v=zQ8Oibbg9Ro',
    'https://youtube.com.evil.example/watch?v=zQ8Oibbg9Ro', 'https://user@www.youtube.com/watch?v=zQ8Oibbg9Ro',
    'https://www.youtube.com:8443/watch?v=zQ8Oibbg9Ro', 'https://www.youtube.com:bad/watch?v=zQ8Oibbg9Ro',
    'https://www.youtube.com/watch?v=--exec=calc', 'https://www.youtube.com/watch?v=a b c d e',
    'file:///etc/passwd', '--exec=calc', '', None, 'https://www.twitch.tv/a/b/c', 'https://youtu.be/' + 'a' * 400,
])
def test_everything_else_is_refused(url):
    with pytest.raises(streams.StreamError):
        streams.parse_source(url)


def test_vod_window_brackets_the_match():
    assert streams.vod_window(10_000, 1_000, 20_000) == (8_970, 9_340)
    assert streams.vod_window(1_010, 1_000, None) == (0, 350)
    assert streams.vod_window(10_000, 1_000, 9_100)[1] == 9_100


@pytest.mark.parametrize('actual,duration', [(500, 20_000), (30_000, 20_000)])
def test_match_on_another_days_stream_is_not_clipped(actual, duration):
    with pytest.raises(streams.StreamError):
        streams.vod_window(actual, 1_000, duration)


def segment(folder: Path, t: float):
    folder.mkdir(exist_ok=True, parents=True)
    p = folder / (datetime.fromtimestamp(t).strftime(streams.SEGMENT_TIME) + '.ts')
    p.write_bytes(b'x')
    return p


def test_cut_clip_refuses_a_gap_in_the_recording(tmp_path):
    t = 1_800_000_000
    for dt in (0, 20, 200, 220):
        segment(tmp_path, t + dt)
    with pytest.raises(streams.StreamError, match='dropped'):
        streams.cut_clip(tmp_path, t, t + 230, tmp_path / 'out.mp4')


def test_cut_clip_refuses_an_uncovered_window(tmp_path):
    segment(tmp_path, 1_800_000_000)
    with pytest.raises(streams.StreamError, match='cover'):
        streams.cut_clip(tmp_path, 1_800_100_000, 1_800_100_100, tmp_path / 'out.mp4')


def test_cut_clip_joins_exactly_the_overlapping_segments(tmp_path, monkeypatch):
    t = 1_800_000_000
    made = [segment(tmp_path, t + 20 * i) for i in range(6)]
    seen = {}

    def fake_run(cmd, **_):
        seen['list'] = Path(cmd[cmd.index('-i') + 1]).read_text()
        Path(cmd[-1]).write_bytes(b'mp4')
        return subprocess.CompletedProcess(cmd, 0, '', '')

    monkeypatch.setattr(streams.subprocess, 'run', fake_run)
    out = streams.cut_clip(tmp_path, t + 25, t + 65, tmp_path / 'clip.mp4')
    assert out.read_bytes() == b'mp4'
    assert [p.name in seen['list'] for p in made] == [False, True, True, True, False, False]


def test_prune_keeps_recent_segments(tmp_path):
    old, new = segment(tmp_path, 1_800_000_000), segment(tmp_path, 1_800_009_000)
    streams.prune(tmp_path, 3600, now=1_800_009_100)
    assert not old.exists() and new.exists()


class FakeTBA:
    def __init__(self, matches=None, videos=None):
        self.matches, self.videos = matches or [], videos or []

    def get(self, path, max_age_s=None):
        if path.startswith('/match/'):
            return next((m for m in self.matches if m['key'] == path.split('/')[-1]), {})
        return self.matches

    def match(self, key):
        return {'videos': self.videos}


def result(key):
    return {'match_key': key, 'teams': {}}


def test_stream_runner_clips_then_processes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(B, 'cached_result', lambda k: None)
    monkeypatch.setattr(B.streams, 'clip_vod', lambda url, a, b, out, log: calls.append((a, b)) or out)
    runner = B.Runner(tmp_path, FakeTBA([{'key': MK, 'actual_time': 10_000}]),
                      process=lambda k, f, l, v=None: calls.append(v.name.split('_')[1]) or result(k))
    monkeypatch.setattr(B.streams, 'probe', lambda url: B.streams.StreamInfo('was_live', 1_000, 40_000))
    source = {'stream': True, 'url': 'https://www.youtube.com/watch?v=zQ8Oibbg9Ro'}
    assert runner(MK, False, tmp_path / 'log', source)['match_key'] == MK
    assert calls == [(8_970, 9_340), 'qm40']


@pytest.mark.parametrize('tba,source,message', [
    (FakeTBA([{'key': MK}]), {'stream': True, 'url': 'u'}, 'no start time'),
    (FakeTBA([{'key': MK, 'actual_time': 5}]), {'stream': True, 'url': 'https://www.youtube.com/watch?v=abcdefghijk',
                                                'stream_starts': {'abcdefghijk': 10}}, 'not in that stream'),
    (FakeTBA([{'key': MK}]), {}, 'no video'),
])
def test_a_bad_match_is_skipped_with_a_reason(tmp_path, monkeypatch, tba, source, message):
    monkeypatch.setattr(B, 'cached_result', lambda k: None)
    monkeypatch.setattr(B.streams, 'probe', lambda url: B.streams.StreamInfo('was_live', 1, None))
    with pytest.raises(MatchSkipped, match=message):
        B.Runner(tmp_path, tba, process=None)(MK, False, tmp_path / 'log', source)


def test_failed_processing_never_leaks_the_command(tmp_path, monkeypatch):
    def boom(*_):
        raise subprocess.CalledProcessError(1, ['python', 'C:/secret/path'])

    with pytest.raises(MatchSkipped) as e:
        B.Runner(tmp_path, FakeTBA(), process=boom)(MK, True, tmp_path / 'log', {'clip': 'x.mp4'})
    assert 'secret' not in str(e.value)


def test_skipped_match_does_not_stop_the_job(tmp_path):
    q = Queue(tmp_path)
    jid = q.enqueue(json.dumps({'match_keys': ['2026cc_qm1', '2026cc_qm2'], 'force': False,
                                'source': {'url': 'u'}}), 'k')['id']

    def runner(mk, force, log, source):
        if mk.endswith('qm1'):
            raise MatchSkipped('not in that stream')
        return result(mk)

    q.run_next(runner)
    job = q.get(jid)
    assert job['status'] == 'succeeded' and job['completed_matches'] == ['2026cc_qm2']
    assert q.match_errors(jid) == {'2026cc_qm1': 'not in that stream'}


def test_job_with_nothing_processed_fails(tmp_path):
    q = Queue(tmp_path)
    jid = q.enqueue(json.dumps({'match_keys': ['2026cc_qm1'], 'force': False, 'source': {'url': 'u'}}), 'k')['id']

    def runner(*_):
        raise MatchSkipped('nope')

    q.run_next(runner)
    assert q.get(jid)['status'] == 'failed'


def test_cancel_lands_between_matches_and_keeps_finished_work(tmp_path):
    q = Queue(tmp_path)
    jid = q.enqueue(json.dumps({'match_keys': ['2026cc_qm1', '2026cc_qm2'], 'force': False}), 'k')['id']

    def runner(mk, force, log):
        q.cancel(jid)
        return result(mk)

    q.run_next(runner)
    job = q.get(jid)
    assert job['status'] == 'cancelled' and job['completed_matches'] == ['2026cc_qm1']


def test_cancel_of_queued_job_is_immediate(tmp_path):
    q = Queue(tmp_path)
    jid = q.enqueue(json.dumps({'match_keys': ['2026cc_qm1'], 'force': False}), 'k')['id']
    q.cancel(jid)
    assert q.get(jid)['status'] == 'cancelled'
    assert q.run_next(lambda *_: pytest.fail('cancelled job ran')) is False


class FakeRecorder:
    status = 'recording'

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def live(tmp_path, matches, reach, enqueued=None, cut=None, started=1_000):
    job = {'id': 'j' * 36, 'event_key': EVENT, 'source_url': 'u'}
    s = B.LiveSession(job, tmp_path, lambda k, c: (enqueued.append(k) if enqueued is not None else None) or f'local-{k}',
                      tba=FakeTBA(matches), recorder=FakeRecorder(),
                      cut=cut or (lambda f, a, b, out: out), covered=lambda f: reach)
    s.started = started
    return s


def test_live_waits_for_the_whole_match_then_queues_it_once(tmp_path):
    got = []
    m = [{'key': MK, 'actual_time': 2_000}]
    s = live(tmp_path, m, 2_000 + streams.LIVE_AFTER_S - 1, got)
    s.tick(now=5_000)
    assert got == []
    s.covered = lambda f: 2_000 + streams.LIVE_AFTER_S
    s.next_poll = 0
    s.tick(now=5_001)
    s.next_poll = 0
    s.tick(now=5_002)
    assert got == [MK]


def test_live_ignores_matches_from_before_recording_and_unplayed_ones(tmp_path):
    got = []
    s = live(tmp_path, [{'key': '2026cc_qm1', 'actual_time': 900}, {'key': '2026cc_qm2'}], 99_999, got)
    s.tick(now=5_000)
    assert got == []


def test_live_reports_a_stream_drop_for_that_match_and_keeps_going(tmp_path):
    def cut(f, a, b, out):
        if '2026cc_qm1_' in out.name:
            raise streams.StreamError('The stream dropped during this match')
        return out

    got = []
    s = live(tmp_path, [{'key': '2026cc_qm1', 'actual_time': 2_000}, {'key': '2026cc_qm2', 'actual_time': 3_000}],
             99_999, got, cut)
    s.tick(now=5_000)
    assert got == ['2026cc_qm2'] and 'dropped' in s.errors['2026cc_qm1']


def test_live_survives_a_restart_without_reprocessing(tmp_path):
    got = []
    m = [{'key': MK, 'actual_time': 2_000}]
    s = live(tmp_path, m, 99_999, got)
    s.tick(now=5_000)
    again = live(tmp_path, m, 99_999, got, started=1_000)
    again.tick(now=5_100)
    assert got == [MK]


class FakeBackend:
    def __init__(self, jobs=None, cancel=None):
        self.jobs, self.cancels, self.calls = jobs or [], cancel or [], []

    def call(self, action, **body):
        self.calls.append((action, body))
        if action == 'sync':
            jobs, self.jobs = self.jobs, []
            cancels, self.cancels = self.cancels, []
            return {'jobs': jobs, 'cancel': cancels}
        return {'ok': True}

    def syncs(self):
        return [b for a, b in self.calls if a == 'sync']

    def results(self):
        return [b for a, b in self.calls if a == 'result']


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(B, 'calibrated', lambda e: e == EVENT)
    monkeypatch.setattr(B, 'free_gb', lambda: 100.0)
    monkeypatch.setattr(B, 'calibrated_events', lambda: [EVENT])
    backend = FakeBackend()
    return backend, B.Bridge(backend, root=tmp_path, runner=lambda mk, *_: result(mk))


def job(kind='matches', **extra):
    return {'id': '11111111-1111-4111-8111-111111111111', 'kind': kind, 'event_key': EVENT,
            'match_keys': [MK], 'force': False, **extra}


def test_job_runs_and_every_result_is_uploaded_once(world):
    backend, b = world
    backend.jobs = [job()]
    b.tick()
    b.queue.run_next(b.runner)
    b.tick()
    b.tick()
    assert [r['match_key'] for r in backend.results()] == [MK]
    assert backend.results()[0]['envelope']['source'] == 'autoscout'
    last = [j for s in backend.syncs() for j in s['jobs']][-1]
    assert last['status'] == 'succeeded' and last['completed_matches'] == [MK]
    assert backend.syncs()[-1]['jobs'] == []  # final state is reported once, then dropped


def test_the_same_job_offered_twice_runs_once(world):
    backend, b = world
    b.accept(job())
    b.accept(job())
    b.queue.run_next(b.runner)
    assert b.queue.run_next(b.runner) is False


def test_result_upload_retries_after_a_network_failure(world):
    backend, b = world
    backend.jobs = [job()]
    b.tick()
    b.queue.run_next(b.runner)
    real, boom = backend.call, {'left': 1}

    def flaky(action, **body):
        if action == 'result' and boom['left']:
            boom['left'] -= 1
            raise B.requests.ConnectionError()
        return real(action, **body)

    backend.call = flaky
    with pytest.raises(B.requests.ConnectionError):
        b.tick()
    b.tick()
    assert [r['match_key'] for r in backend.results()] == [MK]


def refusing(backend, status):
    real = backend.call

    def call(action, **body):
        if action == 'result':
            response = B.requests.Response()
            response.status_code = status
            raise B.requests.HTTPError(response=response)
        return real(action, **body)

    backend.call = call


def test_a_refused_result_is_reported_and_does_not_stop_the_heartbeat(world):
    backend, b = world
    backend.jobs = [job()]
    b.tick()
    b.queue.run_next(b.runner)
    refusing(backend, 422)
    b.tick()
    assert 'refused' in [j for s in backend.syncs() for j in s['jobs']][-1]['match_errors'][MK]


def test_an_outage_keeps_the_result_for_later(world):
    backend, b = world
    backend.jobs = [job()]
    b.tick()
    b.queue.run_next(b.runner)
    refusing(backend, 502)
    with pytest.raises(B.requests.HTTPError):
        b.tick()
    assert b.db().execute('SELECT COUNT(*) FROM uploaded').fetchone()[0] == 0


@pytest.mark.parametrize('bad,reason', [
    ({'event_key': '2026nocal', 'match_keys': ['2026nocal_qm1']}, 'not calibrated'),
    ({'event_key': '2024cc', 'match_keys': ['2024cc_qm1']}, 'supports 2025'),
    ({'match_keys': ['2026other_qm1']}, 'Pick up to 100'),
    ({'match_keys': ['../../x']}, 'Pick up to 100'),
    ({'kind': 'vod', 'source_url': 'https://evil.example/x'}, 'YouTube or Twitch'),
    ({'kind': 'live', 'source_url': 'file:///c:/x'}, 'YouTube or Twitch'),
])
def test_rejected_jobs_fail_with_a_reason_and_never_queue(world, bad, reason):
    backend, b = world
    backend.jobs = [{**job(), **bad}]
    b.tick()
    b.tick()
    report = [j for s in backend.syncs() for j in s['jobs']][-1]
    assert report['status'] == 'failed' and reason in report['error']
    assert b.queue.run_next(b.runner) is False


def test_cancel_of_a_job_never_received_is_confirmed_so_it_cannot_hang(world):
    backend, b = world
    backend.cancels = [job()['id']]
    b.tick()
    b.tick()
    report = [j for s in backend.syncs() for j in s['jobs']][-1]
    assert report['id'] == job()['id'] and report['status'] == 'cancelled'


def test_low_disk_refuses_work(world, monkeypatch):
    backend, b = world
    monkeypatch.setattr(B, 'free_gb', lambda: 3.0)
    assert 'disk' in b.reject_reason(job())


def test_vod_job_carries_the_leads_calibration(world):
    _, b = world
    j = job('vod', source_url='https://youtu.be/zQ8Oibbg9Ro', stream_starts={'zQ8Oibbg9Ro': '2026-10-03T16:00:00+00:00'})
    b.accept(j)
    local = b.db().execute('SELECT local_id FROM remote_jobs').fetchone()['local_id']
    request = json.loads(b.queue.connect().execute('SELECT request FROM jobs WHERE id=?', (local,)).fetchone()['request'])
    assert request['source'] == {'stream': True, 'url': 'https://www.youtube.com/watch?v=zQ8Oibbg9Ro', 'stream_starts': {
        'zQ8Oibbg9Ro': datetime.fromisoformat('2026-10-03T16:00:00+00:00').timestamp()}}


def live_bridge(world):
    backend, b = world
    sessions = []

    def factory(job_, root, enqueue):
        s = B.LiveSession(job_, root, enqueue, tba=FakeTBA([{'key': MK, 'actual_time': time.time() + 100}]),
                          recorder=FakeRecorder(), cut=lambda f, a, c, out: out, covered=lambda f: time.time() + 9_999)
        sessions.append(s)
        return s

    b.session_factory = factory
    return backend, b, sessions


def test_live_job_flows_to_results_and_stops_cleanly(world):
    backend, b, sessions = live_bridge(world)
    backend.jobs = [job('live', source_url='https://www.twitch.tv/firstinspires', match_keys=[])]
    b.tick()
    assert sessions[0].recorder.started
    b.tick()
    b.queue.run_next(lambda mk, f, l, src: result(mk))
    b.tick()
    assert [r['match_key'] for r in backend.results()] == [MK]
    running = backend.syncs()[-1]['jobs'][-1]
    assert running['status'] == 'running' and running['progress']['recorder'] == 'recording'
    backend.cancels = [job()['id']]
    b.tick()
    assert sessions[0].recorder.stopped
    b.tick()
    assert backend.syncs()[-1]['jobs'][-1]['status'] == 'cancelled'


def test_live_job_resumes_after_a_restart(world, tmp_path):
    backend, b, sessions = live_bridge(world)
    backend.jobs = [job('live', source_url='https://www.twitch.tv/firstinspires', match_keys=[])]
    b.tick()
    second = B.Bridge(backend, root=tmp_path, runner=b.runner, session_factory=b.session_factory)
    second.resume()
    assert len(sessions) == 2 and job()['id'] in second.live


WEBCASTS = [
    {'type': 'twitch', 'channel': 'firstinspires'},
    {'type': 'youtube', 'channel': 'daytwo12345', 'date': '2026-10-04'},
    {'type': 'youtube', 'channel': 'UCxxxxxxxxxxxx', 'file': 'dayone12345', 'date': '2026-10-03'},
    {'type': 'livestream', 'channel': 'nope'},
]


@pytest.mark.parametrize('day,vod,want', [
    ('2026-10-03', False, 'https://www.youtube.com/watch?v=dayone12345'),
    ('2026-10-04', True, 'https://www.youtube.com/watch?v=daytwo12345'),
    ('2026-10-09', False, 'https://www.twitch.tv/firstinspires'),  # no dated stream: the whole-event one
    ('2026-10-09', True, 'https://www.youtube.com/watch?v=daytwo12345'),  # Twitch channels have no history
])
def test_the_right_days_stream_is_found(day, vod, want):
    assert streams.event_stream(WEBCASTS, day, vod) == want


def test_no_usable_stream_is_none():
    assert streams.event_stream([{'type': 'livestream', 'channel': 'x'}], '2026-10-03') is None
    assert streams.event_stream([{'type': 'twitch', 'channel': 'x1'}], '2026-10-03', vod=True) is None


def test_event_day_is_event_local():
    assert streams.event_day(1_791_000_000, 'America/Los_Angeles') != streams.event_day(1_791_000_000, 'Asia/Tokyo')


def test_stream_runner_finds_each_matchs_stream_from_tba(tmp_path, monkeypatch):
    class EventTBA(FakeTBA):
        def get(self, path, max_age_s=None):
            if path == '/event/2026cc':
                return {'timezone': 'UTC', 'webcasts': WEBCASTS}
            return {'actual_time': datetime.fromisoformat('2026-10-04T18:00:00+00:00').timestamp()}

    urls = []
    monkeypatch.setattr(B, 'cached_result', lambda k: None)
    monkeypatch.setattr(B.streams, 'probe', lambda url: urls.append(url) or B.streams.StreamInfo('was_live', 1_700_000_000, None))
    monkeypatch.setattr(B.streams, 'clip_vod', lambda url, a, b, out, log: out)
    B.Runner(tmp_path, EventTBA(), process=lambda k, f, l, v=None: result(k))(MK, False, tmp_path / 'log', {'stream': True})
    assert urls == ['https://www.youtube.com/watch?v=daytwo12345']


def test_automatic_source_falls_back_to_tbas_video(tmp_path, monkeypatch):
    class NoStream(FakeTBA):
        def get(self, path, max_age_s=None):
            return {'webcasts': []} if path.startswith('/event/') else {'actual_time': 5}

    used = []
    monkeypatch.setattr(B, 'cached_result', lambda k: None)
    runner = B.Runner(tmp_path, NoStream(videos=['abc']), process=lambda k, f, l, v=None: used.append(v) or result(k))
    runner(MK, False, tmp_path / 'log', {'stream': True})
    assert used == [None]


def test_a_pasted_link_is_never_silently_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(B, 'cached_result', lambda k: None)
    monkeypatch.setattr(B.streams, 'probe', lambda url: B.streams.StreamInfo('was_live', 1_000_000, None))
    runner = B.Runner(tmp_path, FakeTBA([{'key': MK, 'actual_time': 5}], videos=['abc']), process=None)
    with pytest.raises(MatchSkipped, match='not in that stream'):
        runner(MK, False, tmp_path / 'log', {'stream': True, 'url': 'https://www.youtube.com/watch?v=abcdefghijk'})


def test_a_dead_link_is_skipped_for_what_is_actually_on_air():
    state = {'https://www.youtube.com/watch?v=dayone12345': 'was_live', 'https://www.twitch.tv/firstinspires': 'is_live'}
    check = lambda url: B.streams.StreamInfo(state.get(url, 'not_live'), None, None)
    assert streams.live_stream(WEBCASTS, '2026-10-03', check) == 'https://www.twitch.tv/firstinspires'


def test_nothing_on_air_is_none():
    assert streams.live_stream(WEBCASTS, '2026-10-03', lambda url: B.streams.StreamInfo('was_live', None, None)) is None


def test_a_match_missing_from_todays_stream_is_found_in_the_twitch_archive(tmp_path, monkeypatch):
    class EventTBA(FakeTBA):
        def get(self, path, max_age_s=None):
            if path.startswith('/event/'):
                return {'timezone': 'UTC', 'webcasts': WEBCASTS}
            return {'actual_time': 5_000}

    archive = 'https://www.twitch.tv/videos/2922204988'
    infos = {archive: B.streams.StreamInfo('was_live', 4_000, 2_000)}
    used = []
    monkeypatch.setattr(B, 'cached_result', lambda k: None)
    monkeypatch.setattr(B.streams, 'twitch_archives', lambda w: [archive])
    monkeypatch.setattr(B.streams, 'probe', lambda url: infos.get(url, B.streams.StreamInfo('was_live', 4_900, 50)))
    monkeypatch.setattr(B.streams, 'clip_vod', lambda url, a, b, out, log: used.append(url) or out)
    B.Runner(tmp_path, EventTBA(), process=lambda k, f, l, v=None: result(k))(MK, False, tmp_path / 'log', {'stream': True})
    assert used == [archive]


GOOD_POINTS = {'blue_hub_center': [610, 766], 'red_hub_center': [1256, 777], 'center_line_yW': [935, 711],
               'center_line_y0': [931, 859], 'blue_hub_top': [609, 668]}


def test_field_points_accept_a_good_calibration():
    points, view = B.field_points('2026cc', {'points': GOOD_POINTS, 'view': [0, 560, 1920, 350]})
    assert view == (0, 560, 1920, 350) and points['blue_hub_top'] == (609, 668)


@pytest.mark.parametrize('payload', [
    {'points': {'nonsense': [1, 2]}, 'view': [0, 0, 10, 10]},
    {'points': {'blue_hub_center': [5000, 1]}, 'view': [0, 0, 10, 10]},
    {'points': {'blue_hub_center': 'x'}, 'view': [0, 0, 10, 10]},
    {'points': GOOD_POINTS, 'view': [0, 900, 1920, 350]},
    {'points': GOOD_POINTS, 'view': [0, 0]},
])
def test_field_points_refuse_bad_input(payload):
    with pytest.raises(ValueError):
        B.field_points('2026cc', payload)


def command(kind, **payload):
    return {'id': 'c1', 'event_key': EVENT, 'kind': kind, 'payload': payload}


def test_save_calibration_writes_the_event_and_reports_the_fit(world, tmp_path, monkeypatch):
    backend, b = world
    monkeypatch.setattr(B.config, 'EVENTS_DIR', tmp_path)
    monkeypatch.setattr(B.config, 'ROOT', tmp_path)
    b.commands.handle({**command('save_calibration', points=GOOD_POINTS, view=[0, 560, 1920, 350]), 'event_key': '2026zz'})
    report = b.commands.take_reports()[0]
    assert report['status'] == 'done' and report['result']['error_m'] == 0
    assert (tmp_path / '2026zz.yaml').exists()
    assert [a for a, _ in backend.calls] == ['calibration']


def test_too_few_landmarks_are_refused(world):
    _, b = world
    b.commands.handle(command('save_calibration', points={'blue_hub_center': [10, 10]}, view=[0, 0, 100, 100]))
    report = b.commands.take_reports()[0]
    assert report['status'] == 'failed' and 'at least 4' in report['error']


def test_an_unknown_command_or_foreign_match_runs_nothing(world):
    _, b = world
    b.commands.run = lambda *a, **k: pytest.fail('ran a command')
    b.commands.handle(command('rm -rf', match_keys=[MK]))
    b.commands.handle(command('reanalyze', match_keys=['2026other_qm1']))
    assert [r['status'] for r in b.commands.take_reports()] == ['failed', 'failed']


def test_match_op_refreshes_the_result_in_the_app(world, monkeypatch):
    backend, b = world
    from autoscout import db
    monkeypatch.setattr(db, 'connect', lambda: __import__('sqlite3').connect(':memory:'))
    monkeypatch.setattr(db, 'load_result', lambda c, k: {'match_key': k, 'teams': {}})
    seen = []
    b.commands.run = lambda cmd, **k: seen.append(cmd[-2:]) or subprocess.CompletedProcess(cmd, 0)
    b.commands.handle(command('rescore', match_keys=[MK]))
    assert seen == [['rescore', MK]]
    assert b.commands.take_reports()[0]['result'] == {'done': [MK], 'failed': []}
    assert [a for a, _ in backend.calls] == ['refresh']


def test_a_command_is_taken_once_and_its_report_survives_an_outage(world):
    backend, b = world
    cmd = command('rm -rf')
    b.commands.offer(cmd)
    b.commands.offer(cmd)
    assert b.commands.inbox.qsize() == 1
    b.commands.handle(b.commands.inbox.get())
    real = backend.call
    backend.call = lambda *a, **k: (_ for _ in ()).throw(B.requests.ConnectionError())
    with pytest.raises(B.requests.ConnectionError):
        b.tick()
    backend.call = real
    b.tick()
    assert backend.syncs()[-1]['commands'][0]['id'] == 'c1'


def test_b_teams_use_their_fms_number():
    from autoscout.tba import team_number

    event = {'remap_teams': {'frc9990': 'frc2073B', 'frc9991': 'frc581B'}}
    assert team_number('frc254', '2026mttd') == 254
    assert team_number('frc2073B', '2026mttd', event) == 9990
    assert team_number('frc581B', '2026mttd', event) == 9991


def test_an_unmapped_b_team_never_loses_the_match():
    from autoscout.tba import team_number

    assert team_number('frc2073B', '2026mttd', {}) == 2073
    with pytest.raises(ValueError):
        team_number('frcX', '2026mttd', {})


def test_a_recording_of_the_match_beats_any_stream(tmp_path, monkeypatch):
    t = 1_800_000_000
    for i in range(30):
        segment(tmp_path / 'safety' / 'ev', t + 20 * i)
    clips = tmp_path / 'clips'
    clips.mkdir()
    monkeypatch.setattr(B, 'cached_result', lambda k: None)
    monkeypatch.setattr(B.streams, 'cut_clip', lambda f, a, b, out: out)
    monkeypatch.setattr(B.streams, 'probe', lambda url: pytest.fail('asked a stream'))
    used = []
    tba = FakeTBA([{'key': MK, 'actual_time': t + 50}])
    B.Runner(clips, tba, process=lambda k, f, l, v=None: used.append(v.name) or result(k))(MK, False, tmp_path / 'log', {'stream': True})
    assert used == [f'{MK}_rec.mp4']


def test_covers_needs_the_whole_window_without_holes(tmp_path):
    t = 1_800_000_000
    for i in list(range(10)) + list(range(14, 30)):
        segment(tmp_path, t + 20 * i)
    assert streams.covers(tmp_path, t + 300, t + 500)
    assert not streams.covers(tmp_path, t + 100, t + 400)  # the hole
    assert not streams.covers(tmp_path, t - 100, t + 100)  # before recording
    assert not streams.covers(tmp_path, t + 400, t + 900)  # past the end


@pytest.mark.parametrize('texts,want', [
    (['Madtown Throwdown', 'Qualification 2'], 2),
    (['Qualification', '41'], 41),
    (['Qual 7'], 7),
    (['Madtown Throwdown'], None),
    (['Semifinal 3'], None),
    ([], None),
])
def test_reading_the_match_off_the_overlay(texts, want):
    from autoscout.overlay import qual_number

    assert qual_number(texts) == want


@pytest.mark.parametrize('line,want', [
    ('RuntimeError: WRONG_MATCH: this footage shows Qualification 2, not 1', 'this footage shows Qualification 2, not 1'),
    ('RuntimeError: could not locate scoreboard (timer + scores) in overlay', 'Could not read the scoreboard'),
    ('something else', 'Processing failed'),
])
def test_a_lead_sees_why_a_match_failed(tmp_path, line, want):
    log = tmp_path / 'job.log'
    log.write_text('noise\n' + line + '\n')
    assert want in B.failure_reason(log)


def test_clips_only_live_job_cuts_but_never_queues(tmp_path):
    got = []
    s = live(tmp_path, [{'key': MK, 'actual_time': 2_000}], 99_999, got)
    s.job['scout'] = False
    s.tick(now=5_000)
    assert got == [] and s.seen == {MK: ''}


def test_kept_clip_is_used_when_the_lead_scouts_it_later(tmp_path):
    clip = tmp_path / f'{MK}_live_abc.mp4'
    clip.write_bytes(b'x')
    used = []
    B.Runner(tmp_path, FakeTBA(), process=lambda k, f, l, v=None: used.append(v) or result(k))(MK, True, tmp_path / 'log', {})
    assert used == [clip]


def test_live_backlog_is_not_capped(tmp_path):
    q = Queue(tmp_path)
    for i in range(1, 31):
        q.enqueue(json.dumps({'match_keys': [f'2026cc_qm{i}'], 'force': False}), f'k{i}', limit=None)
    assert sum(1 for _ in q.connect().execute('SELECT id FROM jobs')) == 30


def test_live_job_reports_each_matchs_state(world):
    backend, b, sessions = live_bridge(world)
    backend.jobs = [job('live', match_keys=[])]  # no link: the event's stream
    b.tick()
    b.tick()
    assert backend.syncs()[-1]['jobs'][-1]['progress']['queued'] == [MK]


def test_clip_cache_keeps_the_newest(tmp_path):
    for i, name in enumerate(('old', 'mid', 'new')):
        p = tmp_path / f'{name}.mp4'
        p.write_bytes(b'x' * 100)
        import os
        os.utime(p, (1_000 + i, 1_000 + i))
    B.prune_clips(tmp_path, keep_gb=250 / 1e9)
    assert sorted(p.stem for p in tmp_path.glob('*.mp4')) == ['mid', 'new']
