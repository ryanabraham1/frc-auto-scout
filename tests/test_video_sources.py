import json

from autoscout import config
from autoscout.tba import MatchInfo
from autoscout.video import match_video


def raw_match():
    return dict(key='2026test_qm40',event_key='2026test',comp_level='qm',match_number=40,
                alliances={a:dict(team_keys=['frc1','frc2','frc3'],score=0) for a in ('red','blue')},videos=[])


def test_verified_archive_source_fills_only_missing_tba_video(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'EVENTS_DIR',tmp_path)
    path=tmp_path/'2026test_videos.json'
    path.write_text(json.dumps({'2026test_qm40':{'youtube_id':'verified-id'}}))
    raw=raw_match()
    assert MatchInfo.from_tba(raw).videos==['verified-id']
    raw['videos']=[dict(type='youtube',key='official-id')]
    info=MatchInfo.from_tba(raw)
    assert info.videos==['official-id'] and info.video_source=={}


def test_local_excerpt_is_resolved_without_downloading_entire_stream(tmp_path,monkeypatch):
    from autoscout import video
    monkeypatch.setattr(config,'EVENTS_DIR',tmp_path)
    monkeypatch.setattr(config,'ROOT',tmp_path)
    source={'local_path':'data/videos/qm40.mp4','source_youtube_id':'official-stream','source_start_s':100}
    (tmp_path/'2026test_videos.json').write_text(json.dumps({'2026test_qm40':source}))
    path=tmp_path/'data/videos/qm40.mp4'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'excerpt')
    monkeypatch.setattr(video,'ensure_decodable',lambda p:p)
    monkeypatch.setattr(video,'download',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('full stream download')))
    info=MatchInfo.from_tba(raw_match())
    assert match_video(info)==path
    assert info.video_source==source
