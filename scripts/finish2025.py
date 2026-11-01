"""Finish the 2025casj accuracy data pass after an existing event worker exits.

Runs sequentially: retry failed available videos, fit cameras, fill reach samples,
reanalyze, evaluate a pinned subset, build review sheets, and render four demos.
Stage and command logs are in data/accuracy_2025/completion/. No git changes are
published unless --publish is explicitly supplied.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
from filelock import FileLock

from autoscout import config, db, pipeline
from autoscout.tba import TBA

EVENT = '2025casj'
OUT = config.DATA_DIR / 'accuracy_2025' / 'completion'


def run(name, args):
    print(name, flush=True)
    with (OUT / f'{name}.log').open('w', encoding='utf-8') as log:
        subprocess.run(args, cwd=config.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def finish(args):
    OUT.mkdir(parents=True, exist_ok=True)
    status = {'event': EVENT, 'state': 'running', 'completed_stages': []}
    def stage(name):
        status['stage'] = name
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        (OUT / 'status.json').write_text(json.dumps(status, indent=2))
        print(name, flush=True)
    try:
        stage('waiting_for_event_run')
        if args.wait_pid and psutil.pid_exists(args.wait_pid):
            proc = psutil.Process(args.wait_pid)
            if 'run-event' not in ' '.join(proc.cmdline()):
                raise RuntimeError('The requested PID is not an event run')
            try:
                while proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    time.sleep(5)
            except psutil.NoSuchProcess:
                pass
        tba = TBA()
        official = [m for m in tba.event_matches(EVENT) if m['comp_level'] == 'qm']
        # qm19's only supplied video is blocked by its copyright owner.
        blocked = [m['key'] for m in official if m['key'] == f'{EVENT}_qm19']
        expected = [m['key'] for m in official if m['key'] not in blocked]
        con = db.connect()
        states = {r['match_key']: r['status'] for r in con.execute(
            'SELECT match_key,status FROM matches WHERE event_key=?', (EVENT,))}
        retry = [k for k in expected if states.get(k) != 'done' or not (pipeline.RESULTS_DIR / f'{k}.json').exists()]
        # qm6 failed after frame processing in full2 and was omitted from full3.
        # A later reanalysis can mark its older cached result done, so DB status
        # alone cannot establish that this match was freshly reprocessed.
        if f'{EVENT}_qm6' in expected and f'{EVENT}_qm6' not in retry:
            retry.append(f'{EVENT}_qm6')
        stage('retry_failed_matches')
        if retry:
            run('retry', [sys.executable, '-m', 'autoscout.cli', 'run-event', EVENT,
                          '--workers', '1', '--matches', ','.join(retry)])
        states = {r['match_key']: r['status'] for r in con.execute(
            'SELECT match_key,status FROM matches WHERE event_key=?', (EVENT,))}
        failed = [k for k in expected if states.get(k) != 'done'
                  or not (pipeline.RESULTS_DIR / f'{k}.json').exists()]
        if failed:
            raise RuntimeError(f'Available matches still failed: {failed}')
        status['blocked_matches'] = blocked
        status['completed_stages'].append('processing')
        stage('fit_closeup_cameras')
        cameras = pipeline.calibrate_closeups(EVENT)
        (OUT / 'cameras.json').write_text(json.dumps(cameras, indent=2))
        status['completed_stages'].append('camera_fit')
        stage('reach_and_reanalysis')
        for key in sorted(expected):
            r = json.loads((pipeline.RESULTS_DIR / f'{key}.json').read_text())
            if not any(r.get('lift_obs', {}).values()) or not r.get('aux_reach'):
                r = pipeline.lift_pass(key)
            else:
                r = pipeline.reanalyze(key, tba)
            db.save_result(con, r, str(pipeline.RESULTS_DIR / f'{key}.json'))
            print('reanalyzed', key, flush=True)
        status['completed_stages'].append('reanalysis')
        stage('disjoint_match_evaluation')
        run('evaluation', [sys.executable, 'scripts/eval2025.py', EVENT, '--saved',
                           '--matches', str(args.matches), '--out', str(OUT / 'evaluation.json')])
        status['completed_stages'].append('evaluation')
        stage('review_sheets')
        for number in (2, 11, 12):
            key = f'{EVENT}_qm{number}'
            run(f'review_qm{number}', [sys.executable, 'scripts/credit_review.py', 'sheet', key,
                                      str(OUT / key)])
        status['completed_stages'].append('review_sheets')
        stage('debug_videos')
        demos = config.ROOT / 'demos'
        demos.mkdir(exist_ok=True)
        for number in (2, 11, 12, 20):
            key = f'{EVENT}_qm{number}'
            r = json.loads((pipeline.RESULTS_DIR / f'{key}.json').read_text())
            src = next(pipeline.VIDEO_DIR.glob(f"{r['video']}.*"))
            dest = demos / f'{key}_annotated.mp4'
            run(f'render_qm{number}', [sys.executable, 'scripts/render_demo.py', key, str(src), str(dest)])
            import cv2
            cap = cv2.VideoCapture(str(dest))
            ok, _ = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f'Rendered video is unreadable: {dest}')
        status['completed_stages'].append('debug_videos')
        metrics = json.loads((OUT / 'evaluation.json').read_text())
        report = ['# 2025casj final data pass', '',
                  f"Processed and reanalyzed {len(expected)} qualification matches. Blocked video: qm19.",
                  '', f"OPR uses {len(metrics['reference_matches'])} matches disjoint from the pinned "
                  f"{len(metrics['matches'])}-match evaluation subset.",
                  'This is a development diagnostic, not an independent human scoring benchmark.', '',
                  '| Metric | Correlation | MAE |', '|---|---:|---:|']
        report += [f"| {k} | {v['r']:.3f} | {v['mae']:.2f} |" for k, v in metrics['metrics'].items()]
        report += ['', 'Camera fit, evaluation, and review sheets: `data/accuracy_2025/completion/`.',
                   'Videos: `demos/2025casj_qm{2,11,12,20}_annotated.mp4`.',
                   'Generated review sheets still require visual inspection; generation is not a visual accuracy check.', '']
        report_path = config.ROOT / 'docs' / '2025_ACCURACY_FINAL_DATA.md'
        report_path.write_text('\n'.join(report))
        progress = config.ROOT / 'docs' / '2025_ACCURACY_PROGRESS.md'
        text = progress.read_text()
        heading = '## Full event data pass still running'
        if heading in text:
            start = text.index(heading)
            end = text.index('## Deferred work', start)
            summary = ('## Full event data pass finished (qm19 blocked)\n\n'
                       f'Processed and reanalyzed {len(expected)} available qualification matches. '
                       'qm19 remains blocked by YouTube. Close-up cameras were refitted, the pinned '
                       'subset was evaluated, review sheets were regenerated, and qm2/qm11/qm12/qm20 '
                       'debug videos were rendered and checked for decoding.\n\n'
                       'See [the final data report](2025_ACCURACY_FINAL_DATA.md) and '
                       '`data/accuracy_2025/completion/status.json`. Generated final sheets still '
                       'need visual inspection; the earlier snapshot spot-check is documented above.\n\n')
            progress.write_text(text[:start] + summary + text[end:])
        if args.publish:
            stage('publish_data_report')
            paths = ['configs/events/2025casj_cams.json', 'docs/2025_ACCURACY_FINAL_DATA.md',
                     'docs/2025_ACCURACY_PROGRESS.md']
            run('git_add', ['git', 'add', '--', *paths])
            changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', *paths], cwd=config.ROOT).returncode
            if changed:
                run('git_commit', ['git', 'commit', '--only', '-m', 'Record 2025casj final accuracy data pass', '--', *paths])
                run('git_push', ['git', 'push', 'origin', 'HEAD'])
        status['state'] = 'complete_except_blocked_video'
        stage('finished')
    except Exception as exc:
        status['state'] = 'failed'
        status['error'] = str(exc)
        stage('failed')
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wait-pid', type=int)
    ap.add_argument('--matches', type=Path, default=config.ROOT / 'docs' / '2025_ACCURACY_MATCHES.json')
    ap.add_argument('--publish', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUT / 'finish.lock'), timeout=0):
        finish(args)
