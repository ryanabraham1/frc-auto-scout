"""Resume qualification processing after an existing GPU job, then save evaluations."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import psutil
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait-pid', type=int)
    args = ap.parse_args()
    with FileLock(str(ROOT / 'data/complete_2026cc.lock'), timeout=0):
        if args.wait_pid:
            try:
                job = psutil.Process(args.wait_pid)
                print(f'Waiting for GPU job {job.pid}: {job.cmdline()}', flush=True)
                job.wait()
            except psutil.NoSuchProcess:
                pass
        for attempt in range(2):
            print(f'Qualification processing pass {attempt + 1}', flush=True)
            subprocess.run([sys.executable, '-u', '-m', 'autoscout.cli', 'run-event',
                            '2026cc', '--levels', 'qm', '--workers', '1'], cwd=ROOT, check=True)
        from autoscout import db
        from autoscout.tba import TBA
        expected = {m['key'] for m in TBA().event_matches('2026cc') if m['comp_level'] == 'qm'}
        con = db.connect()
        done = {r['match_key'] for r in con.execute("SELECT match_key FROM matches WHERE status='done'")}
        missing = sorted(expected - done)
        report = {'expected': len(expected), 'done': len(expected & done), 'missing': missing}
        (ROOT / 'data/2026cc_completion.json').write_text(json.dumps(report, indent=2))
        print(report, flush=True)
        files = [str(ROOT / 'data/results' / f'{k}.json') for k in sorted(expected & done)]
        with (ROOT / 'data/2026cc_fuel_opr.log').open('w') as out:
            subprocess.run([sys.executable, 'scripts/fuel_opr_check.py', '2026cc', *files],
                           cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, 'scripts/audit_2026_identity.py'], cwd=ROOT, check=True)
        if missing:
            raise SystemExit('Some qualifications remain unprocessed; see data/2026cc_completion.json')


if __name__ == '__main__':
    main()
