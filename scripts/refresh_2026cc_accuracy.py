"""Apply verified scoring fixes to cached 2026cc results, backing up every input."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import psutil
from filelock import FileLock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autoscout import config,db
from autoscout.pipeline import RESULTS_DIR,reanalyze,_json_default
from autoscout.tba import TBA


def refresh(done):
    con=db.connect()
    keys=[r['match_key'] for r in con.execute(
        "SELECT match_key FROM matches WHERE event_key='2026cc' AND comp_level='qm' AND status='done'")]
    backup=config.DATA_DIR/'2026cc_before_accuracy_fixes'
    backup.mkdir(exist_ok=True)
    tba=TBA(max_age_s=1e12)
    for key in keys:
        if key in done:
            continue
        path=RESULTS_DIR/f'{key}.json'
        previous=json.loads(path.read_text())
        target=backup/path.name
        if not target.exists():
            shutil.copy2(path,target)
        result=reanalyze(key,tba,save=False)
        for alliance in ('red','blue'):
            summary=result['alliances'][alliance]
            assigned=sum(v['fuel_scored'] for v in result['teams'].values() if v['alliance']==alliance)
            assert assigned+summary['hp_fuel_scored']+summary['unattributed_fuel']==summary['official_fuel']
        temporary = path.with_suffix('.accuracy.tmp')
        temporary.write_text(json.dumps(result,default=_json_default))
        temporary.replace(path)
        db.save_result(con,result,str(path))
        changed=sum(v['fuel_scored']!=previous['teams'].get(str(k),{}).get('fuel_scored')
                    for k,v in result['teams'].items())
        done.add(key)
        print(key,'refreshed;',changed,'team fuel totals changed',flush=True)
        (config.DATA_DIR/'2026cc_accuracy_refresh.json').write_text(json.dumps({'refreshed':sorted(done)},indent=2))
    con.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--wait-pid',type=int)
    args=ap.parse_args()
    with FileLock(str(config.DATA_DIR/'2026cc_accuracy_refresh.lock'),timeout=0):
        done=set()
        refresh(done)
        if args.wait_pid:
            try:
                print('Waiting for completion job',args.wait_pid,flush=True)
                psutil.Process(args.wait_pid).wait()
            except psutil.NoSuchProcess:
                pass
            refresh(done)
        from scripts.audit_2026_identity import main as audit
        audit()


if __name__=='__main__':
    main()
