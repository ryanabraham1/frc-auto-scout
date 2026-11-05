"""Fetch bounded excerpts for missing archived matches from the official RSN broadcast.

The windows are estimated from TBA actual_time and the stream release timestamp.
Verify the match number and alliance roster in the excerpt before registering it.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoscout import config

SOURCE = 'zQ8Oibbg9Ro'
WINDOWS = {'2026cc_qm40': (27700,28150), '2026cc_qm56': (36520,36946)}


def fetch(item):
    key, (start,end) = item
    out = config.DATA_DIR/'videos'/f'{key}.mp4'
    if out.exists():
        return key, str(out)
    cmd = [sys.executable,'-m','yt_dlp','--no-playlist','--no-progress',
           '--js-runtimes','node','--remote-components','ejs:github',
           '--download-sections',f'*{start}-{end}',
           '-f','bv*[height<=1080][vcodec^=avc1]+ba/b[height<=1080]/b',
           '--merge-output-format','mp4','-o',str(out)]
    if cookies := config.yt_cookies():
        cmd += ['--cookies',str(cookies)]
    elif browser := config.yt_cookies_browser():
        cmd += ['--cookies-from-browser',browser]
    cmd.append(f'https://www.youtube.com/watch?v={SOURCE}')
    log = config.DATA_DIR/'runs'/f'{key}_excerpt_download.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('w') as stream:
        subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
    info = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                           '-of','json',str(out)],capture_output=True,text=True,check=True)
    duration = float(json.loads(info.stdout)['format']['duration'])
    if duration > end-start+10 or duration < 180:
        raise RuntimeError(f'Unexpected excerpt duration for {key}: {duration}')
    print(key, 'downloaded',duration,'seconds; verify overlay before processing',flush=True)
    return key,str(out)


if __name__ == '__main__':
    with ThreadPoolExecutor(2) as pool:
        for result in pool.map(fetch,WINDOWS.items()):
            print(result,flush=True)
