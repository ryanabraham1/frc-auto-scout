"""Add explicitly reviewed hard-frame boxes to the existing detector training set."""
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autoscout import config
from autoscout.train.sam_label import write_label


def main():
    data=json.loads((config.ROOT/'benchmarks/attribution/2026cc_hard_robot_frames.json').read_text())
    for r in data['frames']:
        result=json.loads((config.DATA_DIR/'results'/f"{r['match']}.json").read_text())
        cap=cv2.VideoCapture(str(config.DATA_DIR/'videos'/f"{result['video']}.mp4"))
        cap.set(cv2.CAP_PROP_POS_MSEC,r['t']*1000)
        ok,frame=cap.read();cap.release()
        if not ok:raise RuntimeError(f"Missing frame {r['match']} {r['t']}")
        x,y,w,h=map(int,result['calibration']['view'])
        h=min(h,r.get('crop_bottom',y+h)-y)
        boxes=[(a,x0-x,y0-y,x1-x,y1-y) for a,x0,y0,x1,y1 in r['robots']]
        stem=f"{r['match']}_reviewed_{round(r['t']*1000)}"
        write_label(frame[y:y+h,x:x+w],boxes,stem,dataset=config.DATA_DIR/'datasets/hand')
        preview=frame[y:y+h,x:x+w].copy()
        for a,x0,y0,x1,y1 in boxes:
            color=(0,0,255) if a=='red' else (255,100,0)
            cv2.rectangle(preview,(x0,y0),(x1,y1),color,2)
        cv2.imwrite(str(config.DATA_DIR/'runs'/f'{stem}.jpg'),preview)
        print(stem,len(boxes),'robots')


if __name__=='__main__':main()
