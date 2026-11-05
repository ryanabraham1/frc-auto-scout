"""Measure same-resolution batching against the current per-frame GPU inference."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from autoscout import config
from autoscout.vision.detect import (RobotDetector,PersonVeto,merge_scales,is_person_box,
                                    precision_args,Det,iou)


def main():
    result=json.loads((config.DATA_DIR/'results/2026cc_qm62.json').read_text())
    cap=cv2.VideoCapture(str(config.DATA_DIR/'videos'/f"{result['video']}.mp4"))
    x,y,w,h=map(int,result['calibration']['view'])
    images=[]
    for t in np.arange(40,43,.25):
        cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000)
        ok,f=cap.read()
        if ok:images.append(f[y:y+h,x:x+w].copy())
    cap.release()
    robot=RobotDetector(conf=.1,imgsz=1600)
    veto=PersonVeto()
    def run(batch_size):
        out=[]
        for i in range(0,len(images),batch_size):
            frames=images[i:i+batch_size]
            scales=[robot(frames,conf=.1,imgsz=s) for s in (1280,1600)]
            people=veto.model.predict(frames,conf=veto.conf,classes=[0],imgsz=veto.imgsz,
                                      verbose=False,rect=True,device=veto.device,
                                      **precision_args(veto.device=='cuda'))
            for j,p in enumerate(people):
                boxes=[Det(*b,0.,'') for b in p.boxes.xyxy.tolist()]
                out.append([d for d in merge_scales([s[j] for s in scales],.1)
                            if not any(is_person_box(d,b) for b in boxes)])
        torch.cuda.synchronize()
        return out
    run(1);run(4)
    timings={1:[],2:[],4:[]}
    outputs={}
    for size in (1,4,2,4,1,2):
        t=time.perf_counter();outputs[size]=run(size);timings[size].append(time.perf_counter()-t)
    out={'frames':len(images),'seconds':{k:sum(v)/len(v) for k,v in timings.items()},'comparison':{}}
    for size in (2,4):
        mismatches=0
        max_delta=0.
        for a,b in zip(outputs[1],outputs[size]):
            if len(a)!=len(b):mismatches+=1;continue
            for d in a:
                candidates=[e for e in b if e.alliance==d.alliance]
                if not candidates or max(iou(d,e) for e in candidates)<.98:
                    mismatches+=1;continue
                e=max(candidates,key=lambda e:iou(d,e))
                max_delta=max(max_delta,max(abs(v-u) for v,u in zip((d.x0,d.y0,d.x1,d.y1),(e.x0,e.y0,e.x1,e.y1))))
        out['comparison'][size]={'frames_or_boxes_differing':mismatches,'max_box_delta_px':max_delta}
    path=config.DATA_DIR/'inference_batch_benchmark.json'
    path.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
