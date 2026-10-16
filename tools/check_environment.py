"""Reproducible dependency, model, and optional live camera diagnostics."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',action='store_true',help='Run real models on an official public sample; downloads ~140KB')
    p.add_argument('--camera',type=int,help='Probe live camera for 5 seconds. No camera frames are saved.')
    args=p.parse_args()
    report={'python':platform.python_version(),'platform':platform.platform()}
    for name in ('numpy','opencv-python','pygame-ce','mediapipe','torch','torchvision','ultralytics'):
        try: report[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report[name]='MISSING'
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    import torch
    report['cuda_available']=torch.cuda.is_available()
    if args.models:
        from config import Settings
        from hand_tracker import HandTracker
        from object_detector import ObjectDetector
        from segmenter import Segmenter
        cache=ROOT/'artifacts'
        cache.mkdir(exist_ok=True)
        sample=cache/'bus.jpg'
        if not sample.exists():
            url='https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg'
            with urllib.request.urlopen(url,timeout=30) as r: sample.write_bytes(r.read(1_000_000))
        frame=cv2.imread(str(sample))
        if frame is None: raise RuntimeError('Public sample could not be decoded')
        frame=cv2.resize(frame,(480,640))
        settings=Settings()
        for cls in (ObjectDetector,Segmenter):
            model=cls(settings,classes=(5,))  # Public sample is a bus, not an everyday game prop.
            model.detect(frame)  # Warmup is reported separately from steady inference.
            times=[]
            for _ in range(3):
                start=time.perf_counter(); targets=model.detect(frame)
                times.append((time.perf_counter()-start)*1000)
            assert targets, f'{cls.__name__} did not detect the public sample bus'
            report[cls.__name__]={'targets':[t.label for t in targets],'mean_ms':round(sum(times)/len(times),1),'mask_pixels':[int(t.mask.sum()) for t in targets if t.mask is not None]}
            if cls is Segmenter:
                assert all(t.mask.shape==frame.shape[:2] and t.mask.any() for t in targets)
                t=targets[0]
                yy,xx=np.where(t.mask)
                index=len(xx)//2
                assert t.contains((xx[index]/frame.shape[1],yy[index]/frame.shape[0]),precision=True)
                overlay=frame.copy(); overlay[t.mask]=(60,210,245)
                cv2.imwrite(str(cache/'real-segmentation-check.jpg'),cv2.addWeighted(frame,.55,overlay,.45,0))
        tracker=HandTracker()
        try:
            start=time.perf_counter(); result=tracker.detect(np.zeros((480,640,3),np.uint8),time.monotonic())
            report['hand_model']={'blank_frame_ms':round((time.perf_counter()-start)*1000,1),'correctly_empty':result is None}
            assert result is None
        finally: tracker.close()
    if args.camera is not None:
        from config import Settings
        from vision import VisionSystem
        system=VisionSystem(Settings(camera=args.camera))
        began=time.monotonic(); frames=0; tracked=0
        try:
            while time.monotonic()-began<5:
                snapshot=system.poll()
                frames+=snapshot.frame is not None
                tracked+=snapshot.hand.tracked
                time.sleep(.05)
            report['camera']={'image_available':snapshot.frame is not None,'samples_with_image':frames,'samples_with_hand':tracked,'status':snapshot.status,'metrics':snapshot.metrics}
        finally: system.close()
    print(json.dumps(report,indent=2))
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts'/'environment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__': main()
