"""Bounded latest-frame workers. Only the current chapter performs heavy inference."""
from collections import deque
from dataclasses import dataclass, field, replace
import logging
import os
import threading
import time

import cv2
import numpy as np
from config import MODEL_DIR
from hand_tracker import HandInput, HandTracker, GestureInterpreter
from optical_flow import FlowTracker

log=logging.getLogger(__name__)


@dataclass
class Snapshot:
    hand: HandInput = field(default_factory=HandInput)
    frame: object = None
    targets: list = field(default_factory=list)
    status: str = ''
    metrics: dict = field(default_factory=dict)
    availability: dict = field(default_factory=dict)


class CameraFeed:
    def __init__(self,index):
        self.index=index
        self.frame=None
        self.stamp=0.0
        self.sequence=0
        self.status='Opening camera...'
        self.failed=False
        self.lock=threading.Lock()
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,name='camera',daemon=True)
        self.thread.start()

    def run(self):
        cap=None
        try:
            cap=cv2.VideoCapture(self.index,cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY)
            if not cap.isOpened() and os.name=='nt':
                cap.release()
                cap=cv2.VideoCapture(self.index)
            if not cap.isOpened(): raise RuntimeError(f'Camera {self.index} unavailable. Check camera permission, close other apps, or use --camera 1 / --demo.')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            cap.set(cv2.CAP_PROP_FPS,30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            failures=0
            while not self.stop.is_set():
                ok,frame=cap.read()
                if not ok:
                    failures+=1
                    if failures>10: raise RuntimeError('Camera disconnected. Reconnect and restart; ESC exits safely.')
                    self.stop.wait(.05)
                    continue
                failures=0
                frame=cv2.flip(frame,1)
                with self.lock:
                    self.frame=frame
                    self.stamp=time.monotonic()
                    self.sequence+=1
                    self.status='Camera online'
        except Exception as exc:
            with self.lock:
                self.failed=True
                self.status=str(exc)
                self.frame=None
            log.warning('%s',exc)
        finally:
            if cap is not None: cap.release()

    def latest(self):
        with self.lock: return self.frame,self.stamp,self.sequence

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive(): log.warning('Camera driver did not return within 3s; daemon will exit with the process.')


class VisionSystem:
    def __init__(self,settings):
        cv2.setNumThreads(2)
        self.settings=settings
        self.camera=CameraFeed(settings.camera)
        self.stop=threading.Event()
        self.lock=threading.Lock()
        self.hand=HandInput()
        self.hand_stamp=0.0
        self.events=deque(maxlen=24)
        self.targets=[]
        self.target_stamp=0.0
        self.target_generation=-1
        self.chapter=0
        # 0: index-controlled menus, -1: palm calibration, 1..3: gameplay.
        self.generation=0
        self.hand_error=''
        self.model_errors={}
        self.model_status=''
        self.metrics={}
        self.workers=[threading.Thread(target=self.hand_loop,name='hand-flow',daemon=True),threading.Thread(target=self.object_loop,name='objects',daemon=True)]
        for worker in self.workers: worker.start()

    def set_chapter(self,chapter):
        with self.lock:
            if chapter!=self.chapter:
                self.chapter=chapter
                self.generation+=1
                self.targets=[]
                self.events.clear()

    def hand_loop(self):
        tracker=None
        try:
            tracker=HandTracker()
            interpreter=GestureInterpreter(self.settings)
            flow=FlowTracker()
            previous=-1
            while not self.stop.is_set():
                frame,stamp,sequence=self.camera.latest()
                if frame is None or sequence==previous:
                    self.stop.wait(.01)
                    continue
                previous=sequence
                started=time.monotonic()
                points=tracker.detect(frame,stamp)
                with self.lock:
                    gameplay = self.chapter in (1,2,3)
                    calibrating = self.chapter == -1
                    generation = self.generation
                hand=interpreter.update(points,stamp,frame.shape[1]/frame.shape[0],gameplay=gameplay,calibrating=calibrating)
                roi=None
                if points is not None:
                    lo=np.clip(points[:,:2].min(axis=0)-.045,0,1)
                    hi=np.clip(points[:,:2].max(axis=0)+.045,0,1)
                    roi=(*lo,*hi)
                motion=flow.update(frame,roi,stamp)
                interpreter.apply_swipe(hand,motion.swipe,stamp)
                with self.lock:
                    if generation != self.generation:
                        continue
                    self.hand=replace(hand,fire=False,pause=False,swipe=0,shot_aim=None)
                    self.hand_stamp=stamp
                    if hand.fire or hand.pause or hand.swipe:
                        self.events.append((stamp,hand.fire,hand.pause,hand.swipe,hand.shot_aim))
                    self.metrics.update(hand_ms=(time.monotonic()-started)*1000,flow=motion.speed,flow_points=motion.points)
                self.stop.wait(max(0,1/30-(time.monotonic()-started)))
        except Exception as exc:
            with self.lock: self.hand_error=f'Hand tracking unavailable: {exc}'
            log.exception('Hand tracking worker stopped')
        finally:
            if tracker is not None: tracker.close()

    def object_loop(self):
        models={}
        previous=-1
        while not self.stop.is_set():
            with self.lock: chapter,generation=self.chapter,self.generation
            if chapter not in (2,3) or chapter in self.model_errors:
                self.stop.wait(.05)
                continue
            frame,stamp,sequence=self.camera.latest()
            if frame is None or sequence==previous:
                self.stop.wait(.02)
                continue
            previous=sequence
            started=time.monotonic()
            try:
                if chapter not in models:
                    with self.lock: self.model_status=f'Loading Chapter {chapter} model on CPU...'
                    if chapter==2:
                        from object_detector import ObjectDetector
                        models[chapter]=ObjectDetector(self.settings)
                    else:
                        from segmenter import Segmenter
                        models[chapter]=Segmenter(self.settings)
                    # Loading can be slow; use a fresh frame after initialization.
                    frame,stamp,previous=self.camera.latest()
                    if frame is None: continue
                targets=models[chapter].detect(frame)
                with self.lock:
                    if generation==self.generation:
                        self.targets=targets
                        self.target_stamp=stamp
                        self.target_generation=generation
                        self.model_status=''
                        self.metrics['model_ms']=(time.monotonic()-started)*1000
            except Exception as exc:
                with self.lock:
                    self.model_errors[chapter]=f'Chapter {chapter} model error: {exc}'
                    self.model_status=self.model_errors[chapter]
                    self.targets=[]
                log.exception('Object inference stopped for chapter %s',chapter)
            self.stop.wait(max(.01,1/self.settings.infer_hz-(time.monotonic()-started)))
        models.clear()

    def poll(self):
        now=time.monotonic()
        frame,camera_stamp,_=self.camera.latest()
        with self.lock:
            hand=replace(self.hand)
            if now-self.hand_stamp>.30 or now-camera_stamp>.5:
                hand=HandInput(aim=hand.aim)
            while self.events:
                stamp,fire,pause,swipe,shot_aim=self.events.popleft()
                if hand.tracked and now-stamp<=.25:
                    hand.fire |= fire
                    hand.pause |= pause
                    hand.swipe = swipe or hand.swipe
                    if fire: hand.shot_aim=shot_aim
            if hand.swipe: hand.fire=hand.pause=False
            targets=list(self.targets) if self.target_generation==self.generation and now-self.target_stamp<=.65 and now-camera_stamp<.5 else []
            status=self.hand_error or self.camera.status if self.camera.failed else self.hand_error or self.model_status
            if not status:
                if self.chapter == -1:
                    status='Open hand: hold your palm at each corner / point to CANCEL'
                elif self.chapter:
                    status='Open hand: aim / clench fist: fire / V hold: pause / V swipe: EMP'
                else:
                    status='Point and hold to select / hold V to go back'
                if not hand.tracked: status='Show one hand. '+status
            availability={1:'',2:'',3:''}
            for chapter,name in ((2,'yolo11n.pt'),(3,'yolo11n-seg.pt')):
                if not (MODEL_DIR/name).is_file(): availability[chapter]=f'Missing {name}. Run python tools/download_models.py'
                if chapter in self.model_errors: availability[chapter]=self.model_errors[chapter]
            return Snapshot(hand,frame if now-camera_stamp<.5 else None,targets,status,dict(self.metrics),availability)

    def close(self):
        self.stop.set()
        self.camera.close()
        for worker in self.workers:
            worker.join(timeout=4)
            if worker.is_alive(): log.warning('%s is still finishing native inference; process exit will stop it.',worker.name)
