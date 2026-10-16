"""Actual Lucas-Kanade image flow, with background-motion rejection."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FlowResult:
    swipe: int = 0
    speed: float = 0.0
    points: int = 0
    vector: tuple = (0.0,0.0)


class FlowTracker:
    def __init__(self):
        self.previous = self.roi = self.last_time = None
        self.last_swipe = -10.0

    def reset(self):
        self.previous = self.roi = self.last_time = None

    @staticmethod
    def roi_mask(shape,roi):
        h,w=shape
        x0,y0,x1,y1=roi
        mask=np.zeros(shape,np.uint8)
        mask[max(0,int(y0*h)):min(h,int(y1*h)),max(0,int(x0*w)):min(w,int(x1*w))]=255
        return mask

    @staticmethod
    def vectors(previous,current,mask):
        points=cv2.goodFeaturesToTrack(previous,maxCorners=65,qualityLevel=.015,minDistance=6,mask=mask,blockSize=5)
        if points is None or len(points)<6: return np.empty((0,2))
        options=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.02))
        moved,status,error=cv2.calcOpticalFlowPyrLK(previous,current,points,None,**options)
        if moved is None: return np.empty((0,2))
        back,back_status,_=cv2.calcOpticalFlowPyrLK(current,previous,moved,None,**options)
        if back is None: return np.empty((0,2))
        good=(status.ravel()==1)&(back_status.ravel()==1)&(np.linalg.norm(points-back,axis=2).ravel()<1.5)&(error.ravel()<35)
        return (moved-points).reshape(-1,2)[good]

    def update(self,bgr,roi,now):
        h,w=bgr.shape[:2]
        gray=cv2.cvtColor(cv2.resize(bgr,(320,max(1,int(h*320/w)))),cv2.COLOR_BGR2GRAY)
        result=FlowResult()
        dt=now-self.last_time if self.last_time is not None else 0
        if (self.previous is not None and self.previous.shape==gray.shape and self.roi is not None
                and roi is not None and .01<=dt<=.25):
            mask=self.roi_mask(gray.shape,self.roi)
            vectors=self.vectors(self.previous,gray,mask)
            if len(vectors)>=6:
                foreground=np.median(vectors,axis=0)
                # Features well outside the hand distinguish camera pan from a hand swipe.
                exclusion=cv2.dilate(mask,np.ones((25,25),np.uint8))
                outside=self.vectors(self.previous,gray,255-exclusion)
                background=np.median(outside,axis=0) if len(outside)>=6 else np.zeros(2)
                motion=foreground-background
                coherence=np.mean(np.linalg.norm(vectors-foreground,axis=1)<4)
                speed=float(np.linalg.norm(motion)/gray.shape[1]/dt)
                result=FlowResult(speed=speed,points=len(vectors),vector=tuple(float(x) for x in motion))
                if (abs(motion[0])>3 and abs(motion[0])>abs(motion[1])*1.7 and speed>.38
                        and coherence>.60 and now-self.last_swipe>=1.1):
                    result.swipe=1 if motion[0]>0 else -1
                    self.last_swipe=now
        self.previous,self.roi,self.last_time=gray,roi,now
        return result
