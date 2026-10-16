"""Real pretrained YOLO11 nano detections; normalized coordinates and light IoU IDs."""
from config import MODEL_DIR
from game_state import Target

# Everyday, stationary props. Exclude people and hands as shooting targets.
TARGET_CLASSES=(24,26,28,39,41,42,43,44,46,47,49,56,57,58,59,60,62,63,64,65,66,67,73,74,75,76,77,79)


def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1])
    x1,y1=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,x1-x0)*max(0,y1-y0)
    area=lambda box:max(0,box[2]-box[0])*max(0,box[3]-box[1])
    return intersection/max(1e-8,area(a)+area(b)-intersection)


def assign_ids(current,previous,next_id):
    used=set()
    for target in current:
        matches=[(iou(target.box,p.box),p) for p in previous if p.label==target.label and p.id not in used]
        overlap,match=max(matches,key=lambda pair:pair[0],default=(0,None))
        if overlap>.3:
            target.id=match.id
            used.add(match.id)
        else:
            target.id=next_id
            next_id+=1
    return current,next_id


class ObjectDetector:
    filename='yolo11n.pt'
    precision=False

    def __init__(self,settings,model_path=None,classes=TARGET_CLASSES):
        path=model_path or MODEL_DIR/self.filename
        if not path.is_file():
            raise FileNotFoundError(f'{path.name} missing. Run: python tools/download_models.py')
        import torch
        torch.set_num_threads(2)
        from ultralytics import YOLO
        self.model=YOLO(str(path),task='segment' if self.precision else 'detect')
        self.settings=settings
        self.classes=classes
        self.previous=[]
        self.next_id=1

    def detect(self,frame):
        result=self.model.predict(source=frame,imgsz=self.settings.infer_size,conf=self.settings.confidence,
                                  device='cpu',classes=list(self.classes) if self.classes is not None else None,
                                  max_det=12,verbose=False,retina_masks=self.precision)[0]
        targets=[]
        if result.boxes is not None:
            boxes=result.boxes.xyxyn.cpu().numpy()
            scores=result.boxes.conf.cpu().numpy()
            classes=result.boxes.cls.cpu().numpy().astype(int)
            masks=self.masks(result,frame.shape[:2]) if self.precision else None
            for i,(box,score,cls) in enumerate(zip(boxes,scores,classes)):
                if cls==0: continue
                mask=masks[i] if masks is not None and i<len(masks) else None
                if self.precision and mask is None: continue
                targets.append(Target(0,str(result.names[int(cls)]),tuple(float(x) for x in box),float(score),mask))
        targets,self.next_id=assign_ids(targets,self.previous,self.next_id)
        self.previous=targets
        return targets

    def masks(self,result,shape):
        raise NotImplementedError('Use Segmenter for instance masks')
