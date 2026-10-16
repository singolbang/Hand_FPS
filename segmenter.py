"""YOLO11n-seg instance masks; never approximate masks with bounding boxes."""
import cv2
from object_detector import ObjectDetector


class Segmenter(ObjectDetector):
    filename='yolo11n-seg.pt'
    precision=True

    def masks(self,result,shape):
        if result.masks is None: return None
        h,w=shape
        masks=result.masks.data.cpu().numpy()
        # retina_masks=True gives source-sized masks, including undoing letterboxing.
        return [(mask>.5) if mask.shape==(h,w) else (cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)>.5) for mask in masks]
