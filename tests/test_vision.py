import unittest
import cv2
import numpy as np
from collections import deque
from types import SimpleNamespace
from threading import Lock
from unittest.mock import patch

from game_state import Target
from optical_flow import FlowTracker
from object_detector import assign_ids
from hand_tracker import HandInput, GestureInterpreter
from test_open_fist import hand_points, v_points
from vision import VisionSystem


class VisionTests(unittest.TestCase):
    def test_bbox_and_mask_have_different_hit_regions(self):
        mask=np.zeros((100,100),bool)
        mask[30:70,40:60]=True
        target=Target(1,'bottle',(.2,.2,.8,.8),mask=mask)
        self.assertTrue(target.contains((.3,.3)))
        self.assertFalse(target.contains((.3,.3),precision=True))
        self.assertTrue(target.contains((.5,.5),precision=True))
        self.assertFalse(target.contains((1,1),precision=True))
        self.assertFalse(Target(2,'box',(0,0,1,1)).contains((.5,.5),precision=True))

    def test_optical_flow_uses_image_motion_and_has_cooldown(self):
        rng=np.random.default_rng(3)
        frame=np.zeros((240,320,3),np.uint8)
        patch=rng.integers(0,255,(100,100,3),dtype=np.uint8)
        frame[70:170,70:170]=patch
        flow=FlowTracker()
        flow.update(frame,(.18,.2,.65,.8),1)
        shifted=np.zeros_like(frame); shifted[70:170,82:182]=patch
        result=flow.update(shifted,(.22,.2,.69,.8),1.05)
        self.assertEqual(result.swipe,1)
        self.assertGreater(result.speed,.3)
        interpreter=GestureInterpreter()
        for i in range(8): hand=interpreter.update(hand_points(),1+i*.04,aspect=1,gameplay=True)
        interpreter.apply_swipe(hand,result.swipe,1.28)
        self.assertEqual(hand.swipe,0)  # Open-hand aiming cannot spend EMP.
        for i in range(8): hand=interpreter.update(v_points(),2+i*.04,aspect=1,gameplay=True)
        interpreter.apply_swipe(hand,result.swipe,2.28)
        self.assertEqual(hand.swipe,1)  # Real LK result activates a deliberate V command.
        self.assertFalse(hand.fire or hand.pause)
        shifted2=np.zeros_like(frame); shifted2[70:170,94:194]=patch
        result2=flow.update(shifted2,(.26,.2,.73,.8),1.1)
        self.assertEqual(result2.swipe,0)

    def test_static_scene_and_camera_pan_do_not_trigger(self):
        rng=np.random.default_rng(7)
        frame=rng.integers(0,255,(240,320,3),dtype=np.uint8)
        flow=FlowTracker()
        flow.update(frame,(.3,.2,.7,.8),1)
        self.assertEqual(flow.update(frame,(.3,.2,.7,.8),1.05).swipe,0)
        shifted=cv2.warpAffine(frame,np.float32([[1,0,12],[0,1,0]]),(320,240))
        self.assertEqual(flow.update(shifted,(.3,.2,.7,.8),1.1).swipe,0)

    def test_repeated_detection_preserves_ids_one_to_one(self):
        previous=[Target(9,'cup',(.1,.2,.3,.4)),Target(10,'cup',(.6,.2,.8,.4))]
        current=[Target(0,'cup',(.11,.2,.31,.4)),Target(0,'cup',(.59,.2,.79,.4))]
        targets,next_id=assign_ids(current,previous,11)
        self.assertEqual([t.id for t in targets],[9,10])
        self.assertEqual(next_id,11)

    def make_buffer(self):
        # Exercise the real polling code without starting a physical camera or threads.
        system=VisionSystem.__new__(VisionSystem)
        frame=np.zeros((16,16,3),np.uint8)
        system.camera=SimpleNamespace(latest=lambda:(frame,10.0,1),failed=False,status='online')
        system.lock=Lock()
        system.hand=HandInput(tracked=True,aiming=True,fist=True)
        system.hand_stamp=10.0
        system.events=deque()
        system.targets=[Target(1,'cup',(.2,.2,.6,.6))]
        system.target_stamp=10.0
        system.chapter=2; system.generation=4; system.target_generation=4
        system.hand_error=''; system.model_errors={}; system.model_status=''; system.metrics={}
        return system

    def test_old_and_wrong_chapter_targets_are_rejected(self):
        system=self.make_buffer()
        with patch('vision.time.monotonic',return_value=10.1):
            self.assertEqual(len(system.poll().targets),1)
            system.target_stamp=9.0
            self.assertEqual(system.poll().targets,[])
            system.target_stamp=10.0; system.target_generation=3
            self.assertEqual(system.poll().targets,[])

    def test_events_are_once_only_and_expire_after_interruption(self):
        system=self.make_buffer()
        system.events.extend([(9.0,True,False,0,(.2,.3)),(10.0,True,False,0,(.4,.5))])
        with patch('vision.time.monotonic',return_value=10.1):
            event=system.poll().hand
            self.assertTrue(event.fire)
            self.assertEqual(event.shot_aim,(.4,.5))
            self.assertFalse(system.poll().hand.fire)
        system.events.append((10.0,True,False,0,(.4,.5)))
        with patch('vision.time.monotonic',return_value=10.4):
            sample=system.poll()
            self.assertFalse(sample.hand.tracked)
            self.assertFalse(sample.hand.fire)
        system.set_chapter(3)
        self.assertEqual(system.targets,[])
        self.assertEqual(system.generation,5)


if __name__=='__main__':
    unittest.main()
