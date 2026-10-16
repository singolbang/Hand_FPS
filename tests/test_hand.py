import unittest
import numpy as np

from hand_tracker import TriggerGate, GestureInterpreter, classify_hand
from config import WIDTH, HEIGHT


def gun_points():
    p = np.zeros((21, 3), dtype=float)
    p[0] = [.5, .85, 0]
    for base, x in [(5,.42),(9,.5),(13,.58),(17,.66)]:
        p[base:base+4] = [[x,.6,0],[x,.48,0],[x,.56,0],[x,.68,0]]
    p[5:9] = [[.42,.6,0],[.42,.43,0],[.42,.29,0],[.42,.17,0]]
    p[1:5] = [[.4,.76,0],[.31,.65,0],[.25,.57,0],[.2,.5,0]]
    return p


class HandTests(unittest.TestCase):
    def test_small_fingertip_jitter_does_not_move_cursor(self):
        tracker=GestureInterpreter()
        points=gun_points()
        start=np.array(tracker.update(points,1,aspect=1).aim)
        jittered=points.copy(); jittered[8,0]+=.002
        for n in range(12):
            sample=tracker.update(jittered,1+(n+1)/30,aspect=1)
        np.testing.assert_allclose(sample.aim,start,atol=1e-9)

    def test_large_landmark_jump_is_rate_limited(self):
        tracker=GestureInterpreter()
        points=gun_points()
        before=np.array(tracker.update(points,1,aspect=1).aim)
        moved=points.copy(); moved[8,:2]=[.9,.8]
        after=np.array(tracker.update(moved,1+1/30,aspect=1).aim)
        distance=np.linalg.norm((after-before)*[WIDTH,HEIGHT])
        self.assertLessEqual(distance,50.001)

    def test_reacquired_hand_does_not_snap_cursor_across_screen(self):
        tracker=GestureInterpreter()
        points=gun_points()
        before=np.array(tracker.update(points,1,aspect=1).aim)
        tracker.update(None,1.1)
        points[8,:2]=[.9,.8]
        after=np.array(tracker.update(points,1.15,aspect=1).aim)
        self.assertLessEqual(np.linalg.norm((after-before)*[WIDTH,HEIGHT]),75.001)

    def test_stable_aim_reaches_distant_target_without_losing_screen_range(self):
        tracker=GestureInterpreter()
        points=gun_points(); tracker.update(points,1,aspect=1)
        points[8,:2]=[1,1]
        for n in range(90):
            sample=tracker.update(points,1+(n+1)/30,aspect=1)
        self.assertLess(np.linalg.norm((np.array(sample.aim)-1)*[WIDTH,HEIGHT]),7)

    def test_trigger_requires_release_and_debounce(self):
        gate = TriggerGate()
        self.assertFalse(gate.update(.2, 0))
        self.assertFalse(gate.update(.9, .1))
        self.assertFalse(gate.update(.9, .3))
        self.assertFalse(gate.update(.2, .4))
        self.assertTrue(gate.update(.2, .5))
        self.assertFalse(gate.update(.2, 2))
        gate.update(.9, 2.1)
        gate.update(.9, 2.3)
        gate.update(.2, 2.4)
        self.assertTrue(gate.update(.2, 2.5))

    def test_loss_disarms_trigger(self):
        gate = TriggerGate()
        gate.update(.9, 0)
        gate.update(.9, .2)
        gate.reset()
        gate.update(.2, .3)
        self.assertFalse(gate.update(.2, 1))

    def test_geometry_is_scale_and_rotation_invariant(self):
        p = gun_points()
        a = classify_hand(p, aspect=1)
        theta = .7
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        q = p.copy()
        q[:,:2] = p[:,:2] @ rot.T * .45 + [.2,.1]
        b = classify_hand(q, aspect=1)
        self.assertTrue(a.gun)
        self.assertTrue(b.gun)
        self.assertAlmostEqual(a.thumb_ratio, b.thumb_ratio)

    def test_missing_hand_preserves_cursor_but_blocks_actions(self):
        interpreter = GestureInterpreter()
        first = interpreter.update(gun_points(), 1, aspect=1)
        lost = interpreter.update(None, 1.1)
        self.assertFalse(lost.tracked)
        self.assertFalse(lost.fire)
        self.assertEqual(lost.aim, first.aim)

    def test_frame_gap_disarms_without_explicit_missing_landmarks(self):
        interpreter=GestureInterpreter()
        open_hand=gun_points()
        closed_hand=open_hand.copy(); closed_hand[4]=[.42,.59,0]
        interpreter.update(open_hand,1.0,aspect=1)
        interpreter.update(open_hand,1.2,aspect=1)
        interpreter.update(closed_hand,2.0,aspect=1)
        self.assertFalse(interpreter.update(closed_hand,2.1,aspect=1).fire)
        interpreter.update(open_hand,2.2,aspect=1)
        interpreter.update(open_hand,2.35,aspect=1)
        interpreter.update(closed_hand,2.4,aspect=1)
        self.assertTrue(interpreter.update(closed_hand,2.5,aspect=1).fire)


if __name__ == '__main__':
    unittest.main()
