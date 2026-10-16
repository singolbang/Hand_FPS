import tempfile
import unittest
from pathlib import Path
import numpy as np

from config import Settings, Profile
from game_state import Game, Target
from hand_tracker import GestureInterpreter, FistShotTrigger, classify_hand
from test_hand import gun_points


def hand_points(closed=False):
    """Same palm anchor; four straight fingers or four visibly curled fingers."""
    p=gun_points()
    for base,x in ((5,.42),(9,.50),(13,.58),(17,.66)):
        if closed:
            p[base:base+4]=[[x,.6,0],[x,.48,0],[x,.56,0],[x,.68,0]]
        else:
            p[base:base+4]=[[x,.6,0],[x,.44,0],[x,.31,0],[x,.18,0]]
    if closed: p[3:5]=[[.41,.58,0],[.50,.62,0]]
    return p


def v_points():
    p=hand_points()
    folded=hand_points(True)
    p[13:21]=folded[13:21]
    return p


class OpenFistTests(unittest.TestCase):
    def ready(self, hz=25):
        tracker=GestureInterpreter()
        for i in range(int(hz*.4)+1):
            hand=tracker.update(hand_points(),1+i/hz,aspect=1,gameplay=True)
        self.assertTrue(hand.ready)
        return tracker,hand

    def test_clench_fires_once_then_requires_open_hand(self):
        tracker,before=self.ready()
        shots=[]
        for i in range(12):
            hand=tracker.update(hand_points(True),1.44+i*.04,aspect=1,gameplay=True)
            if hand.fire: shots.append(hand)
        self.assertEqual(len(shots),1)
        np.testing.assert_allclose(shots[0].shot_aim,before.aim)
        for i in range(5): tracker.update(hand_points(),2+i*.04,aspect=1,gameplay=True)
        shots=[tracker.update(hand_points(True),2.20+i*.04,aspect=1,gameplay=True).fire for i in range(8)]
        self.assertEqual(sum(shots),1)

    def test_initial_fist_tracking_loss_and_mode_change_require_open_hand(self):
        tracker=GestureInterpreter()
        for i in range(10): self.assertFalse(tracker.update(hand_points(True),i*.04,aspect=1,gameplay=True).fire)
        for interrupt in ('loss','gap','menu'):
            with self.subTest(interrupt=interrupt):
                tracker,_=self.ready()
                if interrupt=='loss': tracker.update(None,1.42,gameplay=True)
                if interrupt=='menu': tracker.update(hand_points(),1.42,aspect=1,gameplay=False)
                start=2 if interrupt=='gap' else 1.44
                for i in range(8):
                    self.assertFalse(tracker.update(hand_points(True),start+i*.04,aspect=1,gameplay=True).fire)

    def test_palm_aim_ignores_finger_clenching(self):
        tracker,before=self.ready()
        # Centroid of wrist and four MCP joints is (.532, .65), not index tip (.42, .68).
        np.testing.assert_allclose(before.raw_aim,(.532,.65))
        after=tracker.update(hand_points(True),1.44,aspect=1,gameplay=True)
        np.testing.assert_allclose(after.raw_aim,before.raw_aim)
        np.testing.assert_allclose(after.aim,before.aim)
        moved=hand_points(); moved[:,0]+=.12
        for i in range(20): after=tracker.update(moved,1.48+i*.04,aspect=1,gameplay=True)
        self.assertGreater(after.aim[0],before.aim[0]+.10)

    def test_low_and_normal_observation_rates_allow_preparing_and_firing(self):
        for hz in (8,9,10,11,15,30):
            with self.subTest(hz=hz):
                tracker,_=self.ready(hz)
                shots=[tracker.update(hand_points(True),1.6+i/hz,aspect=1,gameplay=True).fire for i in range(hz)]
                self.assertEqual(sum(shots),1)

    def test_fist_spike_does_not_fire_but_release_during_cooldown_is_kept(self):
        tracker,_=self.ready()
        self.assertFalse(tracker.update(hand_points(True),1.44,aspect=1,gameplay=True).fire)
        self.assertFalse(tracker.update(hand_points(),1.48,aspect=1,gameplay=True).fire)
        shots=[]
        for t in (1.52,1.56,1.60):
            if tracker.update(hand_points(True),t,aspect=1,gameplay=True).fire: shots.append(t)
        for t in (1.64,1.68,1.72,1.76): tracker.update(hand_points(),t,aspect=1,gameplay=True)
        for i in range(10):
            t=1.80+i*.04
            if tracker.update(hand_points(True),t,aspect=1,gameplay=True).fire: shots.append(t)
        self.assertEqual(len(shots),2)

    def test_pose_classification_survives_mirroring_scale_and_rotation(self):
        theta=.5
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        for closed in (False,True):
            for mirrored in (False,True):
                p=hand_points(closed)
                if mirrored: p[:,0]=1-p[:,0]
                p[:,:2]=p[:,:2]@rotation.T*.6+[.2,.1]
                f=classify_hand(p,aspect=1)
                self.assertEqual(getattr(f,'palm',None),not closed)
                self.assertEqual(getattr(f,'fist',None),closed)

    def test_confirmed_fist_shot_reaches_game_without_index_extended(self):
        tracker,before=self.ready()
        with tempfile.TemporaryDirectory() as directory:
            game=Game(Settings(),Profile(Path(directory)/'save.json'))
            game.start(1,0)
            x,y=before.aim
            game.targets=[Target(99,'drone',(x-.04,y-.04,x+.04,y+.04),radius=50)]
            for i in range(8):
                hand=tracker.update(hand_points(True),1.44+i*.04,aspect=1,gameplay=True)
                game.update(0,1.44+i*.04,hand)
            self.assertEqual((game.shots,game.hits),(1,1))

    def test_fist_shot_is_not_a_menu_selection_or_recoil_shot(self):
        tracker=GestureInterpreter()
        for i in range(8): tracker.update(hand_points(),1+i*.04,aspect=1)
        for i in range(8): self.assertFalse(tracker.update(hand_points(True),1.4+i*.04,aspect=1).fire)
        tracker,_=self.ready()
        moved=hand_points(); moved[:,1]-=.1
        for i in range(8): self.assertFalse(tracker.update(moved,1.44+i*.04,aspect=1,gameplay=True).fire)

    def test_slow_clench_keeps_open_hand_aim(self):
        gate=FistShotTrigger()
        for t in (0,.04,.08,.12): gate.update(True,False,(.4,.5),t)
        for i in range(10):
            self.assertFalse(gate.update(False,False,(.7,.8),.16+i*.1))
        self.assertFalse(gate.update(False,True,(.8,.9),1.16))
        self.assertTrue(gate.update(False,True,(.8,.9),1.21))
        self.assertEqual(gate.shot_aim,(.4,.5))

    def test_v_sign_cancels_shot_only_after_deliberate_hold(self):
        for count,expected in ((1,1),(12,0)):
            tracker,_=self.ready()
            for i in range(count): tracker.update(v_points(),1.44+i*.04,aspect=1,gameplay=True)
            shots=[]
            for i in range(10):
                shots.append(tracker.update(hand_points(True),1.44+(count+i)*.04,aspect=1,gameplay=True).fire)
            self.assertEqual(sum(shots),expected)

    def test_calibration_saves_palm_range_and_menus_keep_index_control(self):
        tracker=GestureInterpreter()
        p=hand_points()
        menu=tracker.update(p,1,aspect=1)
        np.testing.assert_allclose(menu.raw_aim,(.42,.18))
        with tempfile.TemporaryDirectory() as directory:
            game=Game(Settings(),Profile(Path(directory)/'save.json'))
            game.activate('calibrate',0)
            for i in range(16):
                hand=tracker.update(p,2.1+i*.1,aspect=1,calibrating=True)
                self.assertFalse(hand.fire)
                game.update(.1,2.1+i*.1,hand)
            self.assertEqual(game.cal_step,1)
            q=p.copy(); q[:,:2]+=.25
            for i in range(16):
                hand=tracker.update(q,5.6+i*.1,aspect=1,calibrating=True)
                game.update(.1,5.6+i*.1,hand)
            np.testing.assert_allclose(game.profile.calibration,(.532,.65,.782,.90))

    def test_open_hand_aim_never_pauses_or_fires(self):
        tracker=GestureInterpreter()
        for i in range(60):
            hand=tracker.update(hand_points(),1+i*.04,aspect=1,gameplay=True)
            self.assertFalse(hand.pause)
            self.assertFalse(hand.fire)
        self.assertTrue(hand.ready)
        self.assertTrue(hand.aiming)

    def test_thumb_position_does_not_change_open_or_fist_classification(self):
        for closed in (False,True):
            for thumb_tip in ((.2,.5,0),(.5,.62,0)):
                p=hand_points(closed); p[4]=thumb_tip
                f=classify_hand(p,aspect=1)
                self.assertEqual(f.palm,not closed)
                self.assertEqual(f.fist,closed)

    def test_partly_bent_hand_is_neither_open_nor_fist(self):
        p=hand_points()
        for base in (5,9,13,17):
            x=p[base,0]
            p[base+3]=[x+.19,.24,0]  # About 137 degrees; still far from wrist.
        f=classify_hand(p,aspect=1)
        self.assertFalse(f.palm)
        self.assertFalse(f.fist)

    def test_v_hold_pauses_once_and_does_not_select_menu(self):
        tracker=GestureInterpreter()
        events=[tracker.update(v_points(),1+i*.04,aspect=1,gameplay=True) for i in range(60)]
        self.assertEqual(sum(h.pause for h in events),1)
        self.assertFalse(any(h.fire for h in events))
        with tempfile.TemporaryDirectory() as directory:
            game=Game(Settings(),Profile(Path(directory)/'save.json'))
            for i in range(60):
                hand=tracker.update(v_points(),4+i*.04,aspect=1)
                hand.aim=(290/1280,357/720)
                game.update(.04,4+i*.04,hand)
            from game_state import State
            self.assertEqual(game.state,State.MAIN)

    def test_only_stable_v_routes_swipe_and_delays_pause(self):
        tracker,_=self.ready()
        hand=tracker.update(hand_points(),1.44,aspect=1,gameplay=True)
        tracker.apply_swipe(hand,1,1.44)
        self.assertEqual(hand.swipe,0)
        hand=tracker.update(v_points(),1.48,aspect=1,gameplay=True)
        tracker.apply_swipe(hand,1,1.48)
        self.assertEqual(hand.swipe,0)
        for i in range(1,7): hand=tracker.update(v_points(),1.48+i*.04,aspect=1,gameplay=True)
        tracker.apply_swipe(hand,1,1.72)
        self.assertEqual(hand.swipe,1)
        self.assertFalse(hand.pause or hand.fire)
        for i in range(1,18):
            self.assertFalse(tracker.update(v_points(),1.72+i*.04,aspect=1,gameplay=True).pause)
        self.assertTrue(tracker.update(v_points(),2.44,aspect=1,gameplay=True).pause)

    def test_pause_latch_survives_tracking_loss_until_v_is_released(self):
        tracker=GestureInterpreter()
        events=[tracker.update(v_points(),1+i*.04,aspect=1,gameplay=True) for i in range(25)]
        self.assertEqual(sum(h.pause for h in events),1)
        tracker.update(None,2,gameplay=True)
        events=[tracker.update(v_points(),2.04+i*.04,aspect=1,gameplay=True) for i in range(35)]
        self.assertFalse(any(h.pause for h in events))
        tracker.update(hand_points(),3.5,aspect=1,gameplay=True)
        events=[tracker.update(v_points(),3.54+i*.04,aspect=1,gameplay=True) for i in range(25)]
        self.assertEqual(sum(h.pause for h in events),1)


if __name__=='__main__': unittest.main()
