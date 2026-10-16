import tempfile
import unittest
from pathlib import Path
import numpy as np

from config import Settings,Profile
from game_state import Game,Target
from hand_tracker import HandInput


class AimAssistTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.game=Game(Settings(),Profile(Path(self.tmp.name)/'save.json'),seed=9)
        self.game.profile.unlocked=3

    def test_near_circle_locks_and_hits_but_far_point_misses(self):
        g=self.game; g.start(1,0)
        target=Target(99,'drone',(.45,.45,.55,.55),radius=50)
        g.targets=[target]
        hand=HandInput(tracked=True,aiming=True,aim=(.5+75/1280,.5))
        g.update(0,1,hand)
        self.assertEqual(g.locked_target_id,99)
        self.assertTrue(target.contains(g.assisted_aim))
        hand.fire=True
        g.update(0,1.1,hand)
        self.assertEqual(g.hits,1)
        hand.aim=(0,0)
        g.update(0,1.5,hand)
        self.assertEqual(g.hits,1)
        self.assertIsNone(g.locked_target_id)

    def test_shoots_at_saved_aim_instead_of_hand_moved_during_thumb_fold(self):
        g=self.game; g.start(1,0)
        g.targets=[Target(99,'drone',(.45,.45,.55,.55),radius=50)]
        # Latest pose may already be relaxed; the queued confirmed shot is still valid.
        g.update(0,1,HandInput(tracked=True,aiming=False,fire=True,aim=(.5,.1),shot_aim=(.5,.5)))
        self.assertEqual(g.hits,1)
        self.assertEqual(next(e.pos for e in g.effects if e.kind=='shot'),g.assisted_aim)

    def test_box_assist_excludes_hit_and_missing_targets(self):
        g=self.game; g.start(2,0)
        target=Target(99,'cup',(.4,.4,.6,.6))
        hand=HandInput(tracked=True,aiming=True,aim=(.62,.5))
        g.update(0,1,hand,[target])
        self.assertTrue(target.contains(g.assisted_aim))
        g.hit_until[99]=5
        g.update(0,1.1,hand,[target])
        self.assertIsNone(g.locked_target_id)
        self.assertEqual(g.assisted_aim,hand.aim)
        g.update(0,1.2,hand,[])
        self.assertIsNone(g.locked_target_id)

    def test_precision_assist_always_uses_a_real_mask_pixel(self):
        g=self.game; g.start(3,0)
        mask=np.zeros((100,100),bool); mask[40:60,45:55]=True
        target=Target(99,'bottle',(.2,.2,.8,.8),mask=mask)
        hand=HandInput(tracked=True,aiming=True,aim=(.56,.5))
        self.assertFalse(target.contains(hand.aim,precision=True))
        g.update(0,1,hand,[target])
        self.assertTrue(target.contains(g.assisted_aim,precision=True))
        hand.aim=(.25,.25)
        g.update(0,1.1,hand,[target])
        self.assertIsNone(g.locked_target_id)
        hand.aim=(.5,.5); target.mask=None
        g.update(0,1.2,hand,[target])
        self.assertIsNone(g.locked_target_id)

    def test_assist_does_not_activate_without_a_tracked_aiming_pose(self):
        g=self.game; g.start(1,0)
        for hand in (HandInput(),HandInput(tracked=True)):
            hand.aim=g.targets[0].center
            g.update(0,1,hand)
            self.assertIsNone(g.locked_target_id)


if __name__=='__main__': unittest.main()
