import tempfile
import unittest
from pathlib import Path

from config import Settings, Profile
from game_state import Game, State
from hand_tracker import HandInput


class GameTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.game = Game(Settings(), Profile(Path(self.tmp.name) / 'save.json'), seed=7)

    def test_hit_miss_results_and_unlock(self):
        g = self.game
        g.start(1, 0)
        for n in range(3):
            aim = g.targets[0].center
            g.update(0, n + 1, HandInput(tracked=True, aiming=True, fire=True, aim=aim))
        g.update(0, 5, HandInput(tracked=True, aiming=True, fire=True, aim=(0, 0)))
        self.assertEqual(g.hits, 3)
        self.assertEqual(g.shots, 4)
        self.assertGreaterEqual(g.score, 300)
        g.update(46, 60, HandInput(tracked=True))
        self.assertEqual(g.state, State.RESULTS)
        self.assertEqual(g.accuracy, 75)
        self.assertEqual(g.profile.unlocked, 2)

    def test_pause_freezes_time_and_restart_resets(self):
        g = self.game
        g.start(1, 0)
        g.update(2, 2, HandInput(tracked=True, pause=True))
        self.assertEqual(g.state, State.PAUSED)
        before = g.remaining
        g.update(10, 12, HandInput())
        self.assertEqual(g.remaining, before)
        g.activate('resume', 13)
        g.update(1, 14, HandInput(tracked=True))
        self.assertEqual(g.remaining, before - 1)
        g.activate('retry', 15)
        self.assertEqual(g.score, 0)
        self.assertEqual(g.remaining, 45)

    def test_untracked_and_non_aiming_cannot_shoot(self):
        g = self.game
        g.start(1, 0)
        for i in [HandInput(fire=True), HandInput(tracked=True, fire=True)]:
            g.update(.1, 2, i)
        self.assertEqual(g.shots, 0)

    def test_dwell_selects_and_transition_is_guarded(self):
        g = self.game
        b = next(b for b in g.buttons() if b.action == 'chapters')
        x, y, w, h = b.rect
        hand = HandInput(tracked=True, pointing=True, aim=((x+w/2)/1280, (y+h/2)/720))
        for n in range(20):
            g.update(.1, 1+n*.1, hand)
        self.assertEqual(g.state, State.CHAPTERS)
        self.assertEqual(g.shots, 0)

    def test_tracking_loss_auto_pauses(self):
        g = self.game
        g.start(1, 0)
        g.update(.1, 1, HandInput())
        g.update(1.5, 2.5, HandInput())
        self.assertEqual(g.state, State.PAUSED)

    def test_stationary_dwell_does_not_select_next_screen(self):
        g=self.game
        hand=HandInput(tracked=True,pointing=True,aim=(290/1280,426.5/720))
        for n in range(100):
            g.update(.05,n*.05,hand)
        self.assertEqual(g.state,State.CHAPTERS)
        # Deliberate cursor movement rearms dwell on the new screen.
        hand.aim=(.05,.15)
        g.update(.05,5.1,hand)
        hand.aim=(.2,.5)
        for n in range(int(g.settings.dwell/.05)+4): g.update(.05,5.2+n*.05,hand)
        self.assertEqual(g.state,State.PLAYING)

    def test_stationary_dwell_toggles_setting_once(self):
        g=self.game
        g.activate('settings',0)
        hand=HandInput(tracked=True,pointing=True,aim=(300/1280,421.5/720))
        for n in range(160): g.update(.05,n*.05,hand)
        self.assertFalse(g.settings.sound)
        self.assertEqual(g.hover_time,0)

    def test_menu_requires_deliberate_hold_longer_than_one_second(self):
        g=self.game
        hand=HandInput(tracked=True,pointing=True,aim=(290/1280,426.5/720))
        for n in range(12): g.update(.1,1+n*.1,hand)
        self.assertEqual(g.state,State.MAIN)
        for n in range(5): g.update(.1,2.2+n*.1,hand)
        self.assertEqual(g.state,State.CHAPTERS)

    def test_corrupt_profile_uses_defaults(self):
        p = Path(self.tmp.name) / 'bad.json'
        p.write_text('{broken', encoding='utf-8')
        profile = Profile(p)
        self.assertEqual(profile.unlocked, 1)
        profile.record(1, 500)
        self.assertEqual(Profile(p).unlocked, 2)


if __name__ == '__main__':
    unittest.main()
