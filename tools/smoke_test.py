"""Exercise every UI state and Chapter 1 end-to-end without saving user progress."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
from pathlib import Path
import sys
import tempfile
import pygame
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import Settings,Profile
from game_state import Game,State,Target
from hand_tracker import HandInput
from renderer import Renderer


def main():
    pygame.init(); pygame.display.set_mode((1280,720))
    output=ROOT/'artifacts'; output.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as directory:
            game=Game(Settings(sound=False),Profile(Path(directory)/'save.json'),seed=8)
            renderer=Renderer()
            hand=HandInput(tracked=True,pointing=True,gun=True,aiming=True,gesture='TEST / SYNTHETIC INPUT')
            now=2.0

            def capture(name):
                surface=renderer.draw(game,hand,now,status='AUTOMATED UI CHECK / synthetic inputs, no real camera',demo=True)
                pygame.image.save(surface,str(output/f'{name}.png'))

            capture('01-main')
            game.activate('chapters',now); now+=1; capture('02-chapters')
            game.activate('settings',now); now+=1; capture('03-settings')
            game.activate('calibrate',now); now+=3; capture('04-calibration')
            game.activate('start',now); now+=1
            target=game.targets[0]
            hand.aim=(target.center[0]+(target.radius+30)/1280,target.center[1])
            hand.ready=True; hand.gesture='FIST READY'
            game.update(0,now,hand)
            assert game.locked_target_id==target.id and target.contains(game.assisted_aim)
            capture('05-training')
            for _ in range(4):
                hand.aim=game.targets[0].center; hand.fire=True
                game.update(0,now,hand); now+=.4
            hand.fire=False; capture('06-hit')
            hand.pause=True; game.update(.01,now,hand); hand.pause=False
            assert game.state==State.PAUSED
            now+=1; capture('07-paused')
            game.activate('resume',now); now+=1
            game.update(46,now,hand)
            assert game.state==State.RESULTS and game.score>=300 and game.profile.unlocked==2
            now+=1; capture('08-results')
            game.activate('next',now); now+=1
            assert game.chapter==2 and game.state==State.PLAYING
            target=Target(99,'TEST ONLY',(.4,.3,.6,.7))
            hand.aim=(.5,.5); hand.fire=True
            game.update(.01,now,hand,[target])
            assert game.hits==1
            game.profile.unlocked=3
            game.start(3,now); now+=1
            mask=np.zeros((480,640),bool); mask[180:300,270:370]=True
            target.mask=mask
            hand.aim=(.41,.31)
            game.update(0,now,hand,[target]); assert game.hits==0
            now+=.4; hand.aim=(.5,.5)
            game.update(0,now,hand,[target]); assert game.hits==1
            capture('09-mask-hit-test')
            print('PASS: all 7 screens, training -> pause -> resume -> results -> next, box/mask integration and rendering')
    finally: pygame.quit()


if __name__=='__main__': main()
