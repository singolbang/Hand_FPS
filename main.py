"""Finger Gun FPS. Run --demo for explicit mouse debugging without a webcam."""
import argparse
import os
import time
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import pygame

from config import Settings, Profile, WIDTH, HEIGHT
from hand_tracker import HandInput
from game_state import Game, State
from renderer import Renderer


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo',action='store_true',help='Explicit mouse/keyboard debug mode; not real vision')
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--fullscreen',action='store_true')
    p.add_argument('--debug',action='store_true')
    p.add_argument('--mute',action='store_true')
    p.add_argument('--duration',type=float,default=45)
    p.add_argument('--infer-hz',type=float,default=4)
    p.add_argument('--headless',action='store_true',help='SDL dummy driver (smoke checks)')
    p.add_argument('--frames',type=int,default=0,help='Exit after N frames for smoke testing')
    p.add_argument('--screenshot',help='Save final rendered frame to this file')
    args=p.parse_args()
    if not 5<=args.duration<=180: p.error('--duration must be 5..180 seconds')
    if not .5<=args.infer_hz<=12: p.error('--infer-hz must be 0.5..12')
    if args.frames<0: p.error('--frames cannot be negative')
    return args


def run(args):
    if args.headless:
        os.environ['SDL_VIDEODRIVER']='dummy'
        os.environ['SDL_AUDIODRIVER']='dummy'
    pygame.init()
    display=pygame.display.set_mode((WIDTH,HEIGHT),pygame.FULLSCREEN if args.fullscreen else pygame.RESIZABLE)
    pygame.display.set_caption('Finger Gun FPS / Neural Arcade')
    pygame.mouse.set_visible(False)
    clock=pygame.time.Clock()
    settings=Settings(camera=args.camera,duration=args.duration,debug=args.debug,sound=not args.mute,infer_hz=args.infer_hz)
    profile=Profile()
    settings.bounds=profile.calibration
    game=Game(settings,profile)
    renderer=Renderer()
    vision=None
    frame_count=0
    start=time.monotonic()
    frame=None
    try:
        if not args.demo:
            from vision import VisionSystem
            vision=VisionSystem(settings)
        while game.running:
            dt=min(clock.tick(60)/1000,.1)
            now=time.monotonic()-start
            fire=pause=False; swipe=0
            for event in pygame.event.get():
                if event.type==pygame.QUIT: game.running=False
                elif event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_ESCAPE: game.running=False
                    elif event.key==pygame.K_F1: settings.debug=not settings.debug
                    elif event.key==pygame.K_F11:
                        args.fullscreen=not args.fullscreen
                        display=pygame.display.set_mode((WIDTH,HEIGHT),pygame.FULLSCREEN if args.fullscreen else pygame.RESIZABLE)
                    elif args.demo:
                        if event.key in (pygame.K_SPACE,pygame.K_RETURN): fire=True
                        elif event.key==pygame.K_p: pause=True
                        elif event.key in (pygame.K_LEFT,pygame.K_RIGHT): swipe=-1 if event.key==pygame.K_LEFT else 1
                        elif event.key==pygame.K_e: swipe=1
                elif args.demo and event.type==pygame.MOUSEBUTTONDOWN and event.button==1: fire=True
            if args.demo:
                mx,my=pygame.mouse.get_pos(); sw,sh=display.get_size()
                hand=HandInput(tracked=True,aim=(mx/sw,my/sh),raw_aim=(mx/sw,my/sh),pointing=True,gun=True,aiming=True,palm=game.state==State.CALIBRATION and not fire,ready=True,fire=fire,pause=pause,swipe=swipe,gesture='DEMO / MOUSE')
                targets=[]; metrics={}; status='DEMO: mouse / click or space / P pause / E EMP. No vision inference.'
                game.availability={1:'',2:'Live camera required; restart without --demo.',3:'Live camera required; restart without --demo.'}
            else:
                vision.set_chapter(game.chapter if game.state==State.PLAYING else -1 if game.state==State.CALIBRATION else 0)
                sample=vision.poll()
                hand,frame,targets=sample.hand,sample.frame,sample.targets
                metrics=sample.metrics; status=sample.status
                game.availability=sample.availability
            game.update(dt,now,hand,targets)
            metrics['fps']=clock.get_fps()
            surface=renderer.draw(game,hand,now,frame,status,metrics,args.demo)
            display.blit(pygame.transform.smoothscale(surface,display.get_size()),(0,0))
            pygame.display.flip()
            frame_count+=1
            if args.frames and frame_count>=args.frames: break
        if args.screenshot:
            from pathlib import Path
            Path(args.screenshot).parent.mkdir(parents=True,exist_ok=True)
            pygame.image.save(renderer.surface,args.screenshot)
        print(f'Clean exit: {frame_count} frames, state={game.state.name}, score={game.score}')
    finally:
        if vision is not None: vision.close()
        pygame.mouse.set_visible(True)
        pygame.quit()


if __name__=='__main__':
    run(arguments())
