"""Procedural neon art, camera compositing and a consistent 1280x720 UI."""
import math
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import pygame
import numpy as np

from config import WIDTH, HEIGHT, CHAPTERS, DESCRIPTIONS
from game_state import State

BG=(7,12,24)
PANEL=(13,24,40)
LINE=(35,61,78)
CYAN=(65,230,238)
ORANGE=(255,151,78)
WHITE=(230,242,250)
MUTED=(118,146,164)
GREEN=(99,247,174)
CONNECTIONS=((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17))


class SoundBank:
    def __init__(self):
        self.sounds={}
        self.played=set()
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(22050,-16,2,512)
            frequency,_,channels=pygame.mixer.get_init()
            rng=np.random.default_rng(4)
            for name,pitch,duration in [('shot',180,.09),('hit',960,.17),('miss',125,.12),('select',660,.12),('emp',60,.6)]:
                t=np.arange(int(duration*frequency))/frequency
                signal=np.sin(2*np.pi*(pitch*t+(100 if name=='hit' else -60)*t*t))
                if name in ('shot','emp'): signal=signal*.4+rng.uniform(-1,1,len(t))*.6
                samples=(signal*np.exp(-t/duration*5)*6500).astype(np.int16)
                if channels==2: samples=np.repeat(samples[:,None],2,axis=1)
                self.sounds[name]=pygame.sndarray.make_sound(samples.copy())
        except (pygame.error,ValueError):
            pass

    def update(self,game):
        active={(e.kind,e.born,e.pos) for e in game.effects}
        for e in game.effects:
            key=(e.kind,e.born,e.pos)
            if key not in self.played and game.settings.sound and e.kind in self.sounds:
                self.sounds[e.kind].play()
        self.played=active


class Renderer:
    def __init__(self):
        pygame.font.init()
        self.surface=pygame.Surface((WIDTH,HEIGHT))
        self.fonts={}
        self.background=pygame.Surface((WIDTH,HEIGHT))
        for y in range(HEIGHT):
            t=y/HEIGHT
            pygame.draw.line(self.background,(int(9-3*t),int(20-9*t),int(35-12*t)),(0,y),(WIDTH,y))
        self.sound=SoundBank()
        self.cached_frame=None
        self.camera_surface=None
        self.mask_cache={}

    def text(self,text,pos,size=20,color=WHITE,bold=False,mono=False,center=False):
        key=(size,bold,mono)
        if key not in self.fonts:
            self.fonts[key]=pygame.font.SysFont('consolas' if mono else 'segoeui',size,bold=bold)
        image=self.fonts[key].render(str(text),True,color)
        rect=image.get_rect(center=pos) if center else image.get_rect(topleft=pos)
        self.surface.blit(image,rect)
        return rect

    def panel(self,rect,color=LINE,fill=PANEL,width=1):
        pygame.draw.rect(self.surface,fill,rect,border_radius=10)
        pygame.draw.rect(self.surface,color,rect,width,border_radius=10)

    def line(self,a,b,color=LINE,width=1):
        pygame.draw.line(self.surface,color,a,b,width)

    def circle(self,pos,r,color=CYAN,width=1):
        pygame.draw.circle(self.surface,color,(int(pos[0]),int(pos[1])),max(1,int(r)),width)

    def grid(self,now,arena=False):
        horizon=250 if arena else 430
        for i in range(-12,13):
            self.line((WIDTH/2+i*32,horizon),(WIDTH/2+i*150,HEIGHT),(17,42,55))
        for n in range(12):
            p=((n/12+now*.025)%1)**2
            y=horizon+(HEIGHT-horizon)*p
            self.line((0,y),(WIDTH,y),(17,42,55))
        self.line((0,horizon),(WIDTH,horizon),(26,60,73))
        if arena:
            for i in range(12):
                x=i*113-20
                h=35+(i*47)%110
                pygame.draw.rect(self.surface,(10,25,39),(x,horizon-h,80,h))
                pygame.draw.rect(self.surface,(21,52,66),(x,horizon-h,80,h),1)
                for j in range(3): self.line((x+15+j*22,horizon-h+12),(x+15+j*22,horizon-10),(18,42,57))

    def heading(self,kicker,title,subtitle):
        self.text(kicker,(78,105),15,CYAN,mono=True)
        self.text(title,(76,137),48,WHITE,True)
        self.text(subtitle,(79,197),18,MUTED)

    def header(self,game,demo,fps):
        self.line((38,76),(1242,76))
        self.text('F / G',(40,24),27,CYAN,True)
        self.text('NEURAL ARCADE',(135,29),16,WHITE,mono=True)
        self.text('VISION SYSTEM  /  '+('DEBUG SIMULATION' if demo else 'LIVE INPUT'),(820,31),14,ORANGE if demo else CYAN,mono=True)
        self.text(f'{fps:02.0f} FPS',(1150,28),17,MUTED,mono=True)

    def button(self,b,game,now):
        x,y,w,h=b.rect
        hot=game.hover==b.action
        color=CYAN if hot else LINE
        self.panel(b.rect,color,(19,47,61) if hot else (13,26,42))
        if hot:
            pygame.draw.rect(self.surface,CYAN,(x,y+10,3,h-20))
            pygame.draw.rect(self.surface,CYAN,(x+8,y+h-4,int((w-16)*game.hover_progress),2))
        self.text(b.label,(x+23,y+9 if b.subtitle else y+(h-24)/2),20,WHITE if b.enabled else MUTED,True)
        if b.subtitle: self.text(b.subtitle,(x+24,y+34),10,MUTED,mono=True)
        self.text('>' if b.enabled else '-',(x+w-32,y+h/2-13),23,CYAN if hot else MUTED)

    def hero(self,now,cx=900,cy=325):
        self.circle((cx,cy),179,(23,62,78))
        self.circle((cx,cy),164,(29,80,96))
        for i in range(48):
            a=i*math.tau/48+now*.03
            r=174 if i%4 else 167
            self.line((cx+math.cos(a)*r,cy+math.sin(a)*r),(cx+math.cos(a)*181,cy+math.sin(a)*181),CYAN if i%4==0 else LINE,2)
        for offset in (0,math.pi):
            pygame.draw.arc(self.surface,CYAN,(cx-148,cy-148,296,296),now*.3+offset,now*.3+offset+1.5,3)
        # Open-hand silhouette: visibly extended fingers, no external assets.
        pts=[(-53,91),(-72,42),(-111,-7),(-109,-25),(-92,-30),(-58,0),(-58,-83),(-49,-95),(-37,-91),(-33,-26),(-31,-113),(-20,-124),(-8,-117),(-4,-27),(0,-102),(12,-112),(24,-102),(24,-17),(35,-73),(48,-81),(58,-71),(52,28),(39,72),(22,91)]
        pts=[(cx+x,cy+y) for x,y in pts]
        pygame.draw.polygon(self.surface,(18,61,75),pts)
        pygame.draw.polygon(self.surface,CYAN,pts,3)
        self.line((cx-47,cy+16),(cx+28,cy+16),(51,137,151),2)
        self.line((cx-35,cy+45),(cx+12,cy+57),(51,137,151),2)
        self.circle((cx-12,cy+26),8,ORANGE,2)
        for angle in (now*.4,now*.4+2.3):
            p=(cx+math.cos(angle)*180,cy+math.sin(angle)*180)
            self.circle(p,5,ORANGE,0)
        self.text('GESTURE-POWERED COMBAT',(cx,cy+216),16,CYAN,mono=True,center=True)

    def main_menu(self,game,now):
        self.text('YOUR HAND IS THE CONTROLLER',(92,116),15,CYAN,mono=True)
        self.text('FINGER GUN',(84,146),77,WHITE,True)
        self.text('AIM WITH INTENT.',(91,242),22,ORANGE,True)
        self.text('A computer vision arcade. No controller required.',(92,281),17,MUTED)
        self.hero(now)
        self.panel((667,577,532,66))
        self.text('OPEN HAND / AIM    CLENCH FIST / FIRE',(695,590),16,WHITE,mono=True)
        self.text('HOLD V / PAUSE    V + SWIPE / EMP',(695,614),13,MUTED,mono=True)
        for b in game.buttons(): self.button(b,game,now)

    def chapter_cards(self,game,now):
        self.heading('MISSION DIRECTORY','CHOOSE YOUR CHAPTER','Earn 300 points and finish a mission to unlock the next chapter.')
        for b in game.buttons():
            if not b.action.startswith('chapter:'):
                self.button(b,game,now)
                continue
            i=int(b.action[-1]); x,y,w,h=b.rect
            hot=game.hover==b.action
            accent=ORANGE if i==3 else CYAN
            self.panel(b.rect,accent if hot or game.selected==i else LINE,(14,30,46))
            self.text(f'0{i}',(x+24,y+16),44,accent,True,mono=True)
            self.text('READY' if b.enabled else 'LOCKED' if i>game.profile.unlocked else 'MODEL OFFLINE',(x+w-148,y+32),13,accent if b.enabled else MUTED,mono=True)
            cx,cy=x+w/2,y+124
            if i==1:
                for r in (48,33,14): self.circle((cx,cy),r,accent,2)
                self.line((cx-65,cy),(cx+65,cy),accent)
                self.line((cx,cy-60),(cx,cy+60),accent)
            elif i==2:
                for ox,oy,sw,sh in [(-60,-35,60,78),(10,-16,50,60)]:
                    pygame.draw.rect(self.surface,accent,(cx+ox,cy+oy,sw,sh),2)
                self.text('0.94',(cx-55,cy-52),11,accent,mono=True)
            else:
                pts=[(cx-53,cy+38),(cx-43,cy-25),(cx-12,cy-48),(cx+22,cy-36),(cx+51,cy+8),(cx+30,cy+43)]
                pygame.draw.polygon(self.surface,(73,48,39),pts)
                pygame.draw.polygon(self.surface,accent,pts,2)
                for px,py in pts: self.circle((px,py),4,accent,0)
            self.text(CHAPTERS[i-1],(x+23,y+200),22,WHITE if b.enabled else MUTED,True)
            self.text(DESCRIPTIONS[i-1],(x+23,y+238),14,MUTED)
            self.line((x+24,y+273),(x+w-24,y+273))
            self.text(('LANDMARKS + FLOW','OBJECT DETECTION','INSTANCE SEGMENTATION')[i-1],(x+24,y+291),12,accent,mono=True)
            if hot: pygame.draw.rect(self.surface,accent,(x+10,y+h-5,int((w-20)*game.hover_progress),3))
        self.text('Make a V, then swipe: highlight adjacent mission',(545,624),15,MUTED)

    def camera(self,frame):
        if frame is not self.cached_frame:
            rgb=np.ascontiguousarray(frame[:,:,::-1])
            self.camera_surface=pygame.image.frombuffer(rgb.tobytes(),(frame.shape[1],frame.shape[0]),'RGB').convert()
            self.cached_frame=frame
        return self.camera_surface

    def mask_overlay(self,target):
        key=(target.id,id(target.mask))
        if key not in self.mask_cache:
            import cv2
            mask=target.mask.astype(np.uint8)
            h,w=mask.shape
            rgba=np.zeros((h,w,4),np.uint8)
            rgba[mask>0]=(*CYAN,65)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgba,contours,-1,(*CYAN,235),2)
            self.mask_cache[key]=pygame.image.frombuffer(rgba.tobytes(),(w,h),'RGBA').convert_alpha()
        return self.mask_cache[key]

    def arena(self,game,hand,now,frame):
        if game.chapter>1 and frame is not None:
            self.surface.blit(pygame.transform.scale(self.camera(frame),(WIDTH,HEIGHT)),(0,0))
            shade=pygame.Surface((WIDTH,HEIGHT),pygame.SRCALPHA)
            shade.fill((4,13,25,85))
            self.surface.blit(shade,(0,0))
        else:
            self.grid(now,True)
        keep={(t.id,id(t.mask)) for t in game.targets if t.mask is not None}
        self.mask_cache={k:v for k,v in self.mask_cache.items() if k in keep}
        for t in game.targets:
            if now<game.hit_until.get(t.id,0): continue
            cx,cy=t.center; px,py=cx*WIDTH,cy*HEIGHT
            locked=t.id==game.locked_target_id
            accent=GREEN if locked else CYAN
            if t.radius:
                self.circle((px,py),t.radius+10,GREEN if locked else (24,69,85),3)
                self.circle((px,py),t.radius,accent,2)
                self.circle((px,py),t.radius*.52,(40,119,137),1)
                self.circle((px,py),5,ORANGE,0)
                for a in range(4):
                    theta=a*math.pi/2+now*.7+t.id
                    self.line((px+math.cos(theta)*(t.radius+6),py+math.sin(theta)*(t.radius+6)),(px+math.cos(theta)*(t.radius+17),py+math.sin(theta)*(t.radius+17)),accent,3)
                self.text('LOCKED' if locked else f'T-{t.id:02}',(px,py+t.radius+26),11,GREEN if locked else MUTED,mono=True,center=True)
            else:
                x0,y0,x1,y1=t.box
                rect=(x0*WIDTH,y0*HEIGHT,(x1-x0)*WIDTH,(y1-y0)*HEIGHT)
                if game.chapter==3 and t.mask is not None:
                    self.surface.blit(pygame.transform.scale(self.mask_overlay(t),(WIDTH,HEIGHT)),(0,0))
                else:
                    pygame.draw.rect(self.surface,accent,rect,2,border_radius=5)
                label_y=max(93,y0*HEIGHT-28)
                label_x=min(WIDTH-232,max(12,x0*WIDTH))
                self.panel((label_x,label_y,220,25),accent)
                label=f'{t.label.upper()}  '+('LOCKED' if locked else f'{t.confidence:.0%}')
                self.text(label,(label_x+8,label_y+3),12,accent,mono=True)
        self.panel((28,92,250,76))
        self.text('SCORE',(46,103),12,MUTED,mono=True)
        self.text(f'{game.score:06}',(43,120),31,WHITE,True,mono=True)
        self.text(f'CHAPTER 0{game.chapter}  /  {CHAPTERS[game.chapter-1]}',(640,112),16,CYAN,mono=True,center=True)
        self.text(f'{game.remaining:04.1f}',(1140,127),42,ORANGE if game.remaining<10 else WHITE,True,mono=True,center=True)
        pygame.draw.rect(self.surface,LINE,(330,142,620,4))
        pygame.draw.rect(self.surface,CYAN,(330,142,int(620*game.remaining/game.settings.duration),4))
        self.text(f'COMBO  x{game.combo}',(48,184),17,ORANGE,mono=True)
        self.text(f'ACCURACY  {game.accuracy:.0f}%',(48,211),13,MUTED,mono=True)
        if game.chapter>1 and not game.targets:
            self.panel((367,288,546,111))
            self.text('SEARCHING FOR REAL OBJECTS',(640,317),20,CYAN,True,center=True)
            self.text('Show a cup, bottle, phone, book or chair to the camera.',(640,352),16,MUTED,center=True)
            self.text('Targets appear only after a real model detection.',(640,379),13,MUTED,center=True)
        ready=now-game.last_emp>=5
        self.panel((968,577,280,65),CYAN if ready else LINE)
        self.text('EMP  /  '+('READY' if ready else f'{5-(now-game.last_emp):.1f}s'),(988,588),19,CYAN if ready else MUTED,True)
        self.text('V SIGN + FAST SWIPE',(988,618),12,MUTED,mono=True)
        if frame is not None and game.chapter==1:
            self.surface.blit(pygame.transform.scale(self.camera(frame),(208,156)),(32,471))
            pygame.draw.rect(self.surface,LINE,(32,471,208,156),1)
            self.text('LIVE HAND FEED',(43,634),11,CYAN,mono=True)
            if game.settings.debug: self.landmarks(hand,(32,471,208,156))
        elif game.settings.debug:
            self.landmarks(hand,(0,0,WIDTH,HEIGHT))
        hint='V HOLD / PAUSE - SWIPE / EMP' if hand.v_sign else 'READY / CLENCH FIST TO FIRE' if hand.ready else 'OPEN HAND TO READY'
        self.text(hint,(640,603),17,GREEN if hand.ready else ORANGE,True,mono=True,center=True)
        self.text('OPEN > AIM  /  FIST > FIRE  /  HOLD V > PAUSE',(640,637),13,MUTED,mono=True,center=True)

    def landmarks(self,hand,rect):
        if hand.landmarks is None: return
        x,y,w,h=rect
        pts=[(x+float(p[0])*w,y+float(p[1])*h) for p in hand.landmarks]
        for a,b in CONNECTIONS: self.line(pts[a],pts[b],GREEN,2)
        for p in pts: self.circle(p,3,ORANGE,0)

    def settings_page(self,game,hand,now,frame):
        self.heading('INPUT LAB','MAKE IT FEEL NATURAL','Keep your hand visible, with the palm facing the camera. Use one hand.')
        for b in game.buttons(): self.button(b,game,now)
        self.panel((690,245,480,354))
        if frame is not None:
            self.surface.blit(pygame.transform.scale(self.camera(frame),(424,238)),(718,267))
            self.landmarks(hand,(718,267,424,238))
        else:
            self.text('WAITING FOR CAMERA',(930,375),20,MUTED,mono=True,center=True)
        self.text(f'{hand.gesture}  /  THUMB {hand.thumb_ratio:.2f}',(718,528),16,CYAN,mono=True)
        self.text('In game: open hand to aim, clench fist to fire.',(718,558),14,MUTED)
        self.text(f'Point + hold {game.settings.dwell:g}s to select a menu item.',(718,580),13,MUTED)
        if game.notice: self.text(game.notice,(112,641),15,ORANGE)

    def calibration(self,game,hand,now,frame):
        self.heading('INPUT LAB / RANGE CALIBRATION',f'0{game.cal_step+1} / '+('UPPER LEFT' if game.cal_step==0 else 'LOWER RIGHT'),'Open your hand. Move your palm to a comfortable corner and hold still for 1.3 seconds.')
        if frame is not None:
            self.surface.blit(pygame.transform.scale(self.camera(frame),(640,360)),(320,235))
            self.landmarks(hand,(320,235,640,360))
        pos=(340,255) if game.cal_step==0 else (940,575)
        self.circle(pos,25,ORANGE,3)
        self.line((pos[0]-35,pos[1]),(pos[0]+35,pos[1]),ORANGE,2)
        self.line((pos[0],pos[1]-35),(pos[0],pos[1]+35),ORANGE,2)
        self.text('HOLD STILL' if now>=game.lock_until else 'GET READY',(640,625),20,CYAN,True,center=True)
        pygame.draw.rect(self.surface,LINE,(460,658,360,5))
        pygame.draw.rect(self.surface,CYAN,(460,658,int(360*game.cal_progress),5))
        for b in game.buttons(): self.button(b,game,now)

    def results(self,game,now):
        self.text('MISSION COMPLETE',(640,131),16,CYAN,mono=True,center=True)
        self.text('NICE SHOOTING.' if game.score>=300 else 'KEEP TRAINING.',(640,180),52,WHITE,True,center=True)
        self.text(CHAPTERS[game.chapter-1],(640,232),17,MUTED,mono=True,center=True)
        self.panel((280,277,720,198))
        self.text(f'{game.score:06}',(640,331),68,CYAN,True,mono=True,center=True)
        for x,label,value in [(410,'ACCURACY',f'{game.accuracy:.0f}%'),(640,'HITS / SHOTS',f'{game.hits} / {game.shots}'),(865,'PLAY TIME',f'{game.elapsed:.0f}s')]:
            self.text(label,(x,405),12,MUTED,mono=True,center=True)
            self.text(value,(x,437),25,WHITE,True,mono=True,center=True)
        unlock='NEXT CHAPTER UNLOCKED' if game.score>=300 and game.chapter<3 else '300 POINTS TO UNLOCK THE NEXT CHAPTER' if game.chapter<3 else 'ALL SYSTEMS MASTERED'
        self.text(unlock,(640,492),13,ORANGE,mono=True,center=True)
        for b in game.buttons(): self.button(b,game,now)

    def effects(self,game,now):
        for e in game.effects:
            age=now-e.born; x,y=e.pos[0]*WIDTH,e.pos[1]*HEIGHT
            if e.kind=='shot' and age<.09:
                self.line((WIDTH*.72,HEIGHT),(x,y),ORANGE,2)
                self.circle((x,y),10+age*180,ORANGE,2)
            elif e.kind=='hit':
                if e.target is not None and e.target.mask is not None and age<.25:
                    mask=e.target.mask
                    rgba=np.zeros((*mask.shape,4),np.uint8)
                    rgba[mask]=(*ORANGE,int(120*(1-age/.25)))
                    flash=pygame.image.frombuffer(rgba.tobytes(),(mask.shape[1],mask.shape[0]),'RGBA')
                    self.surface.blit(pygame.transform.scale(flash,(WIDTH,HEIGHT)),(0,0))
                color=tuple(int(c*max(.1,1-age)) for c in ORANGE)
                if age<.6:
                    for i in range(12):
                        a=i*math.tau/12+e.born
                        dist=12+age*(80+i%3*65)
                        px,py=x+math.cos(a)*dist,y+math.sin(a)*dist+age*age*80
                        self.circle((px,py),max(1,4-age*5),color,0)
                    self.circle((x,y),15+age*80,color,2)
                self.text(e.label,(x,y-28-age*40),24,color,True,mono=True,center=True)
            elif e.kind=='miss' and age<.4:
                self.line((x-12,y-12),(x+12,y+12),ORANGE,2)
                self.line((x+12,y-12),(x-12,y+12),ORANGE,2)
                self.text('MISS',(x,y+30),12,ORANGE,mono=True,center=True)
            elif e.kind=='emp':
                self.circle((WIDTH/2,HEIGHT/2),age*1000,CYAN,3)
                self.text(e.label,(640,220),24,CYAN,True,mono=True,center=True)

    def cursor(self,game,hand,now):
        playing=game.state==State.PLAYING
        aim=game.assisted_aim if playing else hand.aim
        x,y=aim[0]*WIDTH,aim[1]*HEIGHT
        color=CYAN if hand.tracked else MUTED
        locked=playing and hand.tracked and game.locked_target_id is not None
        if playing and not hand.aiming: color=ORANGE
        if locked:
            color=GREEN
            raw=(hand.aim[0]*WIDTH,hand.aim[1]*HEIGHT)
            if math.dist(raw,(x,y))>8:
                self.line(raw,(x,y),(52,112,95))
                self.circle(raw,4,MUTED,1)
        self.circle((x,y),19 if hand.aiming or hand.gun else 13,color,1)
        self.circle((x,y),2,color,0)
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            self.line((x+dx*23,y+dy*23),(x+dx*32,y+dy*32),color,2)
        if game.hover_progress>0:
            pygame.draw.arc(self.surface,ORANGE,(x-25,y-25,50,50),-math.pi/2,-math.pi/2+math.tau*game.hover_progress,3)

    def draw(self,game,hand,now,frame=None,status='',metrics=None,demo=False):
        metrics=metrics or {}
        self.surface.blit(self.background,(0,0))
        if game.state in (State.PLAYING,State.PAUSED):
            self.arena(game,hand,now,frame)
            if game.state==State.PAUSED:
                veil=pygame.Surface((WIDTH,HEIGHT),pygame.SRCALPHA); veil.fill((4,10,20,215))
                self.surface.blit(veil,(0,0))
                self.text('MISSION PAUSED',(640,203),44,WHITE,True,center=True)
                self.text(game.notice or 'Take a breath. Your mission is waiting.',(640,264),16,MUTED,center=True)
                for b in game.buttons(): self.button(b,game,now)
        else:
            self.grid(now)
            if game.state==State.MAIN: self.main_menu(game,now)
            elif game.state==State.CHAPTERS: self.chapter_cards(game,now)
            elif game.state==State.SETTINGS: self.settings_page(game,hand,now,frame)
            elif game.state==State.CALIBRATION: self.calibration(game,hand,now,frame)
            elif game.state==State.RESULTS: self.results(game,now)
        self.header(game,demo,metrics.get('fps',0))
        self.effects(game,now)
        pygame.draw.rect(self.surface,(6,14,24),(0,679,WIDTH,41))
        self.circle((34,700),4,GREEN if hand.tracked else ORANGE,0)
        self.text(hand.gesture,(47,689),14,CYAN if hand.tracked else ORANGE,mono=True)
        self.text(status[:100] or f'Point at a menu item. Hold {game.settings.dwell:g}s or fold your thumb.',(226,690),13,MUTED)
        self.text('ESC  /  EXIT',(1140,690),12,MUTED,mono=True)
        if game.settings.debug:
            self.text(f'HAND {metrics.get("hand_ms",0):.0f}ms | MODEL {metrics.get("model_ms",0):.0f}ms | FLOW {metrics.get("flow",0):.2f}',(510,83),12,ORANGE,mono=True)
        fade=max(0,1-(now-game.transition_at)/.35)
        if fade>0:
            veil=pygame.Surface((WIDTH,HEIGHT),pygame.SRCALPHA); veil.fill((CYAN[0],CYAN[1],CYAN[2],int(38*fade)))
            self.surface.blit(veil,(0,0))
        self.cursor(game,hand,now)
        self.sound.update(game)
        return self.surface
