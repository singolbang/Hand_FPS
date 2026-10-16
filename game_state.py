"""Game rules, UI actions and hit tests. No camera/model/pygame dependency."""
from dataclasses import dataclass
from enum import Enum, auto
import math
import random
import numpy as np

from config import WIDTH, HEIGHT, CHAPTERS, DEFAULT_BOUNDS


class State(Enum):
    MAIN = auto()
    CHAPTERS = auto()
    SETTINGS = auto()
    CALIBRATION = auto()
    PLAYING = auto()
    PAUSED = auto()
    RESULTS = auto()


@dataclass
class Button:
    action: str
    label: str
    rect: tuple
    subtitle: str = ''
    enabled: bool = True


@dataclass
class Target:
    id: int
    label: str
    box: tuple
    confidence: float = 1.0
    mask: object = None
    radius: float = 0.0

    @property
    def center(self):
        x0,y0,x1,y1 = self.box
        return ((x0+x1)/2, (y0+y1)/2)

    def contains(self, point, precision=False):
        x,y = point
        if not (0 <= x < 1 and 0 <= y < 1):
            return False
        if self.radius:
            cx,cy = self.center
            return math.hypot((x-cx)*WIDTH,(y-cy)*HEIGHT) <= self.radius
        if precision:
            if self.mask is None:
                return False
            h,w = self.mask.shape
            return bool(self.mask[min(h-1,int(y*h)), min(w-1,int(x*w))])
        x0,y0,x1,y1 = self.box
        return x0 <= x <= x1 and y0 <= y <= y1

    def assist_point(self, point, reach, precision=False):
        """Return distance and a genuine hittable point, never an expanded hitbox."""
        x,y=point
        if not (0<=x<1 and 0<=y<1) or (precision and self.mask is None):
            return None
        if self.contains(point,precision):
            return 0.0,point
        if self.radius:
            cx,cy=self.center
            dx,dy=(x-cx)*WIDTH,(y-cy)*HEIGHT
            distance=math.hypot(dx,dy)
            if distance-self.radius>reach: return None
            fraction=self.radius*.85/distance
            return distance-self.radius,(cx+dx*fraction/WIDTH,cy+dy*fraction/HEIGHT)
        x0,y0,x1,y1=self.box
        nearest=(max(x0,min(x1,x)),max(y0,min(y1,y)))
        if math.hypot((x-nearest[0])*WIDTH,(y-nearest[1])*HEIGHT)>reach:
            return None
        if precision:
            h,w=self.mask.shape
            # Search only the local assist neighborhood, not the full mask each frame.
            left=max(0,int((x-reach/WIDTH)*w))
            right=min(w,int((x+reach/WIDTH)*w)+1)
            top=max(0,int((y-reach/HEIGHT)*h))
            bottom=min(h,int((y+reach/HEIGHT)*h)+1)
            rows,cols=np.nonzero(self.mask[top:bottom,left:right])
            if not len(rows): return None
            xs,ys=(cols+left+.5)/w,(rows+top+.5)/h
            distances=((xs-x)*WIDTH)**2+((ys-y)*HEIGHT)**2
            index=int(np.argmin(distances))
            distance=float(math.sqrt(distances[index]))
            if distance>reach: return None
            return distance,(float(xs[index]),float(ys[index]))
        pad_x=min(2/WIDTH,(x1-x0)/4)
        pad_y=min(2/HEIGHT,(y1-y0)/4)
        aim=(max(x0+pad_x,min(x1-pad_x,x)),max(y0+pad_y,min(y1-pad_y,y)))
        return math.hypot((x-nearest[0])*WIDTH,(y-nearest[1])*HEIGHT),aim


@dataclass
class Effect:
    kind: str
    pos: tuple
    born: float
    label: str = ''
    target: object = None


class Game:
    def __init__(self, settings, profile, seed=None):
        self.settings, self.profile = settings, profile
        self.rng = random.Random(seed)
        self.state = State.MAIN
        self.chapter = self.selected = 1
        self.score = self.shots = self.hits = self.combo = 0
        self.emp_hits = 0
        self.remaining = settings.duration
        self.elapsed = 0.0
        self.targets = []
        self.effects = []
        self.hit_until = {}
        self.running = True
        self.hover = None
        self.hover_time = 0.0
        self.dwell_anchor = None
        self.transition_at = 0.0
        self.lock_until = .5
        self.notice = ''
        self.last_seen = 0.0
        self.last_emp = -10.0
        self.target_seq = 0
        self.cal_step = 0
        self.cal_samples = []
        self.cal_progress = 0.0
        self.availability = {1: '', 2: '', 3: ''}
        self.assisted_aim = (.5,.5)
        self.locked_target_id = None

    @property
    def accuracy(self):
        return 100*self.hits/self.shots if self.shots else 0.0

    @property
    def hover_progress(self):
        return min(1, self.hover_time/self.settings.dwell)

    def transition(self, state, now):
        self.state = state
        self.transition_at = now
        self.lock_until = now+.6
        self.hover, self.hover_time = None, 0
        self.locked_target_id = None

    def buttons(self):
        if self.state == State.MAIN:
            return [Button(a,l,(90,330+i*69,400,55),s) for i,(a,l,s) in enumerate([
                ('start','START GAME','01 / ENTER THE TRAINING GROUND'),
                ('chapters','CHAPTER SELECT','02 / CHOOSE YOUR MISSION'),
                ('settings','SETTINGS / CALIBRATION','03 / TUNE YOUR HAND INTERFACE'),
                ('quit','QUIT','04 / DISCONNECT')])]
        if self.state == State.CHAPTERS:
            buttons = [Button(f'chapter:{i}',CHAPTERS[i-1],(76+(i-1)*380,230,360,330),
                              f'CHAPTER 0{i}',i <= self.profile.unlocked and not self.availability[i]) for i in range(1,4)]
            return buttons+[Button('main','BACK TO MENU',(76,607,265,54))]
        if self.state == State.SETTINGS:
            return [Button(a,l,(112,245+i*74,500,57)) for i,(a,l) in enumerate([
                ('calibrate','CALIBRATE AIM RANGE'),
                ('smooth',f'AIM STABILITY  /  {"HIGH" if self.settings.smoothing > .09 else "NORMAL"}'),
                ('sound',f'SOUND  /  {"ON" if self.settings.sound else "OFF"}'),
                ('debug',f'LANDMARK OVERLAY  /  {"ON" if self.settings.debug else "OFF"}'),
                ('main','BACK TO MENU')])]
        if self.state == State.CALIBRATION:
            return [Button('settings','CANCEL',(80,610,250,52)),Button('reset_cal','RESET RANGE',(950,610,250,52))]
        if self.state == State.PAUSED:
            return [Button(a,l,(440,300+i*76,400,58)) for i,(a,l) in enumerate([
                ('resume','RESUME MISSION'),('retry','RESTART MISSION'),('main','BACK TO MENU')])]
        if self.state == State.RESULTS:
            buttons = [Button('retry','RETRY MISSION',(280,518,345,62)),Button('main','MAIN MENU',(655,518,345,62))]
            if self.chapter < 3:
                enabled = self.profile.unlocked > self.chapter and not self.availability[self.chapter+1]
                buttons.append(Button('next','NEXT CHAPTER' if enabled else 'NEXT CHAPTER / LOCKED',(440,606,400,54),enabled=enabled))
            return buttons
        return []

    def start(self, chapter, now):
        if chapter > self.profile.unlocked or self.availability.get(chapter):
            self.notice = self.availability.get(chapter) or 'Earn 300 points and finish the previous mission.'
            return
        self.chapter = chapter
        self.score = self.shots = self.hits = self.combo = self.emp_hits = 0
        self.remaining = self.settings.duration
        self.elapsed = 0
        self.targets, self.effects, self.hit_until = [], [], {}
        self.last_seen = now
        self.last_emp = now-5
        if chapter == 1:
            self.targets = [self.new_target() for _ in range(4)]
        self.notice = ''
        self.transition(State.PLAYING, now)

    def new_target(self):
        self.target_seq += 1
        for _ in range(40):
            cx,cy = self.rng.uniform(.25,.82),self.rng.uniform(.32,.70)
            if all(math.hypot((cx-t.center[0])*WIDTH,(cy-t.center[1])*HEIGHT)>150 for t in self.targets):
                break
        r = self.rng.uniform(44,58)
        return Target(self.target_seq, 'DRONE', (cx-r/WIDTH,cy-r/HEIGHT,cx+r/WIDTH,cy+r/HEIGHT),radius=r)

    def activate(self, action, now):
        if action == 'quit':
            self.running = False
        elif action in ('main','chapters','settings'):
            self.notice = ''
            self.transition({'main':State.MAIN,'chapters':State.CHAPTERS,'settings':State.SETTINGS}[action],now)
        elif action == 'start':
            self.start(1,now)
        elif action.startswith('chapter:'):
            self.start(int(action.split(':')[1]),now)
        elif action == 'next':
            self.start(min(3,self.chapter+1),now)
        elif action == 'retry':
            self.start(self.chapter,now)
        elif action == 'resume':
            self.last_seen = now
            self.notice = ''
            self.transition(State.PLAYING,now)
        elif action == 'calibrate':
            self.cal_step, self.cal_samples, self.cal_progress = 0, [], 0
            self.transition(State.CALIBRATION,now)
            self.lock_until = now+2
        elif action == 'reset_cal':
            self.settings.bounds = self.profile.calibration = DEFAULT_BOUNDS
            self.profile.save()
            self.transition(State.SETTINGS,now)
        elif action in ('smooth','sound','debug'):
            if action == 'smooth':
                self.settings.smoothing = .075 if self.settings.smoothing > .09 else .15
            else:
                setattr(self.settings,action,not getattr(self.settings,action))
            self.lock_until = now+.8
            self.hover, self.hover_time = None, 0

    def shoot(self, point, now):
        self.shots += 1
        self.effects.append(Effect('shot',point,now))
        # Prefer the visibly locked target, then the smallest overlap; require a real mask in Chapter 3.
        candidates = sorted(self.targets, key=lambda t:(t.id!=self.locked_target_id,(t.box[2]-t.box[0])*(t.box[3]-t.box[1])))
        hit = next((t for t in candidates if now >= self.hit_until.get(t.id,0) and t.contains(point,self.chapter==3)),None)
        if hit:
            self.hits += 1
            self.combo += 1
            points = 100 + min(self.combo-1,5)*20
            self.score += points
            self.effects.append(Effect('hit',point,now,f'+{points}',hit))
            self.remove_target(hit,now)
        else:
            self.combo = 0
            self.effects.append(Effect('miss',point,now,'MISS'))

    def remove_target(self,target,now):
        if self.chapter == 1:
            self.targets = [t for t in self.targets if t.id != target.id]
            self.targets.append(self.new_target())
        else:
            self.hit_until[target.id] = now+2.0

    def special(self,now):
        if now-self.last_emp < 5:
            return
        self.last_emp = now
        self.effects.append(Effect('emp',(.5,.5),now,'OPTICAL FLOW / EMP'))
        for target in list(self.targets):
            if now >= self.hit_until.get(target.id,0):
                self.score += 40
                self.emp_hits += 1
                self.effects.append(Effect('hit',target.center,now,'+40 EMP',target))
                self.remove_target(target,now)

    def resolve_aim(self,point,now):
        reach=self.settings.precision_assist_px if self.chapter==3 else self.settings.aim_assist_px
        candidates=[]
        for target in self.targets:
            if now<self.hit_until.get(target.id,0): continue
            result=target.assist_point(point,reach,self.chapter==3)
            if result is not None:
                distance,aim=result
                candidates.append((distance,(target.box[2]-target.box[0])*(target.box[3]-target.box[1]),target.id,aim))
        if not candidates: return point,None
        best=min(candidates,key=lambda item:item[:2])
        # Avoid flickering between adjacent targets under small tracking jitter.
        previous=next((item for item in candidates if item[2]==self.locked_target_id),None)
        if previous is not None and previous[0]<=best[0]+10:
            best=previous
        return best[3],best[2]

    def update(self,dt,now,hand,vision_targets=None):
        self.effects = [e for e in self.effects if now-e.born < 1.2]
        self.hit_until = {k:v for k,v in self.hit_until.items() if v>now}
        if self.state == State.PLAYING:
            if self.chapter > 1:
                self.targets = list(vision_targets or [])
            if hand.tracked:
                self.last_seen = now
            elif now-self.last_seen > 1.2:
                self.notice = 'Tracking lost. Show your hand, then select RESUME.'
                self.transition(State.PAUSED,now)
                return
            if hand.pause and now >= self.lock_until:
                self.notice = 'Mission suspended'
                self.transition(State.PAUSED,now)
                return
            self.remaining = max(0,self.remaining-dt)
            self.elapsed += dt
            if self.remaining <= 0:
                self.profile.record(self.chapter,self.score)
                self.transition(State.RESULTS,now)
                return
            if self.chapter == 1:
                for target in self.targets:
                    dx = math.sin(now*.9+target.id)*.008*dt
                    dy = math.cos(now*1.2+target.id)*.010*dt
                    cx,cy=target.center
                    if not .12 < cx+dx < .88: dx=0
                    if not .23 < cy+dy < .8: dy=0
                    a,b,c,d=target.box
                    target.box=(a+dx,b+dy,c+dx,d+dy)
            source_aim=hand.shot_aim if hand.fire and hand.shot_aim is not None else hand.aim
            # A confirmed event remains valid if the latest pose changes before polling.
            shooting_pose=hand.aiming or (hand.fire and hand.shot_aim is not None)
            if hand.tracked and shooting_pose:
                self.assisted_aim,self.locked_target_id=self.resolve_aim(source_aim,now)
            else:
                self.assisted_aim,self.locked_target_id=source_aim,None
            if now >= self.lock_until and hand.tracked:
                if hand.swipe:
                    self.special(now)
                elif hand.fire and shooting_pose:
                    self.shoot(self.assisted_aim,now)
            return
        if now < self.lock_until:
            return
        if hand.pause and self.state in (State.CHAPTERS,State.SETTINGS,State.RESULTS):
            self.activate('main',now)
            return
        if self.state == State.CHAPTERS and hand.swipe:
            self.selected = max(1,min(3,self.selected+hand.swipe))
            self.effects.append(Effect('navigate',(.5,.5),now))
        if self.state == State.CALIBRATION:
            self.calibrate(hand,now)
        point = (hand.aim[0]*WIDTH,hand.aim[1]*HEIGHT)
        if self.dwell_anchor is not None and hand.tracked:
            moved = math.hypot(point[0]-self.dwell_anchor[0],point[1]-self.dwell_anchor[1]) > 55
            if moved or not hand.pointing or hand.palm or hand.v_sign:
                self.dwell_anchor = None
        hovered = None
        if hand.tracked and hand.pointing and not hand.palm and not hand.v_sign:
            for b in self.buttons():
                x,y,w,h = b.rect
                if b.enabled and x <= point[0] <= x+w and y <= point[1] <= y+h:
                    hovered=b
                    break
        if hovered is None:
            self.hover,self.hover_time=None,0
        else:
            if self.hover != hovered.action:
                self.hover,self.hover_time=hovered.action,0
            self.hover_time = self.hover_time+dt if self.dwell_anchor is None else 0
            if hand.fire or self.hover_time >= self.settings.dwell:
                self.dwell_anchor = point
                self.effects.append(Effect('select',hand.aim,now))
                self.activate(hovered.action,now)

    def calibrate(self,hand,now):
        if not hand.tracked or not hand.palm:
            self.cal_samples=[]
            self.cal_progress=0
            return
        if self.cal_samples and math.dist(hand.raw_aim,self.cal_samples[0][1])>.035:
            self.cal_samples=[]
        self.cal_samples.append((now,hand.raw_aim))
        self.cal_progress=min(1,(now-self.cal_samples[0][0])/1.3)
        if self.cal_progress<1:
            return
        point=tuple(sum(p[i] for _,p in self.cal_samples)/len(self.cal_samples) for i in (0,1))
        if self.cal_step==0:
            self.cal_first=point
            self.cal_step=1
            self.cal_samples=[]
            self.cal_progress=0
            self.lock_until=now+2
        else:
            x0,y0=self.cal_first
            x1,y1=point
            if x1-x0 < .2 or y1-y0 < .2:
                self.notice='Range too small or reversed. Try again with wider movements.'
            else:
                self.settings.bounds=self.profile.calibration=(max(0,x0),max(0,y0),min(1,x1),min(1,y1))
                self.profile.save()
                self.notice='Calibration saved.'
            self.transition(State.SETTINGS,now)
