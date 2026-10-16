"""MediaPipe adapter and camera-independent, geometric gesture recognition."""
from dataclasses import dataclass
import math
import numpy as np

from config import Settings, MODEL_DIR, WIDTH, HEIGHT


@dataclass
class HandInput:
    tracked: bool = False
    aim: tuple = (.5, .5)
    raw_aim: tuple = (.5, .5)
    pointing: bool = False
    gun: bool = False
    palm: bool = False
    fire: bool = False
    pause: bool = False
    swipe: int = 0
    thumb_ratio: float = 0.0
    landmarks: object = None
    gesture: str = 'NO HAND'
    shot_aim: tuple | None = None
    ready: bool = False
    aiming: bool = False
    fist: bool = False
    v_sign: bool = False
    command_ready: bool = False


@dataclass
class Features:
    pointing: bool
    gun: bool
    palm: bool
    thumb_ratio: float
    fist: bool
    v_sign: bool


def angle(a, b, c):
    u, v = a-b, c-b
    return math.degrees(math.acos(float(np.clip(np.dot(u,v) / max(np.linalg.norm(u)*np.linalg.norm(v), 1e-9), -1, 1))))


def classify_hand(landmarks, aspect=4/3):
    p = np.asarray(landmarks, dtype=float).copy()
    p[:, 0] *= aspect
    p[:, 2] *= aspect
    scale = max(np.linalg.norm(p[0]-p[9]), np.linalg.norm(p[5]-p[17]), .001)
    extended = []
    folded = []
    for base in (5,9,13,17):
        bend = angle(p[base], p[base+1], p[base+3])
        straight = bend > 150
        reaches = np.linalg.norm(p[base+3]-p[0]) > np.linalg.norm(p[base+1]-p[0]) * 1.12
        extended.append(bool(straight and reaches))
        folded.append(bool(bend < 130 and np.linalg.norm(p[base+3]-p[0]) < np.linalg.norm(p[base+1]-p[0])*1.1))
    # Four fingers provide a large geometric change. Thumb position is irrelevant
    # to gameplay; retain its ratio only for the existing index-controlled menu.
    fist = all(folded)
    v_sign = extended[0] and extended[1] and folded[2] and folded[3]
    return Features(extended[0], extended[0] and sum(extended[1:]) <= 1,
                    all(extended), float(np.linalg.norm(p[4]-p[5])/scale),
                    fist, bool(v_sign))


class TriggerGate:
    """Require open -> closed with stable holds; a held trigger fires once."""
    def __init__(self):
        self.last_fire = -10.0
        self.reset()

    def reset(self):
        self.armed = False
        self.open_since = self.closed_since = None

    def update(self, value, now):
        if value >= .70:
            self.closed_since = None
            if self.open_since is None:
                self.open_since = now
            if now-self.open_since >= .12:
                self.armed = True
        elif value <= .45:
            self.open_since = None
            if self.closed_since is None:
                self.closed_since = now
            if self.armed and now-self.closed_since >= .065 and now-self.last_fire >= .28:
                self.armed = False
                self.last_fire = now
                return True
        else:
            self.open_since = self.closed_since = None
        return False


class FistShotTrigger:
    """Open hand -> fist fires once, preserving the last open-hand aim."""
    def __init__(self):
        self.last_fire = -10.0
        self.reset()

    def reset(self):
        self.armed = False
        self.open_since = self.closed_since = None
        self.last_time = self.saved_aim = self.shot_aim = None

    def update(self, palm, fist, aim, now):
        self.shot_aim = None
        if self.last_time is not None and now-self.last_time > .30:
            self.reset()
        self.last_time = now
        if palm:
            self.closed_since = None
            self.saved_aim = tuple(aim)
            if self.open_since is None: self.open_since = now
            # Record release even during cooldown; no sliding time window.
            if now-self.open_since >= .10-1e-9:
                self.armed = True
        elif fist:
            self.open_since = None
            if self.closed_since is None: self.closed_since = now
            if self.armed and now-self.closed_since >= .04-1e-9 and now-self.last_fire >= .28-1e-9:
                self.armed = False
                self.last_fire = now
                self.shot_aim = self.saved_aim
                return True
        else:
            # Allow intermediate finger poses on the way to a complete fist.
            self.open_since = self.closed_since = None
        return False


class GestureInterpreter:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.trigger = TriggerGate()
        self.shot_trigger = FistShotTrigger()
        self.gameplay = False
        self.calibrating = False
        self.cursor = np.array([.5,.5])
        self.previous_time = None
        self.cursor_initialized = False
        self.command_since = None
        self.command_latched = False
        self.last_pause = -10.0

    def update(self, landmarks, now, aspect=4/3, gameplay=False, calibrating=False):
        if (self.gameplay,self.calibrating) != (gameplay,calibrating):
            self.gameplay = gameplay
            self.calibrating = calibrating
            self.trigger.reset()
            self.shot_trigger.reset()
            self.command_since = None
        if self.previous_time is not None and now-self.previous_time > .30:
            # A camera/worker stall need not produce an explicit missing-hand frame.
            self.trigger.reset()
            self.shot_trigger.reset()
            self.command_since = None
        dt = max(.001, min(.15, now-(self.previous_time or now-.03)))
        self.previous_time = now
        if landmarks is None:
            self.trigger.reset()
            self.shot_trigger.reset()
            self.command_since = None
            # Preserve the command latch across tracking loss: release visibly to rearm.
            return HandInput(aim=tuple(self.cursor))
        p = np.asarray(landmarks)
        if p.shape != (21,3) or not np.isfinite(p).all():
            return self.update(None, now, aspect=aspect, gameplay=gameplay,calibrating=calibrating)
        f = classify_hand(p, aspect)
        # Menus still use the index; calibration uses the same palm
        # anchor as gameplay, so finger folding cannot move the input anchor.
        raw = p[[0,5,9,13,17],:2].mean(axis=0) if gameplay or calibrating else p[8,:2]
        x0,y0,x1,y1 = self.settings.bounds
        mapped = np.clip((raw-[x0,y0])/[x1-x0,y1-y0], 0, 1)
        if not self.cursor_initialized:
            self.cursor = mapped
            self.cursor_initialized = True
        else:
            # Filter in screen pixels so horizontal/vertical jitter feels the same.
            # Keep the last cursor through tracking loss instead of snapping on return.
            delta_px = (mapped-self.cursor)*[WIDTH,HEIGHT]
            if np.linalg.norm(delta_px) > self.settings.cursor_deadzone_px:
                alpha = 1-math.exp(-dt/max(.015,self.settings.smoothing))
                step_px = delta_px*alpha
                distance = np.linalg.norm(step_px)
                limit = self.settings.cursor_speed_px*dt
                if distance > limit:
                    step_px *= limit/distance
                self.cursor += step_px/[WIDTH,HEIGHT]
        pause = False
        if f.v_sign:
            if self.command_since is None:
                self.command_since = now
            if not self.command_latched and now-self.command_since >= .7 and now-self.last_pause > 1.2:
                pause = True
                self.command_latched = True
                self.last_pause = now
        else:
            self.command_since = None
            self.command_latched = False
        command_ready = f.v_sign and now-self.command_since >= .18
        if gameplay:
            # A fleeting V shape while clenching is a transition, not a command.
            if command_ready:
                self.shot_trigger.reset()
                fire = False
            else:
                fire = self.shot_trigger.update(f.palm,f.fist,tuple(self.cursor),now)
        elif f.pointing and not f.palm and not f.v_sign:
            fire = self.trigger.update(f.thumb_ratio, now)
        else:
            self.trigger.reset()
            fire = False
        gesture = 'OPEN PALM' if f.palm else ('FIRE' if fire else ('ARMED' if self.trigger.armed else 'GUN')) if f.gun else 'POINT' if f.pointing else 'RELAXED'
        if f.v_sign:
            gesture = 'V / COMMAND' if command_ready else 'V SIGN'
        elif gameplay:
            gesture = 'FIRE' if fire else 'FIST READY' if self.shot_trigger.armed else 'OPEN HAND' if f.fist else 'HOLD OPEN HAND'
        elif f.fist:
            gesture = 'FIST'
        return HandInput(True, tuple(self.cursor), tuple(raw), f.pointing, f.gun, f.palm,
                         fire, pause, thumb_ratio=f.thumb_ratio, landmarks=p,
                         gesture=gesture,shot_aim=self.shot_trigger.shot_aim if gameplay and fire else None,
                         ready=self.shot_trigger.armed if gameplay else self.trigger.armed,
                         aiming=gameplay and not command_ready and (f.palm or f.fist or self.shot_trigger.armed),
                         fist=f.fist,v_sign=f.v_sign,command_ready=command_ready)

    def apply_swipe(self,hand,swipe,now):
        """Route actual optical flow only after a deliberate V pose."""
        if hand.command_ready and swipe:
            hand.swipe = swipe
            hand.fire = hand.pause = False
            hand.shot_aim = None
            self.command_since = now
            self.command_latched = False


class HandTracker:
    """Called on a worker; VIDEO mode uses MediaPipe's temporal tracking."""
    def __init__(self, model_path=MODEL_DIR/'hand_landmarker.task'):
        if not model_path.is_file():
            raise FileNotFoundError('Hand model missing. Run: python tools/download_models.py --hand')
        import mediapipe as mp
        self.mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=.55, min_hand_presence_confidence=.55,
            min_tracking_confidence=.55)
        self.model = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.last_ms = -1

    def detect(self, bgr, now):
        import cv2
        stamp = max(self.last_ms+1, int(now*1000))
        self.last_ms = stamp
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.model.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), stamp)
        if not result.hand_landmarks:
            return None
        return np.array([(p.x,p.y,p.z) for p in result.hand_landmarks[0]])

    def close(self):
        self.model.close()
