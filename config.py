"""Small, validated settings and local progression storage."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 720
DEFAULT_BOUNDS = (.06, .04, .94, .94)
MODEL_DIR = ROOT / 'models'
CHAPTERS = ('TRAINING GROUND', 'REAL WORLD HUNT', 'PRECISION STRIKE')
DESCRIPTIONS = ('Master your aim. Build your streak.', 'Your surroundings become the arena.', 'Every pixel counts. Strike the silhouette.')


@dataclass
class Settings:
    camera: int = 0
    duration: float = 45.0
    smoothing: float = .15
    cursor_deadzone_px: float = 6.0
    cursor_speed_px: float = 1500.0
    aim_assist_px: float = 64.0
    precision_assist_px: float = 20.0
    dwell: float = 1.5
    bounds: tuple = DEFAULT_BOUNDS
    debug: bool = False
    sound: bool = True
    infer_size: int = 320
    infer_hz: float = 4.0
    confidence: float = .4


@dataclass
class Profile:
    path: Path = ROOT / 'data' / 'profile.json'
    unlocked: int = 1
    best: dict = field(default_factory=dict)
    calibration: tuple = DEFAULT_BOUNDS
    warning: str = ''

    def __post_init__(self):
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            self.unlocked = max(1, min(3, int(data.get('unlocked', 1))))
            self.best = {str(k): max(0, int(v)) for k, v in data.get('best', {}).items() if str(k) in ('1','2','3')}
            bounds = data.get('calibration', self.calibration)
            if (len(bounds) == 4 and all(isinstance(v, (int,float)) and math.isfinite(v) and 0 <= v <= 1 for v in bounds)
                    and bounds[2]-bounds[0] >= .2 and bounds[3]-bounds[1] >= .2):
                self.calibration = tuple(bounds)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError, OverflowError):
            self.unlocked, self.best = 1, {}
            self.warning = 'Save could not be read. Using defaults.'

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps({'unlocked': self.unlocked, 'best': self.best, 'calibration': self.calibration}, indent=2), encoding='utf-8')
            tmp.replace(self.path)
        except OSError:
            self.warning = 'Progress is in memory only: save folder is not writable.'

    def record(self, chapter, score):
        self.best[str(chapter)] = max(self.best.get(str(chapter), 0), score)
        if score >= 300:
            self.unlocked = max(self.unlocked, min(3, chapter+1))
        self.save()
