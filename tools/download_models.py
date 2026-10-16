"""Explicit, bounded model downloads from the official model distributors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
MODELS={
    'hand_landmarker.task': ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',12_000_000),
    'yolo11n.pt': ('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',10_000_000),
    'yolo11n-seg.pt': ('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt',12_000_000),
}


def download(name,directory):
    url,limit=MODELS[name]
    path=directory/name
    if path.is_file() and path.stat().st_size>100_000:
        print(f'Present: {name} ({path.stat().st_size/1e6:.1f} MB)')
    else:
        directory.mkdir(parents=True,exist_ok=True)
        partial=path.with_suffix(path.suffix+'.part')
        try:
            print(f'Downloading {name} from official distributor...',flush=True)
            req=urllib.request.Request(url,headers={'User-Agent':'FingerGunFPS/1.0'})
            total=0
            with urllib.request.urlopen(req,timeout=30) as response, partial.open('wb') as out:
                while chunk:=response.read(256*1024):
                    total+=len(chunk)
                    if total>limit: raise ValueError(f'{name} exceeds expected download size')
                    out.write(chunk)
            if total<100_000: raise ValueError('Downloaded file is unexpectedly small')
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return {'url':url,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hand',action='store_true',help='Download only the hand model (Chapter 1)')
    p.add_argument('--directory',type=Path,default=ROOT/'models')
    args=p.parse_args()
    names=list(MODELS)[:1] if args.hand else list(MODELS)
    try:
        manifest={name:download(name,args.directory) for name in names}
        path=args.directory/'manifest.json'
        if path.exists():
            try: manifest={**json.loads(path.read_text()),**manifest}
            except (ValueError,OSError): pass
        path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('Models ready. No downloads occur while playing.')
    except Exception as exc:
        print(f'Model preparation failed: {exc}\nCheck your connection, or download the listed official URLs into models/.',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
