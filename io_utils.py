import os
import glob
from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf


REQ_FILES = ('accompaniment.wav', 'mixture.wav', 'vocals.wav')


def has_all_three(d: str) -> bool:
    return all(os.path.isfile(os.path.join(d, f)) for f in REQ_FILES)


def _list_subdirs(root: str, sub: str) -> List[str]:
    base = os.path.join(root, sub)
    return sorted([p for p in glob.glob(os.path.join(base, '*')) if os.path.isdir(p) and has_all_three(p)])


def list_train_samples(root: str) -> List[str]:
    return _list_subdirs(root, 'train')


def list_test_samples(root: str) -> List[str]:
    return _list_subdirs(root, 'multi_yen_2-6_testDOA')


def safe_read_wav_mc(path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    try:
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            return None, None, "file missing or empty"
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            return None, None, "invalid audio info"
        x, sr = sf.read(path, always_2d=True)
        if x.size == 0:
            return None, None, "decoded empty"
        return x.astype(np.float32), sr, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def read_wav_mc(path: str) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=True)
    return x.astype(np.float32), sr


def write_wav_mc(path: str, x: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x, sr)