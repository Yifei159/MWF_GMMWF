from dataclasses import dataclass
import json

@dataclass
class STFTConfig:
    n_fft: int = 1024
    hop: int = 256
    win: int = 1024
    window: str = 'hann'

@dataclass
class GMMConfig:
    K_s: int = 6
    K_n: int = 13
    stage1_k: int = 7
    ema_eta: float = 0.3

@dataclass
class ParamWFConfig:
    beta1: float = 0.5
    gamma1: float = 3.2
    beta2: float = 2.0
    gamma2: float = 1.0

@dataclass
class Config:
    root: str = "./multi_yen_2-6"
    process_sr: int = 96000
    ref_ch: int = 0
    n_channels: int = 5
    stft: STFTConfig = STFTConfig()
    gmm: GMMConfig = GMMConfig()
    wf: ParamWFConfig = ParamWFConfig()
    lowfreq_stage1_hz: float = 300.0
    stats_dir: str = './stats'
    out_dir: str = './outputs'

def cfg_to_json(cfg: Config) -> str:
    d = {
        "root": cfg.root,
        "process_sr": cfg.process_sr,
        "ref_ch": cfg.ref_ch,
        "n_channels": cfg.n_channels,
        "stats_dir": cfg.stats_dir,
        "out_dir": cfg.out_dir,
    }
    return json.dumps(d, indent=2)