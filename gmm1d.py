import os
import json
import numpy as np

from .config import Config
from .dsp_utils import (
    resample_audio, do_stft_mc, sample_spatial_cov,
    compute_mwf_weights, apply_mwf, select_noise_only_frames
)
from .io_utils import safe_read_wav_mc

class GMM1D:
    def __init__(self, K: int, max_iter: int = 100, tol: float = 1e-6, seed: int = 0):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.mu = None
        self.var = None
        self.pi = None

    def _init_params(self, x: np.ndarray):
        qs = np.linspace(0, 1, self.K + 2)[1:-1]
        self.mu = np.quantile(x, qs)
        v = np.var(x)
        self.var = np.full(self.K, v / self.K + 1e-6)
        self.pi = np.full(self.K, 1.0 / self.K)

    def fit(self, x: np.ndarray):
        x = x.reshape(-1).astype(np.float64)
        x = x[np.isfinite(x)]
        x = x[x > 0]
        if len(x) < self.K:
            x = np.pad(x, (0, self.K - len(x) + 1), mode='edge')
        self._init_params(x)
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            N = np.zeros((len(x), self.K))
            for k in range(self.K):
                var = max(self.var[k], 1e-12)
                N[:, k] = (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - self.mu[k]) ** 2 / var)
            num = N * self.pi[None, :]
            denom = np.sum(num, axis=1, keepdims=True) + 1e-18
            r = num / denom
            Nk = r.sum(axis=0) + 1e-12
            self.pi = Nk / len(x)
            self.mu = (r.T @ x) / Nk
            diff = x[:, None] - self.mu[None, :]
            self.var = np.sum(r * (diff ** 2), axis=0) / Nk
            ll = np.sum(np.log(np.sum(num, axis=1) + 1e-18))
            if ll - prev_ll < self.tol:
                break
            prev_ll = ll
        idx = np.argsort(self.mu)
        self.mu = self.mu[idx]
        self.var = self.var[idx]
        self.pi = self.pi[idx]
        return self

def train_source_gmm(cfg: Config, train_dirs):
    print('[train] source GMM')
    freqs = None
    all_vals_per_f = None
    for d in train_dirs:
        v_path = os.path.join(d, 'vocals.wav')
        v, sr, err = safe_read_wav_mc(v_path)
        if err is not None:
            continue
        if sr != cfg.process_sr:
            v = resample_audio(v, sr, cfg.process_sr)
        Z_v, freqs, _ = do_stft_mc(v, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)
        ref = Z_v[:, :, cfg.ref_ch]
        PSD = (np.abs(ref) ** 2)
        F = PSD.shape[0]
        if all_vals_per_f is None:
            all_vals_per_f = [PSD[f, :].astype(np.float32) for f in range(F)]
        else:
            for f in range(F):
                all_vals_per_f[f] = np.concatenate([all_vals_per_f[f], PSD[f, :].astype(np.float32)])
    if all_vals_per_f is None:
        raise RuntimeError('no clean vocals for GMM_s')

    K = cfg.gmm.K_s
    U_s = np.zeros((len(all_vals_per_f), K), dtype=np.float32)
    for f, vals in enumerate(all_vals_per_f):
        gm = GMM1D(K=K, max_iter=200, tol=1e-5, seed=1234).fit(vals)
        U_s[f, :] = gm.mu.astype(np.float32)

    os.makedirs(cfg.stats_dir, exist_ok=True)
    meta = {'freqs': freqs.tolist(), 'K_s': K, 'process_sr': cfg.process_sr}
    np.savez(os.path.join(cfg.stats_dir, 'source_gmm.npz'), U_s=U_s, meta=json.dumps(meta))
    print('[train] saved source_gmm.npz')
    return {'U_s': U_s, 'freqs': freqs}

def train_noise_gmm_per_channel(cfg: Config, train_dirs):
    print('[train] noise GMM per channel')
    nC = cfg.n_channels
    accum = None
    lowfreq_bin = None
    freqs = None

    for d in train_dirs:
        n_path = os.path.join(d, 'accompaniment.wav')
        m_path = os.path.join(d, 'mixture.wav')
        v_path = os.path.join(d, 'vocals.wav')
        n, srn, en = safe_read_wav_mc(n_path)
        m, srm, em = safe_read_wav_mc(m_path)
        v, srv, ev = safe_read_wav_mc(v_path)
        if any(e is not None for e in [en, em, ev]):
            continue

        if srn != cfg.process_sr: n = resample_audio(n, srn, cfg.process_sr)
        if srm != cfg.process_sr: m = resample_audio(m, srm, cfg.process_sr)
        if srv != cfg.process_sr: v = resample_audio(v, srv, cfg.process_sr)

        Z_n, freqs, _ = do_stft_mc(n, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)
        Z_m, _, _ = do_stft_mc(m, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)
        Z_v, _, _ = do_stft_mc(v, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)

        F, _, _ = Z_m.shape
        if accum is None:
            accum = [[np.empty((0,), dtype=np.float32) for _ in range(F)] for __ in range(nC)]
            lowfreq_bin = np.max(np.where(freqs <= cfg.lowfreq_stage1_hz)[0]) if np.any(freqs <= cfg.lowfreq_stage1_hz) else 0

        Phi_pp = sample_spatial_cov(Z_m)
        Phi_vv = sample_spatial_cov(Z_n)
        W = compute_mwf_weights(Phi_pp, Phi_vv)
        Y = apply_mwf(Z_m, W)

        mask_noise = select_noise_only_frames(Z_v[:, :, cfg.ref_ch], thr_db=-40.0)
        for q in range(nC):
            PSD = (np.abs(Y[:, :, q]) ** 2)[:, mask_noise]
            for f in range(F):
                accum[q][f] = np.concatenate([accum[q][f], PSD[f, :].astype(np.float32)])

    if accum is None:
        raise RuntimeError('no training samples for GMM_n')

    K = cfg.gmm.K_n
    results = []
    for q in range(nC):
        U_n = np.zeros((len(accum[q]), K), dtype=np.float32)
        for f, vals in enumerate(accum[q]):
            gm = GMM1D(K=K, max_iter=200, tol=1e-5, seed=5678).fit(vals)
            U_n[f, :] = gm.mu.astype(np.float32)

        low_bin = max(0, lowfreq_bin if lowfreq_bin is not None else 0)
        scores = [(k, float(U_n[:low_bin + 1, k].mean()) if low_bin >= 0 else float(U_n[:, k].mean())) for k in range(K)]
        scores.sort(key=lambda x: x[1], reverse=True)
        stage1_idx = [k for k, _ in scores[:cfg.gmm.stage1_k]]
        stage2_idx = [k for k, _ in scores[cfg.gmm.stage1_k:]]

        os.makedirs(cfg.stats_dir, exist_ok=True)
        meta = {
            'freqs': freqs.tolist(),
            'K_n': K,
            'stage1_idx': stage1_idx,
            'stage2_idx': stage2_idx,
            'process_sr': cfg.process_sr
        }
        np.savez(os.path.join(cfg.stats_dir, f'noise_gmm_ch{q}.npz'), U_n=U_n, meta=json.dumps(meta))
        results.append({'U_n': U_n, 'stage1_idx': stage1_idx, 'stage2_idx': stage2_idx, 'freqs': freqs})
        print(f'[train] saved noise_gmm_ch{q}.npz')
    return results

def load_source_U_s(stats_dir: str):
    d = np.load(os.path.join(stats_dir, 'source_gmm.npz'), allow_pickle=True)
    U_s = d['U_s']
    meta = d['meta'].item() if hasattr(d['meta'], 'item') else d['meta']
    meta = json.loads(meta)
    freqs = np.array(meta['freqs'])
    return U_s, freqs

def load_noise_U_n(stats_dir: str, q: int):
    d = np.load(os.path.join(stats_dir, f'noise_gmm_ch{q}.npz'), allow_pickle=True)
    U_n = d['U_n']
    meta = d['meta'].item() if hasattr(d['meta'], 'item') else d['meta']
    meta = json.loads(meta)
    freqs = np.array(meta['freqs'])
    idx = {'stage1': meta['stage1_idx'], 'stage2': meta['stage2_idx']}
    return U_n, idx, freqs
