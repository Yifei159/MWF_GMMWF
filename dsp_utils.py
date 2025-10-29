import math
import numpy as np
from scipy.signal import stft, istft, resample_poly
from scipy.linalg import pinv

def resample_audio(x, orig_sr, target_sr):
    if orig_sr == target_sr:
        return x
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    x2d = x[:, None] if x.ndim == 1 else x
    out = [resample_poly(x2d[:, c], up, down) for c in range(x2d.shape[1])]
    y = np.stack(out, axis=1)
    return y[:, 0] if x.ndim == 1 else y

def do_stft_mc(x, sr, n_fft, hop, win, window):
    x2d = x[:, None] if x.ndim == 1 else x
    Fs = []
    for c in range(x2d.shape[1]):
        f, t, Zc = stft(
            x2d[:, c],
            fs=sr,
            nperseg=win,
            noverlap=win - hop,
            nfft=n_fft,
            window=window,
            boundary="zeros",
            padded=True,
        )
        Fs.append(Zc)
    Z = np.stack(Fs, axis=2).astype(np.complex64)  # (F, T, C)
    return Z, f, t

def do_istft_mc(Z, sr, n_fft, hop, win, window):
    F, T, C = Z.shape
    outs = []
    for c in range(C):
        _, x_c = istft(
            Z[:, :, c],
            fs=sr,
            nperseg=win,
            noverlap=win - hop,
            nfft=n_fft,
            window=window,
            input_onesided=True,
            boundary=True,
        )
        outs.append(x_c.astype(np.float32))
    L = min(len(xc) for xc in outs)
    y = np.stack([xc[:L] for xc in outs], axis=1)
    return y[:, 0] if C == 1 else y

def sample_spatial_cov(Z):
    F, T, C = Z.shape
    Phi = np.zeros((F, C, C), dtype=np.complex64)
    for f in range(F):
        X = Z[f, :, :]  # (T, C)
        Phi[f] = (X.conj().T @ X) / (T + 1e-9)
    return Phi

def compute_mwf_weights(Phi_pp, Phi_vv, eps=1e-6):
    F, C, _ = Phi_pp.shape
    I = np.eye(C, dtype=np.complex64)
    W = np.zeros_like(Phi_pp, dtype=np.complex64)
    for f in range(F):
        Phi_v = Phi_vv[f] + eps * I
        A = pinv(Phi_v) @ Phi_pp[f]
        denom = (1.0 + np.trace(A) - C).real
        if abs(denom) < 1e-12:
            denom = 1e-12
        W[f] = (A - I) / denom
    return W

def apply_mwf(Z_mix, W):
    F, T, C = Z_mix.shape
    Y = np.zeros_like(Z_mix)
    for f in range(F):
        Y[f] = (W[f].conj().T @ Z_mix[f].T).T
    return Y

def frame_energy(Z):
    return (np.abs(Z) ** 2).mean(axis=0)

def select_noise_only_frames(Z_voc_ref, thr_db=-40.0):
    e = frame_energy(Z_voc_ref)
    e_db = 10.0 * np.log10(e + 1e-12)
    return e_db < (np.max(e_db) + thr_db)

def param_wiener_gain(phi_s, phi_n, beta, gamma, eps=1e-10):
    return ((phi_s + eps) / (phi_s + beta * phi_n + eps)) ** gamma