import os
import argparse
import shutil
import numpy as np

from config import Config, cfg_to_json
from io_utils import list_test_samples, safe_read_wav_mc, write_wav_mc
from dsp_utils import (
    resample_audio, do_stft_mc, do_istft_mc,
    sample_spatial_cov, compute_mwf_weights, apply_mwf, param_wiener_gain
)
from gmm1d import load_source_U_s, load_noise_U_n


def solve_alphas(phi_y: np.ndarray, U_s: np.ndarray, U_n: np.ndarray, ema: np.ndarray, eta: float):
    A = np.concatenate([U_s, U_n], axis=1)
    sol, *_ = np.linalg.lstsq(A, phi_y, rcond=None)
    sol = np.maximum(sol, 0.0)
    if ema is not None:
        sol = eta * ema + (1 - eta) * sol
    Ks = U_s.shape[1]
    return sol[:Ks], sol[Ks:], sol


def mwf_only(Z_m, Z_n):
    Phi_pp = sample_spatial_cov(Z_m)
    Phi_vv = sample_spatial_cov(Z_n)
    W = compute_mwf_weights(Phi_pp, Phi_vv)
    return apply_mwf(Z_m, W)


def run_stage1(cfg: Config, Y, U_s, U_n_list, stage_idx_list):
    F, T, C = Y.shape
    Y1 = np.copy(Y)
    for q in range(C):
        U_n_full = U_n_list[q]
        idx = stage_idx_list[q].get('stage1', [])
        U_n = U_n_full[:, idx] if idx else U_n_full
        ema = None
        for t in range(T):
            phi_y = (np.abs(Y[:, t, q]) ** 2)
            a_s, a_n, ema = solve_alphas(phi_y, U_s, U_n, ema, cfg.gmm.ema_eta)
            phi_s = U_s @ a_s
            phi_n = U_n @ a_n
            G = param_wiener_gain(phi_s, phi_n, cfg.wf.beta1, cfg.wf.gamma1)
            Y1[:, t, q] = G * Y[:, t, q]
    return Y1


def run_stage2(cfg: Config, Y1, U_s, U_n_list, stage_idx_list):
    F, T, C = Y1.shape
    Y2 = np.copy(Y1)
    for q in range(C):
        U_n_full = U_n_list[q]
        idx = stage_idx_list[q].get('stage2', [])
        U_n = U_n_full[:, idx] if idx else U_n_full
        ema = None
        for t in range(T):
            phi_y = (np.abs(Y1[:, t, q]) ** 2)
            a_s, a_n, ema = solve_alphas(phi_y, U_s, U_n, ema, cfg.gmm.ema_eta)
            phi_s = U_s @ a_s
            phi_n = U_n @ a_n
            G = param_wiener_gain(phi_s, phi_n, cfg.wf.beta2, cfg.wf.gamma2)
            Y2[:, t, q] = G * Y1[:, t, q]
    return Y2


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--root', type=str, default="./multi_yen_2-6")
    p.add_argument('--process_sr', type=int, default=96000)
    p.add_argument('--n_channels', type=int, default=5)
    p.add_argument('--ref_ch', type=int, default=0)
    p.add_argument('--stats_dir', type=str, default='./stats')
    p.add_argument('--out_dir', type=str, default='./outputs')
    p.add_argument('--mode', type=str, default='stage2')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        root=args.root, process_sr=args.process_sr, ref_ch=args.ref_ch,
        n_channels=args.n_channels, stats_dir=args.stats_dir, out_dir=args.out_dir
    )
    print(cfg_to_json(cfg))

    U_s, _ = load_source_U_s(cfg.stats_dir)
    U_n_list, stage_idx_list = [], []
    for q in range(cfg.n_channels):
        U_n_q, idxs, _ = load_noise_U_n(cfg.stats_dir, q)
        if U_n_q.shape[0] != U_s.shape[0]:
            raise RuntimeError('freq bins mismatch between U_s and U_n')
        U_n_list.append(U_n_q)
        stage_idx_list.append(idxs)

    tests = list_test_samples(cfg.root)
    if not tests:
        print('no test samples'); return

    for d in tests:
        rel = os.path.basename(d.rstrip(os.sep))
        out_dir = os.path.join(cfg.out_dir, rel)
        os.makedirs(out_dir, exist_ok=True)

        n_path = os.path.join(d, 'accompaniment.wav')
        m_path = os.path.join(d, 'mixture.wav')
        v_path = os.path.join(d, 'vocals.wav')

        try:
            shutil.copy2(m_path, os.path.join(out_dir, 'orig_mixture.wav'))
            shutil.copy2(v_path, os.path.join(out_dir, 'orig_vocals.wav'))
            shutil.copy2(n_path, os.path.join(out_dir, 'orig_accompaniment.wav'))
        except Exception:
            pass

        n, srn, en = safe_read_wav_mc(n_path)
        m, srm, em = safe_read_wav_mc(m_path)
        v, srv, ev = safe_read_wav_mc(v_path)
        if any(e is not None for e in [en, em, ev]):
            continue

        if srn != cfg.process_sr: n = resample_audio(n, srn, cfg.process_sr)
        if srm != cfg.process_sr: m = resample_audio(m, srm, cfg.process_sr)
        if srv != cfg.process_sr: v = resample_audio(v, srv, cfg.process_sr)

        Z_n, _, _ = do_stft_mc(n, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)
        Z_m, _, _ = do_stft_mc(m, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)
        Z_v, _, _ = do_stft_mc(v, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)

        if args.mode == 'mwf':
            Y = mwf_only(Z_m, Z_n)
            x = do_istft_mc(Y, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)
            write_wav_mc(os.path.join(out_dir, 'mwf.wav'), x, cfg.process_sr)

        elif args.mode == 'stage1':
            Y = mwf_only(Z_m, Z_n)
            Y1 = run_stage1(cfg, Y, U_s, U_n_list, stage_idx_list)
            x1 = do_istft_mc(Y1, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)
            write_wav_mc(os.path.join(out_dir, 'stage1.wav'), x1, cfg.process_sr)

        else:
            Y = mwf_only(Z_m, Z_n)
            Y1 = run_stage1(cfg, Y, U_s, U_n_list, stage_idx_list)
            Y2 = run_stage2(cfg, Y1, U_s, U_n_list, stage_idx_list)
            x0 = do_istft_mc(Y,  cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)
            x1 = do_istft_mc(Y1, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)
            x2 = do_istft_mc(Y2, cfg.process_sr, cfg.stft.n_fft, cfg.stft.hop, cfg.stft.win, cfg.stft.window)
            write_wav_mc(os.path.join(out_dir, 'mwf.wav'),    x0, cfg.process_sr)
            write_wav_mc(os.path.join(out_dir, 'stage1.wav'), x1, cfg.process_sr)
            write_wav_mc(os.path.join(out_dir, 'stage2.wav'), x2, cfg.process_sr)
            mono = x2.mean(axis=1, keepdims=True)
            write_wav_mc(os.path.join(out_dir, 'final_mono.wav'), mono, cfg.process_sr)

        print('[done]', rel, '->', args.mode)


if __name__ == '__main__':
    main()