import csv
import math
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import resample_poly
from pystoi import stoi
from pesq import pesq


def read_wav(path: Path):
    x, sr = sf.read(str(path), always_2d=True)
    return x.astype(np.float32), sr


def align_lengths(*arrays):
    L = min(a.shape[0] for a in arrays)
    return [a[:L] for a in arrays]


def crop_channels_to_match(a: np.ndarray, b: np.ndarray):
    C = min(a.shape[1], b.shape[1])
    return a[:, :C], b[:, :C]


def to_mono(x: np.ndarray):
    return x if x.ndim == 1 else x.mean(axis=1)


def compute_snr(clean: np.ndarray, noise: np.ndarray, eps: float = 1e-12) -> float:
    clean = clean if clean.ndim == 2 else clean[:, None]
    noise = noise if noise.ndim == 2 else noise[:, None]
    L = min(clean.shape[0], noise.shape[0])
    C = min(clean.shape[1], noise.shape[1])
    clean = clean[:L, :C]
    noise = noise[:L, :C]
    p_clean = np.sum(clean ** 2, axis=0)
    p_noise = np.sum(noise ** 2, axis=0) + eps
    snr_ch = 10.0 * np.log10((p_clean + eps) / p_noise)
    return float(np.mean(snr_ch))


def safe_pesq(sr: int, ref: np.ndarray, deg: np.ndarray) -> float:
    ref = np.clip(to_mono(ref), -1.0, 1.0)
    deg = np.clip(to_mono(deg), -1.0, 1.0)
    try:
        if sr == 16000:
            return float(pesq(sr, ref, deg, 'wb'))
        if sr == 8000:
            return float(pesq(sr, ref, deg, 'nb'))
        ref16 = resample_poly(ref, 16000, sr)
        deg16 = resample_poly(deg, 16000, sr)
        ref16 = np.clip(ref16, -1, 1)
        deg16 = np.clip(deg16, -1, 1)
        return float(pesq(16000, ref16, deg16, 'wb'))
    except Exception as e:
        warnings.warn(f"PESQ failed (sr={sr}): {e}")
        return float('nan')


def safe_stoi(sr: int, ref: np.ndarray, deg: np.ndarray) -> float:
    ref = to_mono(ref)
    deg = to_mono(deg)
    try:
        if sr <= 48000:
            return float(stoi(ref, deg, sr, extended=True))
        ref48 = resample_poly(ref, 48000, sr)
        deg48 = resample_poly(deg, 48000, sr)
        return float(stoi(ref48, deg48, 48000, extended=True))
    except Exception as e:
        warnings.warn(f"STOI failed (sr={sr}): {e}")
        return float('nan')


def round_bucket(x: float, step: float = 0.5) -> float:
    return round(x / step) * step


def evaluate_sample(sample_dir: Path, methods: dict):
    rows = []
    in_snr_db = float('nan')

    mix_p = sample_dir / 'orig_mixture.wav'
    voc_p = sample_dir / 'orig_vocals.wav'
    if not mix_p.exists() or not voc_p.exists():
        return rows, in_snr_db

    mix, sr_mix = read_wav(mix_p)
    voc, sr_voc = read_wav(voc_p)

    if sr_mix != sr_voc:
        mix = resample_poly(mix, sr_voc, sr_mix).astype(np.float32)
        sr_mix = sr_voc
    sr = sr_voc
    mix, voc = align_lengths(mix, voc)
    mix, voc = crop_channels_to_match(mix, voc)

    try:
        in_snr_db = compute_snr(voc, mix - voc)
    except Exception:
        in_snr_db = float('nan')

    for method_name, fname in methods.items():
        proc_p = sample_dir / fname
        if not proc_p.exists():
            continue
        try:
            est, sr_est = read_wav(proc_p)
            if sr_est != sr:
                est = resample_poly(est, sr, sr_est).astype(np.float32)
            est, voc_al = align_lengths(est, voc)
            est, voc_al = crop_channels_to_match(est, voc_al)

            out_snr = compute_snr(voc_al, est - voc_al)
            snr_impr = out_snr - in_snr_db if not math.isnan(in_snr_db) else float('nan')
            s_stoi = safe_stoi(sr, voc_al, est)
            s_pesq = safe_pesq(sr, voc_al, est)

            rows.append({
                'test_id': sample_dir.name,
                'sr': sr,
                'method': method_name,
                'in_snr_db': in_snr_db,
                'out_snr_db': out_snr,
                'snr_impr_db': snr_impr,
                'stoi': s_stoi,
                'pesq': s_pesq,
            })
        except Exception as e:
            warnings.warn(f"Eval failed: {method_name} @ {sample_dir} ({e})")
            continue

    return rows, in_snr_db


def aggregate_and_plot(rows, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    per_csv = save_dir / 'per_sample_metrics.csv'
    with open(per_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'test_id', 'sr', 'method', 'in_snr_db', 'out_snr_db',
            'snr_impr_db', 'stoi', 'pesq'
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    buckets = {}
    methods = sorted(set(r['method'] for r in rows))
    for m in methods:
        buckets[m] = defaultdict(lambda: {'snr_impr_db': [], 'stoi': [], 'pesq': []})

    for r in rows:
        if math.isnan(r['in_snr_db']):
            continue
        b = round_bucket(r['in_snr_db'])
        d = buckets[r['method']][b]
        if not math.isnan(r['snr_impr_db']):
            d['snr_impr_db'].append(r['snr_impr_db'])
        if not math.isnan(r['stoi']):
            d['stoi'].append(r['stoi'] * 100.0)
        if not math.isnan(r['pesq']):
            d['pesq'].append(r['pesq'])

    def plot_metric(key, ylab, title, fname):
        plt.figure(figsize=(7, 4.5))
        for m in methods:
            xs = sorted(buckets[m].keys())
            if not xs:
                continue
            ys = [np.mean(buckets[m][x][key]) if buckets[m][x][key] else np.nan for x in xs]
            plt.plot(xs, ys, marker='o', label=m)
        plt.xlabel('Input SNR (dB)')
        plt.ylabel(ylab)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / fname, dpi=150)
        plt.close()

    plot_metric('snr_impr_db', 'Average SNR Improvement (dB)', 'SNR Improvement vs Input SNR', 'snr_improvement_vs_input_snr.png')
    plot_metric('stoi', 'Average STOI (%)', 'STOI vs Input SNR', 'stoi_vs_input_snr.png')
    plot_metric('pesq', 'Average PESQ', 'PESQ vs Input SNR', 'pesq_vs_input_snr.png')

    def safemean(vals):
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float('nan')

    overall = []
    for m in methods:
        m_rows = [r for r in rows if r['method'] == m]
        overall.append({
            'method': m,
            'avg_in_snr_db': safemean([r['in_snr_db'] for r in m_rows]),
            'avg_out_snr_db': safemean([r['out_snr_db'] for r in m_rows]),
            'avg_snr_impr_db': safemean([r['snr_impr_db'] for r in m_rows]),
            'avg_stoi_%': safemean([r['stoi'] * 100.0 for r in m_rows if not math.isnan(r['stoi'])]),
            'avg_pesq': safemean([r['pesq'] for r in m_rows if not math.isnan(r['pesq'])]),
            'count': len(m_rows),
        })

    overall_csv = save_dir / 'overall_summary.csv'
    with open(overall_csv, 'w', newline='') as f:
        if overall:
            header = list(overall[0].keys())
        else:
            header = ['method', 'avg_in_snr_db', 'avg_out_snr_db', 'avg_snr_impr_db', 'avg_stoi_%', 'avg_pesq', 'count']
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in overall:
            w.writerow(r)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, default='./outputs')
    parser.add_argument('--save_dir', type=str, default='./outputs_eval')
    parser.add_argument('--methods', type=str, nargs='*', default=['mwf', 'stage1', 'stage2', 'final_mono'])
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    save_dir = Path(args.save_dir)
    if not outputs_dir.exists():
        raise RuntimeError(f"Outputs dir not found: {outputs_dir}")

    methods = {}
    for m in args.methods:
        methods[m] = 'final_mono.wav' if m == 'final_mono' else f'{m}.wav'

    sample_dirs = [p for p in outputs_dir.iterdir() if p.is_dir()]
    if not sample_dirs:
        raise RuntimeError(f"No sample subdirectories found under {outputs_dir}")

    all_rows = []
    for sd in tqdm(sample_dirs, desc='Evaluating'):
        rows, _ = evaluate_sample(sd, methods)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No metrics computed")

    aggregate_and_plot(all_rows, save_dir)


if __name__ == '__main__':
    main()