# MWF_GMMWF

> **Multichannel Wiener Filtering (MWF) + Dual‑stage GMM‑based Parametric Wiener Post‑filtering** for audio recorded by microphones embedded on a noisy drone.
>
> This repository is an unofficial Python re‑implementation of the method in *Drone audition: audio signal enhancement from drone embedded microphones using multichannel Wiener filtering and Gaussian‑mixture based post‑filtering*.

**Reference paper**  
Manamperi, W. N., Abhayapala, T. D., Samarasinghe, P. N., & Zhang, J. (Aimee). *Drone audition: audio signal enhancement from drone embedded microphones using multichannel Wiener filtering and Gaussian‑mixture based post‑filtering*. **Applied Acoustics**, 216, 109818 (Dec 22, 2023).  
DOI: https://doi.org/10.1016/j.apacoust.2023.109818  

---

## Repository structure

```
MWF_GMMWF/
├─ config.py        
├─ io_utils.py  
├─ dsp_utils.py  
├─ gmm1d.py     
├─ train.py    
└─ enhance.py 
```

---

## Data layout expected by the scripts

```
MWF_GMMWF/
├─ train/
│  ├─ sample_000/
│  │   ├─ mixture.wav
│  │   ├─ vocals.wav          # clean target
│  │   └─ accompaniment.wav   # noise only
│  └─ sample_001/...
└─ multi_yen_2-6_testDOA/
   ├─ case_000/
   │   ├─ mixture.wav
   │   ├─ vocals.wav
   │   └─ accompaniment.wav
   └─ case_001/...
```

- Each `.wav` should be **multi‑channel**.

---

## Quick start

### 1) Fit the GMM

This scans `train/` to fit two GMMs. Including a noise GMM trained by first applying MWF to the mixture signal, then using a vocal-based VAD to select noise-only frames, and finally fitting the noise PSD per channel and per frequency bin, and a source GMM by using only the reference channel (channel 0) to fit the vocal PSD per frequency bin. 

It saves:

- `stats/source_gmm.npz`
- `stats/noise_gmm_ch{q}.npz` for `q = 0 .. n_channels-1`

```bash
python train.py
```

### 2) Inferencing

Outputs are written under `./outputs/<case>/`.

```bash
# Only multichannel Wiener filter
python enhance.py --mode mwf

# MWF + GMM Stage 1
python enhance.py --mode stage1

# MWF + GMM Stage 1 + GMM Stage 2
python enhance.py --mode stage2
```

---

### 3) Evaluation

`evaluation.py` script evaluate the enhancement results (SNR improvement, STOI, PESQ).

```bash
python evaluation.py
```

This produces:
- `per_sample_metrics.csv` — raw SNR/STOI/PESQ for each sample
- `overall_summary.csv` — mean metrics across samples
- 3 plots (SNR‑improvement / STOI / PESQ vs input SNR)
