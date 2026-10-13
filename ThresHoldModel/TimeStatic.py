"""
PPG-based threshold drowsiness detection
========================================
Reverse-engineered from: Liu et al., "Development of Real-Time Driver's Health
Detection System by Using Smart Handle", APHM 2017 (Section 3.1, Fig. 8).

Paper's algorithm (Fig. 8)
--------------------------
  1. Acquire PPG signal
  2. Find peaks -> peak-to-peak (PP) intervals
  3. Statistics over the pulses: average PP interval and dispersion (SD)
  4. Is  (average interval > ref value)  AND  (dispersion < ref value) ?
        YES -> drowsiness_count += 1
        NO  -> drowsiness_count -= 1, and if count < 0 -> count = 0
  5. drowsiness_count == 3 ?  YES -> DROWSINESS DETECTED,  NO -> keep acquiring
  (HR = 60 / PP interval)

Physiological idea: drowsy -> heart rate slows (longer PP) and becomes more
regular (lower dispersion).

The paper does NOT give the reference values, so here they are learned from a
short "alert" calibration window at the start of the recording (per driver),
and the thresholds are expressed as ratios of that baseline.

Usage
-----
    python ppg_drowsiness.py ppg_fusion_output.csv
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    fs: float = 25.0                 # resampling rate (Hz); data is ~24.9 Hz
    band: tuple = (0.5, 4.0)         # band-pass for PPG (Hz) -> 30-240 bpm
    min_pp_s: float = 0.35           # refractory distance between beats
    max_pp_s: float = 1.50           # PP intervals outside [min,max] are rejected
    prominence_sd: float = 0.30      # peak prominence as multiple of signal SD

    calib_s: float = 20.0            # alert-state calibration window
    win_s: float = 10.0              # analysis window (trailing)
    step_s: float = 2.0              # decision update period

    # Thresholds as ratios of the calibration baseline
    mean_ratio: float = 1.08         # drowsy if mean PP  > 1.08 x ref  (HR >= ~8 % slower)
    disp_ratio: float = 0.90         # drowsy if SD(PP)   < 0.90 x ref  (more regular)
    count_trigger: int = 3           # paper: 3 consecutive-ish hits => drowsy


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float):
    """Timestamps are jittery (and have gaps), so interpolate to a fixed grid."""
    tu = np.arange(t[0], t[-1] - 1e-6, 1.0 / fs)
    return tu, np.interp(tu, t, x)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def detect_peaks(t: np.ndarray, x: np.ndarray, cfg: Config):
    """Return peak times (s) of the pulse waveform."""
    idx, _ = find_peaks(
        x,
        distance=max(1, int(cfg.min_pp_s * cfg.fs)),
        prominence=cfg.prominence_sd * np.std(x),
    )
    # Sub-sample refinement (parabolic interpolation). At ~25 Hz one sample is
    # 40 ms, which is the same order as the PP dispersion we want to measure,
    # so without this the dispersion statistic is dominated by quantisation.
    idx = idx[(idx > 0) & (idx < len(x) - 1)]
    y0, y1, y2 = x[idx - 1], x[idx], x[idx + 1]
    denom = y0 - 2 * y1 + y2
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
    return t[idx] + np.clip(delta, -1, 1) / cfg.fs


def pp_intervals(peak_t: np.ndarray, cfg: Config):
    """PP intervals stamped at the time of the *second* peak; physiologically
    implausible intervals (missed / double-detected beats) are dropped."""
    pp = np.diff(peak_t)
    t_pp = peak_t[1:]
    ok = (pp >= cfg.min_pp_s) & (pp <= cfg.max_pp_s)
    return t_pp[ok], pp[ok]


# --------------------------------------------------------------------------- #
# Drowsiness state machine (Fig. 8)
# --------------------------------------------------------------------------- #
@dataclass
class DrowsinessDetector:
    cfg: Config
    ref_mean: float = np.nan
    ref_disp: float = np.nan
    count: int = 0
    log: list = field(default_factory=list)

    def calibrate(self, pp: np.ndarray):
        self.ref_mean = float(np.mean(pp))
        self.ref_disp = float(np.std(pp, ddof=1))

    def update(self, t_now: float, pp_win: np.ndarray):
        """One pass through the Fig. 8 flow chart. Returns (flag, drowsy_now)."""
        if len(pp_win) < 4:                       # not enough pulses to judge
            return None
        mean_pp = float(np.mean(pp_win))
        disp = float(np.std(pp_win, ddof=1))

        cond_mean = mean_pp > self.cfg.mean_ratio * self.ref_mean
        cond_disp = disp < self.cfg.disp_ratio * self.ref_disp
        hit = cond_mean and cond_disp

        if hit:
            self.count += 1                       # "increase drowsiness count"
        else:
            self.count -= 1                       # "decrease drowsiness count"
            if self.count < 0:                    # "if count<0 then count=0"
                self.count = 0

        drowsy = self.count >= self.cfg.count_trigger   # "count = 3 ?"
        self.log.append(dict(
            t=t_now, mean_pp=mean_pp, hr_bpm=60.0 / mean_pp, disp=disp,
            cond_mean=cond_mean, cond_disp=cond_disp, hit=hit,
            count=self.count, drowsy=drowsy,
        ))
        return drowsy


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(t_raw, x_raw, cfg: Config | None = None):
    cfg = cfg or Config()
    t, x = resample_uniform(np.asarray(t_raw), np.asarray(x_raw), cfg.fs)
    xf = bandpass(x - np.mean(x), cfg.fs, *cfg.band)

    peak_t = detect_peaks(t, xf, cfg)
    t_pp, pp = pp_intervals(peak_t, cfg)

    det = DrowsinessDetector(cfg)
    t0 = t[0]
    calib_mask = t_pp < t0 + cfg.calib_s
    if calib_mask.sum() < 8:
        raise ValueError("Not enough beats in the calibration window.")
    det.calibrate(pp[calib_mask])

    # Slide the analysis window over the recording after calibration
    t_eval = np.arange(t0 + cfg.calib_s + cfg.win_s, t[-1], cfg.step_s)
    for te in t_eval:
        m = (t_pp > te - cfg.win_s) & (t_pp <= te)
        det.update(te, pp[m])

    res = pd.DataFrame(det.log)
    return dict(t=t, x=x, xf=xf, peak_t=peak_t, t_pp=t_pp, pp=pp,
                det=det, result=res)


def load_csv(path: str, channel: str = "IR"):
    df = pd.read_csv(path)
    col = {"IR": "IR Fusion", "RED": "RED Fusion"}[channel]
    return df["Elapsed_s"].values, df[col].values


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "ppg_fusion_output.csv"
    t_raw, x_raw = load_csv(path, "IR")
    out = run_pipeline(t_raw, x_raw)
    d = out["det"]
    print(f"Reference (alert) mean PP = {d.ref_mean:.3f} s  "
          f"({60/d.ref_mean:.1f} bpm), SD = {d.ref_disp*1000:.1f} ms")
    r = out["result"]
    print(r.round(3).to_string(index=False))
    print("\nDrowsiness detected:", bool(r["drowsy"].any()))