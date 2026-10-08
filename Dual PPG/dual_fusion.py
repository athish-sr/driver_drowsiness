import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew
import matplotlib.pyplot as plt

INPUT = os.path.join("ppg_results_sg", "ppg_fusion_output.csv")  # output of dual_process.py
OUT = "ppg_fusion_ssqi_results"
os.makedirs(OUT, exist_ok=True)
SHOW_WAVEFORM = True

# ---------------- Parameters ----------------
WINDOW_S = 10.0       # fusion window
SIGMA_FLOOR = 0.5     # BPM floor for HR SD
SEG_S = 3.0           # SSQI segment length inside each window
Q_LO = -0.15          # SSQI at/below this -> quality 0 (static "bad" cut)
Q_HI = 0.10           # SSQI at/above this -> quality 1
Q_FLOOR = 0.05        # minimum quality so a weight never hits exactly 0
GAMMA = 1.0           # SSQI influence: 0 = plain IVW, 1 = default, 2 = strong
MIN_INTERVALS = 3     # fewer valid beat intervals than this -> sensor unusable (weight 0)
Q_WARN = 0.3          # if BOTH sensors have quality below this -> flag window

# SSQI source: output folder of step2_ssqi_dual.py (one subfolder per channel,
# e.g. ssqi_dual/ssqi_IR1/ssqi_classified.csv). If a channel's file is missing,
# SSQI for that channel is computed inside this script (same 3 s skewness method).
SSQI_DIR = "ssqi_dual"


def _load_ssqi():
    pre = {}
    for cols in PAIRS_COLS:
        for col in cols:
            ch = col.replace(" Fusion", "").strip()
            f = os.path.join(SSQI_DIR, f"ssqi_{ch}", "ssqi_classified.csv")
            if os.path.isfile(f):
                pre[col] = pd.read_csv(f)
                print(f"SSQI {col}: loaded {f} ({len(pre[col])} segments)")
            else:
                print(f"SSQI {col}: {f} not found -> computing inside script")
    return pre


PAIRS_COLS = [("IR1 Fusion", "IR2 Fusion"), ("RED1 Fusion", "RED2 Fusion")]
_pre = _load_ssqi()

df = pd.read_csv(INPUT)
t = df["Elapsed_s"].to_numpy(float)
FS = 1.0 / np.median(np.diff(t))

PAIRS = {
    "IR": ("IR1 Fusion", "IR2 Fusion"),
    "RED": ("RED1 Fusion", "RED2 Fusion"),
}

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------- Beat detection ----------------
def detect_troughs(x):
    return find_peaks(
        -x,
        distance=max(1, int(FS * 60 / 180)),
        prominence=max(1e-9, 0.15 * np.std(x)),
    )[0]


def get_beats(x, trough_idx):
    B = []
    for a, b in zip(trough_idx[:-1], trough_idx[1:]):
        pp = (b - a) / FS
        if 60 / 180 <= pp <= 60 / 40:
            s = x[a:b + 1]
            if len(s) >= 5 and np.all(np.isfinite(s)):
                B.append((a, b))
    return B


# ---------------- SSQI quality ----------------
def ssqi_value(seg):
    if len(seg) < 3 or np.std(seg) == 0:
        return np.nan
    return float(skew(seg, bias=True))


def ssqi_to_quality(s):
    """Linear ramp: s<=Q_LO -> 0, s>=Q_HI -> 1, then floored."""
    if np.isnan(s):
        return Q_FLOOR
    q = (s - Q_LO) / (Q_HI - Q_LO)
    return float(np.clip(q, Q_FLOOR, 1.0))


def window_quality(x, ws, we, col=None):
    """Mean SSQI and mean quality over SEG_S-second segments inside [ws, we).
    Uses precomputed SSQI for this sensor (col) if provided, else computes it."""
    ssqis, quals = [], []
    if col in _pre:
        d = _pre[col]
        mid = (d["seg_start_s"] + d["seg_end_s"]) / 2
        for v in d["SSQI"][(mid >= ws) & (mid < we)].to_numpy(float):
            ssqis.append(v)
            quals.append(ssqi_to_quality(v))
    if not quals:  # no precomputed file / no segment in this window -> compute here
        for s0 in np.arange(ws, we, SEG_S):
            if min(s0 + SEG_S, we) - s0 < SEG_S - 1e-9:
                continue  # skip short trailing segment
            idx = (t >= s0) & (t < s0 + SEG_S)
            v = ssqi_value(x[idx])
            ssqis.append(v)
            quals.append(ssqi_to_quality(v))
    if len(quals) == 0:
        return np.nan, Q_FLOOR
    return np.nanmean(ssqis) if not np.all(np.isnan(ssqis)) else np.nan, float(np.mean(quals))


# ---------------- Fusion ----------------
def window_sigma(tt, ws, we):
    z = tt[(tt >= ws) & (tt < we)]
    pp = np.diff(z)
    pp = pp[(pp >= 60 / 180) & (pp <= 60 / 40)]
    hr = 60 / pp
    if len(hr) < MIN_INTERVALS:
        return np.inf  # dead / dropped-out sensor -> weight 0
    return max(float(np.std(hr, ddof=1)), SIGMA_FLOOR)


def fuse(c1, c2, use_ssqi):
    x1 = df[c1].to_numpy(float)
    x2 = df[c2].to_numpy(float)
    tt1, tt2 = t[detect_troughs(x1)], t[detect_troughs(x2)]

    w1 = np.zeros(len(t))
    w2 = np.zeros(len(t))
    rows = []

    for ws in np.arange(t[0], t[-1], WINDOW_S):
        we = ws + WINDOW_S
        s1, s2 = window_sigma(tt1, ws, we), window_sigma(tt2, ws, we)
        ss1, q1 = window_quality(x1, ws, we, c1)
        ss2, q2 = window_quality(x2, ws, we, c2)

        a = 0.0 if np.isinf(s1) else 1 / s1
        b = 0.0 if np.isinf(s2) else 1 / s2
        if use_ssqi:
            a *= q1 ** GAMMA
            b *= q2 ** GAMMA

        if a + b == 0:                      # both sensors unusable
            wa, wb = 0.5, 0.5
            low_quality = True
        else:                               # one dead -> other gets weight 1
            wa, wb = a / (a + b), b / (a + b)
            low_quality = bool(max(q1, q2) < Q_WARN)  # both clean-looking beats but bad shape

        idx = (t >= ws) & (t < we)
        w1[idx], w2[idx] = wa, wb
        rows.append({
            "start_s": ws, "end_s": we,
            f"{c1}_sigma_HR": s1, f"{c2}_sigma_HR": s2,
            f"{c1}_SSQI": ss1, f"{c2}_SSQI": ss2,
            f"{c1}_quality": q1, f"{c2}_quality": q2,
            f"{c1}_weight": wa, f"{c2}_weight": wb,
            "low_quality_window": low_quality,
        })

    return w1 * x1 + w2 * x2, w1, w2, pd.DataFrame(rows)


# ---------------- Features / metrics ----------------
def extract_features(x, method, wavelength):
    B = get_beats(x, detect_troughs(x))
    rows = []
    for beat, (a, b) in enumerate(B, 1):
        s = x[a:b + 1]
        peak = a + int(np.argmax(s))
        amp = float(x[peak] - x[a])
        half = x[a] + 0.5 * amp
        above = np.where(s >= half)[0]
        width50 = (above[-1] - above[0]) / FS if len(above) >= 2 else np.nan
        pp = (b - a) / FS
        rows.append({
            "method": method, "wavelength": wavelength, "beat": beat,
            "start_s": t[a], "peak_s": t[peak], "end_s": t[b],
            "PP_interval_s": pp, "Pulse_Rate_BPM": 60 / pp,
            "Pulse_Amplitude": amp,
            "Rise_Time_s": (peak - a) / FS, "Fall_Time_s": (b - peak) / FS,
            "Width_50pct_s": width50,
            "Pulse_Area": _trapz(s - s.min(), dx=1 / FS),
        })
    return pd.DataFrame(rows)


def ten_second_hr(x):
    tt = t[detect_troughs(x)]
    out = []
    for ws in np.arange(t[0], t[-1], 10):
        z = tt[(tt >= ws) & (tt < ws + 10)]
        pp = np.diff(z)
        pp = pp[(pp >= 60 / 180) & (pp <= 60 / 40)]
        out.append(np.median(60 / pp) if len(pp) else np.nan)
    return np.asarray(out)


def mean_ssqi(x):
    vals = []
    for s0 in np.arange(t[0], t[-1], SEG_S):
        idx = (t >= s0) & (t < s0 + SEG_S)
        vals.append(ssqi_value(x[idx]))
    return float(np.nanmean(vals))


# ---------------- Run both methods ----------------
METHODS = {"IVW": False, "IVW_SSQI": True}
outputs, beat_frames, weight_tables = {}, [], {}

for wavelength, (c1, c2) in PAIRS.items():
    for method, use_ssqi in METHODS.items():
        fused, w1, w2, tab = fuse(c1, c2, use_ssqi)
        outputs[(method, wavelength)] = fused
        df[f"{method}_{wavelength}_Fused"] = fused
        df[f"{method}_{wavelength}_W1"] = w1
        df[f"{method}_{wavelength}_W2"] = w2
        tab.to_csv(os.path.join(OUT, f"{method}_{wavelength}_window_weights.csv"), index=False)
        weight_tables[(method, wavelength)] = tab
        beat_frames.append(extract_features(fused, method, wavelength))

df.to_csv(os.path.join(OUT, "ppg_fusion_results.csv"), index=False)
beat_df = pd.concat(beat_frames, ignore_index=True)
beat_df.to_csv(os.path.join(OUT, "ppg_fused_beat_features.csv"), index=False)

consensus = {
    w: np.nanmedian(np.vstack([ten_second_hr(df[c1]), ten_second_hr(df[c2])]), axis=0)
    for w, (c1, c2) in PAIRS.items()
}

metrics = []
for (method, wavelength), x in outputs.items():
    b = extract_features(x, method, wavelength)
    h = ten_second_hr(x)
    metrics.append({
        "method": method, "wavelength": wavelength, "beats": len(b),
        "median_HR_BPM": b.Pulse_Rate_BPM.median(),
        "SD_HR_BPM": b.Pulse_Rate_BPM.std(),
        "PP_CV": b.PP_interval_s.std() / b.PP_interval_s.mean(),
        "10s_consensus_HR_MAE_BPM": np.nanmean(np.abs(h - consensus[wavelength])),
        "mean_SSQI_of_fused": mean_ssqi(x),  # higher = better shape
    })
metrics = pd.DataFrame(metrics)
metrics.to_csv(os.path.join(OUT, "ppg_fusion_comparison_metrics.csv"), index=False)


# ---------------- Plots ----------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
    plt.close()


def save_fusion_waveform(wavelength, c1, c2, show=False):
    """Save the source sensor waveforms and both fused waveforms."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, df[c1], label=c1, linewidth=0.8, alpha=0.55)
    ax.plot(t, df[c2], label=c2, linewidth=0.8, alpha=0.55)
    ax.plot(
        t,
        outputs[("IVW", wavelength)],
        label="IVW fusion",
        linewidth=1.2,
    )
    ax.plot(
        t,
        outputs[("IVW_SSQI", wavelength)],
        label="IVW+SSQI fusion",
        linewidth=1.2,
    )
    ax.set_title(f"{wavelength} PPG fusion waveform")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Signal Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT, f"{wavelength}_fusion_waveform.png"),
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show(block=True)
    plt.close(fig)


for wavelength, (c1, c2) in PAIRS.items():
    save_fusion_waveform(wavelength, c1, c2, show=SHOW_WAVEFORM)

    # weights: IVW vs IVW+SSQI for sensor 1
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, method in zip(axes, METHODS):
        ax.plot(t, df[f"{method}_{wavelength}_W1"], label=f"{c1} weight")
        ax.plot(t, df[f"{method}_{wavelength}_W2"], label=f"{c2} weight")
        ax.set_title(f"{wavelength} {method} weights")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Weight")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Elapsed Time (s)")
    savefig(f"{wavelength}_weights_IVW_vs_IVW_SSQI.png")

    # 10 s HR comparison
    series = {
        c1: ten_second_hr(df[c1]),
        c2: ten_second_hr(df[c2]),
        "IVW": ten_second_hr(outputs[("IVW", wavelength)]),
        "IVW+SSQI": ten_second_hr(outputs[("IVW_SSQI", wavelength)]),
        "Consensus": consensus[wavelength],
    }
    hr_t = np.arange(len(consensus[wavelength])) * 10 + t[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, h in series.items():
        ax.plot(hr_t, h, label=label, marker="o", markersize=2, linewidth=1)
    ax.set_title(f"{wavelength} 10-second HR comparison")
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("HR (BPM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(f"{wavelength}_10s_HR_comparison.png")

    # fused signals
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, outputs[("IVW", wavelength)], label="IVW", linewidth=1, alpha=0.8)
    ax.plot(t, outputs[("IVW_SSQI", wavelength)], label="IVW+SSQI", linewidth=1, alpha=0.8)
    ax.set_title(f"{wavelength} fused PPG")
    ax.set_xlabel("Elapsed Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(f"{wavelength}_fused_signals.png")

print(metrics.to_string(index=False))
print("Saved results in:", OUT)