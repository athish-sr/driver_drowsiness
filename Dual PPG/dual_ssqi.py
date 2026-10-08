"""
STEP 2 (dual sensor): segmented S_SQI with static + rolling adaptive classification
===================================================================================
Input : ppg_fusion_output.csv from dual_process.py
        (Elapsed_s, IR1 Fusion, RED1 Fusion, IR2 Fusion, RED2 Fusion, ...)

Method: IDENTICAL to step2_ssqi_adaptive.py -- the same functions are copied
        into this file, so it is fully standalone (no import from step2):
          * S_SQI = population skewness per SEG_S-second segment
          * static labels  : >0 excellent, >=-0.15 good, else bad
          * adaptive labels: first ROLLING_N segments use the static thresholds
                             (static-fallback); after that, tertile thresholds
                             (33rd / 67th percentile) from the PRECEDING
                             ROLLING_N segments, then the segment is pushed
                             into the rolling buffer.
        It is run independently on every channel (IR1, IR2, RED1, RED2).

Output (in --outdir):
  ssqi_IR1/ssqi_classified.csv, ssqi_IR2/..., ssqi_RED1/..., ssqi_RED2/...
        same format as step2's output (use these for SSQI_CSV in the IVW fusion script)
  ssqi_dual_combined.csv      all channels side by side, one row per segment
  ssqi_dual_summary.csv       label counts and sensor-1 vs sensor-2 agreement
  ssqi_channels.png           S_SQI trace + thresholds for each channel
  ssqi_sensor_compare.png     sensor 1 vs sensor 2 S_SQI overlay per wavelength

Usage:
    python step2_ssqi_dual.py --csv ppg_results_sg/ppg_fusion_output.csv --outdir ssqi_dual
"""

import argparse
from collections import deque
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew

# ==========================================================================
# Functions below are copied unchanged from step2_ssqi_adaptive.py so this
# script is fully standalone (no import from that file).
# ==========================================================================


def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_column(df, wanted, required=True):
    target = _normalize(wanted)
    for col in df.columns:
        if _normalize(col) == target:
            return col
    if required:
        raise KeyError(f"Could not find a column matching '{wanted}'. "
                       f"Available: {list(df.columns)}")
    return None


def find_time_column(df):
    for c in ["Elapsed_s", "Elapsed s", "Time_s", "Time (s)"]:
        for col in df.columns:
            if _normalize(col) == _normalize(c):
                return col
    raise KeyError(f"Could not find a time column. Available: {list(df.columns)}")


# S_SQI, Eq. 3 of the paper -- population skewness
def ssqi(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x) == 0:
        return np.nan
    return float(skew(x, bias=True))


# Static (paper-derived) 3-way thresholds
STATIC_EXCELLENT_CUT = 0.0     # paper's validated cut (G1 vs rest)
STATIC_BAD_CUT = -0.15         # interpolated between reported G2/G3 means


def classify_static(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value > STATIC_EXCELLENT_CUT:
        return "excellent"
    if value >= STATIC_BAD_CUT:
        return "good"
    return "bad"


# Rolling adaptive (per-person) thresholds: tertiles of the rolling buffer
def adaptive_thresholds(buffer_values):
    vals = np.array([v for v in buffer_values if not np.isnan(v)])
    if len(vals) < 3:
        return None, None
    lo, hi = np.percentile(vals, [33, 67])
    return float(lo), float(hi)


def classify_adaptive(value: float, lo, hi) -> str:
    if np.isnan(value) or lo is None or hi is None:
        return "unknown"
    if value > hi:
        return "excellent"
    if value >= lo:
        return "good"
    return "bad"


# Segment the signal in time (robust to non-uniform sample spacing)
def make_segments(time, seg_s, start_t, end_t):
    edges = np.arange(start_t, end_t, seg_s)
    segs = []
    for lo in edges:
        hi = lo + seg_s
        idx = (time >= lo) & (time < hi)
        if idx.sum() > 0:
            segs.append((lo, hi, idx))
    return segs


CHANNELS = ["IR1", "IR2", "RED1", "RED2"]
PAIRS = {"IR": ("IR1", "IR2"), "RED": ("RED1", "RED2")}


def classify_channel(time, signal, seg_s, rolling_n):
    """Exact per-segment logic of step2_ssqi_adaptive.main() (mode='both')."""
    t0, t_end = time[0], time[-1]
    segments = make_segments(time, seg_s, t0, t_end + 1e-9)
    buf = deque(maxlen=rolling_n)
    rows = []

    for lo, hi, idx in segments:
        val = ssqi(signal[idx])
        row = {"seg_start_s": lo, "seg_end_s": hi, "SSQI": val,
               "label_static": classify_static(val)}

        if len(buf) >= rolling_n:
            lo_thr, hi_thr = adaptive_thresholds(buf)
            row["adaptive_source"] = "rolling"
            row["adaptive_thr_lo"] = lo_thr
            row["adaptive_thr_hi"] = hi_thr
            row["label_adaptive"] = classify_adaptive(val, lo_thr, hi_thr)
        else:
            row["adaptive_source"] = "static-fallback"
            row["adaptive_thr_lo"] = np.nan
            row["adaptive_thr_hi"] = np.nan
            row["label_adaptive"] = classify_static(val)

        buf.append(val)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_channel(ax, ch, d, seg_s, rolling_n):
    mid = (d["seg_start_s"] + d["seg_end_s"]) / 2
    ax.plot(mid, d["SSQI"], color="black", linewidth=1, marker="o",
            markersize=3, label="S_SQI")
    ax.axhline(STATIC_EXCELLENT_CUT, color="green", linestyle="--", linewidth=1,
               label="static excellent (0)")
    ax.axhline(STATIC_BAD_CUT, color="red", linestyle="--", linewidth=1,
               label="static bad (-0.15)")
    ax.plot(mid, d["adaptive_thr_hi"], color="green", linestyle=":", linewidth=1,
            label="adaptive excellent thr")
    ax.plot(mid, d["adaptive_thr_lo"], color="red", linestyle=":", linewidth=1,
            label="adaptive bad thr")
    if len(mid):
        ax.axvspan(mid.iloc[0], mid.iloc[min(rolling_n, len(mid)) - 1],
                   color="gray", alpha=0.15, label="static-fallback warm-up")
    ax.set_title(f"{ch} fused channel ({seg_s:.0f}s segments)")
    ax.set_ylabel("S_SQI (skewness)")
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description="Dual-sensor adaptive SSQI")
    ap.add_argument("--csv", default=os.path.join("ppg_results_sg", "ppg_fusion_output.csv"))
    ap.add_argument("--channels", nargs="+", choices=CHANNELS, default=CHANNELS,
                    help="Channels to process (default: all four)")
    ap.add_argument("--seg-s", type=float, default=3.0)
    ap.add_argument("--seg-ms", type=int, default=3000)
    ap.add_argument("--rolling-n", type=int, default=10)
    ap.add_argument("--outdir", default="ssqi_dual")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    time_col = find_time_column(df)
    time = df[time_col].astype(float).values
    print(f"Loaded {len(time)} samples, t = {time[0]:.1f}s to {time[-1]:.1f}s "
          f"({time[-1] - time[0]:.1f}s)")

    results = {}
    summary = []

    for ch in args.channels:
        col = find_column(df, f"{ch} Fusion")
        d = classify_channel(time, df[col].astype(float).values,
                             args.seg_s, args.rolling_n)
        results[ch] = d

        ch_dir = os.path.join(args.outdir, f"ssqi_{ch}")
        os.makedirs(ch_dir, exist_ok=True)
        d.to_csv(os.path.join(ch_dir, "ssqi_classified.csv"), index=False)

        print(f"\n{ch}: {len(d)} segments | mean S_SQI = {d['SSQI'].mean():.3f}")
        print("  static  :", d["label_static"].value_counts().to_dict())
        print("  adaptive:", d["label_adaptive"].value_counts().to_dict())

        row = {"channel": ch, "segments": len(d), "mean_SSQI": d["SSQI"].mean(),
               "median_SSQI": d["SSQI"].median()}
        for lab in ("excellent", "good", "bad", "unknown"):
            row[f"static_{lab}"] = int((d["label_static"] == lab).sum())
            row[f"adaptive_{lab}"] = int((d["label_adaptive"] == lab).sum())
        summary.append(row)

    # ---- combined wide table ----
    combined = None
    for ch, d in results.items():
        part = d.rename(columns={
            "SSQI": f"SSQI_{ch}",
            "label_static": f"label_static_{ch}",
            "label_adaptive": f"label_adaptive_{ch}",
            "adaptive_source": f"adaptive_source_{ch}",
            "adaptive_thr_lo": f"adaptive_thr_lo_{ch}",
            "adaptive_thr_hi": f"adaptive_thr_hi_{ch}",
        })
        combined = part if combined is None else combined.merge(
            part, on=["seg_start_s", "seg_end_s"], how="outer")
    combined = combined.sort_values("seg_start_s").reset_index(drop=True)

    # ---- sensor 1 vs sensor 2 comparison per wavelength ----
    summary_df = pd.DataFrame(summary)
    for wl, (c1, c2) in PAIRS.items():
        if c1 in results and c2 in results:
            combined[f"{wl}_better_sensor_SSQI"] = np.where(
                combined[f"SSQI_{c1}"] >= combined[f"SSQI_{c2}"], c1, c2)
            both = combined[[f"SSQI_{c1}", f"SSQI_{c2}"]].notna().all(axis=1)
            same = (combined[f"label_adaptive_{c1}"] == combined[f"label_adaptive_{c2}"])
            agree = float(same[both].mean()) if both.any() else np.nan
            corr = float(combined.loc[both, f"SSQI_{c1}"].corr(combined.loc[both, f"SSQI_{c2}"])) \
                if both.sum() > 2 else np.nan
            print(f"\n{wl}: {c1} vs {c2} -> adaptive-label agreement "
                  f"{agree * 100:.1f}%, S_SQI correlation {corr:.3f}")
            summary_df.loc[summary_df.channel.isin([c1, c2]),
                           "pair_adaptive_agreement"] = agree
            summary_df.loc[summary_df.channel.isin([c1, c2]),
                           "pair_SSQI_corr"] = corr

    combined.to_csv(os.path.join(args.outdir, "ssqi_dual_combined.csv"), index=False)
    summary_df.to_csv(os.path.join(args.outdir, "ssqi_dual_summary.csv"), index=False)

    # ---- plots ----
    chs = list(results)
    fig, axes = plt.subplots(len(chs), 1, figsize=(14, 3.6 * len(chs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, ch in zip(axes, chs):
        plot_channel(ax, ch, results[ch], args.seg_s, args.rolling_n)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "ssqi_channels.png"), dpi=150)
    plt.close()

    pairs = [(wl, p) for wl, p in PAIRS.items() if p[0] in results and p[1] in results]
    if pairs:
        fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 4 * len(pairs)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (wl, (c1, c2)) in zip(axes, pairs):
            for ch, colr in ((c1, "tab:blue"), (c2, "tab:orange")):
                d = results[ch]
                mid = (d["seg_start_s"] + d["seg_end_s"]) / 2
                ax.plot(mid, d["SSQI"], marker="o", markersize=3, linewidth=1,
                        color=colr, label=ch)
            ax.axhline(STATIC_EXCELLENT_CUT, color="green", linestyle="--", linewidth=1)
            ax.axhline(STATIC_BAD_CUT, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{wl}: sensor 1 vs sensor 2 S_SQI")
            ax.set_ylabel("S_SQI")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "ssqi_sensor_compare.png"), dpi=150)
        plt.close()

    print(f"\nSaved everything in: {args.outdir}")


if __name__ == "__main__":
    main()