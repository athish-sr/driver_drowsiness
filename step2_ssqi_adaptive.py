"""
STEP 2: Segmented S_SQI (skewness) with adaptive per-person classification
============================================================================
Input: the fusion output CSV produced by step1_adaptive_filter.py
       (columns: Elapsed_s, IR Fusion[, RED Fusion]).

FIRST, THE QUESTION YOU ASKED EARLIER: "is the SSQI class threshold
static, or does it vary per person?"
--------------------------------------------------------------------------
In the paper itself: STATIC. The reported group means (e.g. G1 ~ +0.07 to
+0.11, G2 ~ -0.07 to -0.17, G3 ~ -0.16 to -0.21, see Table 2) differ
across annotators/datasets, but the actual classifier the paper proposes
is a single FIXED, population-wide cut: S_SQI > 0 = "excellent" (G1-like),
S_SQI <= 0 = degraded. That threshold does not adapt per subject or drift
over time in the paper -- it is deliberately simple so it generalizes.
What this script does (rolling per-person recalibration) is your own
extension on top of that, not something the paper evaluated.

NO DOUBLE STABILIZATION
-------------------------
Step 1 already discarded the first 60s of the RAW recording for
stabilization + SG calibration. The fusion file this script reads
therefore already starts on "settled" data -- so this script does NOT
skip anything at the start. Every segment in the fusion file is used.

TIMELINE (revised per your instruction)
-----------------------------------------
  Segments 1..ROLLING_N   (default: first 10 x 3s = 30s of this file)
      -> classified using the STATIC (paper-style) thresholds, because
         there isn't yet a full rolling window of past segments to build
         an adaptive baseline from. These segments' S_SQI values are
         simultaneously fed into the rolling buffer.

  Segments ROLLING_N+1 onward
      -> classified using ADAPTIVE thresholds computed from the
         PRECEDING ROLLING_N segments (default: previous 30s window).
         After classifying segment i, its S_SQI value is pushed into the
         rolling buffer (evicting the oldest) so the very next segment's
         thresholds are recomputed from an updated, still strictly-past,
         30s window. This is the continuously-sliding recalibration you
         described ("use the previous 30 sec window ... use it for next").

The static label is still computed for every segment throughout (not
just the first 10), so you always have both columns to compare.

NOTE ON SEGMENT LENGTH (3s vs 4s)
----------------------------------
Your message said "3 sec segmentation" in one place and "process in 4
sec segment" in another -- I've defaulted to 3s (--seg-s 3), matching
the paper's own optimal window (W=2-3s for G1-vs-G3/G1-vs-G2). Pass
--seg-s 4 if you actually meant 4s.

ADAPTIVE THRESHOLD METHOD
--------------------------
Within the rolling buffer of S_SQI values, thresholds are set at the
33rd and 67th percentiles (tertiles), splitting THIS PERSON's own recent
signal into thirds: bottom third -> "bad", middle third -> "good", top
third -> "excellent". This is a design choice (documented, not from the
paper) -- swap `np.percentile(..., [33, 67])` for something else if you
want a different split.

OUTPUT
------
  <outdir>/ssqi_classified.csv - one row per 3s (or --seg-s) segment,
      with S_SQI value, static label, adaptive label, which mode
      actually drove the adaptive column ("static-fallback" vs
      "rolling"), and the adaptive thresholds in effect at that moment.
  <outdir>/ssqi_summary.png    - S_SQI trace with both classification
      schemes plotted for visual comparison.

USAGE
-----
    python step2_ssqi_adaptive.py --csv ppg_fusion_output.csv --outdir .
"""

import argparse
from collections import deque
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew


DEFAULT_DATA_DIR = "PPG_DATA_COLLECTION"
DEFAULT_INPUT_FILE = os.path.join(DEFAULT_DATA_DIR, "ppg_fusion_output.csv")


# --------------------------------------------------------------------------
# Column helpers
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# S_SQI, Eq. 3 of the paper -- population skewness
# --------------------------------------------------------------------------
def ssqi(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x) == 0:
        return np.nan
    return float(skew(x, bias=True))


# --------------------------------------------------------------------------
# Static (paper-derived) 3-way thresholds
# --------------------------------------------------------------------------
# Approximate cutoffs derived from the G1/G2/G3 mean S_SQI values reported
# in Table 2 across the two annotators and the adjudicator (G1 means
# ranged ~0.07-0.11, G2 means ~ -0.07 to -0.17, G3 means ~ -0.16 to -0.21).
# The paper's own headline result is the binary cut at 0; these two extra
# cutoffs are MY interpolation to get a 3-way excellent/good/bad split,
# not a number published in the paper -- adjust if you have a better
# source.
STATIC_EXCELLENT_CUT = 0.0     # paper's actual, validated cut (G1 vs rest)
STATIC_BAD_CUT = -0.15         # interpolated between reported G2/G3 means


def classify_static(value: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value > STATIC_EXCELLENT_CUT:
        return "excellent"
    if value >= STATIC_BAD_CUT:
        return "good"
    return "bad"


# --------------------------------------------------------------------------
# Rolling adaptive (per-person) thresholds
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Segment the signal in time (robust to non-uniform sample spacing)
# --------------------------------------------------------------------------
def make_segments(time, seg_s, start_t, end_t):
    edges = np.arange(start_t, end_t, seg_s)
    segs = []
    for lo in edges:
        hi = lo + seg_s
        idx = (time >= lo) & (time < hi)
        if idx.sum() > 0:
            segs.append((lo, hi, idx))
    return segs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 2: adaptive SSQI classification")
    ap.add_argument("--csv", default=DEFAULT_INPUT_FILE,
                    help=f"Fusion output from step 1 (default: {DEFAULT_INPUT_FILE})")
    ap.add_argument("--channel", choices=["IR", "RED"], default="IR",
                     help="Which fused channel to compute SSQI on")
    ap.add_argument("--seg-s", type=float, default=3.0,
                     help="Segment length for SSQI, in seconds "
                          "(paper used 2-3s; you also mentioned 4s -- pick one)")
    ap.add_argument("--rolling-n", type=int, default=10,
                     help="Number of past segments used both as the initial "
                          "static-only period AND as the rolling adaptive "
                          "window afterwards (10 segs x 3s = 30s)")
    ap.add_argument("--mode", choices=["static", "adaptive", "both"],
                     default="both")
    ap.add_argument("--outdir", default=DEFAULT_DATA_DIR,
                    help=f"Output directory (default: {DEFAULT_DATA_DIR})")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    time_col = find_time_column(df)
    fusion_col = find_column(df, f"{args.channel} Fusion")

    time = df[time_col].astype(float).values
    signal = df[fusion_col].astype(float).values
    t0, t_end = time[0], time[-1]
    duration = t_end - t0
    print(f"Loaded {len(time)} samples spanning {duration:.1f}s "
          f"(t = {t0:.1f}s to {t_end:.1f}s) from column '{fusion_col}'")
    print("No additional stabilization skip is applied here -- Step 1 "
          "already discarded the first 60s of the raw recording, so this "
          "file is processed from its very first sample.\n")

    warmup_s = args.rolling_n * args.seg_s
    if duration < warmup_s:
        print(f"*** WARNING: this file only spans {duration:.1f}s, but "
              f"{warmup_s:.0f}s ({args.rolling_n} segments) is needed before "
              f"any ADAPTIVE (rolling) classification can begin. All "
              f"segments will fall back to the static classifier. ***\n")

    segments = make_segments(time, args.seg_s, t0, t_end + 1e-9)
    if not segments:
        print("No segments could be formed (file too short). Exiting.")
        return

    rolling_buffer = deque(maxlen=args.rolling_n)
    rows = []

    for i, (lo, hi, idx) in enumerate(segments):
        val = ssqi(signal[idx])
        row = {"seg_start_s": lo, "seg_end_s": hi, "SSQI": val}

        if args.mode in ("static", "both"):
            row["label_static"] = classify_static(val)

        if args.mode in ("adaptive", "both"):
            have_full_window = len(rolling_buffer) >= args.rolling_n
            if have_full_window:
                lo_thr, hi_thr = adaptive_thresholds(rolling_buffer)
                row["adaptive_source"] = "rolling"
                row["adaptive_thr_lo"] = lo_thr
                row["adaptive_thr_hi"] = hi_thr
                row["label_adaptive"] = classify_adaptive(val, lo_thr, hi_thr)
            else:
                # not enough history yet -> fall back to static, per your
                # instruction ("first 30s / 10 segments, use static
                # threshold")
                row["adaptive_source"] = "static-fallback"
                row["adaptive_thr_lo"] = np.nan
                row["adaptive_thr_hi"] = np.nan
                row["label_adaptive"] = classify_static(val)

            # push this segment's value in for the NEXT segment's window
            rolling_buffer.append(val)

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.outdir, "ssqi_classified.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Classified {len(out_df)} segments of {args.seg_s:.0f}s each "
          f"(first {min(args.rolling_n, len(out_df))} used static-fallback "
          f"for the adaptive column while the rolling window filled).")
    print(f"Saved to: {out_csv}")

    if args.mode in ("static", "both"):
        print("\nStatic label counts:")
        print(out_df["label_static"].value_counts().to_string())
    if args.mode in ("adaptive", "both"):
        print("\nAdaptive label counts:")
        print(out_df["label_adaptive"].value_counts().to_string())
        print("\nAdaptive source breakdown:")
        print(out_df["adaptive_source"].value_counts().to_string())

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(14, 6))
    mid_t = (out_df["seg_start_s"] + out_df["seg_end_s"]) / 2
    ax.plot(mid_t, out_df["SSQI"], color="black", linewidth=1, marker="o",
            markersize=3, label="S_SQI")

    if args.mode in ("static", "both"):
        ax.axhline(STATIC_EXCELLENT_CUT, color="green", linestyle="--",
                    linewidth=1, label="static excellent cut (0)")
        ax.axhline(STATIC_BAD_CUT, color="red", linestyle="--",
                    linewidth=1, label="static bad cut (-0.15)")
    if args.mode in ("adaptive", "both"):
        ax.plot(mid_t, out_df["adaptive_thr_hi"], color="green",
                linestyle=":", linewidth=1, label="adaptive excellent thr")
        ax.plot(mid_t, out_df["adaptive_thr_lo"], color="red",
                linestyle=":", linewidth=1, label="adaptive bad thr")
        warmup_end = mid_t.iloc[min(args.rolling_n, len(mid_t)) - 1] \
            if len(mid_t) else None
        if warmup_end is not None:
            ax.axvspan(mid_t.iloc[0], warmup_end, color="gray", alpha=0.15,
                       label="static-fallback warm-up")

    ax.set_title(f"S_SQI over time ({args.channel} channel, "
                 f"{args.seg_s:.0f}s segments) -- static vs adaptive thresholds")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("S_SQI (skewness)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(args.outdir, "ssqi_summary.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to: {out_png}")


if __name__ == "__main__":
    main()

