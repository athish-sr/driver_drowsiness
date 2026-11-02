"""
STEP 2 (v2): Segmented S_SQI (skewness) quality classification
==============================================================
Input : fusion CSV from step1_adaptive_filter.py
        (columns: Elapsed_s, IR Fusion[, RED Fusion]).
Output: <outdir>/ssqi_classified.csv  - one row per window
        <outdir>/ssqi_summary.png     - diagnostic plot

WHAT CHANGED vs v1 (and why)
-----------------------------
v1 problems found in testing:
  * Tertile thresholds were computed from ALL recent segments, so a run of bad
    segments dragged the thresholds down until bad signal was called
    "good"/"excellent"; and a perfectly clean signal was still called "bad"
    ~1/3 of the time (tertiles are a RANKING, not a quality measure).
  * Skewness alone cannot detect noise (white noise has skewness ~ 0, which
    is "excellent" half the time under the paper's cut at 0).
  * 3 s windows give noisy skewness (SD ~0.075 on clean signal, similar to the
    gap between the static cuts) -> label flicker.
  * NaN / flatline / gappy / partial segments were not handled.

v2 design:
  1. HARD GATES run first. A window failing any gate is "bad" no matter what
     its skewness is: too few samples, big time gap, flatline, amplitude
     collapse (vs. the person's own recent amplitude), clipping, and a
     spectral-periodicity check (a PPG must be periodic in the heart-rate
     band; white noise / motion bursts are not).
  2. QUALITY-GATED BASELINE. Only windows that pass the gates AND are not
     "bad" by the static cut may feed the personal baseline. Bad stretches
     therefore FREEZE the baseline instead of dragging it.
  3. PERSONAL BASELINE, not tertiles. baseline = median & robust sigma
     (1.4826*MAD) of accepted windows. Thresholds are
         excellent_cut = median - k_exc * sigma
         bad_cut       = median - k_bad * sigma
     so a consistently excellent signal stays excellent.
  4. GUARD RAILS. Adaptive cuts are clamped to  static cut +/- GUARD (0.07).
     Since 2*GUARD < (static excellent cut - static bad cut), the adaptive
     label can differ from the static label by AT MOST ONE class, and bad
     can never become excellent (or vice versa) through adaptation.
  5. STATIC WARM-UP. The first WARMUP_S (30 s) uses the static cuts, and the
     baseline is only used once >= MIN_BASELINE accepted windows exist.
     Windows are fed to the baseline only after they are strictly in the past
     (no overlap with the window being classified).
  6. WINDOWING. 5 s windows with 1 s hop (default). 5 s measurably cut
     skewness noise and label flicker versus 3 s (see benchmark in the
     accompanying notes); the cost is that a short artifact contaminates
     ~W seconds more. Use --seg-s 3 / --hop-s 3 to get the old behaviour.
     Only FULL windows are classified (no trailing partial window).
  7. DEBOUNCE. A class change from skewness must persist --confirm-n
     consecutive windows to take effect (label_final). Hard-gate failures
     take effect immediately.
  8. POLARITY CHECK. Skewness flips sign if the waveform is inverted. The
     script checks clean pulsatile windows and warns (or flips with
     --polarity auto / invert).

The paper (S_SQI > 0 = excellent) is the only validated cut. The -0.15 bad cut
and every gate threshold below are heuristics: tune them on your own data.

USAGE
-----
    python step2_ssqi_adaptive.py --csv ppg_fusion_output.csv --outdir .
"""

import argparse
import math
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew


DEFAULT_DATA_DIR = "PPG_DATA_COLLECTION"
DEFAULT_INPUT_FILE = os.path.join(DEFAULT_DATA_DIR, "ppg_fusion_output.csv")

# Paper's validated cut (G1 vs rest) and the interpolated "bad" cut.
STATIC_EXCELLENT_CUT = 0.0
STATIC_BAD_CUT = -0.15

LABEL_COLORS = {"excellent": "tab:green", "good": "tab:orange",
                "bad": "tab:red", "unknown": "tab:gray"}


# --------------------------------------------------------------------------
# Column helpers
# --------------------------------------------------------------------------
def _normalize(name):
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
# S_SQI and classification
# --------------------------------------------------------------------------
def ssqi(x):
    """Population skewness (Eq. 3 of the paper). NaN if undefined."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or not np.all(np.isfinite(x)) or np.std(x) == 0:
        return float("nan")
    return float(skew(x, bias=True))


def classify(v, exc_cut, bad_cut):
    if not np.isfinite(v):
        return "unknown"
    if v > exc_cut:
        return "excellent"
    if v >= bad_cut:
        return "good"
    return "bad"


def classify_static(v):
    return classify(v, STATIC_EXCELLENT_CUT, STATIC_BAD_CUT)


def adaptive_thresholds(values, args):
    """Personal thresholds from ACCEPTED windows only, clamped to guard rails."""
    vals = np.asarray(values, dtype=float)
    med = float(np.median(vals))
    sigma = max(1.4826 * float(np.median(np.abs(vals - med))), args.min_sigma)
    exc = med - args.k_exc * sigma
    bad = med - args.k_bad * sigma
    g = args.guard
    exc = float(np.clip(exc, STATIC_EXCELLENT_CUT - g, STATIC_EXCELLENT_CUT + g))
    bad = float(np.clip(bad, STATIC_BAD_CUT - g, STATIC_BAD_CUT + g))
    return exc, bad, med, sigma


# --------------------------------------------------------------------------
# Per-window features
# --------------------------------------------------------------------------
def window_spectral(t, x, lo, seg_s, fs):
    """
    Returns (periodicity, slope_ratio) for one window.
      periodicity : fraction of 0.5-8 Hz power lying within +/-(2/seg_s) Hz of
                    the dominant heart-rate peak (0.7-3 Hz) and its 2nd/3rd
                    harmonics. ~0.9 for clean PPG, ~0.2-0.4 for noise.
      slope_ratio : 98th pct upstroke slope / |2nd pct downstroke slope|.
                    > 1 for a correctly oriented PPG (fast rise, slow fall).
    """
    n = int(round(seg_s * fs))
    if fs < 10 or n < 16 or len(x) < 8:
        return float("nan"), float("nan")
    grid = lo + np.arange(n) / fs
    xu = np.interp(grid, t, x)
    xu = xu - xu.mean()

    # slope ratio on a lightly smoothed copy
    k = 3
    xs = np.convolve(xu, np.ones(k) / k, mode="valid")
    d = np.diff(xs)
    up, dn = np.percentile(d, 98), np.percentile(d, 2)
    slope_ratio = float(up / abs(dn)) if (up > 0 and dn < 0) else float("nan")

    # periodicity
    nfft = int(2 ** math.ceil(math.log2(n * 4)))
    P = np.abs(np.fft.rfft(xu * np.hanning(n), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_hi = min(8.0, 0.45 * fs)
    band = (f >= 0.5) & (f <= f_hi)
    total = P[band].sum()
    if total <= 0:
        return 0.0, slope_ratio
    cand = (f >= 0.7) & (f <= 3.0)
    f0 = f[cand][np.argmax(P[cand])]
    hw = 2.0 / seg_s
    mask = np.zeros_like(band)
    for h in (1, 2, 3):
        mask |= np.abs(f - h * f0) <= hw
    purity = float(P[mask & band].sum() / total)
    return purity, slope_ratio


def extract_features(time, sig, fs, args):
    """Compute window features. Only FULL windows are produced."""
    t0, t_end = time[0], time[-1]
    dur = t_end - t0
    if dur < args.seg_s:
        return []
    n_win = int(math.floor((dur - args.seg_s) / args.hop_s + 1e-9)) + 1
    expected = args.seg_s * fs
    feats = []
    for k in range(n_win):
        lo = t0 + k * args.hop_s
        hi = lo + args.seg_s
        i0 = np.searchsorted(time, lo, "left")
        i1 = np.searchsorted(time, hi, "left")
        tw, xw = time[i0:i1], sig[i0:i1]
        f = {"seg_start_s": lo, "seg_end_s": hi, "n_samples": int(len(tw)),
             "expected": expected, "max_gap": float(args.seg_s),
             "std": float("nan"), "ptp": float("nan"), "clip_frac": 0.0,
             "ssqi": float("nan"), "purity": float("nan"),
             "slope_ratio": float("nan")}
        if len(tw) >= 3:
            f["max_gap"] = float(np.max(np.diff(np.concatenate(([lo], tw, [hi])))))
            f["std"] = float(np.std(xw))
            f["ptp"] = float(np.ptp(xw))
            if f["ptp"] > 0:
                tol = 0.01 * f["ptp"]
                f["clip_frac"] = float(np.mean((xw >= xw.max() - tol) |
                                               (xw <= xw.min() + tol)))
            f["ssqi"] = ssqi(xw)
            f["purity"], f["slope_ratio"] = window_spectral(tw, xw, lo,
                                                            args.seg_s, fs)
        feats.append(f)
    return feats


def hard_gate(f, ref_ptp, flat_std_thr, args):
    """First failing gate wins. 'ok' means the window may be classified."""
    if f["n_samples"] < args.min_coverage * f["expected"]:
        return "too_few_samples"
    if f["max_gap"] > args.max_gap_s:
        return "gap"
    if not np.isfinite(f["std"]) or f["std"] <= flat_std_thr:
        return "flatline"
    if ref_ptp is not None and f["ptp"] < args.min_amp_frac * ref_ptp:
        return "low_amplitude"
    if f["clip_frac"] > args.clip_frac:
        return "clipped"
    if np.isfinite(f["purity"]) and f["purity"] < args.min_purity:
        return "low_periodicity"
    if not np.isfinite(f["ssqi"]):
        return "nan_ssqi"
    return "ok"


def polarity_evidence(feats, args):
    cand = [f for f in feats
            if f["n_samples"] >= args.min_coverage * f["expected"]
            and f["max_gap"] <= args.max_gap_s
            and np.isfinite(f["purity"]) and f["purity"] >= args.min_purity
            and np.isfinite(f["ssqi"])]
    if len(cand) < 5:
        return None
    med_s = float(np.median([f["ssqi"] for f in cand]))
    ratios = [f["slope_ratio"] for f in cand if np.isfinite(f["slope_ratio"])]
    med_r = float(np.median(ratios)) if ratios else float("nan")
    return {"n": len(cand), "median_ssqi": med_s, "median_slope_ratio": med_r}


# --------------------------------------------------------------------------
# Core pipeline (importable for testing)
# --------------------------------------------------------------------------
def prepare_signal(time, signal):
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    ok = np.isfinite(time) & np.isfinite(signal)
    n_bad = int((~ok).sum())
    time, signal = time[ok], signal[ok]
    if np.any(np.diff(time) <= 0):
        order = np.argsort(time, kind="stable")
        time, signal = time[order], signal[order]
        keep = np.concatenate(([True], np.diff(time) > 0))
        print("WARNING: time column was not strictly increasing; sorted and "
              f"dropped {int((~keep).sum())} duplicate timestamps.")
        time, signal = time[keep], signal[keep]
    return time, signal, n_bad


def run_pipeline(time, signal, args, verbose=True):
    if 2 * args.guard >= (STATIC_EXCELLENT_CUT - STATIC_BAD_CUT):
        raise ValueError("guard must be < (excellent_cut - bad_cut)/2 = "
                         f"{(STATIC_EXCELLENT_CUT - STATIC_BAD_CUT) / 2:.3f}, "
                         "otherwise adaptation could turn bad into excellent.")

    time, signal, n_nonfinite = prepare_signal(time, signal)
    if n_nonfinite and verbose:
        print(f"Dropped {n_nonfinite} non-finite samples (these show up as "
              "coverage/gap failures).")
    fs = 1.0 / float(np.median(np.diff(time)))
    t0 = time[0]
    info = {"fs": fs, "flipped": False, "polarity": None}

    feats = extract_features(time, signal, fs, args)
    if not feats:
        return pd.DataFrame(), info

    # ---- polarity check -------------------------------------------------
    ev = polarity_evidence(feats, args)
    info["polarity"] = ev
    flip = args.polarity == "invert"
    if ev is not None:
        likely_inv = ev["median_ssqi"] < 0 and ev["median_slope_ratio"] < 1.0
        strongly_neg = ev["median_ssqi"] < STATIC_BAD_CUT
        if verbose:
            print(f"Polarity check ({ev['n']} clean periodic windows): "
                  f"median S_SQI = {ev['median_ssqi']:+.3f}, median "
                  f"upstroke/downstroke slope ratio = "
                  f"{ev['median_slope_ratio']:.2f}")
        if likely_inv:
            if args.polarity == "auto":
                flip = True
            elif verbose:
                print("  *** WARNING: waveform looks INVERTED (negative skew "
                      "and slow upstroke). Re-run with --polarity auto or "
                      "--polarity invert, or fix it in step 1. ***")
        elif strongly_neg and verbose:
            print("  *** WARNING: clean pulsatile windows have strongly "
                  "negative skewness; check signal polarity. ***")
    elif verbose:
        print("Polarity check skipped: fewer than 5 clean periodic windows.")
    if flip:
        signal = -signal
        feats = extract_features(time, signal, fs, args)
        info["flipped"] = True
        if verbose:
            print("  Signal polarity flipped (x -> -x) before classification.")

    # ---- thresholds / state --------------------------------------------
    baseline_n = max(1, int(round((args.baseline_s - args.seg_s) / args.hop_s)) + 1)
    warmup_s = args.baseline_s if args.warmup_s is None else args.warmup_s
    ok_stds = [f["std"] for f in feats
               if f["n_samples"] >= args.min_coverage * f["expected"]
               and np.isfinite(f["std"])]
    flat_std_thr = max(1e-9, args.flat_frac * (np.median(ok_stds) if ok_stds else 0.0))
    use_adaptive = args.mode in ("adaptive", "both")

    base = deque(maxlen=baseline_n)     # (end_t, ssqi, ptp) of ACCEPTED windows
    pending = deque()                   # windows not yet strictly in the past
    final_label, cand_label, streak = None, None, 0
    rows = []

    for f in feats:
        start, end = f["seg_start_s"], f["seg_end_s"]

        # only windows that ended before this one starts may enter the baseline
        while pending and pending[0][0] <= start + 1e-9:
            e = pending.popleft()
            if e[3]:
                base.append(e[:3])
        while base and base[0][0] < start - args.max_baseline_age_s:
            base.popleft()

        n_base = len(base)
        have_base = n_base >= args.min_baseline
        ref_ptp = float(np.median([b[2] for b in base])) if have_base else None
        gate = hard_gate(f, ref_ptp, flat_std_thr, args)

        in_warmup = (start - t0) < warmup_s
        exc_cut = bad_cut = med = sigma = float("nan")
        if have_base:
            exc_cut, bad_cut, med, sigma = adaptive_thresholds(
                [b[1] for b in base], args)
        use_rolling = use_adaptive and have_base and not in_warmup
        source = "rolling" if use_rolling else ("warmup" if in_warmup else "no-baseline")

        v = f["ssqi"]
        if gate != "ok":
            label_static = label_adaptive = "bad"
            accepted = False
        else:
            label_static = classify_static(v)
            label_adaptive = (classify(v, exc_cut, bad_cut)
                              if use_rolling else label_static)
            accepted = None  # set below

        label_raw = label_adaptive if use_adaptive else label_static
        if gate == "ok":
            accepted = (label_static != "bad") and (label_raw != "bad")

        # debounce (hard-gate failures are immediate)
        if final_label is None or gate != "ok":
            final_label, cand_label, streak = label_raw, None, 0
        elif label_raw == final_label:
            cand_label, streak = None, 0
        else:
            if label_raw == cand_label:
                streak += 1
            else:
                cand_label, streak = label_raw, 1
            if streak >= args.confirm_n:
                final_label, cand_label, streak = label_raw, None, 0

        pending.append((end, v, f["ptp"], bool(accepted)))
        rows.append({
            "seg_start_s": start, "seg_end_s": end, "SSQI": v,
            "periodicity": f["purity"], "ptp": f["ptp"],
            "n_samples": f["n_samples"], "gate": gate,
            "label_static": label_static, "label_raw": label_raw,
            "label_final": final_label, "adaptive_source": source,
            "thr_excellent": exc_cut if use_rolling else STATIC_EXCELLENT_CUT,
            "thr_bad": bad_cut if use_rolling else STATIC_BAD_CUT,
            "baseline_median": med if have_base else float("nan"),
            "baseline_sigma": sigma if have_base else float("nan"),
            "baseline_n": n_base, "used_for_baseline": bool(accepted),
        })

    return pd.DataFrame(rows), info


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def make_plot(out_df, args, path, channel):
    mid = (out_df["seg_start_s"] + out_df["seg_end_s"]) / 2
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3, 0.8]})
    ax = axes[0]
    ax.plot(mid, out_df["SSQI"], color="black", linewidth=0.8, alpha=0.6)
    for lab, c in LABEL_COLORS.items():
        m = out_df["label_final"] == lab
        if m.any():
            ax.scatter(mid[m], out_df["SSQI"][m], s=14, color=c,
                       label=f"final: {lab}", zorder=3)
    g = out_df["gate"] != "ok"
    if g.any():
        ax.scatter(mid[g], out_df["SSQI"][g], s=22, marker="x",
                   color="dimgray", label="hard-gate failure", zorder=4)
    ax.axhline(STATIC_EXCELLENT_CUT, color="green", ls="--", lw=1,
               label="static excellent cut (0)")
    ax.axhline(STATIC_BAD_CUT, color="red", ls="--", lw=1,
               label="static bad cut (-0.15)")
    if args.mode in ("adaptive", "both"):
        r = out_df["adaptive_source"] == "rolling"
        ax.plot(mid.where(r), out_df["thr_excellent"].where(r), color="green",
                ls=":", lw=1.4, label="adaptive excellent cut")
        ax.plot(mid.where(r), out_df["thr_bad"].where(r), color="red",
                ls=":", lw=1.4, label="adaptive bad cut")
    wu = out_df["adaptive_source"] == "warmup"
    if wu.any():
        ax.axvspan(out_df["seg_start_s"][wu].min(), out_df["seg_end_s"][wu].max(),
                   color="gray", alpha=0.12, label="static warm-up")
    ax.set_ylabel("S_SQI (skewness)")
    ax.set_title(f"S_SQI over time ({channel}, {args.seg_s:g}s windows, "
                 f"{args.hop_s:g}s hop)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(mid, out_df["periodicity"], color="tab:blue", lw=1)
    ax.axhline(args.min_purity, color="tab:blue", ls="--", lw=1,
               label=f"min periodicity ({args.min_purity:g})")
    ax.set_ylabel("periodicity")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for yi, col in enumerate(["label_final", "label_static"]):
        for lab, c in LABEL_COLORS.items():
            m = (out_df[col] == lab).values
            if m.any():
                ax.scatter(mid[m], np.full(m.sum(), yi), marker="s", s=40,
                           color=c, linewidths=0)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["final", "static"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Window centre time (s)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="Step 2: robust adaptive S_SQI classification")
    ap.add_argument("--csv", default=DEFAULT_INPUT_FILE)
    ap.add_argument("--channel", choices=["IR1 Fusion", "RED1 Fusion"], default="IR")
    ap.add_argument("--outdir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--mode", choices=["static", "adaptive", "both"], default="both",
                    help="static: final label from static cuts only. "
                         "adaptive/both: use guard-railed personal baseline "
                         "(static columns are always written for comparison)")
    # windowing
    ap.add_argument("--seg-s", type=float, default=5.0, help="window length (s)")
    ap.add_argument("--hop-s", type=float, default=1.0,
                    help="hop between windows (s); set equal to --seg-s for "
                         "non-overlapping windows")
    # adaptive baseline
    ap.add_argument("--baseline-s", type=float, default=30.0,
                    help="history length used for the personal baseline (s)")
    ap.add_argument("--warmup-s", type=float, default=None,
                    help="static-only period at the start (default: baseline-s)")
    ap.add_argument("--min-baseline", type=int, default=10,
                    help="min accepted windows needed before adapting")
    ap.add_argument("--max-baseline-age-s", type=float, default=180.0,
                    help="accepted windows older than this are discarded")
    ap.add_argument("--k-exc", type=float, default=2.0,
                    help="excellent cut = median - k_exc*sigma")
    ap.add_argument("--k-bad", type=float, default=4.0,
                    help="bad cut = median - k_bad*sigma")
    ap.add_argument("--min-sigma", type=float, default=0.03,
                    help="floor for the baseline sigma (avoids a zero-width band)")
    ap.add_argument("--guard", type=float, default=0.07,
                    help="adaptive cuts stay within static cut +/- guard")
    # hard gates
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--max-gap-s", type=float, default=0.5)
    ap.add_argument("--flat-frac", type=float, default=0.02,
                    help="flatline if window std < flat_frac * median window std")
    ap.add_argument("--min-amp-frac", type=float, default=0.2,
                    help="bad if peak-to-peak < this * recent accepted amplitude")
    ap.add_argument("--clip-frac", type=float, default=0.10)
    ap.add_argument("--min-purity", type=float, default=0.5,
                    help="min spectral periodicity (0-1); TUNE ON YOUR DATA")
    # stabilisation / polarity
    ap.add_argument("--confirm-n", type=int, default=2,
                    help="consecutive windows needed to change class")
    ap.add_argument("--polarity", choices=["as-is", "auto", "invert"],
                    default="as-is",
                    help="as-is: warn only; auto: flip if clearly inverted; "
                         "invert: always flip")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    time_col = find_time_column(df)
    fusion_col = find_column(df, f"{args.channel}1 Fusion")
    time = df[time_col].astype(float).values
    signal = df[fusion_col].astype(float).values
    print(f"Loaded {len(time)} samples from column '{fusion_col}', "
          f"t = {np.nanmin(time):.1f}s .. {np.nanmax(time):.1f}s")
    print("No extra stabilisation skip: step 1 already discarded the first "
          "60 s of raw data.\n")

    out_df, info = run_pipeline(time, signal, args)
    if out_df.empty:
        print("File shorter than one window. Nothing to classify.")
        return
    print(f"Estimated sampling rate: {info['fs']:.1f} Hz; "
          f"{len(out_df)} windows of {args.seg_s:g}s (hop {args.hop_s:g}s)")

    out_csv = os.path.join(args.outdir, "ssqi_classified.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}\n")

    print("Hard-gate outcomes:")
    print(out_df["gate"].value_counts().to_string())
    for col in ("label_static", "label_final"):
        print(f"\n{col} counts:")
        print(out_df[col].value_counts().to_string())
    print("\nAdaptive source:")
    print(out_df["adaptive_source"].value_counts().to_string())
    flips = int((out_df["label_final"] != out_df["label_final"].shift()).sum() - 1)
    flips_raw = int((out_df["label_raw"] != out_df["label_raw"].shift()).sum() - 1)
    print(f"\nLabel changes: raw={flips_raw}, after debounce={flips}")

    out_png = os.path.join(args.outdir, "ssqi_summary.png")
    make_plot(out_df, args, out_png, args.channel)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()