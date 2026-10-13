"""
STEP 2 (v2): Segmented S_SQI (skewness) quality classification
==============================================================
Input : fusion CSV from step1_adaptive_filter.py. DUAL PPG: columns
        Elapsed_s, IR1 Fusion, RED1 Fusion, IR2 Fusion, RED2 Fusion
        (the *_sg_window / *_sg_polyorder columns are ignored; a single-PPG
        file with 'IR Fusion' / 'RED Fusion' is treated as sensor 1).
Output: <outdir>/ssqi_classified.csv        - one row per window: per-channel
                                              S_SQI / periodicity / gate /
                                              labels, per-sensor labels,
                                              overall label, HR agreement
        <outdir>/ssqi_channel_details.csv   - long format, every diagnostic
                                              column for every channel
        <outdir>/ssqi_summary.png           - diagnostic plot

DUAL-PPG LOGIC
--------------
  * Each of the 4 channels (IR1, RED1, IR2, RED2) is classified INDEPENDENTLY
    with the full pipeline below (own gates, own personal baseline, own
    polarity check). Channels share one window grid, so rows line up.
  * Sensor label  (S1, S2): IR and RED of the same sensor combined by
    --sensor-policy  (default 'worst': both wavelengths must be good, as
    SpO2 needs; use 'ir' if you only need heart rate).
  * Overall label: sensors combined by --overall-policy (default 'best':
    two sensors give redundancy, one good sensor is enough). 'best_sensor'
    tells you which one to trust; 'sensor_rank_spread' = 2 means one sensor
    is excellent while the other is bad.
  * Cross-check: heart rates (dominant pulse frequency) of the sensors that
    passed their gates must agree within --hr-tol-bpm; otherwise
    hr_agreement = 'disagree' (information only, it does not change labels).

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
  2. QUALITY-GATED BASELINE. Only windows that pass the gates, are not "bad"
     by either rule, AND are clearly pulsatile (periodicity >=
     --baseline-min-purity) may feed the personal baseline. Bad stretches
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
and every gate threshold are heuristics. TO TUNE: look at the 'periodicity'
column over a stretch you KNOW is clean; set --min-purity a bit below its 5th
percentile and --baseline-min-purity a bit higher. (Defaults were calibrated
on synthetic PPG at 12.5 Hz, where clean windows score >= 0.9.)

USAGE
-----
    python step2_ssqi_adaptive.py --csv ppg_fusion_output.csv --outdir .
    python step2_ssqi_adaptive.py --csv f.csv --channels IR1 IR2 --sensor-policy ir
"""

import argparse
import math
import os
import re
import sys
import warnings
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
    Returns (periodicity, slope_ratio, f0_hz) for one window.
      periodicity : chance-corrected fraction of 0.5-f_hi Hz power lying within
                    +/-(2/seg_s) Hz of the dominant heart-rate peak (0.7-3 Hz)
                    and its 2nd harmonic: 1 = purely periodic, ~0 = noise.
      slope_ratio : 98th pct upstroke slope / |2nd pct downstroke slope|.
                    > 1 for a correctly oriented PPG (fast rise, slow fall).
      f0_hz       : dominant pulse frequency (60*f0 = heart rate in bpm).
    """
    nan = float("nan")
    n = int(round(seg_s * fs))
    if fs < 10 or n < 16 or len(x) < 8:
        return nan, nan, nan
    grid = lo + np.arange(n) / fs
    xu = np.interp(grid, t, x)
    xu = xu - xu.mean()

    # slope ratio; smoothing length scales with fs (none at ~12 Hz, so the
    # upstroke is not blurred away)
    k = max(1, int(round(0.04 * fs)))
    xs = np.convolve(xu, np.ones(k) / k, mode="valid")
    d = np.diff(xs)
    up, dn = np.percentile(d, 98), np.percentile(d, 2)
    slope_ratio = float(up / abs(dn)) if (up > 0 and dn < 0) else nan

    # periodicity
    nfft = int(2 ** math.ceil(math.log2(n * 4)))
    P = np.abs(np.fft.rfft(xu * np.hanning(n), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    f_hi = min(8.0, 0.45 * fs)
    band = (f >= 0.5) & (f <= f_hi)
    total = P[band].sum()
    if total <= 0:
        return 0.0, slope_ratio, nan
    cand = (f >= 0.7) & (f <= 3.0)
    f0 = float(f[cand][np.argmax(P[cand])])
    hw = 2.0 / seg_s
    mask = np.zeros_like(band)
    for h in (1, 2):
        mask |= np.abs(f - h * f0) <= hw
    mask &= band
    raw = P[mask].sum() / total
    chance = mask.sum() / band.sum()       # share of the band covered by chance
    purity = float(max(0.0, (raw - chance) / (1.0 - chance)))
    return purity, slope_ratio, f0


def extract_features(time, sig, fs, args, t0, t_end):
    """Compute window features on a grid starting at t0. Only FULL windows."""
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
             "slope_ratio": float("nan"), "f0": float("nan")}
        if len(tw) >= 3:
            f["max_gap"] = float(np.max(np.diff(np.concatenate(([lo], tw, [hi])))))
            f["std"] = float(np.std(xw))
            f["ptp"] = float(np.ptp(xw))
            if f["ptp"] > 0:
                tol = 0.01 * f["ptp"]
                f["clip_frac"] = float(np.mean((xw >= xw.max() - tol) |
                                               (xw <= xw.min() + tol)))
            f["ssqi"] = ssqi(xw)
            f["purity"], f["slope_ratio"], f["f0"] = window_spectral(
                tw, xw, lo, args.seg_s, fs)
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


def run_pipeline(time, signal, args, verbose=True, grid=None, tag=""):
    if 2 * args.guard >= (STATIC_EXCELLENT_CUT - STATIC_BAD_CUT):
        raise ValueError("guard must be < (excellent_cut - bad_cut)/2 = "
                         f"{(STATIC_EXCELLENT_CUT - STATIC_BAD_CUT) / 2:.3f}, "
                         "otherwise adaptation could turn bad into excellent.")

    pre = f"[{tag}] " if tag else ""
    time, signal, n_nonfinite = prepare_signal(time, signal)
    if n_nonfinite and verbose:
        print(f"{pre}Dropped {n_nonfinite} non-finite samples (these show up "
              "as coverage/gap failures).")
    info = {"fs": float("nan"), "flipped": False, "polarity": None}
    if len(time) < 3:
        return pd.DataFrame(), info
    fs = 1.0 / float(np.median(np.diff(time)))
    info["fs"] = fs
    # common window grid across channels (so rows line up channel-to-channel)
    t0, t_end = grid if grid is not None else (time[0], time[-1])

    feats = extract_features(time, signal, fs, args, t0, t_end)
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
            print(f"{pre}Polarity check ({ev['n']} clean periodic windows): "
                  f"median S_SQI = {ev['median_ssqi']:+.3f}, median "
                  f"upstroke/downstroke slope ratio = "
                  f"{ev['median_slope_ratio']:.2f}")
        if likely_inv:
            if args.polarity == "auto":
                flip = True
            elif verbose:
                print(f"  {pre}*** WARNING: waveform looks INVERTED (negative skew "
                      "and slow upstroke). Re-run with --polarity auto or "
                      "--polarity invert, or fix it in step 1. ***")
        elif strongly_neg and verbose:
            print(f"  {pre}*** WARNING: clean pulsatile windows have strongly "
                  "negative skewness; check signal polarity. ***")
    elif verbose:
        print(f"{pre}Polarity check skipped: fewer than 5 clean periodic windows.")
    if flip:
        signal = -signal
        feats = extract_features(time, signal, fs, args, t0, t_end)
        info["flipped"] = True
        if verbose:
            print(f"  {pre}Signal polarity flipped (x -> -x) before classification.")

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
            # the baseline may only learn from windows that are clearly
            # pulsatile (high periodicity) and not bad by either rule
            clear = (not np.isfinite(f["purity"])) or f["purity"] >= args.baseline_min_purity
            accepted = (label_static != "bad") and (label_raw != "bad") and clear

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
            "periodicity": f["purity"], "hr_bpm": 60.0 * f["f0"], "ptp": f["ptp"],
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
# Multi-channel handling (dual PPG: IR1, RED1, IR2, RED2)
# --------------------------------------------------------------------------
CH_PATTERN = re.compile(r"^(ir|red)(\d*)fusion$")
RANK = {"bad": 0, "good": 1, "excellent": 2}
INV_RANK = {0: "bad", 1: "good", 2: "excellent"}


def detect_channels(df, wanted=None):
    """Find '<IR|RED><n> Fusion' columns. 'IR Fusion' (no number) = sensor 1."""
    found = []
    for col in df.columns:
        m = CH_PATTERN.match(_normalize(col))
        if m:
            wl = m.group(1).upper()
            sid = int(m.group(2)) if m.group(2) else 1
            found.append({"name": f"{wl}{sid}", "wavelength": wl,
                          "sensor": sid, "col": col})
    names = [c["name"] for c in found]
    if len(set(names)) != len(names):
        raise KeyError(f"Duplicate channel names detected: {names}")
    found.sort(key=lambda c: (c["sensor"], 0 if c["wavelength"] == "IR" else 1))
    if wanted:
        w = {x.upper() for x in wanted}
        missing = w - set(names)
        if missing:
            raise KeyError(f"Requested channels not in file: {sorted(missing)}. "
                           f"Available: {names}")
        found = [c for c in found if c["name"] in w]
    if not found:
        raise KeyError("No '<IR|RED><n> Fusion' columns found. "
                       f"Columns: {list(df.columns)}")
    return found


def _ranks(series):
    return series.map(RANK).fillna(0).astype(int).values


def combine_channels(chan_dfs, channels, args):
    """
    channel labels -> sensor labels -> overall label, plus cross-sensor
    heart-rate agreement. All inputs are row-aligned (common window grid).
    """
    ref = chan_dfs[channels[0]["name"]]
    out = {"seg_start_s": ref["seg_start_s"].values,
           "seg_end_s": ref["seg_end_s"].values}
    sensors = sorted({c["sensor"] for c in channels})
    s_rank, s_score, s_hr = {}, {}, {}

    for sid in sensors:
        chs = [c for c in channels if c["sensor"] == sid]
        d = [chan_dfs[c["name"]] for c in chs]
        R = np.vstack([_ranks(x["label_final"]) for x in d])
        if args.sensor_policy == "best":
            r = R.max(axis=0)
        elif args.sensor_policy == "ir":
            ir_rows = [i for i, c in enumerate(chs) if c["wavelength"] == "IR"]
            r = R[ir_rows[0]] if ir_rows else R.min(axis=0)
        else:  # worst
            r = R.min(axis=0)
        ok = np.vstack([(x["gate"].values == "ok") for x in d])
        per = np.where(ok, np.vstack([x["periodicity"].values for x in d]), np.nan)
        hr = np.where(ok, np.vstack([x["hr_bpm"].values for x in d]), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_per = np.nan_to_num(np.nanmean(per, axis=0))
            hr_med = np.nanmedian(hr, axis=0)
        s_rank[sid] = r
        s_score[sid] = r + 0.5 * mean_per      # periodicity only breaks ties
        s_hr[sid] = hr_med
        out[f"S{sid}_label"] = [INV_RANK[v] for v in r]
        out[f"S{sid}_hr_bpm"] = hr_med

    RS = np.vstack([s_rank[s] for s in sensors])
    SS = np.vstack([s_score[s] for s in sensors])
    overall = RS.max(axis=0) if args.overall_policy == "best" else RS.min(axis=0)
    out["overall_label"] = [INV_RANK[v] for v in overall]
    best_idx = SS.argmax(axis=0)
    out["best_sensor"] = [f"S{sensors[b]}" if o > 0 else "none"
                          for b, o in zip(best_idx, overall)]
    out["sensor_rank_spread"] = RS.max(axis=0) - RS.min(axis=0)

    H = np.vstack([s_hr[s] for s in sensors])
    n_valid = np.isfinite(H).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.where(n_valid >= 2, np.nanmax(H, axis=0) - np.nanmin(H, axis=0),
                          np.nan)
    out["hr_bpm_spread"] = spread
    out["hr_agreement"] = np.where(
        n_valid < 2, "n/a", np.where(spread <= args.hr_tol_bpm, "agree", "disagree"))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def make_plot(chan_dfs, comb, channels, args, path):
    n_ch = len(channels)
    sensors = sorted({c["sensor"] for c in channels})
    strip = ([(c["name"], chan_dfs[c["name"]]["label_final"].values) for c in channels]
             + [(f"S{s}", comb[f"S{s}_label"].values) for s in sensors]
             + [("OVERALL", comb["overall_label"].values)])
    fig, axes = plt.subplots(
        n_ch + 2, 1, figsize=(14, 2.5 * n_ch + 2.5 + 0.3 * len(strip)), sharex=True,
        gridspec_kw={"height_ratios": [2.2] * n_ch + [1.3, 0.35 * len(strip) + 0.6]})

    mid = ((comb["seg_start_s"] + comb["seg_end_s"]) / 2).values
    for i, c in enumerate(channels):
        ax, d = axes[i], chan_dfs[c["name"]]
        v = d["SSQI"].values
        ax.plot(mid, v, color="black", lw=0.7, alpha=0.5)
        for lab, col in LABEL_COLORS.items():
            m = (d["label_final"] == lab).values
            if m.any():
                ax.scatter(mid[m], v[m], s=12, color=col, zorder=3,
                           label=f"final: {lab}" if i == 0 else None)
        g = (d["gate"] != "ok").values
        if g.any():
            ax.scatter(mid[g], v[g], s=20, marker="x", color="dimgray", zorder=4,
                       label="hard-gate failure" if i == 0 else None)
        ax.axhline(STATIC_EXCELLENT_CUT, color="green", ls="--", lw=1)
        ax.axhline(STATIC_BAD_CUT, color="red", ls="--", lw=1)
        if args.mode in ("adaptive", "both"):
            r = (d["adaptive_source"] == "rolling").values
            ax.plot(np.where(r, mid, np.nan), np.where(r, d["thr_excellent"], np.nan),
                    color="green", ls=":", lw=1.4)
            ax.plot(np.where(r, mid, np.nan), np.where(r, d["thr_bad"], np.nan),
                    color="red", ls=":", lw=1.4)
        wu = (d["adaptive_source"] == "warmup").values
        if wu.any():
            ax.axvspan(d["seg_start_s"][wu].min(), d["seg_end_s"][wu].max(),
                       color="gray", alpha=0.12)
        ax.set_ylabel(f"{c['name']}\nS_SQI")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title(f"S_SQI per channel ({args.seg_s:g}s windows, {args.hop_s:g}s hop). "
                         "Dashed = static cuts, dotted = adaptive cuts, "
                         "grey band = static warm-up")
            ax.legend(loc="upper right", fontsize=7, ncol=3)

    ax = axes[n_ch]
    for c in channels:
        ax.plot(mid, chan_dfs[c["name"]]["periodicity"], lw=1, label=c["name"])
    ax.axhline(args.min_purity, color="k", ls="--", lw=1,
               label=f"min periodicity ({args.min_purity:g})")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("periodicity")
    ax.legend(loc="lower right", fontsize=7, ncol=n_ch + 1)
    ax.grid(alpha=0.3)

    ax = axes[n_ch + 1]
    for yi, (name, labs) in enumerate(strip):
        for lab, col in LABEL_COLORS.items():
            m = labs == lab
            if m.any():
                ax.scatter(mid[m], np.full(m.sum(), yi), marker="s", s=38,
                           color=col, linewidths=0)
    ax.set_yticks(range(len(strip)))
    ax.set_yticklabels([n for n, _ in strip], fontsize=8)
    ax.set_ylim(-0.7, len(strip) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Window centre time (s)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description="Step 2 (v2, dual PPG): robust adaptive S_SQI classification")
    ap.add_argument("--csv", default=DEFAULT_INPUT_FILE)
    ap.add_argument("--channels", nargs="+", default=None,
                    help="subset of channels, e.g. --channels IR1 IR2 "
                         "(default: every '<IR|RED><n> Fusion' column)")
    ap.add_argument("--outdir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--mode", choices=["static", "adaptive", "both"], default="both",
                    help="static: final label from static cuts only. "
                         "adaptive/both: use guard-railed personal baseline "
                         "(static columns are always written for comparison)")
    # combining channels
    ap.add_argument("--sensor-policy", choices=["worst", "best", "ir"], default="worst",
                    help="how a sensor's IR and RED labels combine: worst = "
                         "both wavelengths must be good (needed for SpO2); "
                         "ir = IR only (enough for heart rate); best = either")
    ap.add_argument("--overall-policy", choices=["best", "worst"], default="best",
                    help="how sensors combine: best = redundancy (one good "
                         "sensor is enough); worst = both sensors must be good")
    ap.add_argument("--hr-tol-bpm", type=float, default=10.0,
                    help="sensors' heart rates within this many bpm = 'agree'")
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
    ap.add_argument("--baseline-min-purity", type=float, default=0.7,
                    help="a window may feed the baseline only if its periodicity "
                         "is at least this (stricter than --min-purity)")
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
                    help="as-is: warn only; auto: flip a channel if clearly "
                         "inverted; invert: always flip every channel")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    time_col = find_time_column(df)
    channels = detect_channels(df, args.channels)
    time = df[time_col].astype(float).values
    finite_t = time[np.isfinite(time)]
    grid = (float(finite_t.min()), float(finite_t.max()))
    print(f"Loaded {len(df)} rows, t = {grid[0]:.1f}s .. {grid[1]:.1f}s")
    print("Channels: " + ", ".join(f"{c['name']} ('{c['col']}')" for c in channels))
    print("No extra stabilisation skip: step 1 already discarded the first "
          "60 s of raw data.\n")

    chan_dfs = {}
    for c in channels:
        sig = df[c["col"]].astype(float).values
        out_df, info = run_pipeline(time, sig, args, grid=grid, tag=c["name"])
        if out_df.empty:
            print("File shorter than one window. Nothing to classify.")
            return
        chan_dfs[c["name"]] = out_df
        print(f"[{c['name']}] fs ~ {info['fs']:.1f} Hz, "
              f"{len(out_df)} windows ({args.seg_s:g}s, hop {args.hop_s:g}s)"
              + (", polarity flipped" if info["flipped"] else ""))
    if len({len(d) for d in chan_dfs.values()}) != 1:
        raise RuntimeError("Channels produced different window counts.")

    comb = combine_channels(chan_dfs, channels, args)

    # ---- outputs -----------------------------------------------------
    wide = comb[["seg_start_s", "seg_end_s"]].copy()
    for c in channels:
        d = chan_dfs[c["name"]]
        n = c["name"]
        wide[f"{n}_SSQI"] = d["SSQI"].values
        wide[f"{n}_periodicity"] = d["periodicity"].values
        wide[f"{n}_gate"] = d["gate"].values
        wide[f"{n}_label_static"] = d["label_static"].values
        wide[f"{n}_label_final"] = d["label_final"].values
    wide = pd.concat([wide, comb.drop(columns=["seg_start_s", "seg_end_s"])], axis=1)
    out_csv = os.path.join(args.outdir, "ssqi_classified.csv")
    wide.to_csv(out_csv, index=False)

    detail = pd.concat([d.assign(channel=n) for n, d in chan_dfs.items()],
                       ignore_index=True)
    detail = detail[["channel"] + [c for c in detail.columns if c != "channel"]]
    out_detail = os.path.join(args.outdir, "ssqi_channel_details.csv")
    detail.to_csv(out_detail, index=False)
    print(f"\nSaved: {out_csv}\nSaved: {out_detail}")

    # ---- summaries ---------------------------------------------------
    for c in channels:
        d = chan_dfs[c["name"]]
        print(f"\n--- {c['name']} ---")
        print("gate outcomes : " + ", ".join(f"{k}={v}" for k, v in
                                             d["gate"].value_counts().items()))
        print("static labels : " + ", ".join(f"{k}={v}" for k, v in
                                             d["label_static"].value_counts().items()))
        print("final labels  : " + ", ".join(f"{k}={v}" for k, v in
                                             d["label_final"].value_counts().items()))
        print("adaptive src  : " + ", ".join(f"{k}={v}" for k, v in
                                             d["adaptive_source"].value_counts().items()))
        ch_final = int((d["label_final"] != d["label_final"].shift()).sum() - 1)
        ch_raw = int((d["label_raw"] != d["label_raw"].shift()).sum() - 1)
        print(f"label changes : raw={ch_raw}, after debounce={ch_final}")
    for s in sorted({c["sensor"] for c in channels}):
        print(f"\nSensor S{s} : " + ", ".join(
            f"{k}={v}" for k, v in comb[f"S{s}_label"].value_counts().items()))
    print("OVERALL     : " + ", ".join(
        f"{k}={v}" for k, v in comb["overall_label"].value_counts().items()))
    print("HR agreement: " + ", ".join(
        f"{k}={v}" for k, v in comb["hr_agreement"].value_counts().items()))

    out_png = os.path.join(args.outdir, "ssqi_summary.png")
    make_plot(chan_dfs, comb, channels, args, out_png)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()