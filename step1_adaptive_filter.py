"""
STEP 1: Adaptive Butterworth -> Savitzky-Golay filtering of raw PPG
=====================================================================
Builds on savGol_filter.py, with two changes you asked for:

  1. Butterworth band-pass fixed at 0.5-5 Hz (was 0.5-8 Hz), order 4.
  2. The Savitzky-Golay (window length, polyorder) pair is no longer a
     fixed constant. It is *re-derived every 30 s* from the PREVIOUS
     30-second window's own pulse rate, then applied to the CURRENT
     window. This lets the smoothing adapt to each person's heart rate
     instead of using one-size-fits-all SG_WINDOW/SG_POLYORDER.

TIMELINE (matches what you described)
--------------------------------------
  Window 0   [0s   -  30s]  STABILIZATION  -> completely skipped.
  Window 1   [30s  -  60s]  CALIBRATION    -> used ONLY to derive the
                                              first SG (window, polyorder)
                                              pair. Not written to output.
  Window 2   [60s  -  90s]  PROCESSED      -> filtered using the config
                                              derived from window 1,
                                              and written to output.
                                              Its own pulse rate is then
                                              used to derive the config
                                              for window 3.
  Window 3   [90s  - 120s]  PROCESSED      -> filtered using config
                                              derived from window 2. ...
  ... and so on: config for window i is always derived from window i-1.

So, exactly as you specified: the first 60 s of the recording never
appear in the output file (0-30s = stabilization, 30-60s = calibration
only).

HOW THE SG (WINDOW, POLYORDER) PAIR IS DERIVED FROM A WINDOW
--------------------------------------------------------------
The paper does not specify this (SavGol tuning is outside its scope) --
this is a heuristic I designed for you, documented here so you can
change it:

  1. Take the window's band-passed AC signal.
  2. FFT it and find the dominant frequency in the physiological PPG
     band (0.6-3.3 Hz, i.e. 36-200 BPM) -> this is the pulse rate.
  3. Convert to a pulse period in samples: period = fs / f_dominant.
  4. SG window length = round(SG_WIN_FRACTION * period), forced odd,
     and clamped to >= 5. Default SG_WIN_FRACTION = 0.30 (i.e. the
     smoothing window spans ~30% of one heartbeat -- short enough to
     preserve the systolic/dicrotic shape, long enough to smooth noise).
  5. SG polyorder = min(default 3, window_length - 2), so it's always
     a valid combination.

If a window's dominant frequency can't be estimated reliably (e.g. too
noisy/flat), the previous valid config is reused and this is logged.

OUTPUT
------
  <outdir>/ppg_fusion_output.csv   - Elapsed_s, IR Fusion, RED Fusion,
                                      plus the SG (window, polyorder)
                                      actually used for each row, for
                                      full traceability.
  <outdir>/filter_config_log.csv   - one row per processed window: time
                                      range, estimated pulse rate, and
                                      the SG config derived FROM it (to
                                      be applied to the NEXT window).

USAGE
-----
    python step1_adaptive_filter.py --csv raw_ppg.csv --outdir .
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter


DEFAULT_DATA_DIR = "PPG_DATA_COLLECTION"
DEFAULT_INPUT_FILE = os.path.join(DEFAULT_DATA_DIR, "ppg_data.csv")


# --------------------------------------------------------------------------
# Column auto-detection (robust to naming variants)
# --------------------------------------------------------------------------
def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_column(df: pd.DataFrame, wanted: str) -> str:
    target = _normalize(wanted)
    for col in df.columns:
        if _normalize(col) == target:
            return col
    raise KeyError(f"Could not find a column matching '{wanted}'. "
                    f"Available columns: {list(df.columns)}")


def find_signal_column(df: pd.DataFrame, channel: str) -> str:
    """Find a raw PPG column written by ppg_data_collection.py."""
    for wanted in (f"{channel} Raw", channel, channel.lower()):
        try:
            return find_column(df, wanted)
        except KeyError:
            continue
    raise KeyError(f"Could not find a raw {channel} column. "
                   f"Available columns: {list(df.columns)}")


def find_time_column(df: pd.DataFrame) -> str:
    for c in ["Elapsed_s", "Elapsed s", "Time_s", "Time (s)"]:
        for col in df.columns:
            if _normalize(col) == _normalize(c):
                return col
    raise KeyError(f"Could not find a time column. Available: {list(df.columns)}")


# --------------------------------------------------------------------------
# Butterworth band-pass (0.5-5 Hz as requested)
# --------------------------------------------------------------------------
def bandpass_ac(signal: np.ndarray, fs: float, lowcut: float, highcut: float,
                 order: int = 4) -> np.ndarray:
    signal = signal - np.median(signal)
    nyq = fs / 2
    highcut = min(highcut, nyq * 0.98)
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    return filtfilt(b, a, signal)


# --------------------------------------------------------------------------
# Adaptive SavGol config: estimate pulse rate -> (window, polyorder)
# --------------------------------------------------------------------------
def estimate_dominant_freq(ac_segment: np.ndarray, fs: float,
                            f_lo: float = 0.6, f_hi: float = 3.3):
    """Dominant frequency (Hz) in the physiological pulse band (36-200 BPM)."""
    n = len(ac_segment)
    if n < fs * 2:  # need at least ~2s to get a meaningful FFT bin
        return None
    windowed = ac_segment * np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(windowed))
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return None
    band_freqs, band_mag = freqs[band], mag[band]
    if band_mag.max() <= 0:
        return None
    return float(band_freqs[np.argmax(band_mag)])


def derive_savgol_config(ac_segment: np.ndarray, fs: float,
                          win_fraction: float = 0.30, default_polyorder: int = 3):
    """
    Returns (sg_window, sg_polyorder, dominant_freq_hz) derived from this
    segment's own pulse rate. Returns (None, None, None) if it could not
    be estimated (segment too short/flat/noisy).
    """
    f_dom = estimate_dominant_freq(ac_segment, fs)
    if f_dom is None or f_dom <= 0:
        return None, None, None
    period_samples = fs / f_dom
    sg_window = int(round(win_fraction * period_samples))
    if sg_window % 2 == 0:
        sg_window += 1
    sg_window = max(sg_window, 5)
    sg_window = min(sg_window, len(ac_segment) - 1 if len(ac_segment) % 2 == 0
                     else len(ac_segment))
    sg_poly = min(default_polyorder, sg_window - 2)
    sg_poly = max(sg_poly, 1)
    return sg_window, sg_poly, f_dom


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 1: adaptive PPG filtering")
    ap.add_argument("--csv", default=DEFAULT_INPUT_FILE,
                    help=f"Raw collector CSV (default: {DEFAULT_INPUT_FILE})")
    ap.add_argument("--channel", choices=["both", "IR", "RED"], default="both")
    ap.add_argument("--butter-low", type=float, default=0.5)
    ap.add_argument("--butter-high", type=float, default=5.0)
    ap.add_argument("--butter-order", type=int, default=4)
    ap.add_argument("--stab-s", type=float, default=30.0,
                     help="Window 0 length: stabilization, fully skipped")
    ap.add_argument("--calib-s", type=float, default=30.0,
                     help="Window 1 length: used only to derive the first SG config")
    ap.add_argument("--proc-window-s", type=float, default=30.0,
                     help="Length of each subsequent processed window")
    ap.add_argument("--sg-win-fraction", type=float, default=0.30)
    ap.add_argument("--sg-polyorder", type=int, default=3)
    ap.add_argument("--outdir", default=DEFAULT_DATA_DIR,
                    help=f"Output directory (default: {DEFAULT_DATA_DIR})")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    time_col = find_time_column(df)
    ir_col = find_signal_column(df, "IR")
    red_col = find_signal_column(df, "RED")

    time = df[time_col].astype(float).values
    fs = 1 / np.median(np.diff(time))
    ir_raw = df[ir_col].astype(float).values
    red_raw = df[red_col].astype(float).values
    t0 = time[0]
    duration = time[-1] - t0

    print(f"Loaded {len(time)} samples, fs = {fs:.2f} Hz, duration = {duration:.1f} s")

    min_needed = args.stab_s + args.calib_s + args.proc_window_s
    if duration < min_needed:
        print(f"\n*** WARNING: recording is only {duration:.1f}s long, but the "
              f"pipeline needs at least {min_needed:.0f}s "
              f"({args.stab_s:.0f}s stabilization + {args.calib_s:.0f}s "
              f"calibration + {args.proc_window_s:.0f}s of processed output) "
              f"to emit even ONE output window. No rows will be produced. ***\n")

    # Build window boundaries in time (not sample count), so it's robust to
    # any small gaps/jitter in the timestamps.
    boundaries = [t0, t0 + args.stab_s, t0 + args.stab_s + args.calib_s]
    while boundaries[-1] < time[-1]:
        boundaries.append(boundaries[-1] + args.proc_window_s)

    def window_slice(lo, hi):
        idx = (time >= lo) & (time < hi)
        return idx

    channels = ["IR", "RED"] if args.channel == "both" else [args.channel]
    raw_by_channel = {"IR": ir_raw, "RED": red_raw}

    out_rows = {"Elapsed_s": []}
    for ch in channels:
        out_rows[f"{ch} Fusion"] = []
        out_rows[f"{ch}_sg_window"] = []
        out_rows[f"{ch}_sg_polyorder"] = []
    config_log = []

    # window 0 = stabilization -> skip entirely (indices 0)
    # window 1 = calibration -> derive first config only
    calib_idx = window_slice(boundaries[1], boundaries[2])
    current_config = {}
    for ch in channels:
        seg = raw_by_channel[ch][calib_idx]
        if len(seg) == 0:
            print(f"*** Not enough data to calibrate {ch} channel. ***")
            return
        ac = bandpass_ac(seg, fs, args.butter_low, args.butter_high, args.butter_order)
        sg_win, sg_poly, f_dom = derive_savgol_config(
            ac, fs, args.sg_win_fraction, args.sg_polyorder)
        if sg_win is None:
            # fall back to the original script's fixed default
            sg_win, sg_poly = 11, 3
            f_dom = float("nan")
        current_config[ch] = (sg_win, sg_poly)
        config_log.append({
            "window_start_s": boundaries[1], "window_end_s": boundaries[2],
            "role": "calibration (window 1, not output)",
            "channel": ch, "estimated_pulse_hz": f_dom,
            "estimated_bpm": f_dom * 60 if f_dom == f_dom else float("nan"),
            "sg_window_derived_for_NEXT_window": sg_win,
            "sg_polyorder_derived_for_NEXT_window": sg_poly,
        })

    # windows 2..N = processed
    n_processed = 0
    for i in range(2, len(boundaries) - 1 + 1):
        lo = boundaries[i]
        hi = boundaries[i + 1] if i + 1 < len(boundaries) else time[-1] + 1e-9
        idx = window_slice(lo, hi)
        if idx.sum() < fs * 2:  # too short a tail window to be useful
            break

        seg_time = time[idx]
        for ch in channels:
            seg_raw = raw_by_channel[ch][idx]
            ac = bandpass_ac(seg_raw, fs, args.butter_low, args.butter_high,
                              args.butter_order)
            sg_win, sg_poly = current_config[ch]
            sg_win = min(sg_win, len(ac) - 1 if len(ac) % 2 == 0 else len(ac))
            sg_poly = min(sg_poly, sg_win - 2) if sg_win > 2 else 1
            fused = savgol_filter(ac, sg_win, sg_poly)

            out_rows[f"{ch} Fusion"].extend(fused.tolist())
            out_rows[f"{ch}_sg_window"].extend([sg_win] * len(fused))
            out_rows[f"{ch}_sg_polyorder"].extend([sg_poly] * len(fused))

            # derive config for the NEXT window from THIS window's own data
            new_win, new_poly, f_dom = derive_savgol_config(
                ac, fs, args.sg_win_fraction, args.sg_polyorder)
            if new_win is not None:
                current_config[ch] = (new_win, new_poly)
            else:
                f_dom = float("nan")  # keep previous config, just log NaN

            config_log.append({
                "window_start_s": lo, "window_end_s": hi,
                "role": "processed & written to output",
                "channel": ch, "estimated_pulse_hz": f_dom,
                "estimated_bpm": f_dom * 60 if f_dom == f_dom else float("nan"),
                "sg_window_derived_for_NEXT_window": current_config[ch][0],
                "sg_polyorder_derived_for_NEXT_window": current_config[ch][1],
            })

        out_rows["Elapsed_s"].extend(seg_time.tolist())
        # Elapsed_s is only appended once per window loop-iteration above,
        # but with 2 channels we'd double it -- fix by only doing it once:
        n_processed += 1

    if n_processed == 0:
        print("No windows were processed (recording too short). Exiting.")
        return

    out_df = pd.DataFrame(out_rows)
    out_csv = os.path.join(args.outdir, "ppg_fusion_output.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Processed {n_processed} window(s) of {args.proc_window_s:.0f}s each.")
    print(f"Saved fusion output ({len(out_df)} rows, starting at "
          f"t={out_df['Elapsed_s'].min():.1f}s) to: {out_csv}")

    log_df = pd.DataFrame(config_log)
    out_log = os.path.join(args.outdir, "filter_config_log.csv")
    log_df.to_csv(out_log, index=False)
    print(f"Saved adaptive SG config log to: {out_log}")


if __name__ == "__main__":
    main()
