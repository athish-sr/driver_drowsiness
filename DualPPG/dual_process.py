"""
preprocess.py

Adaptive Butterworth -> Savitzky-Golay preprocessing
for TWO PPG sensors stored in a single CSV row.

INPUT FORMAT
------------
Timestamp,IR1,RED1,IR2,RED2

Example:
621,143931,98829,154418,114575
661,143990,98831,154424,114534
701,144037,98839,154383,114516

Timestamp is in milliseconds.

TIMELINE
--------
Window 0: 0-30 s
    Stabilization -> completely skipped

Window 1: 30-60 s
    Calibration -> used only to derive the first SG configuration

Window 2 onward:
    Processed -> written to output

For every channel, the SG configuration is independently
derived from the previous window.

CHANNELS
--------
Sensor 1:
    IR1
    RED1

Sensor 2:
    IR2
    RED2

OUTPUT
------
ppg_fusion_output.csv

Columns:
    Elapsed_s

    IR1 Fusion
    IR1_sg_window
    IR1_sg_polyorder

    RED1 Fusion
    RED1_sg_window
    RED1_sg_polyorder

    IR2 Fusion
    IR2_sg_window
    IR2_sg_polyorder

    RED2 Fusion
    RED2_sg_window
    RED2_sg_polyorder

filter_config_log.csv
    Adaptive filtering configuration for every window.
"""

import argparse
import os

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, savgol_filter


# ============================================================
# COLUMN DETECTION
# ============================================================

def normalize(name: str) -> str:
    return "".join(
        ch for ch in str(name).lower()
        if ch.isalnum()
    )


def find_column(df: pd.DataFrame, wanted: str) -> str:
    target = normalize(wanted)

    for col in df.columns:
        if normalize(col) == target:
            return col

    raise KeyError(
        f"Could not find column '{wanted}'. "
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# BUTTERWORTH BANDPASS
# ============================================================

def bandpass_ac(
    signal: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    order: int = 4
) -> np.ndarray:

    signal = np.asarray(signal, dtype=float)

    # Remove DC component
    signal = signal - np.median(signal)

    nyquist = fs / 2.0

    highcut = min(highcut, nyquist * 0.98)

    if lowcut >= highcut:
        raise ValueError(
            f"Invalid bandpass: lowcut={lowcut}, highcut={highcut}, "
            f"sampling frequency={fs}"
        )

    b, a = butter(
        order,
        [lowcut / nyquist, highcut / nyquist],
        btype="bandpass"
    )

    return filtfilt(b, a, signal)


# ============================================================
# DOMINANT FREQUENCY
# ============================================================

def estimate_dominant_freq(
    ac_segment: np.ndarray,
    fs: float,
    f_lo: float = 0.6,
    f_hi: float = 3.3
):
    """
    Estimate dominant PPG frequency between
    0.6 Hz and 3.3 Hz = 36-200 BPM.
    """

    n = len(ac_segment)

    if n < fs * 2:
        return None

    # Remove mean
    signal = ac_segment - np.mean(ac_segment)

    # Check whether signal contains useful variation
    if np.std(signal) < 1e-12:
        return None

    # Hanning window
    windowed = signal * np.hanning(n)

    # FFT
    freqs = np.fft.rfftfreq(
        n,
        d=1 / fs
    )

    magnitude = np.abs(
        np.fft.rfft(windowed)
    )

    # Physiological PPG range
    band = (
        (freqs >= f_lo) &
        (freqs <= f_hi)
    )

    if not np.any(band):
        return None

    band_freqs = freqs[band]
    band_mag = magnitude[band]

    if band_mag.size == 0:
        return None

    if np.max(band_mag) <= 0:
        return None

    dominant_frequency = band_freqs[
        np.argmax(band_mag)
    ]

    return float(dominant_frequency)


# ============================================================
# DERIVE SAVITZKY-GOLAY CONFIGURATION
# ============================================================

def derive_savgol_config(
    ac_segment: np.ndarray,
    fs: float,
    win_fraction: float = 0.30,
    default_polyorder: int = 3
):
    """
    Derive SG window and polynomial order from
    the dominant pulse frequency.
    """

    f_dom = estimate_dominant_freq(
        ac_segment,
        fs
    )

    if f_dom is None or f_dom <= 0:
        return None, None, None

    # Pulse period in samples
    period_samples = fs / f_dom

    # 30% of heartbeat period
    sg_window = int(
        round(
            win_fraction * period_samples
        )
    )

    # Must be odd
    if sg_window % 2 == 0:
        sg_window += 1

    # Minimum window
    sg_window = max(
        sg_window,
        5
    )

    # Window cannot exceed segment length
    max_window = (
        len(ac_segment) - 1
        if len(ac_segment) % 2 == 0
        else len(ac_segment)
    )

    sg_window = min(
        sg_window,
        max_window
    )

    # Polynomial order
    sg_poly = min(
        default_polyorder,
        sg_window - 2
    )

    sg_poly = max(
        sg_poly,
        1
    )

    return (
        sg_window,
        sg_poly,
        f_dom
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Adaptive preprocessing for two PPG sensors"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV file"
    )

    parser.add_argument(
        "--outdir",
        default="ppg_results_sg",
        help="Output directory"
    )

    parser.add_argument(
        "--butter-low",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--butter-high",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--butter-order",
        type=int,
        default=4
    )

    parser.add_argument(
        "--stab-s",
        type=float,
        default=30.0,
        help="Initial stabilization period"
    )

    parser.add_argument(
        "--calib-s",
        type=float,
        default=30.0,
        help="Calibration period"
    )

    parser.add_argument(
        "--proc-window-s",
        type=float,
        default=30.0,
        help="Processing window length"
    )

    parser.add_argument(
        "--sg-win-fraction",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--sg-polyorder",
        type=int,
        default=3
    )

    args = parser.parse_args()

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    # ========================================================
    # LOAD CSV
    # ========================================================

    df = pd.read_csv(args.csv)

    df.columns = df.columns.str.strip()

    if any(normalize(column) == normalize("Elapsed_s") for column in df.columns):
        timestamp_col = find_column(df, "Elapsed_s")
        timestamp_scale = 1.0
    else:
        timestamp_col = find_column(df, "Device Timestamp")
        timestamp_scale = 0.001

    required_channels = [
        "IR1",
        "RED1",
        "IR2",
        "RED2"
    ]

    channel_columns = {}

    for channel in required_channels:
        channel_columns[channel] = find_column(
            df,
            channel
        )

    # ========================================================
    # TIMESTAMP -> ELAPSED SECONDS
    # ========================================================

    timestamp_values = (
        pd.to_numeric(
            df[timestamp_col],
            errors="coerce"
        )
        .values
    )

    # Remove invalid timestamps
    valid = np.isfinite(timestamp_values)

    df = df.loc[valid].copy()

    timestamp_values = timestamp_values[valid]

    # Sort by timestamp
    sort_idx = np.argsort(timestamp_values)

    timestamp_values = timestamp_values[sort_idx]

    df = df.iloc[sort_idx].reset_index(
        drop=True
    )

    # Convert milliseconds -> seconds
    elapsed_s = (
        timestamp_values - timestamp_values[0]
    ) * timestamp_scale

    # ========================================================
    # ESTIMATE SAMPLING FREQUENCY
    # ========================================================

    dt = np.diff(elapsed_s)

    dt = dt[
        np.isfinite(dt) &
        (dt > 0)
    ]

    if len(dt) == 0:
        raise ValueError(
            "Could not determine sampling interval."
        )

    median_dt = np.median(dt)

    fs = 1.0 / median_dt

    duration = (
        elapsed_s[-1]
        - elapsed_s[0]
    )

    print()
    print("=" * 70)
    print("TWO-SENSOR PPG PREPROCESSING")
    print("=" * 70)
    print(
        f"Samples          : {len(elapsed_s)}"
    )
    print(
        f"Sampling rate    : {fs:.2f} Hz"
    )
    print(
        f"Duration         : {duration:.2f} s"
    )
    print(
        f"Stabilization    : {args.stab_s:.0f} s"
    )
    print(
        f"Calibration      : {args.calib_s:.0f} s"
    )
    print(
        f"Processing window: {args.proc_window_s:.0f} s"
    )
    print("=" * 70)

    # ========================================================
    # LOAD RAW CHANNELS
    # ========================================================

    raw = {}

    for channel in required_channels:

        raw[channel] = pd.to_numeric(
            df[channel_columns[channel]],
            errors="coerce"
        ).values.astype(float)

    # ========================================================
    # WINDOW BOUNDARIES
    # ========================================================

    t0 = elapsed_s[0]

    boundaries = [
        t0,
        t0 + args.stab_s,
        t0 + args.stab_s + args.calib_s
    ]

    while boundaries[-1] < elapsed_s[-1]:

        boundaries.append(
            boundaries[-1] + args.proc_window_s
        )

    # ========================================================
    # OUTPUT STRUCTURE
    # ========================================================

    out_rows = {
        "Elapsed_s": []
    }

    for channel in required_channels:

        out_rows[f"{channel} Fusion"] = []
        out_rows[f"{channel}_sg_window"] = []
        out_rows[f"{channel}_sg_polyorder"] = []

    config_log = []

    # ========================================================
    # CALIBRATION WINDOW
    # ========================================================

    calibration_start = boundaries[1]
    calibration_end = boundaries[2]

    calib_idx = (
        (elapsed_s >= calibration_start) &
        (elapsed_s < calibration_end)
    )

    current_config = {}

    print()
    print("CALIBRATION")
    print("-" * 70)

    for channel in required_channels:

        segment = raw[channel][calib_idx]

        if len(segment) < fs * 2:

            print(
                f"{channel}: insufficient calibration data"
            )

            current_config[channel] = (
                11,
                3
            )

            continue

        # Butterworth
        ac = bandpass_ac(
            segment,
            fs,
            args.butter_low,
            args.butter_high,
            args.butter_order
        )

        # Adaptive SG
        sg_window, sg_poly, f_dom = (
            derive_savgol_config(
                ac,
                fs,
                args.sg_win_fraction,
                args.sg_polyorder
            )
        )

        # Fallback
        if sg_window is None:

            sg_window = 11
            sg_poly = 3
            f_dom = np.nan

        current_config[channel] = (
            sg_window,
            sg_poly
        )

        bpm = (
            f_dom * 60
            if np.isfinite(f_dom)
            else np.nan
        )

        print(
            f"{channel}: "
            f"Pulse = {bpm:.2f} BPM, "
            f"SG = ({sg_window}, {sg_poly})"
        )

        config_log.append({

            "window_start_s":
                calibration_start,

            "window_end_s":
                calibration_end,

            "role":
                "calibration",

            "channel":
                channel,

            "estimated_pulse_hz":
                f_dom,

            "estimated_bpm":
                bpm,

            "sg_window_derived_for_next_window":
                sg_window,

            "sg_polyorder_derived_for_next_window":
                sg_poly
        })

    # ========================================================
    # PROCESS WINDOWS
    # ========================================================

    n_processed = 0

    print()
    print("PROCESSING")
    print("-" * 70)

    # Window index 2 onward
    for i in range(
        2,
        len(boundaries) - 1
    ):

        window_start = boundaries[i]

        window_end = boundaries[i + 1]

        # Last window can end at actual recording end
        if window_end > elapsed_s[-1]:
            window_end = elapsed_s[-1] + 1e-9

        idx = (
            (elapsed_s >= window_start) &
            (elapsed_s < window_end)
        )

        if np.sum(idx) < fs * 2:
            break

        segment_time = elapsed_s[idx]

        # Store time ONCE
        out_rows["Elapsed_s"].extend(
            segment_time.tolist()
        )

        print(
            f"\nWindow {i}: "
            f"{window_start:.2f} - "
            f"{window_end:.2f} s"
        )

        # ====================================================
        # PROCESS EACH CHANNEL
        # ====================================================

        for channel in required_channels:

            segment_raw = raw[channel][idx]

            # Butterworth
            ac = bandpass_ac(
                segment_raw,
                fs,
                args.butter_low,
                args.butter_high,
                args.butter_order
            )

            sg_window, sg_poly = (
                current_config[channel]
            )

            # Ensure valid SG window
            max_window = (
                len(ac) - 1
                if len(ac) % 2 == 0
                else len(ac)
            )

            sg_window = min(
                sg_window,
                max_window
            )

            if sg_window < 5:
                sg_window = 5

            if sg_window % 2 == 0:
                sg_window -= 1

            sg_poly = min(
                sg_poly,
                sg_window - 2
            )

            sg_poly = max(
                sg_poly,
                1
            )

            # =================================================
            # SAVITZKY-GOLAY
            # =================================================

            fused = savgol_filter(
                ac,
                sg_window,
                sg_poly
            )

            # Store output
            out_rows[
                f"{channel} Fusion"
            ].extend(
                fused.tolist()
            )

            out_rows[
                f"{channel}_sg_window"
            ].extend(
                [sg_window] * len(fused)
            )

            out_rows[
                f"{channel}_sg_polyorder"
            ].extend(
                [sg_poly] * len(fused)
            )

            # =================================================
            # DERIVE CONFIG FOR NEXT WINDOW
            # =================================================

            (
                new_window,
                new_poly,
                f_dom
            ) = derive_savgol_config(
                ac,
                fs,
                args.sg_win_fraction,
                args.sg_polyorder
            )

            if new_window is not None:

                current_config[channel] = (
                    new_window,
                    new_poly
                )

            bpm = (
                f_dom * 60
                if f_dom is not None
                else np.nan
            )

            # Log
            config_log.append({

                "window_start_s":
                    window_start,

                "window_end_s":
                    window_end,

                "role":
                    "processed",

                "channel":
                    channel,

                "estimated_pulse_hz":
                    f_dom,

                "estimated_bpm":
                    bpm,

                "sg_window_used":
                    sg_window,

                "sg_polyorder_used":
                    sg_poly,

                "sg_window_derived_for_next_window":
                    current_config[channel][0],

                "sg_polyorder_derived_for_next_window":
                    current_config[channel][1]
            })

            print(
                f"  {channel}: "
                f"used SG=({sg_window},{sg_poly}), "
                f"next SG="
                f"({current_config[channel][0]},"
                f"{current_config[channel][1]})"
            )

        n_processed += 1

    # ========================================================
    # CHECK OUTPUT
    # ========================================================

    if n_processed == 0:

        print()
        print(
            "No processed windows were produced."
        )
        print(
            "At least 60 seconds + one processing "
            "window are required."
        )

        return

    # ========================================================
    # CREATE OUTPUT DATAFRAME
    # ========================================================

    out_df = pd.DataFrame(
        out_rows
    )

    # Make sure columns are ordered
    ordered_columns = [
        "Elapsed_s",

        "IR1 Fusion",
        "IR1_sg_window",
        "IR1_sg_polyorder",

        "RED1 Fusion",
        "RED1_sg_window",
        "RED1_sg_polyorder",

        "IR2 Fusion",
        "IR2_sg_window",
        "IR2_sg_polyorder",

        "RED2 Fusion",
        "RED2_sg_window",
        "RED2_sg_polyorder"
    ]

    out_df = out_df[
        ordered_columns
    ]

    # ========================================================
    # SAVE FUSION OUTPUT
    # ========================================================

    output_csv = os.path.join(
        args.outdir,
        "ppg_fusion_output.csv"
    )

    out_df.to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # SAVE CONFIGURATION LOG
    # ========================================================

    log_df = pd.DataFrame(
        config_log
    )

    log_csv = os.path.join(
        args.outdir,
        "filter_config_log.csv"
    )

    log_df.to_csv(
        log_csv,
        index=False
    )

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Processed windows : {n_processed}"
    )

    print(
        f"Output rows       : {len(out_df)}"
    )

    print(
        f"Output start      : "
        f"{out_df['Elapsed_s'].min():.2f} s"
    )

    print(
        f"Output end        : "
        f"{out_df['Elapsed_s'].max():.2f} s"
    )

    print()
    print(
        f"Fusion output     : {output_csv}"
    )

    print(
        f"Configuration log : {log_csv}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()