import serial
import time
import csv
import os
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.signal import butter, filtfilt, find_peaks, welch


# ============================================================
# USER CONFIGURATION
# ============================================================

SERIAL_PORT = "COM15"          # CHANGE THIS
BAUD_RATE = 115200

# ------------------------------------------------------------
# Effective sampling rate
#
# Change this if your ESP32 actually sends another rate.
# ------------------------------------------------------------



FS = 25.0

# ------------------------------------------------------------
# Real-time processing window
# ------------------------------------------------------------

PROCESSING_WINDOW = 10        # seconds

PROCESSING_SAMPLES = int(
    FS * PROCESSING_WINDOW
)

# ------------------------------------------------------------
# Image / metric window
# ------------------------------------------------------------

IMAGE_WINDOW = 30             # seconds

IMAGE_SAMPLES = int(
    FS * IMAGE_WINDOW
)

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = "DTWATCH_PPG_RESULTS"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

RAW_CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "ppg_processed_data.csv"
)

METRICS_CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "30_second_metrics.csv"
)


# ============================================================
# SERIAL CONNECTION
# ============================================================

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)


print()
print("================================================")
print("        DTWATCH PPG PROCESSING SYSTEM")
print("================================================")
print(f"Serial Port       : {SERIAL_PORT}")
print(f"Baud Rate         : {BAUD_RATE}")
print(f"Sampling Rate     : {FS} Hz")
print(f"Processing Window : {PROCESSING_WINDOW} seconds")
print(f"Image Window      : {IMAGE_WINDOW} seconds")
print("================================================")
print()

acquisition_start_time = time.time()


# ============================================================
# 0.5 - 5 Hz BANDPASS FILTER
# ============================================================

def bandpass_filter(
        signal,
        fs,
        lowcut=0.5,
        highcut=5.0,
        order=3):

    signal = np.asarray(signal)

    if len(signal) < 20:
        return signal.copy()

    nyquist = fs / 2.0

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="bandpass"
    )

    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# HEARTBEAT PEAK DETECTION
# ============================================================

def detect_heartbeats(
        filtered_signal,
        fs):

    if len(filtered_signal) < 10:

        return np.array(
            [],
            dtype=int
        )

    signal_std = np.std(
        filtered_signal
    )

    if signal_std <= 0:

        return np.array(
            [],
            dtype=int
        )

    # --------------------------------------------------------
    # Minimum peak separation
    #
    # 0.30 sec corresponds to approximately 200 BPM
    # --------------------------------------------------------

    minimum_distance = int(
        0.30 * fs
    )

    # Adaptive prominence
    prominence = (
        0.5 * signal_std
    )

    peaks, _ = find_peaks(
        filtered_signal,
        distance=minimum_distance,
        prominence=prominence
    )

    return peaks


# ============================================================
# IBI OUTLIER REJECTION
#
# DTWATCH paper:
# Reject IBI if it differs by >30% from the mean of
# the previous four accepted IBIs.
# ============================================================

def remove_ibi_outliers(
        peak_indices,
        fs):

    if len(peak_indices) < 2:

        return (
            np.array([]),
            np.array([]),
            []
        )

    peak_times = (
        peak_indices / fs
    )

    raw_ibi = np.diff(
        peak_times
    )

    accepted_ibi = []

    accepted_peak_times = []

    rejected_indices = []

    # --------------------------------------------------------
    # First IBI
    # --------------------------------------------------------

    accepted_ibi.append(
        raw_ibi[0]
    )

    accepted_peak_times.append(
        peak_times[1]
    )

    # --------------------------------------------------------
    # Remaining IBIs
    # --------------------------------------------------------

    for i in range(
        1,
        len(raw_ibi)
    ):

        current_ibi = raw_ibi[i]

        previous_ibi = (
            accepted_ibi[-4:]
        )

        mean_previous = np.mean(
            previous_ibi
        )

        if mean_previous <= 0:
            continue

        deviation = (
            abs(
                current_ibi -
                mean_previous
            )
            /
            mean_previous
        )

        # ----------------------------------------------------
        # 30% threshold
        # ----------------------------------------------------

        if deviation <= 0.30:

            accepted_ibi.append(
                current_ibi
            )

            accepted_peak_times.append(
                peak_times[i + 1]
            )

        else:

            rejected_indices.append(
                i + 1
            )

    return (
        np.array(accepted_ibi),
        np.array(accepted_peak_times),
        rejected_indices
    )


# ============================================================
# HEART RATE
#
# IMPORTANT:
# The internally calculated HR is obtained from IBI.
# The requested reported value is calculated HR / 2.
#
# No "/2" wording is displayed in the graph/image.
# ============================================================

def calculate_bpm(
        ibi):

    if len(ibi) == 0:

        return None

    raw_bpm = (
        60.0 / ibi
    )

    # Physiological filtering before final calculation
    valid = (
        (raw_bpm >= 40) &
        (raw_bpm <= 200)
    )

    raw_bpm = (
        raw_bpm[valid]
    )

    if len(raw_bpm) == 0:

        return None

    calculated_bpm = np.median(
        raw_bpm
    )

    # Requested output calculation
    # final_bpm = (
    #     calculated_bpm / 2.0
    # )
    final_bpm = (
            calculated_bpm
        )
    return final_bpm


def normalize_bpm(
        bpm,
        bpm_history):

    if bpm is None:

        return None

    if bpm > 100 and bpm_history:

        bpm = np.mean(
            bpm_history[-5:]
        )

    bpm = float(
        bpm
    )

    bpm_history.append(
        bpm
    )

    return bpm


# ============================================================
# BEAT-BY-BEAT BPM VALUES
#
# Used for the BPM graph
# ============================================================

def calculate_bpm_series(
        peak_indices,
        fs):

    if len(peak_indices) < 2:

        return (
            np.array([]),
            np.array([])
        )

    peak_times = (
        peak_indices / fs
    )

    ibi = np.diff(
        peak_times
    )

    bpm = (
        60.0 / ibi
    )

    # Physiological range
    valid = (
        (bpm >= 40) &
        (bpm <= 200)
    )

    bpm = bpm[valid]

    # Requested reported BPM calculation
    # bpm = (
    #     bpm / 2.0
    # )
    
    # Corresponding time is the second peak
    bpm_times = peak_times[1:]

    bpm_times = bpm_times[valid]

    return (
        bpm_times,
        bpm
    )


# ============================================================
# AC / DC
# ============================================================

def calculate_ac_dc(
        raw_signal,
        filtered_signal):

    dc = np.mean(
        raw_signal
    )

    ac = np.std(
        filtered_signal
    )

    return ac, dc


# ============================================================
# R VALUE
#
# Paper:
#
# R = (Green AC / Green DC)
#     ----------------------
#       (IR AC / IR DC)
#
# Your MAX30102:
#
# R = (RED AC / RED DC)
#     -----------------
#       (IR AC / IR DC)
#
# This is the Red/IR adaptation.
# ============================================================

def calculate_R(
        ir_raw,
        red_raw,
        ir_filtered,
        red_filtered):

    ir_ac, ir_dc = calculate_ac_dc(
        ir_raw,
        ir_filtered
    )

    red_ac, red_dc = calculate_ac_dc(
        red_raw,
        red_filtered
    )

    if (
        ir_dc <= 0 or
        red_dc <= 0 or
        ir_ac <= 0 or
        red_ac <= 0
    ):

        return (
            None,
            ir_ac,
            ir_dc,
            red_ac,
            red_dc
        )

    ir_ratio = (
        ir_ac /
        ir_dc
    )

    red_ratio = (
        red_ac /
        red_dc
    )

    R = (
        red_ratio /
        ir_ratio
    )

    return (
        R,
        ir_ac,
        ir_dc,
        red_ac,
        red_dc
    )


# ============================================================
# CALCULATE R SERIES
#
# Calculates a moving R value over the entire 30-sec segment.
# ============================================================

def calculate_R_series(
        ir_signal,
        red_signal,
        fs,
        window_seconds=4):

    window_samples = int(
        window_seconds * fs
    )

    if len(ir_signal) < window_samples:

        return (
            np.array([]),
            np.array([])
        )

    r_values = []
    r_times = []

    # Move the calculation window through the signal
    step = max(
        1,
        int(0.25 * fs)
    )

    for end in range(
        window_samples,
        len(ir_signal) + 1,
        step
    ):

        start = (
            end -
            window_samples
        )

        ir_window = (
            ir_signal[start:end]
        )

        red_window = (
            red_signal[start:end]
        )

        try:

            ir_filtered = bandpass_filter(
                ir_window,
                fs,
                0.5,
                5.0
            )

            red_filtered = bandpass_filter(
                red_window,
                fs,
                0.5,
                5.0
            )

            (
                R,
                _,
                _,
                _,
                _
            ) = calculate_R(
                ir_window,
                red_window,
                ir_filtered,
                red_filtered
            )

            if R is not None:

                r_values.append(
                    R
                )

                r_times.append(
                    end / fs
                )

        except Exception:

            pass

    return (
        np.array(r_times),
        np.array(r_values)
    )


# ============================================================
# HRV FEATURES
# ============================================================

def calculate_hrv_features(
        ibi):

    features = {

        "Mean IBI": np.nan,
        "Median IBI": np.nan,
        "SDNN": np.nan,
        "RMSSD": np.nan,
        "NN50": np.nan,
        "pNN50": np.nan,
        "Mean Abs IBI Diff": np.nan,
        "MAD IBI": np.nan,
        "Min IBI": np.nan,
        "Max IBI": np.nan,
        "Mean HR": np.nan,
        "Max HR": np.nan,
        "LF Power": np.nan,
        "HF Power": np.nan,
        "LF/HF": np.nan,
        "Total Power": np.nan,
        "Normalized LF": np.nan
    }

    if len(ibi) < 2:

        return features


    # ========================================================
    # IBI
    # ========================================================

    features["Mean IBI"] = np.mean(
        ibi
    )

    features["Median IBI"] = np.median(
        ibi
    )

    features["SDNN"] = np.std(
        ibi,
        ddof=1
    )

    features["Min IBI"] = np.min(
        ibi
    )

    features["Max IBI"] = np.max(
        ibi
    )


    # ========================================================
    # HR
    #
    # Use requested calculated value
    # ========================================================

    raw_hr = (
        60.0 / ibi
    )

    features["Mean HR"] = (
        # np.mean(raw_hr) / 2.0
        np.mean(raw_hr) 

    )

    features["Max HR"] = (
        # np.max(raw_hr) / 2.0
        np.max(raw_hr)

    )


    # ========================================================
    # IBI DIFFERENCES
    # ========================================================

    diff_ibi = np.diff(
        ibi
    )

    abs_diff = np.abs(
        diff_ibi
    )


    # ========================================================
    # NN50
    # ========================================================

    features["NN50"] = np.sum(
        abs_diff > 0.050
    )


    # ========================================================
    # pNN50
    # ========================================================

    features["pNN50"] = (
        np.sum(
            abs_diff > 0.050
        )
        /
        len(abs_diff)
        *
        100.0
    )


    # ========================================================
    # MEAN ABSOLUTE IBI DIFFERENCE
    # ========================================================

    features["Mean Abs IBI Diff"] = (
        np.mean(abs_diff)
    )


    # ========================================================
    # RMSSD
    # ========================================================

    features["RMSSD"] = np.sqrt(
        np.mean(
            diff_ibi ** 2
        )
    )


    # ========================================================
    # MAD
    # ========================================================

    median_ibi = np.median(
        ibi
    )

    features["MAD IBI"] = np.median(
        np.abs(
            ibi -
            median_ibi
        )
    )


    # ========================================================
    # FREQUENCY DOMAIN
    # ========================================================

    if len(ibi) >= 4:

        cumulative_time = np.cumsum(
            ibi
        )

        cumulative_time -= (
            cumulative_time[0]
        )

        unique_time, indices = (
            np.unique(
                cumulative_time,
                return_index=True
            )
        )

        unique_ibi = (
            ibi[indices]
        )

        if len(unique_time) >= 4:

            interpolation_fs = 4.0

            regular_time = np.arange(
                unique_time[0],
                unique_time[-1],
                1.0 / interpolation_fs
            )

            if len(regular_time) >= 8:

                interpolated_ibi = np.interp(
                    regular_time,
                    unique_time,
                    unique_ibi
                )

                interpolated_ibi -= (
                    np.mean(
                        interpolated_ibi
                    )
                )

                frequencies, power = welch(
                    interpolated_ibi,
                    fs=interpolation_fs,
                    nperseg=min(
                        256,
                        len(interpolated_ibi)
                    )
                )


                # LF
                lf_mask = (
                    (frequencies >= 0.04) &
                    (frequencies < 0.15)
                )


                # HF
                hf_mask = (
                    (frequencies >= 0.15) &
                    (frequencies <= 0.40)
                )


                # Total
                total_mask = (
                    (frequencies >= 0.0) &
                    (frequencies <= 0.40)
                )


                if np.any(lf_mask):

                    lf_power = np.trapz(
                        power[lf_mask],
                        frequencies[lf_mask]
                    )

                else:

                    lf_power = 0.0


                if np.any(hf_mask):

                    hf_power = np.trapz(
                        power[hf_mask],
                        frequencies[hf_mask]
                    )

                else:

                    hf_power = 0.0


                if np.any(total_mask):

                    total_power = np.trapz(
                        power[total_mask],
                        frequencies[total_mask]
                    )

                else:

                    total_power = 0.0


                features["LF Power"] = (
                    lf_power
                )

                features["HF Power"] = (
                    hf_power
                )

                if hf_power > 0:

                    features["LF/HF"] = (
                        lf_power /
                        hf_power
                    )

                features["Total Power"] = (
                    total_power
                )

                if total_power > 0:

                    features["Normalized LF"] = (
                        lf_power /
                        total_power
                    )


    return features


# ============================================================
# CSV FILE INITIALIZATION
# ============================================================

raw_csv = open(
    RAW_CSV_FILE,
    "w",
    newline=""
)

raw_writer = csv.writer(
    raw_csv
)

raw_writer.writerow([
    "Clock Time",
    "Elapsed_s",
    "Device Timestamp",
    "IR Raw",
    "RED Raw",
    "IR Filtered",
    "RED Filtered",
    "BPM",
    "R"
])


metrics_csv = open(
    METRICS_CSV_FILE,
    "w",
    newline=""
)

metrics_writer = csv.writer(
    metrics_csv
)


# ============================================================
# REAL-TIME PROCESSING BUFFERS
# ============================================================

processing_ir = []
processing_red = []


# ============================================================
# 30 SECOND STORAGE
# ============================================================

segment_ir = []
segment_red = []
segment_timestamp = []


# ============================================================
# CURRENT REAL-TIME METRICS
# ============================================================

current_bpm = None
current_R = None
bpm_history = []


# ============================================================
# SEGMENT NUMBER
# ============================================================

segment_number = 1


# ============================================================
# MAIN LOOP
# ============================================================

print("Waiting for ESP32 PPG data...")
print()

try:

    while True:

        # ====================================================
        # READ SERIAL
        # ====================================================

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()


        if not line:

            continue


        # ====================================================
        # STATUS MESSAGES
        # ====================================================

        if not line[0].isdigit():

            print(line)

            continue


        # ====================================================
        # EXPECTED FORMAT:
        #
        # timestamp,IR,RED
        # ====================================================

        parts = line.split(",")

        if len(parts) != 3:

            continue


        try:

            device_timestamp = int(
                parts[0]
            )

            ir = float(
                parts[1]
            )

            red = float(
                parts[2]
            )

        except ValueError:

            continue


        # ====================================================
        # REAL-TIME PROCESSING BUFFER
        # ====================================================

        processing_ir.append(
            ir
        )

        processing_red.append(
            red
        )


        if len(processing_ir) > PROCESSING_SAMPLES:

            processing_ir.pop(0)

            processing_red.pop(0)


        # ====================================================
        # 30 SECOND BUFFER
        # ====================================================

        clock_time = datetime.now()

        elapsed_time = (
            time.time() -
            acquisition_start_time
        )

        segment_timestamp.append(
            clock_time
        )

        segment_ir.append(
            ir
        )

        segment_red.append(
            red
        )


        # ====================================================
        # REAL-TIME PROCESSING
        # ====================================================

        if len(processing_ir) >= PROCESSING_SAMPLES:

            ir_window = np.array(
                processing_ir
            )

            red_window = np.array(
                processing_red
            )


            # ------------------------------------------------
            # FILTER IR
            # ------------------------------------------------

            ir_filtered = bandpass_filter(
                ir_window,
                FS,
                0.5,
                5.0
            )


            # ------------------------------------------------
            # FILTER RED
            # ------------------------------------------------

            red_filtered = bandpass_filter(
                red_window,
                FS,
                0.5,
                5.0
            )


            # ------------------------------------------------
            # PEAK DETECTION
            # ------------------------------------------------

            peaks = detect_heartbeats(
                ir_filtered,
                FS
            )


            # ------------------------------------------------
            # IBI
            # ------------------------------------------------

            (
                ibi,
                accepted_peaks,
                rejected
            ) = remove_ibi_outliers(
                peaks,
                FS
            )


            # ------------------------------------------------
            # BPM
            # ------------------------------------------------

            bpm = calculate_bpm(
                ibi
            )

            bpm = normalize_bpm(
                bpm,
                bpm_history
            )

            if bpm is not None:

                current_bpm = bpm


            # ------------------------------------------------
            # R
            # ------------------------------------------------

            (
                R,
                _,
                _,
                _,
                _
            ) = calculate_R(
                ir_window,
                red_window,
                ir_filtered,
                red_filtered
            )

            if R is not None:

                current_R = R


            # ------------------------------------------------
            # SAVE RAW PROCESSING DATA
            # ------------------------------------------------

            raw_writer.writerow([
                clock_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                elapsed_time,
                device_timestamp,
                ir,
                red,
                ir_filtered[-1],
                red_filtered[-1],
                current_bpm
                if current_bpm is not None
                else "",
                current_R
                if current_R is not None
                else ""
            ])

            raw_csv.flush()


        # ====================================================
        # 30 SECOND SEGMENT COMPLETE
        # ====================================================

        if len(segment_ir) >= IMAGE_SAMPLES:

            print()
            print(
                "================================================"
            )

            print(
                f"PROCESSING SEGMENT {segment_number}"
            )

            print(
                "================================================"
            )


            # =================================================
            # ARRAYS
            # =================================================

            ir_30 = np.array(
                segment_ir
            )

            red_30 = np.array(
                segment_red
            )


            # =================================================
            # TIME
            # =================================================

            time_axis = np.array(
                segment_timestamp
            )

            segment_start_time = time_axis[0]

            segment_end_time = time_axis[-1]


            # =================================================
            # FILTER COMPLETE 30 SEC
            # =================================================

            ir_filtered_30 = bandpass_filter(
                ir_30,
                FS,
                0.5,
                5.0
            )

            red_filtered_30 = bandpass_filter(
                red_30,
                FS,
                0.5,
                5.0
            )


            # =================================================
            # PEAK DETECTION
            # =================================================

            peaks_30 = detect_heartbeats(
                ir_filtered_30,
                FS
            )


            # =================================================
            # IBI OUTLIER REMOVAL
            # =================================================

            (
                ibi_30,
                accepted_peak_times,
                rejected_indices
            ) = remove_ibi_outliers(
                peaks_30,
                FS
            )


            # =================================================
            # FINAL BPM
            # =================================================

            final_bpm = calculate_bpm(
                ibi_30
            )


            # =================================================
            # BPM SERIES
            # =================================================

            (
                bpm_times,
                bpm_values
            ) = calculate_bpm_series(
                peaks_30,
                FS
            )

            normalized_bpm_values = []

            for bpm_value in bpm_values:

                normalized_bpm_values.append(
                    normalize_bpm(
                        bpm_value,
                        bpm_history
                    )
                )

            bpm_values = np.array(
                normalized_bpm_values
            )

            final_bpm = normalize_bpm(
                final_bpm,
                bpm_history
            )


            # =================================================
            # HRV
            # =================================================

            final_hrv = (
                calculate_hrv_features(
                    ibi_30
                )
            )


            # =================================================
            # R VALUE SERIES
            #
            # A moving R value is calculated throughout
            # the complete 30 second segment.
            # =================================================

            (
                r_times,
                r_values
            ) = calculate_R_series(
                ir_30,
                red_30,
                FS,
                window_seconds=4
            )


            # =================================================
            # FINAL R VALUE
            # =================================================

            (
                final_R,
                final_ir_ac,
                final_ir_dc,
                final_red_ac,
                final_red_dc
            ) = calculate_R(
                ir_30,
                red_30,
                ir_filtered_30,
                red_filtered_30
            )


            # =================================================
            # METRICS
            # =================================================

            detected_beats = len(
                peaks_30
            )

            accepted_beats = len(
                ibi_30
            )

            rejected_beats = len(
                rejected_indices
            )


            # =================================================
            # FORMAT FUNCTION
            # =================================================

            def fmt(value, digits=4):

                if value is None:

                    return "N/A"

                try:

                    if np.isnan(value):

                        return "N/A"

                except (TypeError, ValueError):

                    pass

                if isinstance(
                    value,
                    (float, np.floating)
                ):

                    return f"{value:.{digits}f}"

                return str(value)


            # =================================================
            # CREATE METRICS DICTIONARY
            # =================================================

            metrics = {

                "Segment":
                    segment_number,

                "Duration_sec":
                    IMAGE_WINDOW,

                "BPM":
                    final_bpm,

                "R":
                    final_R,

                "IR_AC":
                    final_ir_ac,

                "IR_DC":
                    final_ir_dc,

                "RED_AC":
                    final_red_ac,

                "RED_DC":
                    final_red_dc,

                "Detected_Beats":
                    detected_beats,

                "Accepted_Beats":
                    accepted_beats,

                "Rejected_Beats":
                    rejected_beats
            }


            metrics.update(
                final_hrv
            )


            # =================================================
            # SAVE CSV
            # =================================================

            if segment_number == 1:

                metrics_writer.writerow(
                    list(
                        metrics.keys()
                    )
                )


            metrics_writer.writerow(
                list(
                    metrics.values()
                )
            )

            metrics_csv.flush()


            # =================================================
            # CREATE FIGURE
            #
            # SIX GRAPHS
            # =================================================

            fig, axes = plt.subplots(
                3,
                2,
                figsize=(20, 15)
            )


            # =================================================
            # GRAPH 1
            # RAW IR
            # =================================================

            ax = axes[0, 0]

            ax.plot(
                time_axis,
                ir_30,
                linewidth=0.7
            )

            ax.set_title(
                "Raw IR PPG"
            )

            ax.set_xlabel(
                "Clock Time"
            )

            ax.set_ylabel(
                "IR ADC"
            )

            ax.set_xlim(
                segment_start_time,
                segment_end_time
            )

            ax.grid(
                True
            )


            # =================================================
            # GRAPH 2
            # FILTERED IR
            # =================================================

            ax = axes[0, 1]

            ax.plot(
                time_axis,
                ir_filtered_30,
                linewidth=0.8
            )


            # Accepted peaks
            if len(
                accepted_peak_times
            ) > 0:

                accepted_indices = (
                    (
                        accepted_peak_times *
                        FS
                    ).astype(int)
                )

                accepted_indices = (
                    accepted_indices[
                        accepted_indices <
                        len(
                            ir_filtered_30
                        )
                    ]
                )

                if len(
                    accepted_indices
                ) > 0:

                    accepted_times = time_axis[
                        accepted_indices
                    ]

                    ax.plot(
                        accepted_times,
                        ir_filtered_30[
                            accepted_indices
                        ],
                        "o",
                        markersize=4,
                        label="Accepted Beats"
                    )


            ax.set_title(
                "Filtered IR PPG (0.5–5 Hz)"
            )

            ax.set_xlabel(
                "Clock Time"
            )

            ax.set_ylabel(
                "Amplitude"
            )

            ax.set_xlim(
                segment_start_time,
                segment_end_time
            )

            ax.grid(
                True
            )

            ax.legend()


            # =================================================
            # GRAPH 3
            # RAW RED
            # =================================================

            ax = axes[1, 0]

            ax.plot(
                time_axis,
                red_30,
                linewidth=0.7
            )

            ax.set_title(
                "Raw RED PPG"
            )

            ax.set_xlabel(
                "Clock Time"
            )

            ax.set_ylabel(
                "RED ADC"
            )

            ax.set_xlim(
                segment_start_time,
                segment_end_time
            )

            ax.grid(
                True
            )


            # =================================================
            # GRAPH 4
            # FILTERED RED
            # =================================================

            ax = axes[1, 1]

            ax.plot(
                time_axis,
                red_filtered_30,
                linewidth=0.8
            )

            ax.set_title(
                "Filtered RED PPG (0.5–5 Hz)"
            )

            ax.set_xlabel(
                "Clock Time"
            )

            ax.set_ylabel(
                "Amplitude"
            )

            ax.set_xlim(
                segment_start_time,
                segment_end_time
            )

            ax.grid(
                True
            )


            # =================================================
            # GRAPH 5
            # R VALUE
            # =================================================

            ax = axes[2, 0]

            if len(r_times) > 0:

                r_clock_times = [
                    segment_start_time + timedelta(seconds=float(t))
                    for t in r_times
                ]

                ax.plot(
                    r_clock_times,
                    r_values,
                    linewidth=1.2
                )

            if final_R is not None:

                ax.axhline(
                    final_R,
                    linestyle="--",
                    linewidth=1.0,
                    label=f"Mean R = {final_R:.4f}"
                )

            ax.set_title(
                "R Value"
            )

            ax.set_xlabel(
                "Clock Time"
            )

            ax.set_ylabel(
                "R"
            )

            ax.set_xlim(
                segment_start_time,
                segment_end_time
            )

            ax.grid(
                True
            )

            ax.legend()


            # =================================================
            # GRAPH 6
            # BPM
            # =================================================

            ax = axes[2, 1]

            if len(
                bpm_times
            ) > 0:

                bpm_clock_times = [
                    segment_start_time + timedelta(seconds=float(t))
                    for t in bpm_times
                ]

                ax.plot(
                    bpm_clock_times,
                    bpm_values,
                    "o-",
                    markersize=4,
                    linewidth=1.0
                )

            if final_bpm is not None:

                ax.axhline(
                    final_bpm,
                    linestyle="--",
                    linewidth=1.0,
                    label=f"Mean BPM = {final_bpm:.2f}"
                )

            ax.set_title(
                "BPM"
            )

            ax.set_xlabel(
                "Clock Time"
            )

            ax.set_ylabel(
                "BPM"
            )

            ax.set_xlim(
                segment_start_time,
                segment_end_time
            )

            ax.grid(
                True
            )

            ax.legend()


            # =================================================
            # FORMAT CLOCK-TIME X AXES
            # =================================================

            for axis in axes.flat:

                axis.xaxis.set_major_formatter(
                    mdates.DateFormatter("%H:%M:%S")
                )


            # =================================================
            # OVERALL TITLE
            # =================================================

            fig.suptitle(
                f"DTWATCH PPG Processing - "
                f"30 Second Segment {segment_number}",
                fontsize=18
            )


            # =================================================
            # METRICS TEXT
            #
            # IMPORTANT:
            # There is NO "BPM/2" anywhere.
            # =================================================

            metrics_text = (
                "CALCULATED METRICS\n"
                "──────────────────────────────\n"
                f"BPM                 : "
                f"{fmt(final_bpm, 2)}\n"
                f"R Value             : "
                f"{fmt(final_R, 4)}\n"
                f"IR AC               : "
                f"{fmt(final_ir_ac, 4)}\n"
                f"IR DC               : "
                f"{fmt(final_ir_dc, 2)}\n"
                f"RED AC              : "
                f"{fmt(final_red_ac, 4)}\n"
                f"RED DC              : "
                f"{fmt(final_red_dc, 2)}\n"
                f"Detected Beats      : "
                f"{detected_beats}\n"
                f"Accepted Beats      : "
                f"{accepted_beats}\n"
                f"Rejected Beats      : "
                f"{rejected_beats}\n"
                "\n"
                "IBI / HRV\n"
                "──────────────────────────────\n"
                f"Mean IBI            : "
                f"{fmt(final_hrv.get('Mean IBI'))} s\n"
                f"Median IBI          : "
                f"{fmt(final_hrv.get('Median IBI'))} s\n"
                f"SDNN                : "
                f"{fmt(final_hrv.get('SDNN'))} s\n"
                f"RMSSD               : "
                f"{fmt(final_hrv.get('RMSSD'))} s\n"
                f"NN50                : "
                f"{fmt(final_hrv.get('NN50'), 0)}\n"
                f"pNN50               : "
                f"{fmt(final_hrv.get('pNN50'), 2)} %\n"
                f"Mean Abs IBI Diff   : "
                f"{fmt(final_hrv.get('Mean Abs IBI Diff'))} s\n"
                f"MAD IBI             : "
                f"{fmt(final_hrv.get('MAD IBI'))} s\n"
                f"Min IBI             : "
                f"{fmt(final_hrv.get('Min IBI'))} s\n"
                f"Max IBI             : "
                f"{fmt(final_hrv.get('Max IBI'))} s\n"
                f"Mean HR             : "
                f"{fmt(final_hrv.get('Mean HR'), 2)}\n"
                f"Max HR              : "
                f"{fmt(final_hrv.get('Max HR'), 2)}\n"
                "\n"
                "FREQUENCY DOMAIN\n"
                "──────────────────────────────\n"
                f"LF Power            : "
                f"{fmt(final_hrv.get('LF Power'))}\n"
                f"HF Power            : "
                f"{fmt(final_hrv.get('HF Power'))}\n"
                f"LF/HF               : "
                f"{fmt(final_hrv.get('LF/HF'))}\n"
                f"Total Power         : "
                f"{fmt(final_hrv.get('Total Power'))}\n"
                f"Normalized LF       : "
                f"{fmt(final_hrv.get('Normalized LF'))}"
            )


            # =================================================
            # ADD METRICS TO FIGURE
            # =================================================

            fig.text(
                0.99,
                0.50,
                metrics_text,
                fontsize=9,
                family="monospace",
                verticalalignment="center",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round",
                    alpha=0.9
                )
            )


            # =================================================
            # ADJUST LAYOUT
            # =================================================

            plt.tight_layout(
                rect=[
                    0.02,
                    0.02,
                    0.80,
                    0.94
                ]
            )


            # =================================================
            # SAVE IMAGE
            # =================================================

            image_file = os.path.join(
                OUTPUT_DIR,
                f"PPG_30sec_{segment_number:03d}.png"
            )

            plt.savefig(
                image_file,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(
                fig
            )


            # =================================================
            # TERMINAL OUTPUT
            # =================================================

            print()
            print(
                "--------------- RESULT ---------------"
            )

            print(
                f"BPM              : "
                f"{fmt(final_bpm, 2)}"
            )

            print(
                f"R Value          : "
                f"{fmt(final_R, 4)}"
            )

            print(
                f"Detected Beats   : "
                f"{detected_beats}"
            )

            print(
                f"Accepted Beats   : "
                f"{accepted_beats}"
            )

            print(
                f"Rejected Beats   : "
                f"{rejected_beats}"
            )

            print(
                f"SDNN             : "
                f"{fmt(final_hrv.get('SDNN'))}"
            )

            print(
                f"RMSSD            : "
                f"{fmt(final_hrv.get('RMSSD'))}"
            )

            print(
                f"pNN50            : "
                f"{fmt(final_hrv.get('pNN50'), 2)} %"
            )

            print(
                f"LF/HF            : "
                f"{fmt(final_hrv.get('LF/HF'))}"
            )

            print(
                f"Image saved      : "
                f"{image_file}"
            )

            print(
                "========================================"
            )


            # =================================================
            # RESET 30 SECOND BUFFER
            # =================================================

            segment_timestamp.clear()

            segment_ir.clear()

            segment_red.clear()

            segment_number += 1


except KeyboardInterrupt:

    print()
    print("Stopping acquisition...")


finally:

    raw_csv.close()

    metrics_csv.close()

    ser.close()

    print()
    print("Serial connection closed.")

    print(
        f"All results saved in: "
        f"{OUTPUT_DIR}"
    )