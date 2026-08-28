import math
import time
from datetime import datetime
from pathlib import Path

import serial
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

PORT = "COM9"
BAUDRATE = 115200

OUTPUT_FILE = "ADS1292R_ECG.csv"
INTERVAL_SECONDS = 5
IMAGE_OUTPUT_DIR = Path("interval_5s_images")


# ============================================================
# OPEN SERIAL PORT
# ============================================================

print("=" * 60)
print("ADS1292R ECG DATA LOGGER")
print("=" * 60)

print()
print(f"Opening {PORT}...")


try:

    ser = serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        timeout=2
    )

except serial.SerialException as e:

    print()
    print("ERROR: Could not open serial port.")
    print(e)

    print()
    print("Make sure:")
    print("1. Arduino Serial Monitor is CLOSED")
    print("2. Correct COM port is selected")
    print("3. ESP8266 is connected")

    raise SystemExit


print("Port opened successfully.")


# ============================================================
# WAIT FOR ESP8266 RESET
# ============================================================

time.sleep(2)


# Remove old serial data
ser.reset_input_buffer()


print()
print("Receiving ADS1292R data...")
print("Press Ctrl+C to stop.")
print()


# ============================================================
# DATA ARRAYS
# ============================================================

timestamps = []

clock_times = []

raw_ecg = []

filtered_ecg = []

bpm_data = []


# ============================================================
# START TIMER
# ============================================================

start_time = time.time()


# ============================================================
# RECEIVE DATA
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # Read one line
        # ----------------------------------------------------

        line = ser.readline()


        # ----------------------------------------------------
        # Serial timeout
        # ----------------------------------------------------

        if not line:

            print(
                "\rWaiting for data...",
                end=""
            )

            continue


        # ----------------------------------------------------
        # Decode serial data
        # ----------------------------------------------------

        line = line.decode(
            "utf-8",
            errors="ignore"
        ).strip()


        # ----------------------------------------------------
        # Ignore empty lines
        # ----------------------------------------------------

        if not line:

            continue


        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        print(
            f"RX: {line}"
        )


        # ----------------------------------------------------
        # Ignore Arduino startup messages
        # ----------------------------------------------------

        if line.startswith("RawECG"):

            continue


        if line.startswith("="):

            continue


        if line.startswith("ADS1292R"):

            continue


        if line.startswith("SPI"):

            continue


        if line.startswith("SCK"):

            continue


        if line.startswith("MISO"):

            continue


        if line.startswith("MOSI"):

            continue


        if line.startswith("Control"):

            continue


        if line.startswith("CS"):

            continue


        if line.startswith("DRDY"):

            continue


        if line.startswith("START"):

            continue


        if line.startswith("PWDN"):

            continue


        if line.startswith("Initializing"):

            continue


        if line.startswith("ADS1292R initialized"):

            continue


        if line.startswith("Starting"):

            continue


        # ----------------------------------------------------
        # Split CSV
        #
        # Expected:
        #
        # RawECG,FilteredECG,BPM
        #
        # Example:
        #
        # 1250,43,76
        # ----------------------------------------------------

        parts = line.split(",")


        # Must contain exactly 3 values

        if len(parts) != 3:

            continue


        # ----------------------------------------------------
        # Convert values
        # ----------------------------------------------------

        try:

            raw = float(parts[0])

            filtered = float(parts[1])

            bpm = float(parts[2])

        except ValueError:

            continue


        # ----------------------------------------------------
        # Generate timestamp
        # ----------------------------------------------------

        timestamp = (
            time.time()
            -
            start_time
        )

        clock_time = datetime.now()


        # ----------------------------------------------------
        # Store data
        # ----------------------------------------------------

        timestamps.append(timestamp)

        clock_times.append(clock_time)

        raw_ecg.append(raw)

        filtered_ecg.append(filtered)

        bpm_data.append(bpm)


        # ----------------------------------------------------
        # Live display
        # ----------------------------------------------------

        print(
            f"Time: {clock_time.strftime('%H:%M:%S.%f')[:-3]} | "
            f"Elapsed: {timestamp:8.2f} s | "
            f"Raw: {raw:8.0f} | "
            f"Filtered: {filtered:8.0f} | "
            f"BPM: {bpm:6.0f}"
        )


except KeyboardInterrupt:

    print()
    print()
    print("Stopping data acquisition...")


finally:

    ser.close()

    print("Serial port closed.")


# ============================================================
# CHECK WHETHER DATA WAS RECEIVED
# ============================================================

if len(timestamps) == 0:

    print()
    print("=" * 60)
    print("NO VALID DATA RECEIVED")
    print("=" * 60)

    print()
    print("Check:")
    print("1. ADS1292R wiring")
    print("2. DRDY connection")
    print("3. ESP8266 COM port")
    print("4. Arduino code")
    print("5. ADS1292R initialization")

    raise SystemExit


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({

    "Time": clock_times,

    "Time_s": timestamps,

    "ADS1292R_Raw": raw_ecg,

    "ADS1292R_Filtered": filtered_ecg,

    "BPM": bpm_data

})


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE ALL 5-SECOND IMAGES
# ============================================================

def save_interval_images(df: pd.DataFrame, interval_seconds: int = INTERVAL_SECONDS):
    """Save ECG charts for each full or partial 5-second interval."""
    IMAGE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    if df.empty:
        return 0

    min_time = df["Time_s"].min()
    max_time = df["Time_s"].max()
    start_bound = math.floor(min_time / interval_seconds) * interval_seconds
    end_bound = math.ceil(max_time / interval_seconds) * interval_seconds
    saved_count = 0

    for interval_start in range(int(start_bound), int(end_bound), interval_seconds):
        interval_end = interval_start + interval_seconds

        segment = df[
            (df["Time_s"] >= interval_start)
            & (df["Time_s"] < interval_end)
        ]

        if segment.empty:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

        axes[0].plot(segment["Time"], segment["ADS1292R_Raw"], color="tab:blue")
        axes[0].set_title("ADS1292R Raw ECG")
        axes[0].set_ylabel("Raw ECG")
        axes[0].grid(True)

        axes[1].plot(segment["Time"], segment["ADS1292R_Filtered"], color="tab:orange")
        axes[1].set_title("ADS1292R Filtered ECG")
        axes[1].set_ylabel("Filtered ECG")
        axes[1].grid(True)

        axes[2].plot(segment["Time"], segment["BPM"], color="tab:green")
        axes[2].set_title("ADS1292R Heart Rate")
        axes[2].set_xlabel("Clock Time")
        axes[2].set_ylabel("BPM")
        axes[2].grid(True)

        fig.suptitle(
            f"ADS1292R ECG Interval {interval_start:.1f}s to {interval_end:.1f}s",
            fontsize=14
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        file_name = f"ecg_interval_{interval_start:.1f}s_to_{interval_end:.1f}s.png"
        fig.savefig(IMAGE_OUTPUT_DIR / file_name, dpi=200)
        plt.close(fig)
        saved_count += 1

    return saved_count


# ============================================================
# DATA INFORMATION
# ============================================================

print()
print("=" * 60)
print("DATA SAVED")
print("=" * 60)

print()

print(
    f"File     : {OUTPUT_FILE}"
)

print(
    f"Samples  : {len(df)}"
)


# ============================================================
# DURATION
# ============================================================

if len(df) > 1:

    duration = (
        df["Time_s"].iloc[-1]
        -
        df["Time_s"].iloc[0]
    )

else:

    duration = 0


print(
    f"Duration : {duration:.2f} seconds"
)


# ============================================================
# SAMPLE RATE
# ============================================================

if duration > 0:

    sample_rate = (
        len(df)
        /
        duration
    )

else:

    sample_rate = 0


print(
    f"Sample Rate : {sample_rate:.2f} Hz"
)

saved_interval_images = save_interval_images(df)
print(
    f"Saved interval images : {saved_interval_images}"
)
print(
    f"Image folder : {IMAGE_OUTPUT_DIR}"
)


# ============================================================
# BPM INFORMATION
# ============================================================

valid_bpm = df[
    df["BPM"] > 0
]["BPM"]


if len(valid_bpm) > 0:

    print()
    print("=" * 60)
    print("HEART RATE")
    print("=" * 60)

    print()

    print(
        f"Average BPM : "
        f"{valid_bpm.mean():.2f}"
    )

    print(
        f"Minimum BPM : "
        f"{valid_bpm.min():.2f}"
    )

    print(
        f"Maximum BPM : "
        f"{valid_bpm.max():.2f}"
    )

else:

    print()
    print("No valid BPM values detected.")


# ============================================================
# GRAPH 1
# RAW ECG
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    df["Time"],
    df["ADS1292R_Raw"]
)

plt.title(
    "ADS1292R Raw ECG"
)

plt.xlabel(
    "Clock Time"
)

plt.ylabel(
    "Raw ECG"
)

plt.grid(True)

plt.tight_layout()


# ============================================================
# GRAPH 2
# FILTERED ECG
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    df["Time"],
    df["ADS1292R_Filtered"]
)

plt.title(
    "ADS1292R Filtered ECG"
)

plt.xlabel(
    "Clock Time"
)

plt.ylabel(
    "Filtered ECG"
)

plt.grid(True)

plt.tight_layout()


# ============================================================
# GRAPH 3
# BPM
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    df["Time"],
    df["BPM"]
)

plt.title(
    "ADS1292R Heart Rate"

)

plt.xlabel(
    "Clock Time"
)

plt.ylabel(
    "BPM"
)

plt.grid(True)

plt.tight_layout()


# ============================================================
# SHOW GRAPHS
# ============================================================

plt.show()