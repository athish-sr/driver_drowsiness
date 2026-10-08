import serial
import csv
import time
import os
from datetime import datetime

# ============================================================
# SERIAL CONFIGURATION
# ============================================================

SERIAL_PORT = "COM9"
BAUD_RATE = 115200

# ============================================================
# CSV FILE
# ============================================================

OUTPUT_FILE = "dual_max30102_data.csv"

CSV_HEADER = [
    "Clock Time",
    "Elapsed_s",
    "Device Timestamp",
    "IR1",
    "Red1",
    "IR2",
    "Red2"
]

# ============================================================
# CONNECT TO ESP32
# ============================================================

print("Connecting to ESP32...")

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("Connected!")
print("Waiting for sensor data...")
print()

# ============================================================
# OPEN CSV FILE
# ============================================================

file_exists = os.path.exists(OUTPUT_FILE)

if file_exists and os.path.getsize(OUTPUT_FILE) > 0:
    with open(OUTPUT_FILE, "r", newline="") as existing_file:
        existing_header = next(csv.reader(existing_file), [])
    if existing_header != CSV_HEADER:
        raise RuntimeError(
            f"{OUTPUT_FILE} uses an old CSV format. Rename or remove it before "
            "starting a new recording."
        )

csv_file = open(
    OUTPUT_FILE,
    "a",
    newline="",
    buffering=1
)

writer = csv.writer(csv_file)

# Write header only if file is new/empty
if not file_exists or os.path.getsize(OUTPUT_FILE) == 0:

    writer.writerow(CSV_HEADER)

    csv_file.flush()

# ============================================================
# DATA COLLECTION
# ============================================================

sample_count = 0
first_device_timestamp = None

try:

    while True:

        # ------------------------------------------------------
        # READ ONE LINE FROM ESP32
        # ------------------------------------------------------

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            continue

        # ------------------------------------------------------
        # DISPLAY ESP32 STATUS MESSAGES
        # ------------------------------------------------------

        if not line[0].isdigit():

            print("ESP32:", line)

            continue

        # ------------------------------------------------------
        # SPLIT CSV DATA
        # ------------------------------------------------------

        values = line.split(",")

        # Expected:
        #
        # timestamp
        # IR1
        # RED1
        # IR2
        # RED2
        #
        if len(values) != 5:

            print("Invalid data:", line)

            continue

        try:

            timestamp = int(values[0])

            ir1 = int(values[1])

            red1 = int(values[2])

            ir2 = int(values[3])

            red2 = int(values[4])

        except ValueError:

            print("Invalid numeric data:", line)

            continue

        # ------------------------------------------------------
        # SAVE TO CSV
        # ------------------------------------------------------

        if first_device_timestamp is None:
            first_device_timestamp = timestamp

        elapsed_seconds = (
            timestamp - first_device_timestamp
        ) / 1000.0

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"{elapsed_seconds:.3f}",
            timestamp,
            ir1,
            red1,
            ir2,
            red2
        ])

        # Make sure data is written immediately
        csv_file.flush()

        sample_count += 1

        # ------------------------------------------------------
        # DISPLAY DATA
        # ------------------------------------------------------

        print(
            f"{timestamp} ms | "
            f"IR1: {ir1} | "
            f"RED1: {red1} | "
            f"IR2: {ir2} | "
            f"RED2: {red2}"
        )

except KeyboardInterrupt:

    print()
    print("Stopping data collection...")

finally:

    # ========================================================
    # CLOSE EVERYTHING
    # ========================================================

    csv_file.close()

    ser.close()

    print()
    print("Data collection stopped.")
    print(f"Total samples saved: {sample_count}")
    print(f"CSV file: {os.path.abspath(OUTPUT_FILE)}")