import os
import matplotlib.pyplot as plt
import pandas as pd

INPUT_CSV = r"ppg_results_sg\ppg_fusion_output.csv"
OUTPUT_DIR = r"ppg_results_sg\visualisations"

# INPUT_CSV = r"ppg_results_dm\ppg_fusion_output.csv"
# OUTPUT_DIR = r"ppg_results_dm\visualisations"

T_START, T_END = 500,1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read CSV
df = pd.read_csv(INPUT_CSV)

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Print columns for verification
print("Columns:", df.columns.tolist())

# Check required columns
required_columns = [
    "Elapsed_s",
    "IR1 Fusion",
    "IR2 Fusion",
    "RED1 Fusion",
    "RED2 Fusion",
]

missing = [column for column in required_columns if column not in df.columns]

if missing:
    raise ValueError(f"Missing processed columns: {missing}")

# Select required time range
mask = (
    (df["Elapsed_s"] >= T_START) &
    (df["Elapsed_s"] <= T_END)
)

d = df.loc[mask]

print(f"Number of samples in {T_START}-{T_END}s:", len(d))

# ---------------------------------------------------------
# Create figure
# ---------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 9),
    sharex=True
)

# ---------------------------------------------------------
# IR signals
# ---------------------------------------------------------

axes[0].plot(
    d["Elapsed_s"],
    d["IR1 Fusion"],
    linewidth=1.2,
    label="Sensor 1 IR"
)

axes[0].plot(
    d["Elapsed_s"],
    d["IR2 Fusion"],
    linewidth=1.2,
    label="Sensor 2 IR"
)

axes[0].set_title(
    f"Filtered IR Overlay ({T_START}-{T_END} s)"
)

axes[0].set_ylabel("Filtered IR (ADC Counts)")
axes[0].legend(loc="upper right")
axes[0].grid(alpha=0.3)

# ---------------------------------------------------------
# RED signals
# ---------------------------------------------------------

axes[1].plot(
    d["Elapsed_s"],
    d["RED1 Fusion"],
    linewidth=1.2,
    label="Sensor 1 RED"
)

axes[1].plot(
    d["Elapsed_s"],
    d["RED2 Fusion"],
    linewidth=1.2,
    label="Sensor 2 RED"
)

axes[1].set_title(
    f"Filtered RED Overlay ({T_START}-{T_END} s)"
)

axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Red Filtered (ADC Counts)")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.3)

axes[1].set_xlim(T_START, T_END)

# ---------------------------------------------------------
# Overall title
# ---------------------------------------------------------

fig.suptitle(
    "Two-Sensor Filtered PPG Comparison",
    fontsize=14
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

# ---------------------------------------------------------
# SHOW FIGURE
# ---------------------------------------------------------

print("Displaying plot...")
print("Close the Matplotlib window to save the figure.")

plt.show()

# ---------------------------------------------------------
# SAVE AFTER WINDOW IS CLOSED
# ---------------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "ir_red_both_sensors_overlay.png"
)

fig.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {output_path}")