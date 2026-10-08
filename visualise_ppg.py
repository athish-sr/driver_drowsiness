from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "PPG_DATA_COLLECTION" / "ppg_fusion_output.csv"
OUTPUT_DIR = ROOT / "PPG_DATA_COLLECTION" / "visualisations"
T_START, T_END = 60, 90

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(INPUT_CSV)

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Print columns for verification
print("Columns:", df.columns.tolist())

mask = (
    (df['Elapsed_s'] >= T_START) &
    (df['Elapsed_s'] <= T_END)
)

d = df.loc[mask]

print(f"Number of samples in {T_START}-{T_END}s:", len(d))

required_columns = ["IR Fusion", "RED Fusion"]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError(f"Missing processed columns: {missing}")

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

axes[0].plot(d["Elapsed_s"], d["IR Fusion"], color="crimson", linewidth=1.2,
             label="IR")
axes[0].set_title(f"Single-Sensor Filtered IR ({T_START}-{T_END} s)")
axes[0].set_ylabel("Filtered IR (ADC Counts)")
axes[0].legend(loc="upper right")
axes[0].grid(alpha=0.3)

axes[1].plot(d["Elapsed_s"], d["RED Fusion"], color="steelblue", linewidth=1.2,
             label="RED")
axes[1].set_title(f"Single-Sensor Filtered RED ({T_START}-{T_END} s)")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Red Filtered (ADC Counts)")
axes[1].legend(loc="upper right")
axes[1].grid(alpha=0.3)

axes[1].set_xlim(T_START, T_END)
fig.suptitle("Single-Sensor Filtered PPG", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])

output_path = OUTPUT_DIR / "ir_red_single_sensor.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {output_path}")