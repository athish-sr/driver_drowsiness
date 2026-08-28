from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import timedelta


ECG_CSV = Path("ADS1292R_ECG.csv")
PPG_CSV = Path("DTWATCH_PPG_RESULTS") / "ppg_processed_data.csv"
OUTPUT_DIR = Path("ecg_ppg_15s_visualizations")
WINDOW_SECONDS = 15
SHOW_PLOT = True


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_ecg(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ECG file not found: {path}")

    df = pd.read_csv(path)
    require_columns(
        df,
        ["Time", "ADS1292R_Raw", "ADS1292R_Filtered", "BPM"],
        "ECG CSV"
    )

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    numeric_cols = ["ADS1292R_Raw", "ADS1292R_Filtered", "BPM"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Time", "ADS1292R_Raw", "ADS1292R_Filtered"]).copy()
    df = df.sort_values("Time")
    return df


def load_ppg(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PPG file not found: {path}")

    df = pd.read_csv(path)
    require_columns(
        df,
        ["Clock Time", "IR Raw", "IR Filtered", "BPM"],
        "PPG CSV"
    )

    df["Clock Time"] = pd.to_datetime(df["Clock Time"], errors="coerce")

    numeric_cols = ["IR Raw", "IR Filtered", "BPM"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Clock Time", "IR Raw", "IR Filtered"]).copy()
    df = df.sort_values("Clock Time")
    return df


def plot_combined_window(
    ecg_df: pd.DataFrame,
    ppg_df: pd.DataFrame,
    window_start,
    window_end,
    output_path: Path
) -> bool:
    ecg_window = ecg_df[
        (ecg_df["Time"] >= window_start)
        & (ecg_df["Time"] < window_end)
    ]

    ppg_window = ppg_df[
        (ppg_df["Clock Time"] >= window_start)
        & (ppg_df["Clock Time"] < window_end)
    ]

    if ecg_window.empty and ppg_window.empty:
        return False

    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=False)

    axes[0, 0].plot(ecg_window["Time"], ecg_window["ADS1292R_Raw"], label="ECG Raw", color="tab:blue", linewidth=1.0)
    axes[0, 0].set_title("ECG Raw")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(ppg_window["Clock Time"], ppg_window["IR Raw"], label="PPG Raw (IR)", color="tab:orange", linewidth=1.0)
    axes[0, 1].set_title("PPG Raw (IR)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(ecg_window["Time"], ecg_window["ADS1292R_Filtered"], label="ECG Filtered", color="tab:green", linewidth=1.0)
    axes[1, 0].set_title("ECG Filtered")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(ppg_window["Clock Time"], ppg_window["IR Filtered"], label="PPG Filtered (IR)", color="tab:red", linewidth=1.0)
    axes[1, 1].set_title("PPG Filtered (IR)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    ecg_bpm = ecg_window.dropna(subset=["BPM"])
    ppg_bpm = ppg_window.dropna(subset=["BPM"])

    if not ecg_bpm.empty:
        axes[2, 0].plot(ecg_bpm["Time"], ecg_bpm["BPM"], label="ECG BPM", color="tab:purple", linewidth=1.2)
    if not ppg_bpm.empty:
        axes[2, 1].plot(ppg_bpm["Clock Time"], ppg_bpm["BPM"], label="PPG BPM", color="tab:brown", linewidth=1.2)

    axes[2, 0].set_title("ECG BPM")
    axes[2, 0].set_xlabel("Clock Time")
    axes[2, 0].set_ylabel("BPM")
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    axes[2, 1].set_title("PPG BPM")
    axes[2, 1].set_xlabel("Clock Time")
    axes[2, 1].set_ylabel("BPM")
    axes[2, 1].grid(True)
    axes[2, 1].legend()

    for axis in axes.flat:
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axis.set_xlim(left=window_start, right=window_end)

    fig.suptitle(
        f"ECG + PPG Visualization ({window_start.strftime('%H:%M:%S')} to {window_end.strftime('%H:%M:%S')})",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=220)

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)

    return True


def main() -> None:
    ecg_df = load_ecg(ECG_CSV)
    ppg_df = load_ppg(PPG_CSV)

    print(f"Loaded ECG samples: {len(ecg_df)}")
    print(f"Loaded PPG samples: {len(ppg_df)}")

    overall_start = max(ecg_df["Time"].min(), ppg_df["Clock Time"].min())
    overall_end = min(ecg_df["Time"].max(), ppg_df["Clock Time"].max())

    if overall_start >= overall_end:
        raise ValueError("No overlapping time range found between ECG and PPG data.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current_start = overall_start
    saved_count = 0

    while current_start < overall_end:
        current_end = min(current_start + timedelta(seconds=WINDOW_SECONDS), overall_end)
        image_name = (
            f"ecg_ppg_{current_start.strftime('%H%M%S')}_to_{current_end.strftime('%H%M%S')}.png"
        )
        output_path = OUTPUT_DIR / image_name

        saved = plot_combined_window(
            ecg_df,
            ppg_df,
            current_start,
            current_end,
            output_path,
        )

        if saved:
            saved_count += 1
            print(f"Saved: {output_path}")

        current_start = current_end

    print(f"Saved {saved_count} window visualizations in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
