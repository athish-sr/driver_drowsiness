"""
Overlay filtered ECG and filtered PPG using ONLY their own elapsed-time
columns, with NO resampling/interpolation of either signal.

Inputs:
    filtered_ecg.csv        -> "Time_s"    (ECG elapsed time, seconds, ~125 Hz)
    ppg_fusion_output.csv   -> "Elapsed_s" (PPG elapsed time, seconds, ~25 Hz)

Why no resampling:
    ECG (~125 Hz) is much faster than PPG (~25 Hz). Interpolating ECG onto
    PPG's sparser timestamps clips/flattens the sharp QRS peaks (confirmed:
    peak amplitude dropped ~37 -> ~33 units in a spot check). This version
    keeps each signal at its OWN native sample times and simply restricts
    both to the overlapping time window, so what you see is exactly what's
    in the original files - nothing is downsampled, upsampled, or
    interpolated.

HOW IT WORKS
-------------------
    1. Take ECG's "Time_s" and PPG's "Elapsed_s" as directly comparable
       time axes (no raw/wall-clock files used).
    2. Overlap window = [max(ECG_start, PPG_start), min(ECG_end, PPG_end)].
    3. Trim each signal to that window at its OWN native sample points
       (no interpolation onto the other signal's timestamps).
    4. Shift both time axes by the same offset (the window start) so they
       share a common "Sync_Time_s" origin, purely for plotting alignment.
    5. Save each trimmed signal to its own CSV (they have different sample
       counts/timestamps, so they can't be forced into one row-aligned
       table without resampling) and plot a dual-axis overlay.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_ECG_FILE = ROOT / "ecg_validation_results_ecg" / "filtered_ecg.csv"
DEFAULT_PPG_FILE = ROOT / "PPG_DATA_COLLECTION" / "ppg_fusion_output.csv"
DEFAULT_OUTPUT_DIR = ROOT / "SYNC_OUTPUT"


def numeric_column(df: pd.DataFrame, names: list[str], description: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(
        f"Could not find {description}. Expected one of {names}; "
        f"available columns: {list(df.columns)}"
    )


def load_ecg(ecg_file: Path) -> pd.DataFrame:
    ecg = pd.read_csv(ecg_file)
    time_column = numeric_column(ecg, ["Time_s"], "ECG time column")
    signal_column = numeric_column(
        ecg,
        ["Filtered_ECG", "Filtered_ECG_counts", "Filtered_ECG_uV", "Filtered_ECG_mV"],
        "filtered ECG column",
    )

    ecg = ecg[[time_column, signal_column]].copy()
    ecg.columns = ["ECG_Time_s", "Filtered_ECG"]
    ecg["ECG_Time_s"] = pd.to_numeric(ecg["ECG_Time_s"], errors="coerce")
    ecg["Filtered_ECG"] = pd.to_numeric(ecg["Filtered_ECG"], errors="coerce")
    ecg = ecg.dropna().sort_values("ECG_Time_s").drop_duplicates("ECG_Time_s")
    ecg = ecg.reset_index(drop=True)
    return ecg


def load_ppg(ppg_file: Path) -> pd.DataFrame:
    ppg = pd.read_csv(ppg_file)
    time_column = numeric_column(ppg, ["Elapsed_s"], "PPG elapsed-time column")
    signal_column = numeric_column(
        ppg,
        ["IR Fusion", "IR Filtered", "IR Raw", "IR_Raw", "RED Fusion"],
        "filtered PPG column",
    )

    ppg = ppg[[time_column, signal_column]].copy()
    ppg.columns = ["PPG_Elapsed_s", "Filtered_PPG"]
    ppg["PPG_Elapsed_s"] = pd.to_numeric(ppg["PPG_Elapsed_s"], errors="coerce")
    ppg["Filtered_PPG"] = pd.to_numeric(ppg["Filtered_PPG"], errors="coerce")
    ppg = ppg.dropna().sort_values("PPG_Elapsed_s").drop_duplicates("PPG_Elapsed_s")
    ppg = ppg.reset_index(drop=True)
    return ppg


def overlap_trim(ecg: pd.DataFrame, ppg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Restrict both signals to the overlapping time window, at their OWN
    native sample points - no interpolation/resampling of either signal."""
    ecg_time = ecg["ECG_Time_s"].to_numpy()
    ppg_time = ppg["PPG_Elapsed_s"].to_numpy()

    sync_start = max(ecg_time[0], ppg_time[0])
    sync_end = min(ecg_time[-1], ppg_time[-1])
    if sync_start >= sync_end:
        raise ValueError(
            "ECG and PPG have no overlapping elapsed-time range. "
            f"ECG covers {ecg_time[0]:.2f}s to {ecg_time[-1]:.2f}s; "
            f"PPG covers {ppg_time[0]:.2f}s to {ppg_time[-1]:.2f}s."
        )

    ecg_trimmed = ecg[(ecg["ECG_Time_s"] >= sync_start) & (ecg["ECG_Time_s"] <= sync_end)].copy()
    ppg_trimmed = ppg[(ppg["PPG_Elapsed_s"] >= sync_start) & (ppg["PPG_Elapsed_s"] <= sync_end)].copy()

    # Shared origin for plotting only - original sample values untouched.
    ecg_trimmed["Sync_Time_s"] = ecg_trimmed["ECG_Time_s"] - sync_start
    ppg_trimmed["Sync_Time_s"] = ppg_trimmed["PPG_Elapsed_s"] - sync_start

    return ecg_trimmed.reset_index(drop=True), ppg_trimmed.reset_index(drop=True), sync_start


def display_full_overlay(ecg_trimmed: pd.DataFrame, ppg_trimmed: pd.DataFrame) -> None:
    figure, ecg_axis = plt.subplots(figsize=(16, 6))
    ppg_axis = ecg_axis.twinx()

    ecg_rate = len(ecg_trimmed) / (ecg_trimmed["Sync_Time_s"].iloc[-1] - ecg_trimmed["Sync_Time_s"].iloc[0])
    ppg_rate = len(ppg_trimmed) / (ppg_trimmed["Sync_Time_s"].iloc[-1] - ppg_trimmed["Sync_Time_s"].iloc[0])

    ecg_line, = ecg_axis.plot(
        ecg_trimmed["Sync_Time_s"],
        ecg_trimmed["Filtered_ECG"],
        color="tab:blue",
        label=f"Filtered ECG (native ~{ecg_rate:.0f} Hz)",
        linewidth=0.8,
    )
    ppg_line, = ppg_axis.plot(
        ppg_trimmed["Sync_Time_s"],
        ppg_trimmed["Filtered_PPG"],
        color="tab:orange",
        label=f"Filtered PPG (native ~{ppg_rate:.0f} Hz)",
        linewidth=1.0,
    )

    ecg_axis.set_title("ECG + PPG Overlay: native sample rates, no resampling")
    ecg_axis.set_xlabel("Sync time (s)")
    ecg_axis.set_ylabel("Filtered ECG (device units)", color="tab:blue")
    ppg_axis.set_ylabel("Filtered PPG (device units)", color="tab:orange")
    ecg_axis.tick_params(axis="y", labelcolor="tab:blue")
    ppg_axis.tick_params(axis="y", labelcolor="tab:orange")
    ecg_axis.grid(True, alpha=0.3)
    ecg_axis.legend(handles=[ecg_line, ppg_line], loc="upper right")
    figure.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay filtered ECG and PPG at native sample rates (no resampling)."
    )
    parser.add_argument("--ecg", type=Path, default=DEFAULT_ECG_FILE)
    parser.add_argument("--ppg", type=Path, default=DEFAULT_PPG_FILE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot window (still saves the PNG/CSVs).",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    print("Loading filtered ECG and PPG data...")
    ecg = load_ecg(args.ecg)
    ppg = load_ppg(args.ppg)
    ecg_trimmed, ppg_trimmed, sync_start = overlap_trim(ecg, ppg)

    # Each signal keeps its own native sample times - saved separately since
    # they have different lengths/timestamps (no shared row grid without
    # resampling).
    ecg_file = args.outdir / "ECG_native_overlap.csv"
    ppg_file = args.outdir / "PPG_native_overlap.csv"
    ecg_trimmed[["Sync_Time_s", "ECG_Time_s", "Filtered_ECG"]].to_csv(ecg_file, index=False)
    ppg_trimmed[["Sync_Time_s", "PPG_Elapsed_s", "Filtered_PPG"]].to_csv(ppg_file, index=False)

    if args.no_show:
        matplotlib.use("Agg")
    fig_path = args.outdir / "ECG_PPG_overlay_native.png"
    display_full_overlay(ecg_trimmed, ppg_trimmed)
    plt.gcf().savefig(fig_path, dpi=150)

    duration = ecg_trimmed["Sync_Time_s"].iloc[-1]
    print(f"Overlap window duration: {duration:.2f}s")
    print(f"ECG native samples in window: {len(ecg_trimmed)}")
    print(f"PPG native samples in window: {len(ppg_trimmed)}")
    print(f"ECG data saved to: {ecg_file}")
    print(f"PPG data saved to: {ppg_file}")
    print(f"Overlay plot saved to: {fig_path}")


if __name__ == "__main__":
    main()