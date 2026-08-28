"""Prepare labeled ECG windows from the MPDDF EDF and annotation files."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, welch
from scipy.stats import kurtosis, skew


TARGET_FS = 125.0
WINDOW_SECONDS = 5.0


def read_and_preprocess_ecg(edf_path: Path) -> tuple[np.ndarray, datetime]:
    """Read only ECG from an EDF, resample it, and apply ECG preprocessing."""
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = [reader.getLabel(i).strip() for i in range(reader.signals_in_file)]
        ecg_channels = [i for i, label in enumerate(labels) if label.upper() == "ECG"]
        if not ecg_channels:
            raise ValueError(f"ECG channel not found in {edf_path}: {labels}")
        channel = ecg_channels[0]
        signal = reader.readSignal(channel).astype(float)
        source_fs = float(reader.getSampleFrequency(channel))
        recording_start = reader.getStartdatetime()
    finally:
        reader.close()

    valid = np.isfinite(signal)
    if not np.any(valid):
        raise ValueError(f"ECG channel contains no finite samples: {edf_path}")

    source_time = np.arange(signal.size, dtype=float) / source_fs
    uniform_time = np.arange(0.0, source_time[-1] + 0.5 / TARGET_FS, 1.0 / TARGET_FS)
    signal = np.interp(source_time, source_time[valid], signal[valid])
    signal = np.interp(uniform_time, source_time, signal)
    signal -= np.median(signal)

    band_b, band_a = butter(
        4,
        [0.5 / (TARGET_FS / 2), 40.0 / (TARGET_FS / 2)],
        btype="band",
    )
    signal = filtfilt(band_b, band_a, signal)
    notch_b, notch_a = iirnotch(50.0 / (TARGET_FS / 2), 30.0)
    signal = filtfilt(notch_b, notch_a, signal)
    return signal, recording_start


def read_annotations(annotation_path: Path, recording_start: datetime) -> list[tuple[float, int]]:
    annotations = []
    for line in annotation_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            annotation_clock = datetime.strptime(parts[0], "%H:%M:%S").time()
            label = int(parts[2])
        except ValueError:
            continue

        annotation_time = datetime.combine(recording_start.date(), annotation_clock)
        while annotation_time - recording_start < timedelta(hours=-12):
            annotation_time += timedelta(days=1)
        while annotation_time - recording_start > timedelta(hours=12):
            annotation_time -= timedelta(days=1)
        annotations.append(((annotation_time - recording_start).total_seconds(), label))
    return sorted(annotations)


def label_window(start: float, end: float, annotations: list[tuple[float, int]]) -> tuple[int | None, float]:
    """Return the state covering at least 80% of a window, if one exists."""
    overlaps: dict[int, float] = {}
    for index, (annotation_start, label) in enumerate(annotations):
        annotation_end = annotations[index + 1][0] if index + 1 < len(annotations) else math.inf
        overlap = max(0.0, min(end, annotation_end) - max(start, annotation_start))
        overlaps[label] = overlaps.get(label, 0.0) + overlap
    if not overlaps:
        return None, 0.0
    label, coverage_seconds = max(overlaps.items(), key=lambda item: item[1])
    coverage = coverage_seconds / (end - start)
    return (label if coverage >= 0.8 else None), coverage


def detect_r_peaks(signal: np.ndarray) -> np.ndarray:
    return find_peaks(
        signal,
        distance=int(0.30 * TARGET_FS),
        prominence=0.5 * np.std(signal),
    )[0]


def signal_features(window: np.ndarray, peaks: np.ndarray) -> dict[str, float]:
    features = {
        "mean": float(np.mean(window)),
        "median": float(np.median(window)),
        "std": float(np.std(window)),
        "rms": float(np.sqrt(np.mean(window**2))),
        "min": float(np.min(window)),
        "max": float(np.max(window)),
        "peak_to_peak": float(np.ptp(window)),
        "iqr": float(np.percentile(window, 75) - np.percentile(window, 25)),
        "skew": float(skew(window, bias=False)),
        "kurtosis": float(kurtosis(window, bias=False)),
        "energy": float(np.mean(window**2)),
    }

    frequencies, powers = welch(window, fs=TARGET_FS, nperseg=min(512, len(window)))
    total_power = float(np.trapz(powers, frequencies))
    features["total_power"] = total_power
    for name, low, high in (("0_4", 0.5, 4), ("4_8", 4, 8), ("8_15", 8, 15), ("15_40", 15, 40)):
        band_mask = (frequencies >= low) & (frequencies < high)
        band_power = float(np.trapz(powers[band_mask], frequencies[band_mask]))
        features[f"power_{name}"] = band_power
        features[f"relative_power_{name}"] = band_power / total_power if total_power > 0 else np.nan

    probability = powers / np.sum(powers) if np.sum(powers) > 0 else np.zeros_like(powers)
    features["dominant_frequency"] = float(frequencies[np.argmax(powers)])
    features["spectral_entropy"] = float(
        -np.sum(probability[probability > 0] * np.log2(probability[probability > 0]))
    )

    rr_ms = np.diff(peaks) / TARGET_FS * 1000.0
    valid_rr = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
    features["peak_count"] = float(len(peaks))
    features["valid_rr_count"] = float(len(valid_rr))
    features["valid_rr_fraction"] = float(len(valid_rr) / len(rr_ms)) if len(rr_ms) else 0.0
    features["mean_hr"] = float(60000.0 / np.mean(valid_rr)) if len(valid_rr) else np.nan
    features["median_hr"] = float(60000.0 / np.median(valid_rr)) if len(valid_rr) else np.nan
    features["sdnn"] = float(np.std(valid_rr, ddof=1)) if len(valid_rr) > 1 else np.nan
    features["rmssd"] = float(np.sqrt(np.mean(np.diff(valid_rr) ** 2))) if len(valid_rr) > 2 else np.nan
    features["invalid_rr_fraction"] = float(1.0 - features["valid_rr_fraction"])
    features["flatline_fraction"] = float(np.mean(np.isclose(window, window[0])))
    return features


def prepare_dataset(dataset_root: Path, output_path: Path, window_seconds: float) -> None:
    window_samples = int(TARGET_FS * window_seconds)
    rows = []
    psg_dir = dataset_root / "PSG"
    annotation_dir = dataset_root / "Annotation"

    edf_paths = sorted(psg_dir.glob("MPDDF_raw_*_PSG.edf"))
    if not edf_paths:
        raise FileNotFoundError(f"No MPDDF EDF files found in {psg_dir}")

    for edf_path in edf_paths:
        subject_id = edf_path.stem.split("_")[2]
        annotation_path = annotation_dir / f"MPDDF_raw_{subject_id}_Annotation.txt"
        if not annotation_path.exists():
            print(f"Skipping {subject_id}: annotation file not found")
            continue

        signal, recording_start = read_and_preprocess_ecg(edf_path)
        annotations = read_annotations(annotation_path, recording_start)
        peaks = detect_r_peaks(signal)
        for start_sample in range(0, len(signal) - window_samples + 1, window_samples):
            start = start_sample / TARGET_FS
            end = (start_sample + window_samples) / TARGET_FS
            label, coverage = label_window(start, end, annotations)
            if label is None:
                continue
            window_end = start_sample + window_samples
            window_peaks = peaks[(peaks >= start_sample) & (peaks < window_end)] - start_sample
            row = signal_features(signal[start_sample:window_end], window_peaks)
            row.update({
                "subject_id": subject_id,
                "window_start_s": start,
                "window_end_s": end,
                "label": label,
                "label_coverage": coverage,
            })
            rows.append(row)
        print(f"Processed {subject_id}: {len(signal) / TARGET_FS / 60:.1f} minutes")

    if not rows:
        raise RuntimeError("No labeled windows were created; verify annotation timing and label alignment.")
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved {len(frame)} labeled windows to {output_path}")
    print(frame.groupby("label").size().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("Raw Dataset"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mpddf_ecg_prepared.csv"))
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    args = parser.parse_args()
    if args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")
    prepare_dataset(args.dataset_root, args.output, args.window_seconds)


if __name__ == "__main__":
    main()
