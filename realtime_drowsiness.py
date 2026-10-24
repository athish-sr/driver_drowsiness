"""
Collect dual MAX30102 readings and periodically run the fusion detector.

The collector keeps writing the active CSV while run_all_fusion.py processes
the latest flushed contents. Stop with Ctrl+C; the collector is then stopped
and one final detection pass is attempted.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COLLECTOR = ROOT / "DualPPG" / "dual_ppg.py"
DETECTOR = ROOT / "run_all_fusion.py"
HEADER = ["Clock Time", "Elapsed_s", "Device Timestamp", "IR1", "Red1", "IR2", "Red2"]


def latest_elapsed(csv_path: Path) -> float:
    """Return the last complete elapsed time written by the collector."""
    if not csv_path.is_file():
        return 0.0
    with csv_path.open("r", newline="") as stream:
        rows = csv.reader(stream)
        next(rows, None)
        last = 0.0
        for row in rows:
            if len(row) != len(HEADER):
                continue
            try:
                last = float(row[1])
            except ValueError:
                continue
        return last


def run_detection(csv_path: Path, output_root: Path, serial_port: str) -> int:
    env = os.environ.copy()
    env["FUSION_INPUT_FILE"] = csv_path.name
    env["FUSION_PPG_DIR"] = str(csv_path.parent)
    env["FUSION_OUT_ROOT"] = str(output_root)
    env["DUAL_PPG_SERIAL_PORT"] = serial_port
    process = subprocess.run(
        [sys.executable, str(DETECTOR)],
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=os.environ.get("DUAL_PPG_SERIAL_PORT", "COM9"))
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--interval", type=float, default=30.0,
                        help="seconds between detection passes")
    parser.add_argument("--minimum-seconds", type=float, default=95.0,
                        help="wait this long before the first detection pass")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fusion_output_realtime")
    parser.add_argument("--recording", default="dual_max30102_realtime.csv")
    args = parser.parse_args()

    ppg_dir = ROOT / "PPG_reading"
    ppg_dir.mkdir(exist_ok=True)
    csv_path = ppg_dir / args.recording
    if csv_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = ppg_dir / f"{csv_path.stem}_{stamp}{csv_path.suffix}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["DUAL_PPG_SERIAL_PORT"] = args.port
    env["DUAL_PPG_BAUD_RATE"] = str(args.baud)
    env["DUAL_PPG_OUTPUT_FILE"] = str(csv_path)

    collector = subprocess.Popen(
        [sys.executable, str(COLLECTOR)],
        cwd=str(ROOT),
        env=env,
    )
    print(f"Recording to: {csv_path}")
    print(f"Detection output: {args.output_dir}")
    print("Press Ctrl+C to stop collection.")

    last_detection = 0.0
    try:
        while collector.poll() is None:
            elapsed = latest_elapsed(csv_path)
            now = time.monotonic()
            if elapsed >= args.minimum_seconds and now - last_detection >= args.interval:
                print(f"\nRunning detection at recording time {elapsed:.1f}s...")
                run_detection(csv_path, args.output_dir, args.port)
                last_detection = now
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping real-time collection...")
    finally:
        if collector.poll() is None:
            collector.terminate()
            try:
                collector.wait(timeout=10)
            except subprocess.TimeoutExpired:
                collector.kill()
                collector.wait()

    if csv_path.is_file() and latest_elapsed(csv_path) > 0:
        print("\nRunning final detection pass...")
        run_detection(csv_path, args.output_dir, args.port)
    return collector.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
