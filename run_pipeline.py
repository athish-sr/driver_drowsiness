"""
run_pipeline.py  –  Dual PPG end-to-end pipeline orchestrator
==============================================================

USAGE
-----
    python run_pipeline.py --csv <path/to/raw.csv> [--method 1] [--outdir <dir>] [options]

PIPELINE (--method 1)
---------------------
  Step 1  dual_process.py          Raw CSV  ->  ppg_fusion_output.csv
  Step 2a dual_ssqi.py             ppg_fusion_output.csv  ->  ssqi_classified.csv
  Step 2b dual_fusion.py           ppg_fusion_output.csv  ->  ppg_fusion_results.csv
  Step 3  step2_ssqi_adaptive_v2.py  best fused IR signal ->  ssqi_classified_fused.csv
  Step 4  ThresHoldModel/TimeStatic.py   fused IR signal  ->  drowsiness table (stdout + CSV)

All outputs land in --outdir  (default: <csv_stem>_pipeline_<timestamp>/).

All intermediate files are kept so each step can be re-run in isolation.
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent          # MainProject/
DUAL_PPG = ROOT / "DualPPG"
THRESHOLD = ROOT / "ThresHoldModel"

_STEP_WIDTH = 70


def _banner(title: str):
    line = "=" * _STEP_WIDTH
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def _check_output(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Expected output from {label} not found: {path}"
        )
    size = path.stat().st_size
    print(f"  [OK] {path.name}  ({size:,} bytes)")
    return path


# ---------------------------------------------------------------------------
# Step 1 – dual_process.py
# ---------------------------------------------------------------------------

def step1_preprocess(csv: Path, outdir: Path, args) -> Path:
    """
    Runs dual_process.py.
    Returns: path to ppg_fusion_output.csv
    """
    _banner("STEP 1 · Adaptive Butterworth + Savitzky-Golay preprocessing")

    step1_out = outdir / "step1_preprocess"
    step1_out.mkdir(parents=True, exist_ok=True)

    # Build argv for dual_process.main()
    argv = [
        "--csv",        str(csv),
        "--outdir",     str(step1_out),
        "--butter-low", str(args.butter_low),
        "--butter-high",str(args.butter_high),
        "--butter-order",str(args.butter_order),
        "--stab-s",     str(args.stab_s),
        "--calib-s",    str(args.calib_s),
        "--proc-window-s", str(args.proc_window_s),
    ]

    # Import and call; sys.path already has DUAL_PPG injected below
    import dual_process
    _saved_argv = sys.argv[:]
    sys.argv = ["dual_process.py"] + argv
    try:
        dual_process.main()
    finally:
        sys.argv = _saved_argv

    fusion_csv = step1_out / "ppg_fusion_output.csv"
    _check_output(fusion_csv, "dual_process")
    return fusion_csv


# ---------------------------------------------------------------------------
# Step 2a – dual_ssqi.py
# ---------------------------------------------------------------------------

def step2a_ssqi(fusion_csv: Path, outdir: Path, args) -> Path:
    """
    Runs dual_ssqi.py on the four channels.
    Returns: path to ssqi_classified.csv
    """
    _banner("STEP 2a · Dual S_SQI quality classification (dual_ssqi.py)")

    step2a_out = outdir / "step2a_ssqi"
    step2a_out.mkdir(parents=True, exist_ok=True)

    argv = [
        "--csv",            str(fusion_csv),
        "--outdir",         str(step2a_out),
        "--mode",           args.ssqi_mode,
        "--sensor-policy",  args.sensor_policy,
        "--overall-policy", args.overall_policy,
        "--seg-s",          str(args.seg_s),
        "--hop-s",          str(args.hop_s),
        "--polarity",       args.polarity,
    ]

    import dual_ssqi
    dual_ssqi.main(argv)

    classified_csv = step2a_out / "ssqi_classified.csv"
    _check_output(classified_csv, "dual_ssqi")
    return classified_csv


# ---------------------------------------------------------------------------
# Step 2b – dual_fusion.py  (run as subprocess – script is module-level)
# ---------------------------------------------------------------------------

def step2b_fusion(fusion_csv: Path, outdir: Path) -> Path:
    """
    Runs dual_fusion.py as a subprocess because it executes at import time.
    Patches INPUT / OUT by injecting them as environment overrides via a
    thin wrapper written to a temp file.
    Returns: path to ppg_fusion_results.csv
    """
    _banner("STEP 2b · Dual-sensor IVW / IVW+SSQI fusion (dual_fusion.py)")

    step2b_out = outdir / "step2b_fusion"
    step2b_out.mkdir(parents=True, exist_ok=True)

    # Write a small driver that overrides INPUT and OUT before executing the
    # fusion script body so we don't have to modify dual_fusion.py at all.
    driver = step2b_out / "_fusion_driver.py"
    driver.write_text(
        textwrap.dedent(f"""\
        import sys, os, types

        # Redirect INPUT and OUT
        _orig_input = {str(fusion_csv)!r}
        _orig_out   = {str(step2b_out)!r}

        # Inject overrides so the script's module-level assignments pick them up
        import importlib.util, pathlib

        src = pathlib.Path({str(DUAL_PPG / 'dual_fusion.py')!r})
        code = src.read_text(encoding='utf-8')

        # Replace the two hard-coded lines
        code = code.replace(
            'INPUT = os.path.join("ppg_results_sg", "ppg_fusion_output.csv")',
            f'INPUT = r{{_orig_input!r}}'
        )
        code = code.replace(
            'OUT = "ppg_fusion_ssqi_results"',
            f'OUT = r{{_orig_out!r}}'
        )

        # SHOW_WAVEFORM = False so no interactive window blocks the pipeline
        code = code.replace('SHOW_WAVEFORM = True', 'SHOW_WAVEFORM = False')

        exec(compile(code, str(src), 'exec'), {{'__name__': '__main__',
                                                '__file__': str(src)}})
        """),
        encoding="utf-8",
    )

    # Re-write with proper f-string escaping (the textwrap above has placeholders)
    driver.write_text(
        f"""\
import sys, os

_orig_input = {str(fusion_csv)!r}
_orig_out   = {str(step2b_out)!r}

import pathlib

src = pathlib.Path({str(DUAL_PPG / 'dual_fusion.py')!r})
code = src.read_text(encoding='utf-8')

code = code.replace(
    'INPUT = os.path.join("ppg_results_sg", "ppg_fusion_output.csv")',
    f'INPUT = ' + repr(_orig_input)
)
code = code.replace(
    'OUT = "ppg_fusion_ssqi_results"',
    f'OUT = ' + repr(_orig_out)
)
code = code.replace('SHOW_WAVEFORM = True', 'SHOW_WAVEFORM = False')

exec(compile(code, str(src), 'exec'), {{'__name__': '__main__', '__file__': str(src)}})
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(driver)],
        cwd=str(DUAL_PPG),
        capture_output=False,      # let stdout/stderr flow to terminal
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dual_fusion.py exited with code {result.returncode}."
        )

    fusion_results = step2b_out / "ppg_fusion_results.csv"
    _check_output(fusion_results, "dual_fusion")
    return fusion_results


# ---------------------------------------------------------------------------
# Step 3 – step2_ssqi_adaptive_v2.py on the best fused IR signal
# ---------------------------------------------------------------------------

def step3_ssqi_fused(fusion_results_csv: Path, outdir: Path, args) -> Path:
    """
    Extracts the best fused IR signal (IVW_SSQI column) and classifies it
    with step2_ssqi_adaptive_v2.run_pipeline() directly — bypassing the
    broken --channel choices validator in that script's argparse definition.
    Returns: path to ssqi_classified_fused.csv
    """
    _banner("STEP 3 · S_SQI classification on fused IR signal (step2_ssqi_adaptive_v2.py)")

    step3_out = outdir / "step3_ssqi_fused"
    step3_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fusion_results_csv)

    # Prefer IVW_SSQI (quality-weighted) fused IR; fall back to plain IVW
    fused_col = None
    for candidate in ("IVW_SSQI_IR_Fused", "IVW_IR_Fused"):
        if candidate in df.columns:
            fused_col = candidate
            break

    if fused_col is None:
        raise KeyError(
            f"Could not find a fused IR column in {fusion_results_csv}. "
            f"Available: {list(df.columns)}"
        )

    print(f"  Using fused column: '{fused_col}'")

    time   = df["Elapsed_s"].astype(float).values
    signal = df[fused_col].astype(float).values

    import step2_ssqi_adaptive_v2 as v2

    # Build a namespace that mirrors what argparse would produce, avoiding the
    # broken choices=["IR1 Fusion","RED1 Fusion"] validator entirely.
    import types
    ns = types.SimpleNamespace(
        csv=str(fusion_results_csv),    # not used by run_pipeline, but kept for completeness
        channel="IR",
        outdir=str(step3_out),
        mode=args.ssqi_mode,
        seg_s=args.seg_s,
        hop_s=args.hop_s,
        baseline_s=30.0,
        warmup_s=None,
        min_baseline=10,
        max_baseline_age_s=180.0,
        k_exc=2.0,
        k_bad=4.0,
        min_sigma=0.03,
        guard=0.07,
        min_coverage=0.8,
        max_gap_s=0.5,
        flat_frac=0.02,
        min_amp_frac=0.2,
        clip_frac=0.10,
        min_purity=0.5,
        confirm_n=2,
        polarity=args.polarity,
    )

    out_df, info = v2.run_pipeline(time, signal, ns)

    if out_df.empty:
        print("  WARNING: step2_ssqi_adaptive_v2 produced no windows (signal too short).")
        # Write an empty placeholder so downstream steps don't crash
        dst = step3_out / "ssqi_classified_fused.csv"
        out_df.to_csv(dst, index=False)
        return dst

    print(f"  fs ~ {info['fs']:.1f} Hz, {len(out_df)} windows "
          f"({args.seg_s:g}s, hop {args.hop_s:g}s)"
          + (", polarity flipped" if info["flipped"] else ""))

    dst = step3_out / "ssqi_classified_fused.csv"
    out_df.to_csv(dst, index=False)

    # Also save the diagnostic plot
    out_png = step3_out / "ssqi_summary_fused.png"
    v2.make_plot(out_df, ns, str(out_png), "IR (fused)")
    print(f"  Saved plot: {out_png.name}")

    _check_output(dst, "step2_ssqi_adaptive_v2")
    return dst


# ---------------------------------------------------------------------------
# Step 4 – TimeStatic.py  (threshold drowsiness detector)
# ---------------------------------------------------------------------------

def step4_drowsiness(fusion_results_csv: Path, outdir: Path) -> Path:
    """
    Prepares a single-column CSV that TimeStatic.load_csv() expects
    (columns: Elapsed_s, IR Fusion) from the best fused IR signal,
    then calls run_pipeline() directly.
    Returns: path to drowsiness_result.csv
    """
    _banner("STEP 4 · Threshold drowsiness detection (TimeStatic.py)")

    step4_out = outdir / "step4_drowsiness"
    step4_out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fusion_results_csv)

    # Best fused IR
    fused_col = None
    for candidate in ("IVW_SSQI_IR_Fused", "IVW_IR_Fused"):
        if candidate in df.columns:
            fused_col = candidate
            break

    if fused_col is None:
        raise KeyError(
            f"Could not find a fused IR column in {fusion_results_csv}."
        )

    print(f"  Using fused column: '{fused_col}'")

    # TimeStatic.load_csv expects 'IR Fusion' or 'RED Fusion'
    tmp_csv = step4_out / "_fused_ir_for_threshold.csv"
    tmp_df = pd.DataFrame({
        "Elapsed_s": df["Elapsed_s"].values,
        "IR Fusion":  df[fused_col].values,
    })
    tmp_df.to_csv(tmp_csv, index=False)

    # Import and call programmatically
    sys.path.insert(0, str(THRESHOLD))
    import TimeStatic as ts

    t_raw, x_raw = ts.load_csv(str(tmp_csv), "IR")
    out = ts.run_pipeline(t_raw, x_raw)

    det = out["det"]
    result_df = out["result"]

    print(f"\n  Reference (alert) mean PP = {det.ref_mean:.3f} s  "
          f"({60/det.ref_mean:.1f} bpm),  SD = {det.ref_disp*1000:.1f} ms\n")
    print(result_df.round(3).to_string(index=False))

    drowsy_detected = bool(result_df["drowsy"].any())
    print(f"\n  *** Drowsiness detected: {drowsy_detected} ***")

    out_csv = step4_out / "drowsiness_result.csv"
    result_df.to_csv(out_csv, index=False)
    _check_output(out_csv, "TimeStatic")

    return out_csv


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(outdir: Path, outputs: dict):
    _banner("PIPELINE COMPLETE – output files")
    for label, path in outputs.items():
        print(f"  {label:<40s}  {path}")
    print(f"\n  All outputs under: {outdir}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="Dual PPG end-to-end pipeline: preprocess → SSQI → fusion → drowsiness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- required ----
    ap.add_argument("--csv", required=True,
                    help="Raw dual-PPG CSV (columns: Timestamp,IR1,RED1,IR2,RED2)")

    # ---- method selector ----
    ap.add_argument("--method", type=int, choices=[1], default=1,
                    help="Pipeline method (currently only 1 is defined)")

    # ---- output directory ----
    ap.add_argument("--outdir", default=None,
                    help="Root output directory (default: <csv_stem>_pipeline_<timestamp>)")

    # ---- Step 1 pass-throughs ----
    g1 = ap.add_argument_group("Step 1 – preprocessing")
    g1.add_argument("--butter-low",       type=float, default=0.5)
    g1.add_argument("--butter-high",      type=float, default=5.0)
    g1.add_argument("--butter-order",     type=int,   default=4)
    g1.add_argument("--stab-s",           type=float, default=30.0,
                    help="Stabilization period to skip (s)")
    g1.add_argument("--calib-s",          type=float, default=30.0,
                    help="Calibration window (s)")
    g1.add_argument("--proc-window-s",    type=float, default=30.0,
                    help="Processing window length (s)")

    # ---- Step 2 / 3 pass-throughs ----
    g2 = ap.add_argument_group("Steps 2-3 – S_SQI classification")
    g2.add_argument("--ssqi-mode",        choices=["static","adaptive","both"],
                    default="both")
    g2.add_argument("--sensor-policy",    choices=["worst","best","ir"],
                    default="worst")
    g2.add_argument("--overall-policy",   choices=["best","worst"],
                    default="best")
    g2.add_argument("--seg-s",            type=float, default=5.0,
                    help="SSQI window length (s)")
    g2.add_argument("--hop-s",            type=float, default=1.0,
                    help="SSQI hop (s)")
    g2.add_argument("--polarity",         choices=["as-is","auto","invert"],
                    default="as-is")

    return ap


def main():
    args = build_parser().parse_args()

    csv = Path(args.csv).resolve()
    if not csv.exists():
        sys.exit(f"ERROR: input file not found: {csv}")

    # Default output directory
    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = csv.parent / f"{csv.stem}_pipeline_{stamp}"
    else:
        outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nInput  : {csv}")
    print(f"Method : {args.method}")
    print(f"Outdir : {outdir}")

    # Inject module search paths
    for p in (str(ROOT), str(DUAL_PPG), str(THRESHOLD)):
        if p not in sys.path:
            sys.path.insert(0, p)

    outputs: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # METHOD 1
    # ------------------------------------------------------------------
    if args.method == 1:

        # Step 1
        fusion_csv = step1_preprocess(csv, outdir, args)
        outputs["Step 1 – ppg_fusion_output.csv"] = fusion_csv

        # Step 2a
        ssqi_csv = step2a_ssqi(fusion_csv, outdir, args)
        outputs["Step 2a – ssqi_classified.csv"] = ssqi_csv

        # Step 2b
        fusion_results_csv = step2b_fusion(fusion_csv, outdir)
        outputs["Step 2b – ppg_fusion_results.csv"] = fusion_results_csv

        # Step 3
        ssqi_fused_csv = step3_ssqi_fused(fusion_results_csv, outdir, args)
        outputs["Step 3 – ssqi_classified_fused.csv"] = ssqi_fused_csv

        # Step 4
        drowsiness_csv = step4_drowsiness(fusion_results_csv, outdir)
        outputs["Step 4 – drowsiness_result.csv"] = drowsiness_csv

    print_summary(outdir, outputs)


if __name__ == "__main__":
    main()
