"""
PPG-based threshold drowsiness detection
========================================
Reverse-engineered from: Liu et al., "Development of Real-Time Driver's Health
Detection System by Using Smart Handle", APHM 2017 (Section 3.1, Fig. 8).

Paper's algorithm (Fig. 8)
--------------------------
  1. Acquire PPG signal
  2. Find peaks -> peak-to-peak (PP) intervals
  3. Statistics over the pulses: average PP interval and dispersion (SD)
  4. Is  (average interval > ref value)  AND  (dispersion < ref value) ?
        YES -> drowsiness_count += 1
        NO  -> drowsiness_count -= 1, and if count < 0 -> count = 0
  5. drowsiness_count == 3 ?  YES -> DROWSINESS DETECTED,  NO -> keep acquiring
  (HR = 60 / PP interval)

Physiological idea: drowsy -> heart rate slows (longer PP) and becomes more
regular (lower dispersion).

The paper does NOT give the reference values, so here they are learned from a
short "alert" calibration window at the start of the recording (per driver),
and the thresholds are expressed as ratios of that baseline.

Preprocessing (stages 1-4)
---------------------------
Reuses the same pipeline as TImeDynamic_v2.py:
  Stage 1  dual_process.py       Butterworth + adaptive Savitzky-Golay
  Stage 2  dual_ssqi.py          per-channel S_SQI quality scoring
  Stage 3  dual_fusion.py        IVW / IVW+SSQI sensor fusion
The fused column (IVW_SSQI_IR_Fused or IVW_SSQI_RED_Fused) is then fed into
this file's own detector.

Usage
-----
    python TimeStatic.py recording.csv
    python TimeStatic.py recording.csv --channel RED
    python TimeStatic.py recording.csv --method IVW
    python TimeStatic.py recording.csv --workdir out_dir --reuse
    python TimeStatic.py --from-fusion ppg_fusion_results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

# reuse stages 1-4 and PipelineError from TImeDynamic_v2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import TImeDynamic_v2 as TD  # noqa: E402
from TImeDynamic_v2 import PipelineError  # noqa: E402


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    fs: float = 25.0                 # resampling rate (Hz); data is ~24.9 Hz
    band: tuple = (0.5, 4.0)         # band-pass for PPG (Hz) -> 30-240 bpm
    min_pp_s: float = 0.35           # refractory distance between beats
    max_pp_s: float = 1.50           # PP intervals outside [min,max] are rejected
    prominence_sd: float = 0.30      # peak prominence as multiple of signal SD

    calib_s: float = 20.0            # alert-state calibration window
    win_s: float = 10.0              # analysis window (trailing)
    step_s: float = 2.0              # decision update period

    # Thresholds as ratios of the calibration baseline
    mean_ratio: float = 1.08         # drowsy if mean PP  > 1.08 x ref  (HR >= ~8 % slower)
    disp_ratio: float = 0.90         # drowsy if SD(PP)   < 0.90 x ref  (more regular)
    count_trigger: int = 3           # paper: 3 consecutive-ish hits => drowsy


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float):
    """Timestamps are jittery (and have gaps), so interpolate to a fixed grid."""
    tu = np.arange(t[0], t[-1] - 1e-6, 1.0 / fs)
    return tu, np.interp(tu, t, x)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def detect_peaks(t: np.ndarray, x: np.ndarray, cfg: Config):
    """Return peak times (s) of the pulse waveform."""
    idx, _ = find_peaks(
        x,
        distance=max(1, int(cfg.min_pp_s * cfg.fs)),
        prominence=cfg.prominence_sd * np.std(x),
    )
    # Sub-sample refinement (parabolic interpolation). At ~25 Hz one sample is
    # 40 ms, which is the same order as the PP dispersion we want to measure,
    # so without this the dispersion statistic is dominated by quantisation.
    idx = idx[(idx > 0) & (idx < len(x) - 1)]
    y0, y1, y2 = x[idx - 1], x[idx], x[idx + 1]
    denom = y0 - 2 * y1 + y2
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
    return t[idx] + np.clip(delta, -1, 1) / cfg.fs


def pp_intervals(peak_t: np.ndarray, cfg: Config):
    """PP intervals stamped at the time of the *second* peak; physiologically
    implausible intervals (missed / double-detected beats) are dropped."""
    pp = np.diff(peak_t)
    t_pp = peak_t[1:]
    ok = (pp >= cfg.min_pp_s) & (pp <= cfg.max_pp_s)
    return t_pp[ok], pp[ok]


# --------------------------------------------------------------------------- #
# Drowsiness state machine (Fig. 8)
# --------------------------------------------------------------------------- #
@dataclass
class DrowsinessDetector:
    cfg: Config
    ref_mean: float = np.nan
    ref_disp: float = np.nan
    count: int = 0
    log: list = field(default_factory=list)

    def calibrate(self, pp: np.ndarray):
        self.ref_mean = float(np.mean(pp))
        self.ref_disp = float(np.std(pp, ddof=1))

    def update(self, t_now: float, pp_win: np.ndarray):
        """One pass through the Fig. 8 flow chart. Returns (flag, drowsy_now)."""
        if len(pp_win) < 4:                       # not enough pulses to judge
            return None
        mean_pp = float(np.mean(pp_win))
        disp = float(np.std(pp_win, ddof=1))

        cond_mean = mean_pp > self.cfg.mean_ratio * self.ref_mean
        cond_disp = disp < self.cfg.disp_ratio * self.ref_disp
        hit = cond_mean and cond_disp

        if hit:
            self.count += 1                       # "increase drowsiness count"
        else:
            self.count -= 1                       # "decrease drowsiness count"
            if self.count < 0:                    # "if count<0 then count=0"
                self.count = 0

        drowsy = self.count >= self.cfg.count_trigger   # "count = 3 ?"
        self.log.append(dict(
            t=t_now, mean_pp=mean_pp, hr_bpm=60.0 / mean_pp, disp=disp,
            cond_mean=cond_mean, cond_disp=cond_disp, hit=hit,
            count=self.count, drowsy=drowsy,
        ))
        return drowsy


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(t_raw, x_raw, cfg: Config | None = None):
    cfg = cfg or Config()
    t, x = resample_uniform(np.asarray(t_raw), np.asarray(x_raw), cfg.fs)
    xf = bandpass(x - np.mean(x), cfg.fs, *cfg.band)

    peak_t = detect_peaks(t, xf, cfg)
    t_pp, pp = pp_intervals(peak_t, cfg)

    det = DrowsinessDetector(cfg)
    t0 = t[0]
    calib_mask = t_pp < t0 + cfg.calib_s
    if calib_mask.sum() < 8:
        raise ValueError("Not enough beats in the calibration window.")
    det.calibrate(pp[calib_mask])

    # Slide the analysis window over the recording after calibration
    t_eval = np.arange(t0 + cfg.calib_s + cfg.win_s, t[-1], cfg.step_s)
    for te in t_eval:
        m = (t_pp > te - cfg.win_s) & (t_pp <= te)
        det.update(te, pp[m])

    res = pd.DataFrame(det.log)
    return dict(t=t, x=x, xf=xf, peak_t=peak_t, t_pp=t_pp, pp=pp,
                det=det, result=res)


def load_fusion_csv(fusion_csv: Path, channel: str = "auto", method: str = "IVW_SSQI"):
    """Load elapsed time + the chosen fused column from a dual_fusion.py output file."""
    df = pd.read_csv(fusion_csv)
    df.columns = df.columns.str.strip()
    col_map = {
        "IR":  f"{method}_IR_Fused",
        "RED": f"{method}_RED_Fused",
    }
    if channel == "auto":
        # pick whichever fused column is present; prefer IR
        for ch in ("IR", "RED"):
            col = col_map[ch]
            if col in df.columns:
                chosen_col = col
                break
        else:
            raise PipelineError(
                f"No fused column found in {fusion_csv}. Columns: {list(df.columns)}"
            )
    else:
        chosen_col = col_map[channel]
        if chosen_col not in df.columns:
            raise PipelineError(
                f"Column '{chosen_col}' not found in {fusion_csv}. "
                f"Available: {list(df.columns)}"
            )
    return df["Elapsed_s"].values, df[chosen_col].values, chosen_col


def analyze_fusion_csv(fusion_csv, channel="auto", method="IVW_SSQI",
                       cfg: Config | None = None, plot=None):
    """Stages 4-5: run detector on an existing fusion CSV."""
    cfg = cfg or Config()
    t_raw, x_raw, col = load_fusion_csv(Path(fusion_csv), channel, method)
    out = run_pipeline(t_raw, x_raw, cfg)
    d = out["det"]
    result = out["result"]

    print(f"Signal column      : {col}")
    print(f"Reference (alert)  : mean PP = {d.ref_mean:.3f} s "
          f"({60/d.ref_mean:.1f} bpm),  SD = {d.ref_disp*1000:.1f} ms")
    print(result.round(3).to_string(index=False))
    print("\nDrowsiness detected:", bool(result["drowsy"].any()))

    if plot:
        _plot(out, plot)
    return out


def analyze(raw_csv, workdir=None, scripts_dir=None, channel="auto", method="IVW_SSQI",
            reuse=False, cfg: Config | None = None, plot=None, log=print):
    """Full pipeline: raw CSV -> stages 1-3 (via TImeDynamic_v2) -> detector."""
    raw_csv = Path(raw_csv)
    scripts_dir = (Path(scripts_dir) if scripts_dir
                   else Path(__file__).resolve().parent.parent)
    workdir = Path(workdir) if workdir else Path(f"{raw_csv.stem}_pipeline")

    fusion_csv = TD.run_stages_1_to_3(raw_csv, workdir, scripts_dir, reuse, log)
    log("[4/5] detector (TimeStatic threshold model)")
    return analyze_fusion_csv(fusion_csv, channel, method, cfg, plot)


def _plot(out: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, xf = out["t"], out["xf"]
    pk_t = out["peak_t"]
    pk_y = np.interp(pk_t, t, xf)
    res = out["result"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(t, xf, lw=0.7, label="fused PPG (filtered)")
    axes[0].plot(pk_t, pk_y, "g.", ms=4, label="detected beat")
    axes[0].set_ylabel("amplitude"); axes[0].legend(fontsize=7)

    axes[1].plot(res["t"], res["hr_bpm"], "k-", label="window HR (bpm)")
    axes[1].axhline(60 / out["det"].ref_mean, color="g", ls="--", label="ref HR")
    axes[1].set_ylabel("bpm"); axes[1].legend(fontsize=7)

    axes[2].step(res["t"], res["drowsy"].astype(int), where="post", color="red")
    axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["ALERT", "DROWSY"])
    axes[2].set_xlabel("time (s)")

    for ax in axes:
        for _, row in res[res["drowsy"]].iterrows():
            ax.axvspan(row["t"] - 2, row["t"], color="red", alpha=0.12, lw=0)

    title = "DROWSY" if bool(res["drowsy"].any()) else "ALERT"
    fig.suptitle(f"TimeStatic threshold detector — {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TimeStatic: Liu et al. threshold detector on dual-PPG pipeline"
    )
    ap.add_argument("csv", nargs="?", help="raw recording CSV")
    ap.add_argument("--from-fusion",
                    help="skip stages 1-3: provide dual_fusion.py output directly")
    ap.add_argument("--workdir",
                    help="folder for pipeline intermediates (default: <csv stem>_pipeline)")
    ap.add_argument("--scripts-dir",
                    help="folder holding dual_process.py etc. (default: MainProject root)")
    ap.add_argument("--channel", choices=["auto", "IR", "RED"], default="auto")
    ap.add_argument("--method", choices=["IVW_SSQI", "IVW"], default="IVW_SSQI")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse stage outputs already in workdir")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    if not a.csv and not a.from_fusion:
        ap.error("provide a raw CSV or --from-fusion")

    try:
        if a.from_fusion:
            src = Path(a.from_fusion)
            out_dir = Path(a.workdir) if a.workdir else src.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = src.stem
            plot_path = None if a.no_plot else str(out_dir / f"{stem}_static_report.png")
            result = analyze_fusion_csv(src, a.channel, a.method, None, plot_path)
        else:
            src = Path(a.csv)
            out_dir = Path(a.workdir) if a.workdir else Path(f"{src.stem}_pipeline")
            stem = src.stem
            plot_path = None if a.no_plot else str(out_dir / f"{stem}_static_report.png")
            result = analyze(src, out_dir, a.scripts_dir, a.channel, a.method,
                             a.reuse, None, plot_path)

        res_df = result["result"]
        out_dir.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_dir / f"{stem}_static_windows.csv", index=False)
        summary = {
            "drowsy_detected": bool(res_df["drowsy"].any()),
            "prediction": "DROWSY" if bool(res_df["drowsy"].any()) else "ALERT",
            "ref_mean_pp_s": round(float(result["det"].ref_mean), 4),
            "ref_hr_bpm": round(60 / float(result["det"].ref_mean), 1),
            "ref_disp_ms": round(float(result["det"].ref_disp) * 1000, 2),
        }
        (out_dir / f"{stem}_static_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        print(f"\nsaved to: {out_dir}")

    except PipelineError as e:
        sys.exit(f"PIPELINE ERROR: {e}")
    except ValueError as e:
        sys.exit(f"ERROR: {e}")