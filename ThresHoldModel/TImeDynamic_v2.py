"""
TimeDynamic.py - STAGE 5 of the dual-PPG drowsiness pipeline
=============================================================

Pipeline (each stage is YOUR script, run unmodified):

    raw CSV (Clock Time, Elapsed_s, Device Timestamp, IR1, Red1, IR2, Red2)
      |
      | 1  dual_process.py   Butterworth + adaptive Savitzky-Golay, per channel
      |                      -> ppg_results_sg/ppg_fusion_output.csv
      | 2  dual_ssqi.py      per-channel skewness quality (S_SQI) + hard gates
      |                      -> ssqi_dual/ssqi_classified.csv, ssqi_channel_details.csv
      |    (bridge)          re-writes stage-2 SSQI as ssqi_dual/ssqi_<CH>/ssqi_classified.csv,
      |                      the layout dual_fusion.py is written to read
      | 3  dual_fusion.py    IVW and IVW+SSQI fusion of sensor 1 + sensor 2
      |                      -> ppg_fusion_ssqi_results/ppg_fusion_results.csv
      | 4  step2_ssqi_adaptive_v2.py   S_SQI quality of the FUSED signal
      |                      (called as a library: its own CLI cannot take a fused column)
      | 5  THIS FILE         beat detection on the fused signal, quality-gated by stage 4,
      |                      baseline-relative CUSUM threshold detector -> drowsiness state
      v
    ALERT / DROWSY / INSUFFICIENT_DATA  (+ per-window states, events, plot, CSVs)

Usage
-----
    python TimeDynamic.py recording.csv                       # runs stages 1-5
    python TimeDynamic.py recording.csv --workdir out_run1    # keep all intermediates there
    python TimeDynamic.py --from-fusion out_run1/ppg_fusion_ssqi_results/ppg_fusion_results.csv
                                                               # you already ran stages 1-3 yourself
    python TimeDynamic.py recording.csv --channel RED --method IVW

    from TimeDynamic import analyze, analyze_fusion_csv
    rep = analyze("recording.csv")        # rep.summary, rep.events, rep.windows, rep.info

The four stage scripts must sit next to this file (or pass --scripts-dir). Their file
names must be exactly: dual_process.py, dual_ssqi.py, dual_fusion.py,
step2_ssqi_adaptive_v2.py

Detector (stage 5)
------------------
  * The signal is already filtered and fused, so no channel guessing, no re-fusion and
    no polarity heuristics on raw columns. The channel (IR or RED fused) is chosen by
    stage-4 quality unless forced.
  * Beats come from the fused waveform (adaptive refractory + amplitude gate +
    morphology check). An inter-beat interval is only kept if stage 4 says the fused
    signal was usable at both of its ends.
  * Quality rule: stage-4 hard gates always apply. Stage 4's S_SQI "bad" label is used
    only if this person's own skewness is in the range the static cuts were designed
    for; for a person whose clean pulse has a very different skewness (e.g. the
    "nithish" recording, median S_SQI ~ -0.45) a person-relative cut is used instead,
    otherwise 95 % of a clean recording would be thrown away.
  * Evidence = weighted z-score of (heart-rate slowing, HRV drop) against this
    person's own baseline -> one-sided CUSUM -> alarm level that rises with the
    person's own alert-state noise -> hysteresis.
  * Baseline: for the first settle_s (180 s) it is the running MEDIAN of this person's own
    windows (heart rate typically settles for a few minutes after sensor placement; a 30 s
    baseline caused a false alarm on "nithish"). Afterwards it moves slowly (tau_s = 600 s)
    and only while the person looks alert, so a gradual slide into drowsiness is not absorbed.
  * HRV evidence is two-sided (|change in log-RMSSD|): in "drowsy_athish" RMSSD roughly doubles
    when drowsy, the opposite of the "variability drop" the earlier version assumed.
    Set Priors.hrv_sign = -1 / +1 to force a direction.
  * The alarm level is learned from alert-state noise but only from LOW evidence, and the CUSUM
    is capped at 2x the alarm level so an alarm can clear after the person recovers.
  * If a recording is too short or too noisy to learn a baseline the result is
    INSUFFICIENT_DATA instead of an exception. Minimum usable length is about 95 s
    (dual_process.py discards the first 60 s, the detector needs 30 s more).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import skew

STAGE_SCRIPTS = {
    "process": "DualPPG\\dual_process.py",
    "ssqi": "DualPPG\\dual_ssqi.py",
    "fusion": "DualPPG\\dual_fusion.py",
    "step2": "step2_ssqi_adaptive_v2.py",
}
SENSOR_CHANNELS = ("IR1", "RED1", "IR2", "RED2")
STATIC_BAD_CUT = -0.15          # S_SQI value step2/dual_ssqi call "bad" (paper-derived heuristic)


# =========================================================================== #
# Priors (physiology + safety floors) - NOT per-recording settings
# =========================================================================== #
@dataclass(frozen=True)
class Priors:
    hr_range_bpm: tuple = (35, 190)
    max_resample_hz: float = 100.0
    band_lo: float = 0.5
    band_hi_cap: float = 4.0
    refractory_frac: float = 0.65
    amp_gate: float = 0.55
    max_dev: float = 0.30
    min_good_frac: float = 0.70
    min_beats: int = 6
    warmup_min_s: float = 30.0
    warmup_min_beats: int = 25
    settle_s: float = 180.0               # baseline = running MEDIAN of valid windows for this long
    hrv_sign: int = 0                     # HRV evidence: -1 = drop, +1 = rise, 0 = either direction
    win_beats: float = 12.0
    win_clip_s: tuple = (8.0, 20.0)
    tau_s: float = 600.0                  # baseline time constant (was 120: absorbed slow drowsy drift)
    floor_cv_mean: float = 0.03           # min sigma of window mean-PP (3 % of baseline)
    floor_lr: float = 0.20
    freeze_above: float = 1.0
    z_clip: tuple = (-3.0, 6.0)
    w_hr: float = 0.7
    w_hrv: float = 0.3
    k: float = 1.0                        # CUSUM slack
    h_floor: float = 25.0                 # alarm-level safety floor
    h_mult: float = 3.0
    h_off_frac: float = 0.3
    suspect_frac: float = 0.5
    no_signal_s: float = 30.0
    s_cap_mult: float = 2.0               # CUSUM is capped at this x alarm level so an alarm can clear after recovery
    # quality integration (stage 4)
    q_min_cover: float = 0.5              # share of stage-4 windows covering a beat that must be usable
    rel_sigma_k: float = 3.0              # person-relative S_SQI cut = median - k * robust sigma
    rel_sigma_floor: float = 0.05
    # final decision
    min_alarm_s: float = 20.0             # alarm must last this long to count as DROWSY
    min_valid_pct: float = 40.0           # below this the recording is INSUFFICIENT_DATA


P = Priors()


# =========================================================================== #
# Pipeline stages 1-4
# =========================================================================== #
class PipelineError(RuntimeError):
    pass


def _run_stage(name: str, script: Path, args: list, cwd: Path) -> None:
    if not script.is_file():
        raise PipelineError(f"[{name}] script not found: {script} "
                            f"(expected file name '{script.name}'; use --scripts-dir)")
    env = dict(os.environ, MPLBACKEND="Agg")       # dual_fusion calls plt.show(block=True)
    r = subprocess.run([sys.executable, str(script), *args], cwd=str(cwd), env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-12:])
        raise PipelineError(f"[{name}] {script.name} failed (exit {r.returncode}):\n{tail}")


def _bridge_ssqi(workdir: Path) -> None:
    """dual_ssqi.py writes one wide file; dual_fusion.py looks for
    ssqi_dual/ssqi_<CH>/ssqi_classified.csv with columns seg_start_s, seg_end_s, SSQI.
    Re-slice the stage-2 long-format details into that layout."""
    det = workdir / "ssqi_dual" / "ssqi_channel_details.csv"
    if not det.is_file():
        raise PipelineError(f"[bridge] {det} missing - did dual_ssqi.py run?")
    d = pd.read_csv(det)
    for ch in SENSOR_CHANNELS:
        sub = d[d["channel"] == ch][["seg_start_s", "seg_end_s", "SSQI", "gate", "label_final"]]
        if sub.empty:
            continue
        out = workdir / "ssqi_dual" / f"ssqi_{ch}"
        out.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out / "ssqi_classified.csv", index=False)


def run_stages_1_to_3(raw_csv: Path, workdir: Path, scripts_dir: Path, reuse: bool, log=print) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    p1 = workdir / "ppg_results_sg" / "ppg_fusion_output.csv"
    p2 = workdir / "ssqi_dual" / "ssqi_channel_details.csv"
    p3 = workdir / "ppg_fusion_ssqi_results" / "ppg_fusion_results.csv"

    if not (reuse and p1.is_file()):
        log("[1/5] dual_process.py   (Butterworth + adaptive SG)")
        _run_stage("1 process", scripts_dir / STAGE_SCRIPTS["process"],
                   ["--csv", str(raw_csv.resolve()), "--outdir", "ppg_results_sg"], workdir)
    if not p1.is_file():
        raise PipelineError("[1 process] no output. The recording is probably shorter than "
                            "60 s stabilisation/calibration + one 30 s window (needs >= ~92 s).")

    if not (reuse and p2.is_file()):
        log("[2/5] dual_ssqi.py      (per-channel S_SQI)")
        _run_stage("2 ssqi", scripts_dir / STAGE_SCRIPTS["ssqi"],
                   ["--csv", "ppg_results_sg/ppg_fusion_output.csv", "--outdir", "ssqi_dual"], workdir)
    _bridge_ssqi(workdir)

    if not (reuse and p3.is_file()):
        log("[3/5] dual_fusion.py    (IVW / IVW+SSQI)")
        _run_stage("3 fusion", scripts_dir / STAGE_SCRIPTS["fusion"], [], workdir)
    if not p3.is_file():
        raise PipelineError(f"[3 fusion] expected output missing: {p3}")
    return p3


def _load_step2(scripts_dir: Path):
    path = scripts_dir / STAGE_SCRIPTS["step2"]
    if not path.is_file():
        raise PipelineError(f"[4 step2] script not found: {path}")
    spec = importlib.util.spec_from_file_location("step2_ssqi_adaptive_v2", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step2_ssqi_adaptive_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


def usable_mask(q: pd.DataFrame, p: Priors = P):
    """Which stage-4 windows may feed the detector. Returns (usable, mode, info)."""
    ok = (q["gate"] == "ok") & np.isfinite(q["SSQI"])
    info = {"ssqi_median": np.nan, "quality_mode": "no-clean-windows"}
    if ok.sum() < 5:
        return pd.Series(False, index=q.index), "no-clean-windows", info
    s = q.loc[ok, "SSQI"].to_numpy(float)
    med = float(np.median(s))
    sig = max(1.4826 * float(np.median(np.abs(s - med))), p.rel_sigma_floor)
    info["ssqi_median"] = med
    if med >= STATIC_BAD_CUT:
        mode = "step2-labels"                       # person is inside the range the cuts were made for
        usable = ok & (q["label_final"] != "bad")
    else:
        mode = "person-relative"                    # atypical morphology: static S_SQI cuts not meaningful
        usable = ok & (q["SSQI"] >= med - p.rel_sigma_k * sig)
    info["quality_mode"] = mode
    return usable, mode, info


def step4_quality(fusion_df: pd.DataFrame, columns: dict, scripts_dir: Path, p: Priors = P):
    """Run step2's classifier (as a library) on each fused column."""
    step2 = _load_step2(scripts_dir)
    args = step2.build_parser().parse_args([])
    t = fusion_df["Elapsed_s"].to_numpy(float)
    out = {}
    for wl, col in columns.items():
        q, info = step2.run_pipeline(t, fusion_df[col].to_numpy(float), args, verbose=False)
        if q.empty:
            out[wl] = dict(q=q, usable=pd.Series(dtype=bool), frac=0.0, info={"quality_mode": "empty"}, col=col)
            continue
        usable, mode, qinfo = usable_mask(q, p)
        qinfo["gate_counts"] = q["gate"].value_counts().to_dict()
        qinfo["label_counts"] = q["label_final"].value_counts().to_dict()
        out[wl] = dict(q=q, usable=usable, frac=float(usable.mean()), info=qinfo, col=col)
    return out


def quality_lookup(q: pd.DataFrame, usable: pd.Series):
    """f(times) -> share of stage-4 windows covering each time that are usable (NaN = uncovered)."""
    starts = q["seg_start_s"].to_numpy(float)
    seg = float((q["seg_end_s"] - q["seg_start_s"]).iloc[0])
    cs = np.concatenate([[0.0], np.cumsum(usable.to_numpy(float))])

    def f(tt):
        tt = np.asarray(tt, float)
        i0 = np.searchsorted(starts, tt - seg + 1e-9, "right")
        i1 = np.searchsorted(starts, tt + 1e-9, "right")
        n = i1 - i0
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, (cs[i1] - cs[i0]) / np.maximum(n, 1), np.nan)
    return f


# =========================================================================== #
# Beats (stage 5a)
# =========================================================================== #
def _uniformize(t, x, fs):
    tu = np.arange(t[0], t[-1] - 1e-9, 1.0 / fs)
    return tu, np.interp(tu, t, x)


def _bandpass(x, fs, p: Priors = P):
    hi = min(p.band_hi_cap, 0.4 * fs)
    sos = butter(3, [p.band_lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x - np.mean(x))


def _refine_peaks(pk_t, pk_h, p: Priors = P):
    pk_t, pk_h = np.asarray(pk_t, float), np.asarray(pk_h, float)
    if len(pk_t) > 5:
        ref = pd.Series(pk_h).rolling(13, center=True, min_periods=3).quantile(0.8).values
        keep = pk_h >= p.amp_gate * ref
        pk_t, pk_h = pk_t[keep], pk_h[keep]
    for _ in range(12):
        if len(pk_t) < 4:
            break
        g = np.diff(pk_t)
        med = pd.Series(g).rolling(7, center=True, min_periods=2).median().values
        bad = np.where(g < p.refractory_frac * med)[0]
        if len(bad) == 0:
            break
        drop, used = set(), set()
        for i in bad:
            if i in used or i + 1 in used:
                continue
            drop.add(i if pk_h[i] < pk_h[i + 1] else i + 1)
            used.update((i, i + 1))
        m = np.ones(len(pk_t), bool)
        m[list(drop)] = False
        pk_t, pk_h = pk_t[m], pk_h[m]
    return pk_t, pk_h


def _detect(t, xf, fs, p: Priors = P):
    idx, _ = find_peaks(xf, distance=max(1, int(0.30 * fs)), prominence=0.25 * np.std(xf))
    idx = idx[(idx > 0) & (idx < len(xf) - 1)]
    y0, y1, y2 = xf[idx - 1], xf[idx], xf[idx + 1]
    den = y0 - 2 * y1 + y2
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    pk_t = t[idx] + np.clip(delta, -1, 1) / fs
    return _refine_peaks(pk_t, xf[idx], p)


def _beat_quality(t, xf, fs, pk_t):
    h = int(0.30 * fs)
    ii = np.searchsorted(t, pk_t)
    ok = (ii - h >= 0) & (ii + h + 1 <= len(xf))
    seg = np.full((len(ii), 2 * h + 1), np.nan)
    for j in np.where(ok)[0]:
        s = xf[ii[j] - h: ii[j] + h + 1]
        seg[j] = (s - s.mean()) / (s.std() + 1e-12)
    q = np.zeros(len(ii))
    for j in range(len(ii)):
        if not ok[j]:
            continue
        nb = [n for n in range(max(0, j - 7), min(len(ii), j + 8)) if n != j and ok[n]]
        if len(nb) < 3:
            q[j] = 1.0
            continue
        tpl = np.median(seg[nb], axis=0)
        tpl = (tpl - tpl.mean()) / (tpl.std() + 1e-12)
        q[j] = float(np.mean(seg[j] * tpl))
    return q


def _regularity(pk_t):
    if len(pk_t) < 8:
        return np.inf
    pp = np.diff(pk_t)
    return float(np.median(np.abs(np.diff(pp))) / np.median(pp))


def extract_beats(t_raw, x_raw, fs, qual_fn=None, p: Priors = P):
    """Beats from the (already filtered + fused) waveform. Inter-beat intervals are kept only
    if both beats have normal morphology, the interval is physiological, and stage 4 judged
    the signal usable at both ends. (Intervals are identical for either polarity; polarity only
    decides whether crests or troughs are used as fiducials.)"""
    t, x = _uniformize(t_raw, x_raw, fs)
    gaps = [(t_raw[i], t_raw[i + 1]) for i in np.where(np.diff(t_raw) > max(0.30, 6 * np.median(np.diff(t_raw))))[0]]
    xf = _bandpass(x, fs, p)
    sk = float(skew(xf))
    if abs(sk) > 0.1:
        sgn = 1 if sk > 0 else -1
        pk_t, pk_h = _detect(t, sgn * xf, fs, p)
    else:
        best = None
        for sgn in (1, -1):
            a, b = _detect(t, sgn * xf, fs, p)
            sc = _regularity(a)
            if best is None or sc < best[0] * 0.9:
                best = (sc, sgn, a, b)
        _, sgn, pk_t, pk_h = best
    xf = sgn * xf
    mq = _beat_quality(t, xf, fs, pk_t)
    thr = float(np.clip(np.median(mq) - 4 * 1.4826 * np.median(np.abs(mq - np.median(mq))), 0.5, 0.75))
    good = mq >= thr
    qv = qual_fn(pk_t) if qual_fn is not None else np.ones(len(pk_t))
    q_ok = np.where(np.isnan(qv), False, qv >= p.q_min_cover)

    lo, hi = 60 / p.hr_range_bpm[1], 60 / p.hr_range_bpm[0]
    t_pp, pp = [], []
    for i in range(1, len(pk_t)):
        d = pk_t[i] - pk_t[i - 1]
        if not (good[i] and good[i - 1] and q_ok[i] and q_ok[i - 1] and lo <= d <= hi):
            continue
        if any(g0 < pk_t[i] and g1 > pk_t[i - 1] for g0, g1 in gaps):
            continue
        t_pp.append(pk_t[i])
        pp.append(d)
    info = dict(polarity="inverted" if sgn < 0 else "normal", n_beats=len(pk_t),
                beats_rejected_morphology=int((~good).sum()),
                beats_rejected_quality=int((~q_ok).sum()), n_intervals_used=len(pp))
    return dict(t=t, xf=xf, pk_t=pk_t, good=good, q_ok=q_ok, t_pp=np.array(t_pp), pp=np.array(pp), info=info)


# =========================================================================== #
# Window features (stage 5b)
# =========================================================================== #
def window_features(t_pp, pp, te, win, p: Priors = P):
    m = (t_pp > te - win) & (t_pp <= te)
    q, tt = pp[m], t_pp[m]
    out = dict(valid=False, mean=np.nan, lrmssd=np.nan, good_frac=0.0, n=int(len(q)))
    if len(q) < 4:
        return out
    med = np.median(q)
    keep = np.abs(q - med) <= p.max_dev * med
    out["good_frac"] = float(keep.sum() / (win / med))
    if keep.sum() < p.min_beats or out["good_frac"] < p.min_good_frac:
        return out
    out["valid"] = True
    out["mean"] = float(np.mean(q[keep]))
    idx = np.where(keep)[0]
    a, b = idx[:-1], idx[1:]
    consec = (b - a == 1) & (np.abs((tt[b] - tt[a]) - q[b]) < 0.15 * q[b])
    d = (q[b] - q[a])[consec]
    if len(d) >= 4:
        out["lrmssd"] = float(np.log(max(np.sqrt(np.mean(d ** 2)), 1e-3)))
    return out


def _calib_stats(t_pp, pp, t_lo, t_hi):
    m = (t_pp >= t_lo) & (t_pp < t_hi)
    q, tt = pp[m], t_pp[m]
    med = np.median(q)
    keep = np.abs(q - med) <= P.max_dev * med
    idx = np.where(keep)[0]
    a, b = idx[:-1], idx[1:]
    consec = (b - a == 1) & (np.abs((tt[b] - tt[a]) - q[b]) < 0.15 * q[b])
    d = (q[b] - q[a])[consec]
    lr = float(np.log(np.sqrt(np.mean(d ** 2)))) if len(d) >= 4 else np.nan
    return float(np.mean(q[keep])), lr, int(keep.sum())


# =========================================================================== #
# Adaptive threshold detector (stage 5c)
# =========================================================================== #
def run_detector(beats: dict, fs: float, t0: float, t_end: float, p: Priors = P):
    t_pp, pp = beats["t_pp"], beats["pp"]
    info = {}
    t_warm = None
    for te in np.arange(t0 + p.warmup_min_s, t_end, 1.0):
        if ((t_pp >= t0) & (t_pp < te)).sum() >= p.warmup_min_beats:
            t_warm = te
            break
    if t_warm is None or (t_pp < t_warm).sum() < 8:
        return None, {"error": "too short / too few clean beats to learn a baseline "
                               f"(need >= {p.warmup_min_s:.0f} s with >= {p.warmup_min_beats} clean beats)"}, None
    b_m, b_lr, n_cal = _calib_stats(t_pp, pp, t0, t_warm)
    if np.isnan(b_lr):
        b_lr = np.log(0.03)
    win = float(np.clip(p.win_beats * b_m, *p.win_clip_s))
    step = max(1.0, round(win / 5, 1))
    info.update(warmup_s=round(t_warm - t0, 1), window_s=win, step_s=step,
                baseline_hr_bpm=round(60 / b_m, 1), baseline_rmssd_ms=round(1000 * np.exp(b_lr), 1))

    floor_m = p.floor_cv_mean * b_m
    var_m = floor_m ** 2
    var_lr = p.floor_lr ** 2
    alpha = step / p.tau_s
    S, alarm, M = 0.0, False, 0.0
    last_valid_t = t_warm
    settle_m, settle_lr = [], []          # window statistics gathered while the baseline is still being learned
    settled = False
    rows = []
    for te in np.arange(t_warm, t_end, step):
        f = window_features(t_pp, pp, te, win, p)
        h_on = max(p.h_floor, p.h_mult * M)
        if not f["valid"]:
            state = "NO_SIGNAL" if te - last_valid_t > p.no_signal_s else ("DROWSY" if alarm else "HOLD")
            rows.append(dict(t=te, valid=False, state=state, S=S, alarm=alarm, h_on=h_on,
                             b_hr=60 / b_m, hr_bpm=np.nan, score=np.nan, z_hr=np.nan, z_hrv=np.nan,
                             rmssd_ms=np.nan))
            continue
        last_valid_t = te

        # ---- baseline learning phase: running median of this person's own windows
        learning = (te - t0) <= p.settle_s
        if learning:
            settle_m.append(f["mean"])
            if not np.isnan(f["lrmssd"]):
                settle_lr.append(f["lrmssd"])
            b_m = float(np.median(settle_m))
            if len(settle_lr) >= 3:
                b_lr = float(np.median(settle_lr))
            if len(settle_m) >= 5:
                var_m = max(floor_m ** 2, (1.4826 * np.median(np.abs(np.array(settle_m) - b_m))) ** 2)
            if len(settle_lr) >= 5:
                var_lr = max(p.floor_lr ** 2,
                             (1.4826 * np.median(np.abs(np.array(settle_lr) - b_lr))) ** 2)
        elif not settled:
            settled = True

        hrv_ok = bool(np.exp(b_lr) > 0.5 / fs)       # RMSSD must exceed the timing-resolution floor
        w_hr, w_hrv = (p.w_hr, p.w_hrv) if hrv_ok else (1.0, 0.0)
        s_m = max(np.sqrt(var_m), floor_m)
        s_lr = max(np.sqrt(var_lr), p.floor_lr)
        z_hr = float(np.clip((f["mean"] - b_m) / s_m, *p.z_clip))
        if np.isnan(f["lrmssd"]) or not hrv_ok:
            z_hrv = 0.0
        else:
            dz = (f["lrmssd"] - b_lr) / s_lr
            zz = abs(dz) if p.hrv_sign == 0 else (dz if p.hrv_sign > 0 else -dz)
            z_hrv = float(np.clip(zz, *p.z_clip))
        score = w_hr * z_hr + w_hrv * z_hrv
        S = min(max(0.0, S + score - p.k), p.s_cap_mult * h_on)
        if not alarm and S >= h_on:
            alarm = True
        elif alarm and S <= h_on * p.h_off_frac:
            alarm = False
        looks_alert = (not alarm) and score < p.freeze_above and S < p.suspect_frac * h_on
        # alert-state noise ceiling M sets the alarm level. It may only learn from LOW evidence
        # (S < suspect_frac * h_floor); otherwise a slow slide into drowsiness raises its own
        # alarm level and is never flagged (observed in the original on the 'drowsy' recording).
        if looks_alert and S < p.suspect_frac * p.h_floor:
            M = max(M, S)
        if looks_alert and not learning:
            var_m = (1 - alpha) * var_m + alpha * (f["mean"] - b_m) ** 2
            b_m += alpha * (f["mean"] - b_m)
            if not np.isnan(f["lrmssd"]) and hrv_ok:
                var_lr = (1 - alpha) * var_lr + alpha * (f["lrmssd"] - b_lr) ** 2
                b_lr += alpha * (f["lrmssd"] - b_lr)
        state = "DROWSY" if alarm else ("SUSPECT" if S >= p.suspect_frac * h_on else "ALERT")
        rows.append(dict(t=te, valid=True, state=state, S=S, alarm=alarm, h_on=h_on,
                         b_hr=60 / b_m, hr_bpm=60 / f["mean"], score=score, z_hr=z_hr, z_hrv=z_hrv,
                         rmssd_ms=1000 * np.exp(f["lrmssd"]) if not np.isnan(f["lrmssd"]) else np.nan))
    info.update(hrv_used=bool(np.exp(b_lr) > 0.5 / fs), baseline_hr_bpm=round(60 / b_m, 1),
                baseline_rmssd_ms=round(1000 * np.exp(b_lr), 1))
    return pd.DataFrame(rows), info, t_warm


# =========================================================================== #
# Report
# =========================================================================== #
@dataclass
class Report:
    windows: pd.DataFrame
    events: pd.DataFrame
    summary: dict
    info: dict
    beats: dict = field(repr=False, default=None)
    quality: dict = field(repr=False, default=None)

    def text(self) -> str:
        s, i = self.summary, self.info
        L = ["=" * 66, "DUAL-PPG DROWSINESS REPORT", "=" * 66,
             f"  prediction          : {s['prediction']}" + (f"   ({s['reason']})" if s.get('reason') else ""),
             f"  signal              : {i.get('signal_column')}   (stage-4 usable: "
             + ", ".join(f"{k} {v:.0%}" for k, v in i.get('channel_usable_frac', {}).items()) + ")",
             f"  quality rule        : {i.get('quality_mode')}  (median S_SQI of clean windows "
             f"{i.get('ssqi_median', float('nan')):+.2f})",
             f"  stage-4 gates       : {i.get('gate_counts')}",
             f"  beats               : {i.get('n_beats')}  intervals used {i.get('n_intervals_used')}  "
             f"(rejected: morphology {i.get('beats_rejected_morphology')}, quality {i.get('beats_rejected_quality')})"]
        if "baseline_hr_bpm" in i:
            L += [f"  baseline            : {i['baseline_hr_bpm']} bpm, RMSSD {i['baseline_rmssd_ms']} ms "
                  f"(HRV used: {i['hrv_used']}), warm-up {i['warmup_s']} s, window {i['window_s']:.0f} s",
                  f"  windows evaluated   : {s['n_windows']} (usable {s['pct_valid']:.0f} %)",
                  f"  time in states      : " + ", ".join(f"{k} {v:.0f} %" for k, v in s['state_pct'].items()),
                  f"  peak evidence       : {s['max_S']:.1f}  vs alarm level {s['final_alarm_level']:.1f}"]
            if s.get("first_alarm_s") is not None:
                L.append(f"  first alarm         : t = {s['first_alarm_s']:.0f} s")
        if len(self.events):
            L += ["", "Drowsiness events", self.events.round(1).to_string(index=False)]
        return "\n".join(L)


def _events(win_df: pd.DataFrame) -> pd.DataFrame:
    ev, start = [], None
    for _, r in win_df.iterrows():
        if r.alarm and start is None:
            start = r.t
        if (not r.alarm) and start is not None:
            ev.append((start, r.t))
            start = None
    if start is not None:
        ev.append((start, win_df.t.iloc[-1]))
    rows = []
    for a, z in ev:
        seg = win_df[(win_df.t >= a) & (win_df.t <= z) & win_df.valid]
        rows.append(dict(start_s=a, end_s=z, duration_s=z - a,
                         hr_change_pct=100 * (seg.hr_bpm.mean() / seg.b_hr.mean() - 1) if len(seg) else np.nan,
                         rmssd_ms=seg.rmssd_ms.mean() if len(seg) else np.nan))
    return pd.DataFrame(rows, columns=["start_s", "end_s", "duration_s", "hr_change_pct", "rmssd_ms"])


def analyze_fusion_csv(fusion_csv, scripts_dir=None, channel="auto", method="IVW_SSQI",
                       plot=None, p: Priors = P) -> Report:
    """Stages 4-5 on an existing dual_fusion.py output file."""
    scripts_dir = Path(scripts_dir) if scripts_dir else Path(__file__).resolve().parent.parent
    df = pd.read_csv(fusion_csv)
    df.columns = df.columns.str.strip()
    cols = {wl: f"{method}_{wl}_Fused" for wl in ("IR", "RED")}
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise PipelineError(f"{fusion_csv} lacks fused columns {missing}. Columns: {list(df.columns)}")
    t = df["Elapsed_s"].to_numpy(float)
    fs_raw = 1.0 / np.median(np.diff(t))
    fs = float(min(fs_raw, p.max_resample_hz))

    # ---- stage 4: quality of the fused signal
    use_cols = cols if channel == "auto" else {channel: cols[channel]}
    Q = step4_quality(df, use_cols, scripts_dir, p)
    chosen = max(Q, key=lambda k: (Q[k]["frac"], k == "IR"))        # ties -> IR
    qc = Q[chosen]
    info = dict(signal_column=qc["col"], channel_usable_frac={k: v["frac"] for k, v in Q.items()},
                fs_estimated_hz=round(fs_raw, 2), duration_s=round(t[-1] - t[0], 1),
                quality_mode=qc["info"].get("quality_mode"), ssqi_median=qc["info"].get("ssqi_median", np.nan),
                gate_counts=qc["info"].get("gate_counts"), label_counts=qc["info"].get("label_counts"))

    summary_empty = dict(drowsy_detected=False, first_alarm_s=None, n_windows=0, pct_valid=0.0,
                         state_pct={}, max_S=0.0, final_alarm_level=float("nan"))
    if qc["q"].empty:
        summary = dict(summary_empty, prediction="INSUFFICIENT_DATA", reason="signal shorter than one SSQI window")
        return Report(pd.DataFrame(), pd.DataFrame(), summary, info)

    qfn = quality_lookup(qc["q"], qc["usable"])
    b = extract_beats(t, df[qc["col"]].to_numpy(float), fs, qfn, p)
    info.update(b["info"])
    t0, t_end = b["t"][0], b["t"][-1]
    win_df, dinfo, t_warm = run_detector(b, fs, t0, t_end, p)
    if win_df is None:
        summary = dict(summary_empty, prediction="INSUFFICIENT_DATA", reason=dinfo["error"])
        rep = Report(pd.DataFrame(), pd.DataFrame(), summary, info, b, Q)
        return rep
    info.update(dinfo)

    events = _events(win_df)
    pct = (win_df.state.value_counts(normalize=True) * 100).to_dict()
    pct_valid = 100 * win_df.valid.mean()
    long_alarm = bool(len(events) and (events.duration_s >= p.min_alarm_s).any())
    if pct_valid < p.min_valid_pct:
        prediction, reason = "INSUFFICIENT_DATA", f"only {pct_valid:.0f} % of windows had a usable signal"
    elif long_alarm:
        prediction, reason = "DROWSY", ""
    else:
        prediction, reason = "ALERT", ""
    first = None
    if long_alarm:
        first = float(events.loc[events.duration_s >= p.min_alarm_s, "start_s"].min())
    summary = dict(prediction=prediction, reason=reason, drowsy_detected=long_alarm, first_alarm_s=first,
                   n_windows=len(win_df), pct_valid=pct_valid, state_pct=pct,
                   max_S=float(win_df.S.max()), final_alarm_level=float(win_df.h_on.iloc[-1]))
    rep = Report(win_df, events, summary, info, b, Q)
    if plot:
        plot_report(rep, plot)
    return rep


def analyze(raw_csv, workdir=None, scripts_dir=None, channel="auto", method="IVW_SSQI",
            reuse=False, plot=None, p: Priors = P, log=print) -> Report:
    """Full pipeline: raw CSV -> stages 1-4 -> detector."""
    raw_csv = Path(raw_csv)
    scripts_dir = Path(scripts_dir) if scripts_dir else Path(__file__).resolve().parent.parent
    workdir = Path(workdir) if workdir else Path(f"{raw_csv.stem}_pipeline")
    fusion_csv = run_stages_1_to_3(raw_csv, workdir, scripts_dir, reuse, log)
    log("[4/5] step2_ssqi_adaptive_v2  (quality of fused signal)\n[5/5] detector")
    return analyze_fusion_csv(fusion_csv, scripts_dir, channel, method, plot, p)


def plot_report(rep: Report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b, w = rep.beats, rep.windows
    fig, ax = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    ax[0].plot(b["t"], b["xf"], lw=.7)
    pk_y = np.interp(b["pk_t"], b["t"], b["xf"])
    ok = b["good"] & b["q_ok"]
    ax[0].plot(b["pk_t"][ok], pk_y[ok], "g.", ms=4, label="beat used")
    ax[0].plot(b["pk_t"][~ok], pk_y[~ok], "rx", ms=5, label="beat rejected (morphology / stage-4 quality)")
    ax[0].set_ylabel("fused PPG"); ax[0].legend(fontsize=7, loc="upper right")
    ax[1].plot(w.t, w.hr_bpm, "k-", label="window HR"); ax[1].plot(w.t, w.b_hr, "g--", label="adaptive baseline")
    ax[1].set_ylabel("bpm"); ax[1].legend(fontsize=7)
    ax[2].plot(w.t, w.z_hr, label="z heart-rate slowing"); ax[2].plot(w.t, w.z_hrv, label="z HRV change")
    ax[2].axhline(0, c="gray", lw=.5); ax[2].set_ylabel("z"); ax[2].legend(fontsize=7)
    ax[3].plot(w.t, w.S, "k-", label="evidence (CUSUM)"); ax[3].plot(w.t, w.h_on, "r:", label="alarm level")
    ax[3].set_ylabel("evidence"); ax[3].legend(fontsize=7)
    code = {"ALERT": 0, "HOLD": 0, "SUSPECT": 1, "DROWSY": 2, "NO_SIGNAL": -1}
    ax[4].step(w.t, w.state.map(code), where="post", color="k")
    ax[4].set_yticks([-1, 0, 1, 2]); ax[4].set_yticklabels(["NO_SIGNAL", "ALERT", "SUSPECT", "DROWSY"])
    ax[4].set_xlabel("time (s)")
    col = {"DROWSY": "red", "SUSPECT": "orange", "NO_SIGNAL": "gray"}
    for a in ax:
        for st, c in col.items():
            m = (w.state == st).values
            tt = w.t.values
            for i in np.where(m)[0]:
                a.axvspan(tt[i], tt[min(i + 1, len(tt) - 1)], color=c, alpha=.12, lw=0)
    fig.suptitle(f"Dual-PPG drowsiness: {rep.summary['prediction']}   "
                 f"({rep.info.get('signal_column')}, quality rule: {rep.info.get('quality_mode')})")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dual-PPG drowsiness pipeline (stages 1-5)")
    ap.add_argument("csv", nargs="?", help="raw recording CSV")
    ap.add_argument("--from-fusion", help="skip stages 1-3: dual_fusion.py output (ppg_fusion_results.csv)")
    ap.add_argument("--workdir", help="folder for all intermediates (default <csv stem>_pipeline)")
    ap.add_argument("--scripts-dir", help="folder holding the four stage scripts (default: next to this file)")
    ap.add_argument("--channel", choices=["auto", "IR", "RED"], default="auto")
    ap.add_argument("--method", choices=["IVW_SSQI", "IVW"], default="IVW_SSQI")
    ap.add_argument("--reuse", action="store_true", help="reuse stage outputs already in the workdir")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    if not a.csv and not a.from_fusion:
        ap.error("give a raw CSV or --from-fusion")
    try:
        if a.from_fusion:
            src = Path(a.from_fusion)
            out = Path(a.workdir) if a.workdir else src.parent
            out.mkdir(parents=True, exist_ok=True)
            stem = src.parent.parent.name if src.parent.parent.name else "recording"
            rep = analyze_fusion_csv(src, a.scripts_dir, a.channel, a.method,
                                     None if a.no_plot else str(out / f"{stem}_drowsiness_report.png"))
        else:
            src = Path(a.csv)
            out = Path(a.workdir) if a.workdir else Path(f"{src.stem}_pipeline")
            stem = src.stem
            rep = analyze(src, out, a.scripts_dir, a.channel, a.method, a.reuse,
                          None if a.no_plot else str(out / f"{stem}_drowsiness_report.png"))
    except PipelineError as e:
        sys.exit(f"PIPELINE ERROR: {e}")
    print(rep.text())
    if len(rep.windows):
        rep.windows.to_csv(out / f"{stem}_drowsiness_windows.csv", index=False)
        rep.events.to_csv(out / f"{stem}_drowsiness_events.csv", index=False)
    (out / f"{stem}_drowsiness_summary.json").write_text(json.dumps(
        {"summary": rep.summary, "info": {k: v for k, v in rep.info.items()}}, indent=2, default=str))
    print(f"\nsaved to: {out}")


if __name__ == "__main__":
    main()