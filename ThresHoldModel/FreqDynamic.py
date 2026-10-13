"""
FreqDynamic.py - pipeline-adapted FREQUENCY-DOMAIN drowsiness threshold model
=============================================================================
Same input chain as TimeDynamic.py (stages 1-4 reused from it); only the detector differs.
It keeps the paper's idea (Pugliese et al. 2022: drowsiness = autonomic shift, LF down / HF up,
read from the LF/HF ratio of a power spectrum) but is built around what THIS pipeline delivers.

    python FreqDynamic.py recording.csv --workdir out
    python FreqDynamic.py --from-fusion out/ppg_fusion_ssqi_results/ppg_fusion_results.csv

HOW EACH CHOICE FOLLOWS FROM THE PREPROCESSING / DATA
-----------------------------------------------------
  pipeline fact                                   -> design choice
  ----------------------------------------------------------------------------------------------
  dual_process.py band-passes at 0.5-5 Hz, so     -> spectrum is taken from the beat-interval
  the PPG itself has no 0.04-0.40 Hz content         series (tachogram), not from the PPG waveform
  (LF/HF bands are 21-32 dB below the pulse)         (the HRV that the paper's lambda stands for)
  stage-4 quality gating deletes intervals,       -> Lomb-Scargle periodogram on the surviving
  and ~4 % of samples are dropped                    (time, interval) pairs; no interpolation
                                                     across gaps, no resampling grid
  paper window 40.96 s: LF band (0.04 Hz) has     -> 60 s window (>= 2 cycles of the lowest LF
  only 1.6 cycles                                    frequency), 10 s hop
  paper's peak/peak ratio: heavy-tailed           -> ln(LF band POWER / HF band POWER)
  (lambda 1..100 on "nithish")
  paper threshold = max |d lambda| of ONE         -> person's own baseline: median + robust sigma of
  calibration window (undefined, fragile)            the first learn_s seconds; sigma never below the
                                                     sampling-noise floor of a spectral band estimate
                                                     (computed from degrees of freedom, not tuned)
  paper rule |d lambda| is direction-blind and    -> one-sided (LF/HF falling = HF dominance) CUSUM,
  only sees fluctuation, not a new level             so a sustained shift of level accumulates
  60 s start-up skip in dual_process.py           -> baseline learning starts at the first sample the
                                                     pipeline outputs and lasts learn_s (180 s)
  overlapping windows are not independent         -> CUSUM increments are weighted by hop/window
  CUSUM k = 0.5, h = 4 are the textbook choice     (ARL0 ~ 170 independent windows for Gaussian
  (not fitted to these recordings)                   noise)

LIMITS (stated, not hidden)
---------------------------
  * Needs learn_s + one window = 240 s after the pipeline's 60 s skip, i.e. >= ~300 s recordings.
    The three motion files (100-120 s) cannot be evaluated by any LF/HF model on this chain.
  * The baseline is assumed to be an awake period (same assumption as TimeDynamic).
  * LF at 60 s is at the edge of what HRV literature accepts; the ratio is noisy. sigma_floor
    accounts for sampling noise only, not for physiological non-stationarity.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import TImeDynamic_v2 as TD  # noqa: E402  (stages 1-4, beats, quality gating)
from TImeDynamic_v2 import PipelineError  # noqa: E402


# =========================================================================== #
@dataclass(frozen=True)
class FreqConfig:
    window_s: float = 60.0
    hop_s: float = 10.0
    lf_band: tuple = (0.04, 0.15)
    hf_band: tuple = (0.15, 0.40)
    f_step: float = 0.0025                 # Hz grid of the Lomb-Scargle evaluation
    min_intervals: int = 30
    min_coverage: float = 0.70             # share of the window covered by clean intervals
    max_gap_s: float = 10.0                # longest hole inside a window
    max_dev: float = 0.30                  # drop intervals > 30 % off the window median (as TimeDynamic)
    learn_s: float = 180.0                 # baseline period (assumed awake)
    min_learn_windows: int = 5
    k: float = 0.5                         # CUSUM slack (sigma units)
    h: float = 4.0                         # CUSUM alarm level (sigma units, independent-window equivalents)
    h_off_frac: float = 0.3
    suspect_frac: float = 0.5
    s_cap_mult: float = 2.0
    direction: str = "down"                # "down": LF/HF falls (paper physiology) | "up" | "both"
    no_signal_s: float = 120.0
    min_alarm_s: float = 60.0              # alarm must last one window length to count
    min_valid_pct: float = 40.0
    enbw: float = 1.0                      # noise-bandwidth factor used for the sigma floor (no taper => 1)


CFG = FreqConfig()


def sigma_floor(cfg: FreqConfig) -> float:
    """Sampling-noise SD of ln(LF/HF) for a noise-like spectrum: a band estimate with
    dof = 2*B*T/ENBW chi-square degrees of freedom has var(ln P) ~ 2/dof."""
    def v(band):
        dof = 2.0 * (band[1] - band[0]) * cfg.window_s / cfg.enbw
        return 2.0 / dof
    return float(np.sqrt(v(cfg.lf_band) + v(cfg.hf_band)))


# =========================================================================== #
# Input (same objects as TimeDynamic)
# =========================================================================== #
@dataclass
class Prepared:
    t0: float
    t_end: float
    beats: dict
    qc: dict
    Q: dict
    info: dict


def prepare(fusion_csv, scripts_dir=None, channel="auto", method="IVW_SSQI") -> Prepared:
    scripts_dir = Path(scripts_dir) if scripts_dir else Path(__file__).resolve().parent.parent
    P = TD.P
    df = pd.read_csv(fusion_csv)
    df.columns = df.columns.str.strip()
    cols = {wl: f"{method}_{wl}_Fused" for wl in ("IR", "RED")}
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise PipelineError(f"{fusion_csv} lacks fused columns {missing}")
    t = df["Elapsed_s"].to_numpy(float)
    fs_raw = 1.0 / np.median(np.diff(t))
    fs = float(min(fs_raw, P.max_resample_hz))
    use = cols if channel == "auto" else {channel: cols[channel]}
    Q = TD.step4_quality(df, use, scripts_dir, P)
    chosen = max(Q, key=lambda k: (Q[k]["frac"], k == "IR"))
    qc = Q[chosen]
    info = dict(signal_column=qc["col"], channel_usable_frac={k: v["frac"] for k, v in Q.items()},
                fs_estimated_hz=round(fs_raw, 2), duration_s=round(t[-1] - t[0], 1),
                quality_mode=qc["info"].get("quality_mode"), ssqi_median=qc["info"].get("ssqi_median", np.nan))
    beats = None
    if not qc["q"].empty:
        beats = TD.extract_beats(t, df[qc["col"]].to_numpy(float), fs, TD.quality_lookup(qc["q"], qc["usable"]), P)
        info.update(beats["info"])
    return Prepared(float(t[0]), float(t[-1]), beats, qc, Q, info)


# =========================================================================== #
# Feature: ln(LF power / HF power) of the tachogram, Lomb-Scargle
# =========================================================================== #
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _band_power(f, p, band):
    m = (f >= band[0]) & (f <= band[1])
    return float(_trapz(p[m], f[m]))


def window_feature(beats, ws, we, cfg: FreqConfig):
    out = dict(valid=False, reason="", coverage=np.nan, n=0, LF=np.nan, HF=np.nan, lnLFHF=np.nan, hr_bpm=np.nan)
    if beats is None:
        out["reason"] = "no_signal"
        return out
    m = (beats["t_pp"] > ws) & (beats["t_pp"] <= we)
    q, tt = beats["pp"][m], beats["t_pp"][m]
    if len(q) >= 4:
        med = np.median(q)
        keep = np.abs(q - med) <= cfg.max_dev * med
        q, tt = q[keep], tt[keep]
    out["n"] = int(len(q))
    cov = float(min(1.0, q.sum() / cfg.window_s)) if len(q) else 0.0
    out["coverage"] = cov
    if len(q) < cfg.min_intervals:
        out["reason"] = "few_intervals"
        return out
    if cov < cfg.min_coverage:
        out["reason"] = "low_coverage"
        return out
    # largest hole between consecutive clean intervals (and at the window edges)
    holes = (tt - q)[1:] - tt[:-1]
    gap = max(float(np.max(holes)) if len(holes) else 0.0, float(tt[0] - q[0] - ws), float(we - tt[-1]))
    if gap > cfg.max_gap_s:
        out["reason"] = "long_gap"
        return out
    # detrend (slow heart-rate drift would leak into the LF band), then Lomb-Scargle
    x = tt - ws
    y = q - np.polyval(np.polyfit(x, q, 1), x)
    f = np.arange(0.02, 0.50 + 1e-9, cfg.f_step)
    p = lombscargle(x, y, 2 * np.pi * f, precenter=True, normalize=False)
    LF, HF = _band_power(f, p, cfg.lf_band), _band_power(f, p, cfg.hf_band)
    if LF <= 0 or HF <= 0:
        out["reason"] = "zero_power"
        return out
    out.update(valid=True, LF=LF, HF=HF, lnLFHF=float(np.log(LF / HF)), hr_bpm=float(60.0 / np.mean(q)))
    return out


# =========================================================================== #
# Detector
# =========================================================================== #
@dataclass
class FreqReport:
    windows: pd.DataFrame
    events: pd.DataFrame
    summary: dict
    info: dict
    cfg: FreqConfig

    def text(self) -> str:
        s, i, c = self.summary, self.info, self.cfg
        L = ["=" * 66, "DUAL-PPG DROWSINESS REPORT  (frequency-domain, baseline-relative)", "=" * 66,
             f"  prediction          : {s['prediction']}" + (f"   ({s['reason']})" if s.get("reason") else ""),
             f"  signal              : {i.get('signal_column')}   (stage-4 usable "
             + ", ".join(f"{k} {v:.0%}" for k, v in i.get('channel_usable_frac', {}).items()) + ")",
             f"  feature             : ln(LF/HF) of tachogram, Lomb-Scargle, {c.window_s:.0f} s window / {c.hop_s:.0f} s hop"]
        if "baseline" in s:
            L += [f"  baseline            : ln(LF/HF) = {s['baseline']:.2f}  sigma = {s['sigma']:.2f} "
                  f"(floor {s['sigma_floor']:.2f}, learned {s['sigma_learned']:.2f}), learn {c.learn_s:.0f} s, "
                  f"{s['n_learn']} windows",
                  f"  CUSUM               : k = {c.k}, h = {c.h} (direction: {c.direction}); peak S = {s['max_S']:.2f}",
                  f"  windows evaluated   : {s['n_windows']} (usable {s['pct_valid']:.0f} %)",
                  f"  time in states      : " + ", ".join(f"{k} {v:.0f} %" for k, v in s['state_pct'].items())]
        if s.get("first_alarm_s") is not None:
            L.append(f"  first alarm         : decided at t = {s['first_alarm_s']:.0f} s")
        if len(self.events):
            L += ["", "Drowsiness events", self.events.round(1).to_string(index=False)]
        return "\n".join(L)


def run_freq_detector(prep: Prepared, cfg: FreqConfig = CFG) -> FreqReport:
    base = dict(drowsy_detected=False, first_alarm_s=None, n_windows=0, pct_valid=0.0, state_pct={},
                max_S=0.0, sigma_floor=sigma_floor(cfg))
    empty = pd.DataFrame()
    if prep.beats is None:
        return FreqReport(empty, empty, dict(base, prediction="INSUFFICIENT_DATA", reason="no usable signal"),
                          prep.info, cfg)
    W, hop = cfg.window_s, cfg.hop_s
    t0, t_end = prep.t0, prep.t_end
    ends = np.arange(t0 + W, t_end + 1e-9, hop)
    rows = []
    for we in ends:
        f = window_feature(prep.beats, we - W, we, cfg)
        f.update(start_s=we - W, end_s=we)
        rows.append(f)
    if not rows:
        return FreqReport(empty, empty, dict(base, prediction="INSUFFICIENT_DATA",
                          reason=f"recording is {t_end - t0:.0f} s after the pipeline's 60 s start-up skip; "
                                 f"shorter than one {W:.0f} s window"), prep.info, cfg)
    D = pd.DataFrame(rows)
    learn_end = t0 + cfg.learn_s
    L = D[(D.end_s <= learn_end + 1e-9) & D.valid]
    if len(L) < cfg.min_learn_windows or D.end_s.max() <= learn_end + 1e-9:
        need = cfg.learn_s + hop
        summ = dict(base, prediction="INSUFFICIENT_DATA", n_windows=len(D),
                    reason=f"{len(L)} valid learning windows (need {cfg.min_learn_windows}); the model needs "
                           f">= {cfg.learn_s:.0f} s of baseline plus at least one {W:.0f} s window after the "
                           f"pipeline's 60 s start-up skip (recording is {t_end - t0:.0f} s)")
        D["state"] = "UNUSED"
        return FreqReport(D, empty, summ, prep.info, cfg)

    b = float(np.median(L.lnLFHF))
    s_learn = 1.4826 * float(np.median(np.abs(L.lnLFHF - b)))
    sig = max(s_learn, base["sigma_floor"])
    scale = hop / W                                     # overlapping windows: weight by new information
    S, alarm, last_valid = 0.0, False, learn_end
    z_l, S_l, st_l = [], [], []
    for _, r in D.iterrows():
        if r.end_s <= learn_end + 1e-9:
            z_l.append(np.nan); S_l.append(0.0); st_l.append("LEARN")
            continue
        if not r.valid:
            gap_s = r.end_s - last_valid
            st = "NO_SIGNAL" if gap_s > cfg.no_signal_s else ("DROWSY" if alarm else "HOLD")
            z_l.append(np.nan); S_l.append(S); st_l.append(st)
            continue
        last_valid = r.end_s
        d = (r.lnLFHF - b) / sig
        z = -d if cfg.direction == "down" else (d if cfg.direction == "up" else abs(d))
        S = min(max(0.0, S + (z - cfg.k) * scale), cfg.s_cap_mult * cfg.h)
        if not alarm and S >= cfg.h:
            alarm = True
        elif alarm and S <= cfg.h * cfg.h_off_frac:
            alarm = False
        st = "DROWSY" if alarm else ("SUSPECT" if S >= cfg.suspect_frac * cfg.h else "ALERT")
        z_l.append(z); S_l.append(S); st_l.append(st)
    D["z"], D["S"], D["state"] = z_l, S_l, st_l
    D["alarm"] = D.state == "DROWSY"

    ev, start = [], None
    for _, r in D.iterrows():
        if r.alarm and start is None:
            start = r.end_s
        if (not r.alarm) and start is not None:
            ev.append((start, r.end_s)); start = None
    if start is not None:
        ev.append((start, float(D.end_s.iloc[-1])))
    evd = pd.DataFrame([dict(start_s=a, end_s=z, duration_s=z - a,
                             mean_lnLFHF=float(D[(D.end_s >= a) & (D.end_s <= z) & D.valid].lnLFHF.mean()),
                             hr_bpm=float(D[(D.end_s >= a) & (D.end_s <= z) & D.valid].hr_bpm.mean()))
                        for a, z in ev], columns=["start_s", "end_s", "duration_s", "mean_lnLFHF", "hr_bpm"])
    post = D[D.state != "LEARN"]
    pct_valid = 100 * float(post.valid.mean())
    long_alarm = bool(len(evd) and (evd.duration_s >= cfg.min_alarm_s).any())
    if pct_valid < cfg.min_valid_pct:
        pred, reason = "INSUFFICIENT_DATA", f"only {pct_valid:.0f} % of windows usable"
    else:
        pred, reason = ("DROWSY" if long_alarm else "ALERT"), ""
    first = float(evd.loc[evd.duration_s >= cfg.min_alarm_s, "start_s"].min()) if long_alarm else None
    summ = dict(prediction=pred, reason=reason, drowsy_detected=long_alarm, first_alarm_s=first, n_windows=len(post),
                pct_valid=pct_valid, state_pct=(post.state.value_counts(normalize=True) * 100).to_dict(),
                max_S=float(post.S.max()), baseline=b, sigma=sig, sigma_floor=base["sigma_floor"],
                sigma_learned=s_learn, n_learn=int(len(L)))
    return FreqReport(D, evd, summ, prep.info, cfg)


# =========================================================================== #
def plot_freq_report(rep: FreqReport, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, c, s = rep.windows, rep.cfg, rep.summary
    if "z" not in w:
        return
    fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    ax[0].plot(w.end_s, w.hr_bpm, "k-"); ax[0].set_ylabel("window HR (bpm)\n(not used by the detector)")
    ax[1].plot(w.end_s, w.lnLFHF, "k.-", lw=.8)
    ax[1].axhline(s["baseline"], color="g", ls="--", label="baseline")
    ax[1].axhspan(s["baseline"] - s["sigma"], s["baseline"] + s["sigma"], color="g", alpha=.12, label="+/- sigma")
    ax[1].set_ylabel("ln(LF/HF)"); ax[1].legend(fontsize=7)
    ax[2].plot(w.end_s, np.log(w.LF), label="ln LF power"); ax[2].plot(w.end_s, np.log(w.HF), label="ln HF power")
    ax[2].set_ylabel("ln band power"); ax[2].legend(fontsize=7)
    ax[3].plot(w.end_s, w.S, "k-", label="evidence S"); ax[3].axhline(c.h, color="r", ls=":", label="alarm level h")
    ax[3].set_ylabel("CUSUM"); ax[3].legend(fontsize=7)
    code = {"LEARN": -1, "NO_SIGNAL": -1, "HOLD": 0, "ALERT": 0, "SUSPECT": 1, "DROWSY": 2, "UNUSED": 0}
    ax[4].step(w.end_s, w.state.map(code), where="post", color="k")
    ax[4].set_yticks([-1, 0, 1, 2]); ax[4].set_yticklabels(["learn/no signal", "ALERT", "SUSPECT", "DROWSY"])
    ax[4].set_xlabel("window end time (s)")
    for a in ax:
        for st, col in {"DROWSY": "red", "SUSPECT": "orange", "LEARN": "green", "NO_SIGNAL": "gray"}.items():
            m = (w.state == st).to_numpy()
            for i in np.where(m)[0]:
                a.axvspan(w.end_s.iloc[i], w.end_s.iloc[min(i + 1, len(w) - 1)], color=col, alpha=.10, lw=0)
    fig.suptitle(f"Frequency-domain detector: {s['prediction']}  ({rep.info.get('signal_column')})")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def analyze_fusion_csv(fusion_csv, scripts_dir=None, channel="auto", method="IVW_SSQI",
                       cfg: FreqConfig = CFG, plot=None) -> FreqReport:
    rep = run_freq_detector(prepare(fusion_csv, scripts_dir, channel, method), cfg)
    if plot:
        plot_freq_report(rep, plot)
    return rep


def run_stages(raw_csv: Path, workdir: Path, scripts_dir: Path, reuse: bool, process_args=(), log=print) -> Path:
    """TimeDynamic's stage runner, with optional extra arguments for dual_process.py."""
    if not process_args:
        return TD.run_stages_1_to_3(raw_csv, workdir, scripts_dir, reuse, log)
    workdir.mkdir(parents=True, exist_ok=True)
    p1 = workdir / "ppg_results_sg" / "ppg_fusion_output.csv"
    p2 = workdir / "ssqi_dual" / "ssqi_channel_details.csv"
    p3 = workdir / "ppg_fusion_ssqi_results" / "ppg_fusion_results.csv"
    if not (reuse and p1.is_file()):
        log("[1/5] dual_process.py " + " ".join(process_args))
        TD._run_stage("1 process", scripts_dir / TD.STAGE_SCRIPTS["process"],
                      ["--csv", str(raw_csv.resolve()), "--outdir", "ppg_results_sg", *process_args], workdir)
    if not p1.is_file():
        raise PipelineError("[1 process] no output (recording too short for the chosen --stab-s/--calib-s)")
    if not (reuse and p2.is_file()):
        TD._run_stage("2 ssqi", scripts_dir / TD.STAGE_SCRIPTS["ssqi"],
                      ["--csv", "ppg_results_sg/ppg_fusion_output.csv", "--outdir", "ssqi_dual"], workdir)
    TD._bridge_ssqi(workdir)
    if not (reuse and p3.is_file()):
        TD._run_stage("3 fusion", scripts_dir / TD.STAGE_SCRIPTS["fusion"], [], workdir)
    return p3


def analyze(raw_csv, workdir=None, scripts_dir=None, channel="auto", method="IVW_SSQI", reuse=False,
            cfg: FreqConfig = CFG, plot=None, process_args=(), log=print) -> FreqReport:
    raw_csv = Path(raw_csv)
    scripts_dir = Path(scripts_dir) if scripts_dir else Path(__file__).resolve().parent.parent
    workdir = Path(workdir) if workdir else Path(f"{raw_csv.stem}_pipeline")
    fusion_csv = run_stages(raw_csv, workdir, scripts_dir, reuse, process_args, log)
    log("[4/5] step2_ssqi_adaptive_v2  (quality of fused signal)\n[5/5] frequency-domain detector")
    return analyze_fusion_csv(fusion_csv, scripts_dir, channel, method, cfg, plot)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dual-PPG drowsiness: pipeline-adapted frequency-domain model")
    ap.add_argument("csv", nargs="?"); ap.add_argument("--from-fusion")
    ap.add_argument("--workdir"); ap.add_argument("--scripts-dir")
    ap.add_argument("--channel", choices=["auto", "IR", "RED"], default="auto")
    ap.add_argument("--method", choices=["IVW_SSQI", "IVW"], default="IVW_SSQI")
    ap.add_argument("--reuse", action="store_true"); ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--window-s", type=float, default=CFG.window_s)
    ap.add_argument("--hop-s", type=float, default=CFG.hop_s)
    ap.add_argument("--learn-s", type=float, default=CFG.learn_s)
    ap.add_argument("--k", type=float, default=CFG.k); ap.add_argument("--h", type=float, default=CFG.h)
    ap.add_argument("--direction", choices=["down", "up", "both"], default=CFG.direction)
    ap.add_argument("--stab-s", type=float, help="pass-through to dual_process.py (default 30)")
    ap.add_argument("--calib-s", type=float, help="pass-through to dual_process.py (default 30)")
    a = ap.parse_args(argv)
    if not a.csv and not a.from_fusion:
        ap.error("give a raw CSV or --from-fusion")
    cfg = FreqConfig(window_s=a.window_s, hop_s=a.hop_s, learn_s=a.learn_s, k=a.k, h=a.h, direction=a.direction)
    pa = []
    if a.stab_s is not None:
        pa += ["--stab-s", str(a.stab_s)]
    if a.calib_s is not None:
        pa += ["--calib-s", str(a.calib_s)]
    try:
        if a.from_fusion:
            src = Path(a.from_fusion)
            out = Path(a.workdir) if a.workdir else src.parent
            stem = src.parent.parent.name or "recording"
            out.mkdir(parents=True, exist_ok=True)
            rep = analyze_fusion_csv(src, a.scripts_dir, a.channel, a.method, cfg,
                                     None if a.no_plot else str(out / f"{stem}_freq_report.png"))
        else:
            src = Path(a.csv)
            out = Path(a.workdir) if a.workdir else Path(f"{src.stem}_pipeline")
            stem = src.stem
            rep = analyze(src, out, a.scripts_dir, a.channel, a.method, a.reuse, cfg,
                          None if a.no_plot else str(out / f"{stem}_freq_report.png"), pa)
    except PipelineError as e:
        sys.exit(f"PIPELINE ERROR: {e}")
    print(rep.text())
    if len(rep.windows):
        rep.windows.to_csv(out / f"{stem}_freq_windows.csv", index=False)
        rep.events.to_csv(out / f"{stem}_freq_events.csv", index=False)
    (out / f"{stem}_freq_summary.json").write_text(json.dumps(
        {"summary": rep.summary, "info": rep.info, "config": cfg.__dict__}, indent=2, default=str))
    print(f"\nsaved to: {out}")


if __name__ == "__main__":
    main()