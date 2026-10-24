"""
FreqDynamic.py (v2) - pipeline-adapted FREQUENCY-DOMAIN drowsiness threshold model
==================================================================================
Same input chain as TimeDynamic.py (stages 1-4 reused from it); only the detector differs.
Idea taken from Pugliese et al. 2022 (drowsiness = autonomic shift, LF down / HF up, read from
the LF/HF ratio), but built around what THIS pipeline delivers and validated on synthetic data
with known truth, so nothing here is tuned to the six recordings.

    python FreqDynamic.py recording.csv --workdir out
    python FreqDynamic.py --from-fusion out/ppg_fusion_ssqi_results/ppg_fusion_results.csv

DESIGN (each choice and why)
----------------------------
  * Spectrum of the beat-interval series (tachogram), not of the PPG: dual_process.py band-passes
    at 0.5-5 Hz, so the PPG has no 0.04-0.40 Hz content left.
  * Lomb-Scargle on the surviving (time, interval) pairs: stage-4 gating deletes intervals.
  * Feature = ln(LF band power / HF band power), 60 s window, 10 s hop. 60 s gives the lowest LF
    frequency (0.04 Hz) 2.4 cycles.  Sampling SD of this feature was verified on synthetic
    tachograms (0.42-0.53 against a 0.47 theoretical floor).
  * Gating bias guard: with gaps the ratio is biased downward (about -0.08 ln-units per 10 % of
    time lost for LF-dominant spectra, same sign as drowsiness). A window is therefore only used
    as evidence if its coverage is >= 0.80 AND not more than 0.10 below the baseline's coverage.
  * Interpretability gate: LF/HF is only meaningful if the HF band holds an identifiable
    respiratory peak. Per window: peak/mean of the HF band >= ln(M/alpha) (white-noise null,
    M = independent frequencies in the band, alpha = 0.05) and the peak is not at a band edge.
    If fewer than half of the baseline windows pass, the recording is reported
    INSUFFICIENT_DATA instead of producing alarms from a red-noise slope.
  * Baseline: median of the first learn_s, then an exponential tracker (tau_s) updated ONLY on
    alert-looking windows (so a real shift is not absorbed). sigma is tracked the same way and
    never falls below the sampling-noise floor.
  * Evidence: one-sided CUSUM on z = (baseline - feature)/sigma (LF/HF falling = HF dominance),
    increments weighted by hop/window because windows overlap. k and h are set by simulation of
    the WHOLE detector, including baseline-estimation error, to a pre-specified target
    (<= 5 % false alarms per 30 min of alert signal under mild non-stationarity), not on data.

MEASURED BEHAVIOUR (all numbers from simulations/tests in this work, not tuned on the recordings)
-------------------------------------------------------------------------------------------------
  Synthetic tachograms with known LF/HF (my generator; "mild"/"strong" = slow log-amplitude
  modulation with SD 0.2/0.4, an assumption):
    false alarms per 30 min of alert signal (fresh seeds): stationary 0 %, mild 2-3 %, strong 8 %
    (h chosen as the smallest value with <= 5 % under "mild" on a 500-recording calibration set)
    detection within 15 min of an injected ln(LF/HF) drop of 0.5/1.0/1.5/2.0: 4/10/24/34 %
    (median delay about 8-9 min). This is a CONSERVATIVE, SLOW detector: that is the price of
    a usable false-alarm rate on a noisy 60 s spectral feature.
    recordings with no respiratory peak in HF are declined (54/60), not alarmed on.
  Real recordings (this repository's six): see the accompanying write-up; 5 of 6 are declined
  (3 too short, 2 without a respiratory peak in the baseline) and the sixth is a miss.

LIMITS (not fixable with this input)
------------------------------------
  * Needs learn_s + one window = 240 s after the pipeline's 60 s skip (>= ~300 s recordings).
  * Baseline is assumed to be an awake period.
  * Respiration is not available (filtered out upstream), so the HF band cannot be adapted to it.
  * LF/HF may simply not change in a given drowsy episode; it is then undetectable here.
  * Whether a respiratory peak exists can vary within a recording; the accept/decline decision
    then depends on where the baseline falls.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import TImeDynamic_v2 as TD  # noqa: E402  (stages 1-4, beats, quality gating)
from TImeDynamic_v2 import PipelineError  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz


# =========================================================================== #
@dataclass(frozen=True)
class FreqConfig:
    window_s: float = 60.0
    hop_s: float = 10.0
    lf_band: tuple = (0.04, 0.15)
    hf_band: tuple = (0.15, 0.40)
    f_step: float = 0.0025                 # Hz grid of the Lomb-Scargle evaluation
    # --- window validity
    min_intervals: int = 30
    min_coverage: float = 0.80             # share of the window covered by clean intervals
    coverage_drop: float = 0.10            # ... and not more than this below the baseline's coverage
    max_gap_s: float = 10.0                # longest hole inside a window
    max_dev: float = 0.30                  # drop intervals > 30 % off the window median (as TimeDynamic)
    # --- interpretability gate (respiratory peak in the HF band)
    require_rsa: bool = True
    rsa_alpha: float = 0.05
    min_rsa_frac: float = 0.50             # share of baseline windows that must show a HF peak
    # --- baseline
    learn_s: float = 180.0                 # minimum baseline period
    learn_max_s: float = 360.0             # learning is extended (gated windows) up to this, until enough valid data
    min_learn_independent: float = 3.0     # non-overlapping-window equivalents of valid learning data
    tau_s: float = 600.0                   # baseline / sigma tracking time constant (as TimeDynamic)
    track_mode: str = "symmetric"          # "symmetric" | "frozen". (one-sided "selective" tracking was tested: biased)
    track_gate: float = 2.5                # only windows with |z| below this (either side) update the baseline
    # --- CUSUM (k, h set by simulation, see MEASURED BEHAVIOUR in the docstring)
    k: float = 0.5
    h: float = 12.0
    z_clip: float = 3.0                    # robust CUSUM: clip evidence z at this (0 = off)
    k_baseline_se: bool = True             # widen the slack k by the baseline's standard error 1/sqrt(n_eff)
    h_off_frac: float = 0.3
    suspect_frac: float = 0.5
    s_cap_mult: float = 2.0
    direction: str = "down"                # "down": LF/HF falls (paper physiology) | "up" | "both"
    no_signal_s: float = 120.0
    min_valid_pct: float = 40.0
    enbw: float = 1.0                      # noise-bandwidth factor for the sigma floor (no taper => 1)

    @property
    def min_alarm_s(self) -> float:        # an alarm must last one window length to count
        return self.window_s


CFG = FreqConfig()


def sigma_floor(cfg: FreqConfig) -> float:
    """Sampling-noise SD of ln(LF/HF) for a noise-like spectrum: a band estimate with
    dof = 2*B*T/ENBW chi-square degrees of freedom has var(ln P) ~ 2/dof."""
    def v(band):
        return 2.0 / (2.0 * (band[1] - band[0]) * cfg.window_s / cfg.enbw)
    return float(np.sqrt(v(cfg.lf_band) + v(cfg.hf_band)))


def rsa_threshold(cfg: FreqConfig) -> float:
    """Peak/mean level of the HF band that a white-noise spectrum exceeds with probability alpha:
    P(max/mean > c) ~ M*exp(-c), M = independent frequencies in the band."""
    M = max(2.0, (cfg.hf_band[1] - cfg.hf_band[0]) * cfg.window_s)
    return float(np.log(M / cfg.rsa_alpha))


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
def _band_power(f, p, band):
    m = (f >= band[0]) & (f <= band[1])
    return float(_trapz(p[m], f[m]))


def window_feature(beats, ws, we, cfg: FreqConfig):
    out = dict(valid=False, reason="", coverage=np.nan, n=0, LF=np.nan, HF=np.nan, lnLFHF=np.nan,
               hr_bpm=np.nan, rsa_ok=False, hf_peak_hz=np.nan, hf_peak_ratio=np.nan)
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
    holes = (tt - q)[1:] - tt[:-1]                       # holes between clean intervals and at the window edges
    gap = max(float(np.max(holes)) if len(holes) else 0.0, float(tt[0] - q[0] - ws), float(we - tt[-1]))
    if gap > cfg.max_gap_s:
        out["reason"] = "long_gap"
        return out
    x = tt - ws
    y = q - np.polyval(np.polyfit(x, q, 1), x)           # slow HR drift would leak into the LF band
    y = y - y.mean()
    f = np.arange(0.02, 0.50 + 1e-9, cfg.f_step)
    p = lombscargle(x, y, 2 * np.pi * f, normalize=False)
    LF, HF = _band_power(f, p, cfg.lf_band), _band_power(f, p, cfg.hf_band)
    if LF <= 0 or HF <= 0:
        out["reason"] = "zero_power"
        return out
    hm = (f >= cfg.hf_band[0]) & (f <= cfg.hf_band[1])
    ph, fh = p[hm], f[hm]
    ratio = float(ph.max() / ph.mean())
    res = 1.0 / cfg.window_s
    inside = (fh[np.argmax(ph)] > cfg.hf_band[0] + res) and (fh[np.argmax(ph)] < cfg.hf_band[1] - res)
    out.update(valid=True, LF=LF, HF=HF, lnLFHF=float(np.log(LF / HF)), hr_bpm=float(60.0 / np.mean(q)),
               hf_peak_hz=float(fh[np.argmax(ph)]), hf_peak_ratio=ratio,
               rsa_ok=bool(inside and ratio >= rsa_threshold(cfg)))
    return out


def feature_table(beats, t0, t_end, cfg: FreqConfig) -> pd.DataFrame:
    W, hop = cfg.window_s, cfg.hop_s
    rows = []
    for we in np.arange(t0 + W, t_end + 1e-9, hop):
        f = window_feature(beats, we - W, we, cfg)
        f.update(start_s=we - W, end_s=we)
        rows.append(f)
    return pd.DataFrame(rows)


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
            L += [f"  baseline (initial)  : ln(LF/HF) = {s['baseline']:.2f}  sigma = {s['sigma']:.2f} "
                  f"(floor {s['sigma_floor']:.2f}, learned {s['sigma_learned']:.2f}); "
                  f"{s['n_learn']} windows = {s['n_independent']:.1f} independent; coverage {s['baseline_coverage']:.2f}",
                  f"  respiratory peak    : present in {s['rsa_frac_baseline']:.0%} of baseline windows "
                  f"({s['rsa_frac_all']:.0%} overall; need >= {c.min_rsa_frac:.0%})",
                  f"  CUSUM               : k = {c.k}, h = {c.h} (direction: {c.direction}); peak S = {s['max_S']:.2f}",
                  f"  windows evaluated   : {s['n_windows']} (usable {s['pct_valid']:.0f} %)",
                  f"  time in states      : " + ", ".join(f"{k} {v:.0f} %" for k, v in s['state_pct'].items())]
        if s.get("first_alarm_s") is not None:
            L.append(f"  first alarm         : decided at t = {s['first_alarm_s']:.0f} s")
        if len(self.events):
            L += ["", "Drowsiness events", self.events.round(1).to_string(index=False)]
        return "\n".join(L)


def run_freq_detector(prep: Prepared, cfg: FreqConfig = CFG) -> FreqReport:
    floor = sigma_floor(cfg)
    base = dict(drowsy_detected=False, first_alarm_s=None, n_windows=0, pct_valid=0.0, state_pct={},
                max_S=0.0, sigma_floor=floor)
    empty = pd.DataFrame()
    if prep.beats is None:
        return FreqReport(empty, empty, dict(base, prediction="INSUFFICIENT_DATA", reason="no usable signal"),
                          prep.info, cfg)
    W, hop = cfg.window_s, cfg.hop_s
    t0, t_end = prep.t0, prep.t_end
    D = feature_table(prep.beats, t0, t_end, cfg)
    if D.empty:
        return FreqReport(empty, empty, dict(base, prediction="INSUFFICIENT_DATA",
                          reason=f"recording is {t_end - t0:.0f} s after the pipeline's 60 s start-up skip; "
                                 f"shorter than one {W:.0f} s window"), prep.info, cfg)
    def _learning(le):
        Lw = D[(D.end_s <= le + 1e-9) & D.valid]
        n = 0.0
        if len(Lw):                       # independent-window equivalents = length of the union of the windows / W
            iv = sorted(zip(Lw.start_s, Lw.end_s))
            cs, ce, tot = iv[0][0], iv[0][1], 0.0
            for s_, e_ in iv[1:]:
                if s_ <= ce + 1e-9:
                    ce = max(ce, e_)
                else:
                    tot += ce - cs; cs, ce = s_, e_
            n = (tot + ce - cs) / W
        return Lw, n
    # learn for at least learn_s, and keep going (up to learn_max_s) until there are enough VALID independent
    # windows, so a few gated windows cannot veto a recording or leave a thin baseline
    learn_end = t0 + cfg.learn_s
    L, n_ind = _learning(learn_end)
    while n_ind < cfg.min_learn_independent - 1e-9 and learn_end < t0 + cfg.learn_max_s - 1e-9 \
            and learn_end < D.end_s.max():
        learn_end += hop
        L, n_ind = _learning(learn_end)
    short = (len(L) == 0) or (n_ind < cfg.min_learn_independent - 1e-9) or (D.end_s.max() <= learn_end + 1e-9)
    if short:
        D["state"] = "UNUSED"
        reason = (f"{len(L)} valid learning windows = {n_ind:.1f} independent (need {cfg.min_learn_independent:.0f}); "
                  f"the model needs >= {cfg.learn_s:.0f} s of baseline plus at least one {W:.0f} s window after "
                  f"the pipeline's 60 s start-up skip (recording is {t_end - t0:.0f} s)")
        return FreqReport(D, empty, dict(base, prediction="INSUFFICIENT_DATA", reason=reason, n_windows=len(D)),
                          prep.info, cfg)
    rsa_frac_b = float(L.rsa_ok.mean())
    rsa_frac_all = float(D[D.valid].rsa_ok.mean())
    b = float(np.median(L.lnLFHF))
    s_learn = 1.4826 * float(np.median(np.abs(L.lnLFHF - b)))
    cov_b = float(np.median(L.coverage))
    common = dict(n_windows=len(D), sigma_floor=floor, sigma_learned=s_learn, baseline=b,
                  sigma=max(s_learn, floor), n_learn=int(len(L)), n_independent=n_ind, baseline_coverage=cov_b,
                  rsa_frac_baseline=rsa_frac_b, rsa_frac_all=rsa_frac_all)
    if cfg.require_rsa and rsa_frac_b < cfg.min_rsa_frac:
        D["state"] = "UNUSED"
        reason = (f"no identifiable respiratory (HF) peak in the baseline ({rsa_frac_b:.0%} of windows, need "
                  f">= {cfg.min_rsa_frac:.0%}): LF/HF is not interpretable for this recording")
        return FreqReport(D, empty, {**base, **common, "prediction": "INSUFFICIENT_DATA", "reason": reason,
                                     "pct_valid": 0.0, "max_S": 0.0, "state_pct": {}}, prep.info, cfg)

    var = max(s_learn, floor) ** 2
    scale = hop / W                                     # overlapping windows: weight by new information
    alpha = hop / cfg.tau_s
    S, alarm, last_valid = 0.0, False, learn_end
    n_eff, cap_eff = max(n_ind, 1.0), max(cfg.tau_s / W, n_ind)   # independent-sample equivalents behind the baseline
    cols = {k: [] for k in ("z", "S", "state", "b", "sigma", "gated")}
    for _, r in D.iterrows():
        if r.end_s <= learn_end + 1e-9:
            for k, v in zip(cols, (np.nan, 0.0, "LEARN", b, np.sqrt(var), False)):
                cols[k].append(v)
            continue
        usable = bool(r.valid and r.coverage >= max(cfg.min_coverage, cov_b - cfg.coverage_drop)
                      and (r.rsa_ok or not cfg.require_rsa))
        sig = max(np.sqrt(var), floor)
        if not usable:
            gap_s = r.end_s - last_valid
            st = "NO_SIGNAL" if gap_s > cfg.no_signal_s else ("DROWSY" if alarm else "HOLD")
            for k, v in zip(cols, (np.nan, S, st, b, sig, True)):
                cols[k].append(v)
            continue
        last_valid = r.end_s
        d = (r.lnLFHF - b) / sig
        z = -d if cfg.direction == "down" else (d if cfg.direction == "up" else abs(d))
        if cfg.z_clip > 0:
            z = float(np.clip(z, -cfg.z_clip, cfg.z_clip))
        k_eff = cfg.k + (1.0 / np.sqrt(n_eff) if cfg.k_baseline_se else 0.0)
        n_eff = min(n_eff + scale, cap_eff)
        S = min(max(0.0, S + (z - k_eff) * scale), cfg.s_cap_mult * cfg.h)
        if not alarm and S >= cfg.h:
            alarm = True
        elif alarm and S <= cfg.h * cfg.h_off_frac:
            alarm = False
        if cfg.track_mode == "symmetric" and (not alarm) and abs(d) <= cfg.track_gate \
                and S < cfg.suspect_frac * cfg.h:
            var = (1 - alpha) * var + alpha * (r.lnLFHF - b) ** 2     # track the person's alert-state scale
            b += alpha * (r.lnLFHF - b)                                # ... and level, slowly; outliers excluded both ways
        st = "DROWSY" if alarm else ("SUSPECT" if S >= cfg.suspect_frac * cfg.h else "ALERT")
        for k, v in zip(cols, (z, S, st, b, sig, False)):
            cols[k].append(v)
    for k, v in cols.items():
        D[k] = v
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
    pct_valid = 100 * float((~post.gated).mean())
    long_alarm = bool(len(evd) and (evd.duration_s >= cfg.min_alarm_s).any())
    if pct_valid < cfg.min_valid_pct:
        pred, reason = "INSUFFICIENT_DATA", f"only {pct_valid:.0f} % of windows usable"
    else:
        pred, reason = ("DROWSY" if long_alarm else "ALERT"), ""
    first = float(evd.loc[evd.duration_s >= cfg.min_alarm_s, "start_s"].min()) if long_alarm else None
    summ = {**base, **common, "prediction": pred, "reason": reason, "drowsy_detected": long_alarm,
            "first_alarm_s": first, "n_windows": len(post), "pct_valid": pct_valid,
            "state_pct": (post.state.value_counts(normalize=True) * 100).to_dict(), "max_S": float(post.S.max())}
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
    ax[1].plot(w.end_s, w.b, "g--", label="baseline (tracked)")
    ax[1].fill_between(w.end_s, w.b - w.sigma, w.b + w.sigma, color="g", alpha=.12, label="+/- sigma")
    ax[1].set_ylabel("ln(LF/HF)"); ax[1].legend(fontsize=7)
    ax[2].plot(w.end_s, w.hf_peak_ratio, label="HF peak / mean")
    ax[2].axhline(rsa_threshold(c), color="r", ls=":", label="respiratory-peak threshold")
    ax[2].set_ylabel("HF peak prominence"); ax[2].legend(fontsize=7)
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
    if plot and "z" in rep.windows:
        plot_freq_report(rep, plot)
    return rep


def _run_script(name, script: Path, args, cwd: Path):
    if not script.is_file():
        raise PipelineError(f"[{name}] script not found: {script}")
    r = subprocess.run([sys.executable, str(script), *args], cwd=str(cwd),
                       env=dict(os.environ, MPLBACKEND="Agg"), capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-12:])
        raise PipelineError(f"[{name}] {script.name} failed (exit {r.returncode}):\n{tail}")


def run_stages(raw_csv: Path, workdir: Path, scripts_dir: Path, reuse: bool, process_args=(), log=print) -> Path:
    """Stages 1-3. Without extra arguments this is exactly TimeDynamic's runner."""
    if not process_args:
        return TD.run_stages_1_to_3(raw_csv, workdir, scripts_dir, reuse, log)
    workdir.mkdir(parents=True, exist_ok=True)
    p1 = workdir / "ppg_results_sg" / "ppg_fusion_output.csv"
    p2 = workdir / "ssqi_dual" / "ssqi_channel_details.csv"
    p3 = workdir / "ppg_fusion_ssqi_results" / "ppg_fusion_results.csv"
    S = TD.STAGE_SCRIPTS
    if not (reuse and p1.is_file()):
        log("[1/5] dual_process.py " + " ".join(process_args))
        _run_script("1 process", scripts_dir / S["process"],
                    ["--csv", str(raw_csv.resolve()), "--outdir", "ppg_results_sg", *process_args], workdir)
    if not p1.is_file():
        raise PipelineError("[1 process] no output (recording too short for the chosen --stab-s/--calib-s)")
    if not (reuse and p2.is_file()):
        _run_script("2 ssqi", scripts_dir / S["ssqi"],
                    ["--csv", "ppg_results_sg/ppg_fusion_output.csv", "--outdir", "ssqi_dual"], workdir)
    TD._bridge_ssqi(workdir)
    if not (reuse and p3.is_file()):
        _run_script("3 fusion", scripts_dir / S["fusion"], [], workdir)
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
    ap.add_argument("--no-rsa-gate", action="store_true", help="disable the respiratory-peak interpretability gate")
    ap.add_argument("--stab-s", type=float, help="pass-through to dual_process.py (default 30)")
    ap.add_argument("--calib-s", type=float, help="pass-through to dual_process.py (default 30)")
    a = ap.parse_args(argv)
    if not a.csv and not a.from_fusion:
        ap.error("give a raw CSV or --from-fusion")
    cfg = FreqConfig(window_s=a.window_s, hop_s=a.hop_s, learn_s=a.learn_s, k=a.k, h=a.h,
                     direction=a.direction, require_rsa=not a.no_rsa_gate)
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
    out.mkdir(parents=True, exist_ok=True)
    if len(rep.windows):
        rep.windows.to_csv(out / f"{stem}_freq_windows.csv", index=False)
        rep.events.to_csv(out / f"{stem}_freq_events.csv", index=False)
    (out / f"{stem}_freq_summary.json").write_text(json.dumps(
        {"summary": rep.summary, "info": rep.info, "config": cfg.__dict__}, indent=2, default=str))
    print(f"\nsaved to: {out}")


if __name__ == "__main__":
    main()