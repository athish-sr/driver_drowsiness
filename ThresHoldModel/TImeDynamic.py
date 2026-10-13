"""
AUTO-ADAPTIVE PPG DROWSINESS DETECTOR  (zero user configuration)
================================================================
Usage
-----
    python auto_drowsiness.py recording.csv          # prints report, saves PNG + CSV
    from auto_drowsiness import analyze
    rep = analyze("recording.csv")                   # or a pandas DataFrame
    rep.summary, rep.events, rep.windows, rep.info

Nothing has to be configured: sampling rate, time units, which column is the
pulse signal, signal polarity, filter band, beat-detection settings, window
length, the driver's baseline, the natural variability and even the alarm level
are all inferred from the data.  (The constants in `Priors` are physiological
priors / safety floors, not per-dataset settings.)

PIPELINE
--------
 1 INGEST        find time column + unit, estimate fs, find gaps, rank every
                 numeric column by pulse-quality, pick best (or fuse two)
 2 CONDITION     resample to uniform grid, band-pass (upper edge tied to fs),
                 pick polarity (the one giving the most regular beat train)
 3 BEATS         peak detection -> amplitude gate -> adaptive refractory period
                 -> per-beat morphology check (correlation with neighbours)
 4 WINDOWS       window length = ~12 beats of this driver; per window: cleaned
                 mean PP (heart rate) + log-RMSSD (beat-to-beat variability);
                 windows with too few clean beats do not vote
 5 BASELINE      learned from this driver's own first clean beats, then tracked
                 slowly - but ONLY while the driver looks alert (so it can't
                 learn drowsiness). Natural variability tracked the same way.
 6 EVIDENCE      z-scores vs. own baseline -> weighted score -> one-sided CUSUM
 7 DECISION      alarm level = max(safety floor, 3 x highest CUSUM this driver
                 produced while alert) -> noisier drivers automatically get a
                 higher bar; hysteresis; states WARMUP/ALERT/SUSPECT/DROWSY/
                 NO_SIGNAL
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt, welch


# =========================================================================== #
# Priors (physiology + safety floors) - NOT user settings
# =========================================================================== #
@dataclass(frozen=True)
class Priors:
    hr_range_bpm: tuple = (35, 190)       # physiologically possible
    fs_range_hz: tuple = (8, 1000)        # plausible sampling rates
    max_resample_hz: float = 100.0        # never process faster than this
    band_lo: float = 0.5
    band_hi_cap: float = 4.0
    refractory_frac: float = 0.65         # peak closer than 65 % of local PP => double detection
    amp_gate: float = 0.55                # peak < 55 % of local systolic amplitude => bump
    max_dev: float = 0.30                 # beat interval >30 % off window median => artefact
    min_good_frac: float = 0.70           # clean beats / expected beats in window
    min_beats: int = 6
    warmup_min_s: float = 20.0
    warmup_min_beats: int = 20
    win_beats: float = 12.0               # window ~ 12 beats
    win_clip_s: tuple = (8.0, 20.0)
    tau_s: float = 120.0                  # baseline time constant
    floor_cv_mean: float = 0.02           # min sigma of window mean-PP (2 % of baseline)
    floor_lr: float = 0.20                # min sigma of log-RMSSD
    freeze_above: float = 1.0
    z_clip: tuple = (-3.0, 6.0)
    w_hr: float = 0.7
    w_hrv: float = 0.3
    k: float = 1.0                        # CUSUM slack
    h_floor: float = 25.0                 # alarm-level safety floor
    h_mult: float = 3.0                   # alarm level = h_mult x max alert-state CUSUM
    h_off_frac: float = 0.3
    suspect_frac: float = 0.5
    no_signal_s: float = 30.0


P = Priors()


# =========================================================================== #
# 1. INGEST
# =========================================================================== #
@dataclass
class Ingest:
    t: np.ndarray                 # seconds, strictly increasing
    x: np.ndarray                 # chosen/fused signal (raw units)
    fs: float
    gaps: list                    # [(t_start, t_end), ...] where data is missing
    info: dict


_TIME_HINT = re.compile(r"(time|stamp|elapsed|sec|^t$|^ts$|millis|^ms$|micro|epoch)", re.I)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and s.notna().mean() > 0.95


def _detect_time(df: pd.DataFrame):
    """-> (column name | None, scale to seconds)"""
    cands = []
    for c in df.columns:
        s = df[c]
        if not _is_numeric(s):
            continue
        v = s.dropna().values.astype(float)
        if len(v) < 50:
            continue
        d = np.diff(v)
        if (d > 0).mean() < 0.98:
            continue
        hint = bool(_TIME_HINT.search(str(c)))
        cands.append((not hint, c))
    cands.sort(key=lambda z: z[0])        # name-hinted first
    for _, c in cands:
        v = df[c].dropna().values.astype(float)
        dt = np.median(np.diff(v))
        for scale in (1.0, 1e-3, 1e-6, 1e-9):
            fs = 1.0 / (dt * scale)
            if P.fs_range_hz[0] <= fs <= P.fs_range_hz[1]:
                # reject index-like columns (dt exactly constant 1) unless name says time
                if np.allclose(np.diff(v), 1.0) and not _TIME_HINT.search(str(c)):
                    break
                return c, scale
    return None, 1.0


def _uniformize(t, x, fs):
    tu = np.arange(t[0], t[-1] - 1e-9, 1.0 / fs)
    return tu, np.interp(tu, t, x)


def _bandpass(x, fs, lo=None, hi=None):
    lo = P.band_lo if lo is None else lo
    hi = min(P.band_hi_cap, 0.4 * fs) if hi is None else hi
    sos = butter(3, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x - np.mean(x))


def _pulse_quality(t, x, fs):
    """0..1: share of band power sitting on one cardiac fundamental (+ harmonic)."""
    f, pxx = welch(x - np.mean(x), fs=fs, nperseg=min(len(x), int(fs * 12)))
    lo, hi = P.hr_range_bpm[0] / 60, P.hr_range_bpm[1] / 60
    m = (f >= lo) & (f <= hi)
    if m.sum() < 3 or pxx[m].sum() <= 0:
        return 0.0, np.nan
    f0 = f[m][np.argmax(pxx[m])]
    tot = pxx[(f >= 0.3) & (f <= min(8.0, 0.45 * fs))].sum()
    on = pxx[np.abs(f - f0) <= 0.15].sum() + pxx[np.abs(f - 2 * f0) <= 0.2].sum()
    return float(on / tot) if tot > 0 else 0.0, float(f0 * 60)


def ingest(source) -> Ingest:
    df = pd.read_csv(source, sep=None, engine="python") if not isinstance(source, pd.DataFrame) else source.copy()
    info = {}
    tcol, scale = _detect_time(df)
    if tcol is None:
        info["warning_time"] = "no time column found: assuming 25 Hz sample spacing"
        t = np.arange(len(df)) / 25.0
    else:
        t = df[tcol].values.astype(float) * scale
        info["time_column"], info["time_scale_to_s"] = tcol, scale
    ok = np.isfinite(t)
    df, t = df[ok].reset_index(drop=True), t[ok]
    order = np.argsort(t, kind="stable")
    df, t = df.iloc[order].reset_index(drop=True), t[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    df, t = df[keep].reset_index(drop=True), t[keep]

    dt = np.diff(t)
    fs_raw = 1.0 / np.median(dt)
    fs = float(min(fs_raw, P.max_resample_hz))
    gap_thr = max(0.30, 6 * np.median(dt))
    gaps = [(t[i], t[i + 1]) for i in np.where(dt > gap_thr)[0]]
    info.update(fs_estimated_hz=round(fs_raw, 2), fs_used_hz=round(fs, 2),
                duration_s=round(t[-1] - t[0], 1), n_gaps=len(gaps),
                dt_jitter_cv=round(float(np.std(dt) / np.mean(dt)), 3))

    # ---- rank numeric columns by pulse quality
    ranked = []
    for c in df.columns:
        if c == tcol or not _is_numeric(df[c]):
            continue
        v = df[c].interpolate(limit_direction="both").values.astype(float)
        if np.nanstd(v) < 1e-9:                      # constant (e.g. metadata)
            continue
        tu, xu = _uniformize(t, v, fs)
        try:
            q, f0 = _pulse_quality(tu, _bandpass(xu, fs), fs)
        except Exception:
            continue
        ranked.append((q, c, f0, v))
    if not ranked:
        raise ValueError("No usable signal column found (all columns constant/non-numeric).")
    ranked.sort(key=lambda z: -z[0])
    info["channel_quality"] = {c: round(q, 3) for q, c, _, _ in ranked}
    q1, c1, f1, v1 = ranked[0]
    if q1 < 0.15:
        info["warning_quality"] = f"best channel '{c1}' has weak cardiac periodicity (q={q1:.2f})"
    chosen, x = [c1], (v1 - v1.mean()) / v1.std()

    # ---- fuse the second channel only if it is nearly as good and well correlated
    if len(ranked) > 1:
        q2, c2, f2, v2 = ranked[1]
        if q2 >= 0.8 * q1 and q2 >= 0.15:
            tu, a = _uniformize(t, v1, fs)
            _, b = _uniformize(t, v2, fs)
            r = np.corrcoef(_bandpass(a, fs), _bandpass(b, fs))[0, 1]
            if abs(r) > 0.6:
                x = (q1 * x + np.sign(r) * q2 * (v2 - v2.mean()) / v2.std()) / (q1 + q2)
                chosen.append(c2)
    info["channels_used"] = chosen
    return Ingest(t=t, x=x, fs=fs, gaps=gaps, info=info)


# =========================================================================== #
# 2-3. CONDITION + BEATS
# =========================================================================== #
def _refine_peaks(pk_t, pk_h):
    """amplitude gate + adaptive refractory (vectorised passes)."""
    pk_t, pk_h = np.asarray(pk_t, float), np.asarray(pk_h, float)
    if len(pk_t) > 5:
        ref = pd.Series(pk_h).rolling(13, center=True, min_periods=3).quantile(0.8).values
        keep = pk_h >= P.amp_gate * ref
        pk_t, pk_h = pk_t[keep], pk_h[keep]
    for _ in range(12):
        if len(pk_t) < 4:
            break
        g = np.diff(pk_t)
        med = pd.Series(g).rolling(7, center=True, min_periods=2).median().values
        bad = np.where(g < P.refractory_frac * med)[0]
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


def _detect(t, xf, fs):
    idx, _ = find_peaks(xf, distance=max(1, int(0.30 * fs)), prominence=0.25 * np.std(xf))
    idx = idx[(idx > 0) & (idx < len(xf) - 1)]
    y0, y1, y2 = xf[idx - 1], xf[idx], xf[idx + 1]       # parabolic sub-sample refinement
    den = y0 - 2 * y1 + y2
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    pk_t = t[idx] + np.clip(delta, -1, 1) / fs
    return _refine_peaks(pk_t, xf[idx])


def _beat_quality(t, xf, fs, pk_t):
    """correlation of each beat with the median of its neighbours (morphology)."""
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
    med = np.median(pp)
    return float(np.median(np.abs(np.diff(pp))) / med) + 0.02 * abs(len(pk_t) - 1 - len(pk_t))


def extract_beats(ing: Ingest):
    fs = ing.fs
    t, x = _uniformize(ing.t, ing.x, fs)
    xf = _bandpass(x, fs)
    # polarity: beats are the narrow, sharp tips of the waveform -> positive skewness.
    # If skewness is ambiguous, fall back to the sign giving the most regular beat train.
    from scipy.stats import skew
    sk = float(skew(xf))
    cand = (1, -1) if sk >= 0 else (-1, 1)
    if abs(sk) > 0.1:
        sgn = cand[0]
        pk_t, pk_h = _detect(t, sgn * xf, fs)
        best = (0.0, sgn, pk_t, pk_h)
    else:
        best = None
        for sgn in (1, -1):
            pk_t, pk_h = _detect(t, sgn * xf, fs)
            score = _regularity(pk_t)
            if best is None or score < best[0] * 0.9:
                best = (score, sgn, pk_t, pk_h)
    _, sgn, pk_t, pk_h = best
    xf = sgn * xf
    q = _beat_quality(t, xf, fs, pk_t)
    thr = float(np.clip(np.median(q) - 4 * 1.4826 * np.median(np.abs(q - np.median(q))), 0.5, 0.75))
    good = q >= thr

    # PP intervals: both ends good, no gap crossing, physiologic
    lo, hi = 60 / P.hr_range_bpm[1], 60 / P.hr_range_bpm[0]
    t_pp, pp = [], []
    for i in range(1, len(pk_t)):
        d = pk_t[i] - pk_t[i - 1]
        if not (good[i] and good[i - 1] and lo <= d <= hi):
            continue
        if any(g0 < pk_t[i] and g1 > pk_t[i - 1] for g0, g1 in ing.gaps):
            continue
        t_pp.append(pk_t[i]); pp.append(d)
    info = dict(polarity="inverted" if sgn < 0 else "normal", n_beats=len(pk_t),
                beats_rejected_morphology=int((~good).sum()), morphology_threshold=round(thr, 2))
    return dict(t=t, xf=xf, pk_t=pk_t, good=good, t_pp=np.array(t_pp), pp=np.array(pp), info=info)


# =========================================================================== #
# 4. WINDOW FEATURES
# =========================================================================== #
def window_features(t_pp, pp, te, win):
    m = (t_pp > te - win) & (t_pp <= te)
    p, tt = pp[m], t_pp[m]
    out = dict(valid=False, mean=np.nan, lrmssd=np.nan, good_frac=0.0, n=int(len(p)))
    if len(p) < 4:
        return out
    med = np.median(p)
    keep = np.abs(p - med) <= P.max_dev * med
    out["good_frac"] = float(keep.sum() / (win / med))
    if keep.sum() < P.min_beats or out["good_frac"] < P.min_good_frac:
        return out
    out["valid"] = True
    out["mean"] = float(np.mean(p[keep]))
    idx = np.where(keep)[0]
    a, b = idx[:-1], idx[1:]
    consec = (b - a == 1) & (np.abs((tt[b] - tt[a]) - p[b]) < 0.15 * p[b])
    d = (p[b] - p[a])[consec]
    if len(d) >= 4:
        out["lrmssd"] = float(np.log(max(np.sqrt(np.mean(d ** 2)), 1e-3)))
    return out


def _calib_stats(t_pp, pp, t_lo, t_hi):
    m = (t_pp >= t_lo) & (t_pp < t_hi)
    p, tt = pp[m], t_pp[m]
    med = np.median(p)
    keep = np.abs(p - med) <= P.max_dev * med
    idx = np.where(keep)[0]
    a, b = idx[:-1], idx[1:]
    consec = (b - a == 1) & (np.abs((tt[b] - tt[a]) - p[b]) < 0.15 * p[b])
    d = (p[b] - p[a])[consec]
    lr = float(np.log(np.sqrt(np.mean(d ** 2)))) if len(d) >= 4 else np.nan
    return float(np.mean(p[keep])), lr, int(keep.sum())


# =========================================================================== #
# 5-7. ADAPTIVE DETECTOR
# =========================================================================== #
def run_detector(beats: dict, fs: float, t0: float, t_end: float):
    t_pp, pp = beats["t_pp"], beats["pp"]
    info = {}
    # ---- warm-up: first moment with >=20 s AND >=20 clean beats
    t_warm = None
    for te in np.arange(t0 + P.warmup_min_s, t_end, 1.0):
        if ((t_pp >= t0) & (t_pp < te)).sum() >= P.warmup_min_beats:
            t_warm = te
            break
    if t_warm is None or (t_pp < t_warm).sum() < 8:
        raise ValueError("Recording too short / too noisy to learn a baseline "
                         "(need ~20 s with >=20 clean beats).")
    b_m, b_lr, n_cal = _calib_stats(t_pp, pp, t0, t_warm)
    if np.isnan(b_lr):
        b_lr = np.log(0.03)
    win = float(np.clip(P.win_beats * b_m, *P.win_clip_s))
    step = max(1.0, round(win / 5, 1))
    # HRV usable only if RMSSD is above the timing-resolution noise floor
    hrv_ok = np.exp(b_lr) > 0.5 / fs
    w_hr, w_hrv = (P.w_hr, P.w_hrv) if hrv_ok else (1.0, 0.0)
    info.update(warmup_s=round(t_warm - t0, 1), window_s=win, step_s=step,
                baseline_hr_bpm=round(60 / b_m, 1), baseline_rmssd_ms=round(1000 * np.exp(b_lr), 1),
                hrv_used=bool(hrv_ok))

    var_m = (P.floor_cv_mean * b_m) ** 2
    var_lr = P.floor_lr ** 2
    alpha = step / P.tau_s
    S, alarm, M = 0.0, False, 0.0
    last_valid_t = t_warm
    rows = []
    for te in np.arange(t_warm, t_end, step):
        f = window_features(t_pp, pp, te, win)
        h_on = max(P.h_floor, P.h_mult * M)
        if not f["valid"]:
            state = "NO_SIGNAL" if te - last_valid_t > P.no_signal_s else ("DROWSY" if alarm else "HOLD")
            rows.append(dict(t=te, valid=False, state=state, S=S, alarm=alarm, h_on=h_on,
                             b_hr=60 / b_m, hr_bpm=np.nan, score=np.nan, z_hr=np.nan, z_hrv=np.nan))
            continue
        last_valid_t = te
        s_m = max(np.sqrt(var_m), P.floor_cv_mean * b_m)
        s_lr = max(np.sqrt(var_lr), P.floor_lr)
        z_hr = float(np.clip((f["mean"] - b_m) / s_m, *P.z_clip))
        z_hrv = 0.0 if (np.isnan(f["lrmssd"]) or not hrv_ok) else \
            float(np.clip((b_lr - f["lrmssd"]) / s_lr, *P.z_clip))
        score = w_hr * z_hr + w_hrv * z_hrv
        S = max(0.0, S + score - P.k)
        if not alarm and S >= h_on:
            alarm = True
        elif alarm and S <= h_on * P.h_off_frac:
            alarm = False
        looks_alert = (not alarm) and score < P.freeze_above and S < P.suspect_frac * h_on
        if looks_alert:
            M = max(M, S)                                   # driver's own alert-state CUSUM ceiling
            var_m = (1 - alpha) * var_m + alpha * (f["mean"] - b_m) ** 2
            b_m += alpha * (f["mean"] - b_m)
            if not np.isnan(f["lrmssd"]) and hrv_ok:
                var_lr = (1 - alpha) * var_lr + alpha * (f["lrmssd"] - b_lr) ** 2
                b_lr += alpha * (f["lrmssd"] - b_lr)
        state = "DROWSY" if alarm else ("SUSPECT" if S >= P.suspect_frac * h_on else "ALERT")
        rows.append(dict(t=te, valid=True, state=state, S=S, alarm=alarm, h_on=h_on,
                         b_hr=60 / b_m, hr_bpm=60 / f["mean"], score=score, z_hr=z_hr, z_hrv=z_hrv,
                         rmssd_ms=1000 * np.exp(f["lrmssd"]) if not np.isnan(f["lrmssd"]) else np.nan))
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

    def text(self) -> str:
        s, i = self.summary, self.info
        L = ["=" * 64, "AUTO-ADAPTIVE PPG DROWSINESS REPORT", "=" * 64,
             "Auto-detected settings",
             f"  time column      : {i.get('time_column', '(none)')}  (x{i.get('time_scale_to_s', 1)} -> s)",
             f"  sampling rate    : {i['fs_estimated_hz']} Hz (jitter CV {i['dt_jitter_cv']}), gaps: {i['n_gaps']}",
             f"  signal channel(s): {i['channels_used']}   quality: {i['channel_quality']}",
             f"  polarity         : {i['polarity']}",
             f"  beats            : {i['n_beats']}  (rejected by morphology: {i['beats_rejected_morphology']})",
             f"  warm-up          : {i['warmup_s']} s  -> baseline {i['baseline_hr_bpm']} bpm, "
             f"RMSSD {i['baseline_rmssd_ms']} ms (HRV used: {i['hrv_used']})",
             f"  window / step    : {i['window_s']:.1f} s / {i['step_s']} s", "",
             "Result",
             f"  drowsiness detected : {'YES' if s['drowsy_detected'] else 'no'}",
             f"  windows evaluated   : {s['n_windows']} (usable {s['pct_valid']:.0f} %)",
             f"  time in states      : " + ", ".join(f"{k} {v:.0f} %" for k, v in s['state_pct'].items()),
             f"  peak evidence (CUSUM) {s['max_S']:.1f}  vs alarm level {s['final_alarm_level']:.1f}"]
        for w in ("warning_time", "warning_quality"):
            if w in i:
                L.append(f"  WARNING: {i[w]}")
        if len(self.events):
            L += ["", "Drowsiness events", self.events.round(1).to_string(index=False)]
        return "\n".join(L)


def analyze(source, plot: str | None = None) -> Report:
    ing = ingest(source)
    b = extract_beats(ing)
    t0, t_end = b["t"][0], b["t"][-1]
    win_df, dinfo, t_warm = run_detector(b, ing.fs, t0, t_end)
    info = {**ing.info, **b["info"], **dinfo}

    # events
    ev, start = [], None
    for _, r in win_df.iterrows():
        if r.alarm and start is None:
            start = r.t
        if (not r.alarm) and start is not None:
            ev.append((start, r.t)); start = None
    if start is not None:
        ev.append((start, win_df.t.iloc[-1]))
    rows = []
    for a, z in ev:
        seg = win_df[(win_df.t >= a) & (win_df.t <= z) & win_df.valid]
        rows.append(dict(start_s=a, end_s=z, duration_s=z - a,
                         hr_change_pct=100 * (seg.hr_bpm.mean() / seg.b_hr.mean() - 1) if len(seg) else np.nan,
                         rmssd_ms=seg.rmssd_ms.mean() if "rmssd_ms" in seg else np.nan))
    events = pd.DataFrame(rows, columns=["start_s", "end_s", "duration_s", "hr_change_pct", "rmssd_ms"])
    pct = (win_df.state.value_counts(normalize=True) * 100).to_dict()
    summary = dict(drowsy_detected=bool(win_df.alarm.any()),
                   first_alarm_s=float(win_df.loc[win_df.alarm, "t"].min()) if win_df.alarm.any() else None,
                   n_windows=len(win_df), pct_valid=100 * win_df.valid.mean(),
                   state_pct=pct, max_S=float(win_df.S.max()),
                   final_alarm_level=float(win_df.h_on.iloc[-1]))
    rep = Report(win_df, events, summary, info, b)
    if plot:
        plot_report(rep, plot)
    return rep


def plot_report(rep: Report, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b, w = rep.beats, rep.windows
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(b["t"], b["xf"], lw=.7)
    pk_y = np.interp(b["pk_t"], b["t"], b["xf"])
    ax[0].plot(b["pk_t"][b["good"]], pk_y[b["good"]], "g.", ms=4, label="accepted beat")
    ax[0].plot(b["pk_t"][~b["good"]], pk_y[~b["good"]], "rx", ms=5, label="rejected beat")
    ax[0].set_ylabel("PPG (band-passed)"); ax[0].legend(fontsize=7)
    ax[1].plot(w.t, w.hr_bpm, "k-", label="window HR"); ax[1].plot(w.t, w.b_hr, "g--", label="adaptive baseline")
    ax[1].set_ylabel("bpm"); ax[1].legend(fontsize=7)
    ax[2].plot(w.t, w.z_hr, label="z heart-rate slowing"); ax[2].plot(w.t, w.z_hrv, label="z variability drop")
    ax[2].axhline(0, c="gray", lw=.5); ax[2].set_ylabel("z-score"); ax[2].legend(fontsize=7)
    ax[3].plot(w.t, w.S, "k-", label="evidence (CUSUM)"); ax[3].plot(w.t, w.h_on, "r:", label="auto alarm level")
    ax[3].set_ylabel("evidence"); ax[3].set_xlabel("time (s)"); ax[3].legend(fontsize=7)
    col = {"DROWSY": "red", "SUSPECT": "orange", "NO_SIGNAL": "gray"}
    for a in ax:
        for st, c in col.items():
            m = (w.state == st).values
            if m.any():
                tt = w.t.values
                for i in np.where(m)[0]:
                    a.axvspan(tt[i], tt[min(i + 1, len(tt) - 1)], color=c, alpha=.12, lw=0)
    fig.suptitle("Auto-adaptive PPG drowsiness analysis  (red = DROWSY, orange = SUSPECT, grey = no signal)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python auto_drowsiness.py recording.csv")
    src = sys.argv[1]
    stem = Path(src).stem
    rep = analyze(src, plot=f"{stem}_report.png")
    print(rep.text())
    rep.windows.to_csv(f"{stem}_windows.csv", index=False)
    print(f"\nsaved: {stem}_report.png, {stem}_windows.csv")