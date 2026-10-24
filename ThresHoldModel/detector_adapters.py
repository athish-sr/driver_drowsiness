"""
detector_adapters.py
====================
Translate TImeDynamic_v2.Report and FreqDynamic_v2.FreqReport objects into
the normalized DetectorResult schema used by the fusion engine.

This is the ONLY file in the project that imports from TImeDynamic_v2 and
FreqDynamic_v2.  The fusion_engine.py module must NOT import from those modules.

State normalization rules
--------------------------
Time-domain states:
    ALERT       -> "ALERT"
    SUSPECT     -> "SUSPECT"
    DROWSY      -> "DROWSY"
    HOLD        -> "HOLD"   (semantically distinct from ALERT: CUSUM below hysteresis
                              during recovery; NOT equated to ALERT)
    NO_SIGNAL   -> "NO_SIGNAL"

Frequency-domain states:
    LEARN       -> "INSUFFICIENT_DATA"
    UNUSED      -> "INSUFFICIENT_DATA"
    ALERT       -> "ALERT"
    SUSPECT     -> "SUSPECT"
    DROWSY      -> "DROWSY"
    HOLD        -> "HOLD"   (same caution: not equated to ALERT)
    NO_SIGNAL   -> "NO_SIGNAL"

Validity logic
--------------
adapt_time_domain_report:
    valid=False  if prediction == "INSUFFICIENT_DATA"
    valid=False  if windows DataFrame is empty
    valid=False  if prediction == "ALERT" and no window has valid=True
    valid=True   otherwise

adapt_freq_domain_report:
    valid=False  if prediction == "INSUFFICIENT_DATA"
    valid=False  if windows DataFrame is empty
    valid=False  if no window has state not in ("LEARN", "UNUSED")
    valid=True   otherwise
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

# These imports are intentionally placed ONLY in this file.
# fusion_engine.py must not import from these modules.
try:
    from ThresHoldModel.TImeDynamic_v2 import Report
    from ThresHoldModel.FreqDynamic_v2 import FreqReport
except ImportError:
    # Allow relative imports when running inside the package directory
    try:
        import TImeDynamic_v2 as _td_mod
        import FreqDynamic_v2 as _fd_mod
        Report    = _td_mod.Report
        FreqReport = _fd_mod.FreqReport
    except ImportError:
        Report    = None  # type: ignore
        FreqReport = None  # type: ignore

try:
    from ThresHoldModel.fusion_engine import DetectorResult
except ImportError:
    from fusion_engine import DetectorResult  # type: ignore

# ---------------------------------------------------------------------------
# State normalization maps
# ---------------------------------------------------------------------------

_TD_STATE_MAP: dict[str, str] = {
    "ALERT":     "ALERT",
    "SUSPECT":   "SUSPECT",
    "DROWSY":    "DROWSY",
    "HOLD":      "HOLD",
    "NO_SIGNAL": "NO_SIGNAL",
}

_FREQ_STATE_MAP: dict[str, str] = {
    "LEARN":     "INSUFFICIENT_DATA",
    "UNUSED":    "INSUFFICIENT_DATA",
    "ALERT":     "ALERT",
    "SUSPECT":   "SUSPECT",
    "DROWSY":    "DROWSY",
    "HOLD":      "HOLD",
    "NO_SIGNAL": "NO_SIGNAL",
}

_FREQ_SKIP_STATES = frozenset({"LEARN", "UNUSED"})


def _normalize_td_state(raw: str) -> str:
    """Normalize a raw time-domain window state string."""
    return _TD_STATE_MAP.get(str(raw).strip().upper(), "NO_SIGNAL")


def _normalize_freq_state(raw: str) -> str:
    """Normalize a raw frequency-domain window state string."""
    return _FREQ_STATE_MAP.get(str(raw).strip().upper(), "NO_SIGNAL")


def _safe_float(val) -> Optional[float]:
    """Convert to float or return None if not finite."""
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Time-domain adapters
# ---------------------------------------------------------------------------

def adapt_time_domain_report(report, now: float = None) -> Optional[DetectorResult]:
    """Convert TImeDynamic_v2.Report to a DetectorResult.

    Parameters
    ----------
    report : TImeDynamic_v2.Report
        Result from TImeDynamic_v2.analyze() or analyze_fusion_csv().
    now : float, optional
        time.monotonic() at call time. Measured internally if not provided.

    Returns
    -------
    DetectorResult or None if report is None.

    Notes
    -----
    - Uses report.summary['prediction'] for the recording-level state.
    - If report.windows is non-empty, uses the LAST valid window for
      window_start/window_end and alarm_active.  Falls back to the last
      window if no valid windows exist.
    - evidence_score = last window's S value if available.
    - valid=False for INSUFFICIENT_DATA prediction or empty windows.
    """
    if report is None:
        return None
    if now is None:
        now = time.monotonic()

    summary  = report.summary
    windows  = report.windows
    pred     = summary.get("prediction", "INSUFFICIENT_DATA")

    # ---- Validity logic ----
    if pred == "INSUFFICIENT_DATA":
        valid = False
    elif windows is None or (hasattr(windows, "__len__") and len(windows) == 0):
        valid = False
    elif pred == "ALERT":
        # If no window has valid=True, treat as insufficient
        valid_windows = windows[windows["valid"] == True] if "valid" in windows.columns else pd.DataFrame()
        valid = len(valid_windows) > 0
    else:
        valid = True

    # ---- Recording-level state from prediction ----
    pred_upper = str(pred).strip().upper()
    if pred_upper == "INSUFFICIENT_DATA":
        rec_state = "INSUFFICIENT_DATA"
    else:
        rec_state = _normalize_td_state(pred_upper)

    # ---- Window boundary and alarm from the recording result ----
    # Batch reports are fused after both detectors have processed the complete
    # recording. The final window may already be in recovery even though the
    # detector correctly found a drowsy event earlier in the recording.
    window_start = 0.0
    window_end   = 0.0
    alarm_active = False
    evidence     = None
    final_window_alarm_active = False

    if windows is not None and len(windows) > 0:
        # Prefer last valid window; fall back to last window
        if "valid" in windows.columns:
            valid_df = windows[windows["valid"] == True]
            last_w   = valid_df.iloc[-1] if len(valid_df) > 0 else windows.iloc[-1]
        else:
            last_w = windows.iloc[-1]

        t_val       = _safe_float(last_w.get("t")) if hasattr(last_w, "get") else _safe_float(last_w["t"])
        window_end  = t_val if t_val is not None else 0.0
        # Estimate window_start from t (end time) and window duration in summary info
        win_s = 0.0
        if hasattr(report, "info") and isinstance(report.info, dict):
            win_s = _safe_float(report.info.get("window_s", 0.0)) or 0.0
        window_start = window_end - win_s if win_s > 0 else 0.0

        alarm_col = last_w.get("alarm") if hasattr(last_w, "get") else last_w["alarm"]
        final_window_alarm_active = bool(alarm_col)

        s_val = last_w.get("S") if hasattr(last_w, "get") else last_w["S"]
        evidence = _safe_float(s_val)

    alarm_active = bool(pred_upper == "DROWSY" and summary.get("drowsy_detected", False))

    reason = pred
    if summary.get("reason"):
        reason = f"{pred}: {summary['reason']}"

    meta = {
        "prediction":        pred,
        "drowsy_detected":   summary.get("drowsy_detected"),
        "first_alarm_s":     summary.get("first_alarm_s"),
        "n_windows":         summary.get("n_windows"),
        "pct_valid":         summary.get("pct_valid"),
        "state_pct":         summary.get("state_pct"),
        "max_S":             summary.get("max_S"),
        "final_alarm_level": summary.get("final_alarm_level"),
        "final_window_alarm_active": final_window_alarm_active,
    }
    if hasattr(report, "info") and isinstance(report.info, dict):
        meta["info"] = {k: v for k, v in report.info.items()
                        if k not in ("beats", "quality")}

    return DetectorResult(
        model="time_domain",
        state=rec_state,
        valid=valid,
        window_start=window_start,
        window_end=window_end,
        decision_time=now,
        alarm_active=alarm_active,
        evidence_score=evidence,
        reason=reason,
        metadata=meta,
    )


def adapt_time_domain_window(window_row, report_info: dict, now: float = None) -> DetectorResult:
    """Convert a single window row from Report.windows to a DetectorResult.

    For live/streaming integration where windows are processed one at a time.

    Parameters
    ----------
    window_row : dict or pd.Series
        A row from Report.windows with keys: t, valid, state, S, alarm, h_on,
        hr_bpm, b_hr, score, z_hr, z_hrv, rmssd_ms.
    report_info : dict
        Context from Report.info (e.g. window_s for boundary reconstruction).
    now : float, optional
        time.monotonic() at call time.
    """
    if now is None:
        now = time.monotonic()

    def _get(row, key, default=None):
        if hasattr(row, "get"):
            return row.get(key, default)
        try:
            return row[key]
        except (KeyError, IndexError):
            return default

    raw_state  = str(_get(window_row, "state", "NO_SIGNAL"))
    state      = _normalize_td_state(raw_state)
    is_valid   = bool(_get(window_row, "valid", False))
    t_val      = _safe_float(_get(window_row, "t", 0.0)) or 0.0
    win_s      = _safe_float(report_info.get("window_s", 0.0)) or 0.0
    alarm      = bool(_get(window_row, "alarm", False))
    evidence   = _safe_float(_get(window_row, "S"))

    meta = {
        "h_on":     _safe_float(_get(window_row, "h_on")),
        "hr_bpm":   _safe_float(_get(window_row, "hr_bpm")),
        "b_hr":     _safe_float(_get(window_row, "b_hr")),
        "score":    _safe_float(_get(window_row, "score")),
        "z_hr":     _safe_float(_get(window_row, "z_hr")),
        "z_hrv":    _safe_float(_get(window_row, "z_hrv")),
        "rmssd_ms": _safe_float(_get(window_row, "rmssd_ms")),
    }

    return DetectorResult(
        model="time_domain",
        state=state,
        valid=is_valid,
        window_start=t_val - win_s if win_s > 0 else 0.0,
        window_end=t_val,
        decision_time=now,
        alarm_active=alarm,
        evidence_score=evidence,
        reason=raw_state,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Frequency-domain adapters
# ---------------------------------------------------------------------------

def adapt_freq_domain_report(report, now: float = None) -> Optional[DetectorResult]:
    """Convert FreqDynamic_v2.FreqReport to a DetectorResult.

    Parameters
    ----------
    report : FreqDynamic_v2.FreqReport
        Result from FreqDynamic_v2.analyze() or analyze_fusion_csv().
    now : float, optional
        time.monotonic() at call time.

    Returns
    -------
    DetectorResult or None if report is None.

    Notes
    -----
    - Uses report.summary['prediction'] for the recording-level state.
    - Uses the LAST window where state not in ('LEARN', 'UNUSED') for window
      boundaries, alarm_active, and evidence_score.
    - HOLD state is NOT equated to ALERT (stored as 'HOLD' explicitly; also
      recorded in metadata.hold_note).
    - valid=False for INSUFFICIENT_DATA or no post-LEARN windows.
    """
    if report is None:
        return None
    if now is None:
        now = time.monotonic()

    summary = report.summary
    windows = report.windows
    pred    = summary.get("prediction", "INSUFFICIENT_DATA")

    # ---- Validity logic ----
    if pred == "INSUFFICIENT_DATA":
        valid = False
    elif windows is None or (hasattr(windows, "__len__") and len(windows) == 0):
        valid = False
    else:
        # Check for any window beyond the LEARN/UNUSED phase
        if "state" in windows.columns:
            post_learn = windows[~windows["state"].isin(_FREQ_SKIP_STATES)]
        else:
            post_learn = pd.DataFrame()
        valid = len(post_learn) > 0

    # ---- Recording-level state ----
    pred_upper = str(pred).strip().upper()
    if pred_upper == "INSUFFICIENT_DATA":
        rec_state = "INSUFFICIENT_DATA"
    else:
        rec_state = _normalize_freq_state(pred_upper)

    # ---- Window boundary and alarm from the recording result ----
    # The frequency detector can lag the time-domain detector by several
    # windows. Use the recording-level event for batch fusion instead of
    # requiring the final frequency window to remain in alarm.
    window_start = 0.0
    window_end   = 60.0   # default one window length
    alarm_active = False
    evidence     = None
    final_window_alarm_active = False

    if windows is not None and len(windows) > 0 and "state" in windows.columns:
        post_learn_df = windows[~windows["state"].isin(_FREQ_SKIP_STATES)]
        if len(post_learn_df) > 0:
            last_w = post_learn_df.iloc[-1]
            ws = _safe_float(last_w.get("start_s") if hasattr(last_w, "get") else last_w["start_s"])
            we = _safe_float(last_w.get("end_s")   if hasattr(last_w, "get") else last_w["end_s"])
            window_start = ws if ws is not None else 0.0
            window_end   = we if we is not None else 60.0

            alarm_col    = last_w.get("alarm") if hasattr(last_w, "get") else last_w["alarm"]
            final_window_alarm_active = bool(alarm_col)

            s_col    = last_w.get("S") if hasattr(last_w, "get") else last_w["S"]
            evidence = _safe_float(s_col)

    alarm_active = bool(pred_upper == "DROWSY" and summary.get("drowsy_detected", False))

    reason = pred
    if summary.get("reason"):
        reason = f"{pred}: {summary['reason']}"

    meta = {
        "prediction":      pred,
        "drowsy_detected": summary.get("drowsy_detected"),
        "first_alarm_s":   summary.get("first_alarm_s"),
        "n_windows":       summary.get("n_windows"),
        "pct_valid":       summary.get("pct_valid"),
        "state_pct":       summary.get("state_pct"),
        "max_S":           summary.get("max_S"),
        "baseline":        summary.get("baseline"),
        "sigma":           summary.get("sigma"),
        "sigma_floor":     summary.get("sigma_floor"),
        "sigma_learned":   summary.get("sigma_learned"),
        "n_learn":         summary.get("n_learn"),
        "final_window_alarm_active": final_window_alarm_active,
    }
    # HOLD note: HOLD is semantically distinct from ALERT (below hysteresis during recovery)
    if rec_state == "HOLD":
        meta["hold_note"] = (
            "HOLD state: CUSUM is below the hysteresis threshold during alarm recovery. "
            "This is NOT equivalent to ALERT."
        )
    if hasattr(report, "info") and isinstance(report.info, dict):
        meta["info"] = report.info
    if hasattr(report, "cfg"):
        try:
            meta["cfg"] = {
                "window_s": report.cfg.window_s,
                "hop_s":    report.cfg.hop_s,
                "learn_s":  report.cfg.learn_s,
            }
        except AttributeError:
            pass

    return DetectorResult(
        model="frequency_domain",
        state=rec_state,
        valid=valid,
        window_start=window_start,
        window_end=window_end,
        decision_time=now,
        alarm_active=alarm_active,
        evidence_score=evidence,
        reason=reason,
        metadata=meta,
    )


def adapt_freq_domain_window(window_row, report_info: dict, now: float = None) -> DetectorResult:
    """Convert a single frequency-domain window row to a DetectorResult.

    For live integration where windows are processed one at a time.

    Parameters
    ----------
    window_row : dict or pd.Series
        A row from FreqReport.windows with keys: start_s, end_s, valid, state,
        S, alarm, lnLFHF.
    report_info : dict
        Context from FreqReport.info or FreqReport.summary.
    now : float, optional
        time.monotonic() at call time.
    """
    if now is None:
        now = time.monotonic()

    def _get(row, key, default=None):
        if hasattr(row, "get"):
            return row.get(key, default)
        try:
            return row[key]
        except (KeyError, IndexError):
            return default

    raw_state  = str(_get(window_row, "state", "NO_SIGNAL"))
    state      = _normalize_freq_state(raw_state)
    is_valid   = bool(_get(window_row, "valid", False))
    ws         = _safe_float(_get(window_row, "start_s", 0.0)) or 0.0
    we         = _safe_float(_get(window_row, "end_s", 60.0)) or 60.0
    alarm      = bool(_get(window_row, "alarm", False))
    evidence   = _safe_float(_get(window_row, "S"))

    # LEARN/UNUSED windows are not valid evidence
    if raw_state.strip().upper() in _FREQ_SKIP_STATES:
        is_valid   = False
        alarm      = False
        state      = "INSUFFICIENT_DATA"

    meta = {
        "lnLFHF":  _safe_float(_get(window_row, "lnLFHF")),
        "hr_bpm":  _safe_float(_get(window_row, "hr_bpm")),
        "z":       _safe_float(_get(window_row, "z")),
        "raw_state": raw_state,
    }
    if state == "HOLD":
        meta["hold_note"] = (
            "HOLD state: CUSUM is below the hysteresis threshold. NOT equivalent to ALERT."
        )

    return DetectorResult(
        model="frequency_domain",
        state=state,
        valid=is_valid,
        window_start=ws,
        window_end=we,
        decision_time=now,
        alarm_active=alarm,
        evidence_score=evidence,
        reason=raw_state,
        metadata=meta,
    )
