"""
fusion_engine.py
================
Asymmetric fusion engine for the dual-PPG driver drowsiness detection system.

Combines the time-domain (TImeDynamic_v2) and frequency-domain (FreqDynamic_v2)
threshold detectors without modifying either.  The engine receives normalized
DetectorResult objects from detector_adapters.py and applies asymmetric fusion
rules to produce a FusedResult.

Imports
-------
This module must NOT import from TImeDynamic_v2 or FreqDynamic_v2.
All model-specific translation happens in detector_adapters.py.

Fusion strategy (asymmetric)
-----------------------------
A single valid DROWSY from either model immediately activates a fused alarm.
Neither detector can veto the other's positive alarm.
ALERT from one model does not clear a valid alarm raised by the other.
Recovery requires both relevant detectors to show sustained non-alarm evidence
for recovery_interval_s of cumulative valid-evidence time.

See FusionEngine.evaluate() docstring for full rule list.
"""
from __future__ import annotations

import time
import datetime
from dataclasses import dataclass, field
from typing import Optional

try:
    from ThresHoldModel.fusion_config import FusionConfig
except ImportError:
    from fusion_config import FusionConfig  # type: ignore

# ---------------------------------------------------------------------------
# Reason codes (module-level string constants)
# ---------------------------------------------------------------------------
RC_TIME_DOMAIN_ALARM           = "TIME_DOMAIN_ALARM"
RC_FREQUENCY_DOMAIN_ALARM      = "FREQUENCY_DOMAIN_ALARM"
RC_BOTH_DETECTORS_ALARM        = "BOTH_DETECTORS_ALARM"
RC_TIME_DOMAIN_SUSPECT         = "TIME_DOMAIN_SUSPECT"
RC_STALE_FREQUENCY_RESULT      = "STALE_FREQUENCY_RESULT"
RC_TIME_DOMAIN_INVALID         = "TIME_DOMAIN_INVALID"
RC_FREQUENCY_DOMAIN_INVALID    = "FREQUENCY_DOMAIN_INVALID"
RC_ALARM_LATCHED               = "ALARM_LATCHED"
RC_RECOVERY_IN_PROGRESS        = "RECOVERY_IN_PROGRESS"
RC_RECOVERY_COMPLETED          = "RECOVERY_COMPLETED"
RC_INSUFFICIENT_VALID_DATA     = "INSUFFICIENT_VALID_DATA"
RC_FREQ_PROMOTES_SUSPECT_TO_DROWSY = "FREQ_PROMOTES_SUSPECT_TO_DROWSY"
RC_NO_ACTIVE_ALARM             = "NO_ACTIVE_ALARM"

# States (normalized, model-independent)
_DROWSY            = "DROWSY"
_SUSPECT           = "SUSPECT"
_ALERT             = "ALERT"
_HOLD              = "HOLD"
_NO_SIGNAL         = "NO_SIGNAL"
_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# States that must NEVER be interpreted as ALERT evidence
_NON_ALERT_STATES = frozenset({_NO_SIGNAL, _INSUFFICIENT_DATA, "LEARN", "UNUSED"})

# States that represent an active alarm from a model
_ALARM_STATES = frozenset({_DROWSY})

# State hierarchy for output: higher index wins when no alarm is active
_STATE_ORDER = [_INSUFFICIENT_DATA, _NO_SIGNAL, _ALERT, _HOLD, _SUSPECT, _DROWSY]


def _state_rank(s: str) -> int:
    try:
        return _STATE_ORDER.index(s)
    except ValueError:
        return _STATE_ORDER.index(_ALERT)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectorResult:
    """Normalized output from a single detector, produced by detector_adapters.py.

    Fields
    ------
    model : str
        "time_domain" or "frequency_domain".
    state : str
        Normalized state string (ALERT, SUSPECT, DROWSY, HOLD, NO_SIGNAL, INSUFFICIENT_DATA).
    valid : bool
        Whether this result carries usable evidence for alarm decisions.
    window_start : float
        Start of the analyzed interval in recording time (seconds).
    window_end : float
        End of the analyzed interval in recording time (seconds).
    decision_time : float
        time.monotonic() at the moment this result was generated.
    alarm_active : bool
        Whether the detector's own alarm flag is currently set.
    evidence_score : float | None
        Optional CUSUM score or other numeric evidence. None if not available.
    reason : str
        Concise explanation of the result or why it is invalid.
    metadata : dict
        Original model-specific fields for debugging and auditing.
    """
    model: str
    state: str
    valid: bool
    window_start: float
    window_end: float
    decision_time: float
    alarm_active: bool
    evidence_score: Optional[float]
    reason: str
    metadata: dict


@dataclass
class FusedResult:
    """Output of a single fusion evaluation.

    Fields
    ------
    state : str
        Fused state: ALERT, SUSPECT, DROWSY, NO_SIGNAL, or INSUFFICIENT_DATA.
    decision_time : float
        time.monotonic() at which this result was produced.
    time_domain_state : str
        Normalized state from the time-domain detector at evaluation time.
    frequency_domain_state : str
        Normalized state from the frequency-domain detector at evaluation time.
    time_domain_valid : bool
        Whether the stored time-domain result is valid.
    frequency_domain_valid : bool
        Whether the stored frequency-domain result is valid.
    time_domain_result_age_s : float | None
        Seconds since decision_time of the time-domain result (None if no result).
    frequency_domain_result_age_s : float | None
        Seconds since decision_time of the frequency-domain result (None if no result).
    active_alarm_sources : list[str]
        Model names that contributed to the current active alarm.
    reason_code : str
        One of the RC_* constants explaining the primary decision path.
    reason : str
        Human-readable explanation of the decision.
    alarm_active : bool
        Whether the fused alarm is currently active.
    metadata : dict
        Diagnostic metadata (always present; richer when FusionConfig.debug=True).
    """
    state: str
    decision_time: float
    time_domain_state: str
    frequency_domain_state: str
    time_domain_valid: bool
    frequency_domain_valid: bool
    time_domain_result_age_s: Optional[float]
    frequency_domain_result_age_s: Optional[float]
    active_alarm_sources: list
    reason_code: str
    reason: str
    alarm_active: bool
    metadata: dict


# ---------------------------------------------------------------------------
# Fusion engine
# ---------------------------------------------------------------------------

class FusionEngine:
    """Asymmetric fusion engine for the drowsiness detection system.

    Usage
    -----
    engine = FusionEngine(FusionConfig())
    # Feed each detector's result as soon as it becomes available:
    engine.update_detector_result(td_result)
    engine.update_detector_result(freq_result)
    fused = engine.evaluate()

    The engine is synchronous and event-driven.  update_detector_result()
    stores the latest result for its model and returns an immediate evaluation.
    evaluate() can also be called independently at any time.
    """

    def __init__(self, config: FusionConfig):
        self._cfg = config
        self._td_result: Optional[DetectorResult] = None
        self._freq_result: Optional[DetectorResult] = None
        self._alarm_active: bool = False
        self._alarm_sources: set = set()          # model names that triggered the alarm
        self._recovery_start: Optional[float] = None   # monotonic time when recovery started
        self._recovery_valid_s: float = 0.0            # cumulative valid-evidence seconds in recovery
        self._last_evaluate_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_detector_result(self, result: DetectorResult) -> "FusedResult":
        """Validate and store a new detector result, then evaluate immediately.

        Parameters
        ----------
        result : DetectorResult
            Normalized detector output from detector_adapters.py.

        Returns
        -------
        FusedResult
            Fused state immediately after incorporating the new result.
        """
        if result.model == "time_domain":
            self._td_result = result
        elif result.model == "frequency_domain":
            self._freq_result = result
        else:
            raise ValueError(f"Unknown model '{result.model}'; expected 'time_domain' or 'frequency_domain'")
        return self.evaluate(now=result.decision_time)

    def evaluate(self, now: float = None) -> FusedResult:
        """Evaluate the latest usable information and return the current fused output.

        Asymmetric fusion rules applied here:

        Rule 1:  Valid + active time-domain DROWSY -> fused DROWSY immediately.
        Rule 2:  Valid + active freq-domain DROWSY -> fused DROWSY immediately.
        Rule 3:  Freq-domain ALERT must NEVER veto a valid time-domain DROWSY.
        Rule 4:  Time-domain ALERT must NEVER veto a valid freq-domain DROWSY.
        Rule 5:  Time-domain SUSPECT + freq ALERT/NO_SIGNAL/invalid/stale -> remains SUSPECT.
        Rule 6:  Time-domain SUSPECT + fresh valid freq DROWSY -> promotes to fused DROWSY.
        Rule 7:  NO_SIGNAL / INSUFFICIENT_DATA / invalid / warm-up / stale -> NOT treated as ALERT.
        Rule 8:  Missing/stale freq result must never BLOCK a valid time-domain alarm.
        Rule 9:  Active fused alarm not cleared by a single contradictory ALERT.
        Rule 10: Recovery requires both relevant detectors to show sustained recovery for
                 recovery_interval_s of cumulative valid-evidence time.
                 - Time-domain-only alarm: only time-domain recovery is required.
                 - Freq-triggered alarm: both must show recovery.
        Rule 11: Signal loss during active alarm: keep alarm, add diagnostic in metadata.
        Rule 12: A stale positive result must not activate a NEW alarm
                 (when require_fresh_result_for_new_alarm=True).
        Rule 13: Overlapping freq windows do NOT count as independent confirmations
                 (only the most recent result per model is stored).

        Parameters
        ----------
        now : float, optional
            Current time.monotonic() value. Defaults to time.monotonic().

        Returns
        -------
        FusedResult
        """
        if now is None:
            now = time.monotonic()
        self._last_evaluate_time = now

        # --- Compute ages and freshness ---
        td_age    = (now - self._td_result.decision_time)   if self._td_result   is not None else None
        freq_age  = (now - self._freq_result.decision_time) if self._freq_result is not None else None

        td_fresh   = td_age   is not None   # time-domain has no freshness limit (uses adaptive windows)
        freq_fresh = (
            freq_age is not None
            and freq_age < self._cfg.freq_result_freshness_s  # strictly less-than: at-boundary = stale
        )

        td_r    = self._td_result
        freq_r  = self._freq_result

        td_state   = td_r.state   if td_r   is not None else _INSUFFICIENT_DATA
        freq_state = freq_r.state if freq_r is not None else _INSUFFICIENT_DATA

        td_valid   = bool(td_r   is not None and td_r.valid   and td_fresh)
        freq_valid = bool(freq_r is not None and freq_r.valid and freq_fresh)

        # Whether each detector is currently showing an ACTIVE alarm (valid + fresh + alarm flag)
        td_alarm_now   = bool(td_valid   and td_r.alarm_active   and td_r.state == _DROWSY)
        freq_alarm_now = bool(freq_valid and freq_r.alarm_active and freq_r.state == _DROWSY)

        meta = {}
        if self._cfg.debug:
            meta["td_age_s"]   = td_age
            meta["freq_age_s"] = freq_age
            meta["td_fresh"]   = td_fresh
            meta["freq_fresh"] = freq_fresh
            meta["td_alarm_now"]   = td_alarm_now
            meta["freq_alarm_now"] = freq_alarm_now
            meta["alarm_sources"]  = list(self._alarm_sources)

        # =================================================================
        # Step 3: Check for NEW alarm activation (when not already alarming)
        # =================================================================
        if not self._alarm_active:
            new_alarm_sources = set()

            # Rule 1: Valid + active time-domain DROWSY
            if td_alarm_now:
                # Rule 12: stale positive cannot open new alarm
                if self._cfg.require_fresh_result_for_new_alarm and not td_fresh:
                    pass  # stale — cannot open
                else:
                    new_alarm_sources.add("time_domain")

            # Rule 2: Valid + active freq-domain DROWSY
            if freq_alarm_now:
                if self._cfg.require_fresh_result_for_new_alarm and not freq_fresh:
                    pass  # stale — cannot open
                else:
                    new_alarm_sources.add("frequency_domain")

            # Rule 6: Time-domain SUSPECT + fresh valid freq DROWSY promotes to DROWSY
            if (
                not new_alarm_sources                      # not already triggering
                and td_valid
                and td_r.state == _SUSPECT
                and freq_valid
                and freq_r.state == _DROWSY
                and freq_r.alarm_active
            ):
                new_alarm_sources.add("frequency_domain")  # freq promoted the suspect

            if new_alarm_sources:
                self._alarm_active  = True
                self._alarm_sources = new_alarm_sources
                self._recovery_start    = None
                self._recovery_valid_s  = 0.0

        # =================================================================
        # Step 4: Recovery management (when alarm is active)
        # =================================================================
        if self._alarm_active:
            # Determine which detectors need to show recovery
            # Rule 10: For time-domain-only alarm, only TD needs to clear.
            #           For freq-triggered alarm, both must clear.
            need_td_recovery   = True   # always track TD recovery
            need_freq_recovery = "frequency_domain" in self._alarm_sources

            # Is each relevant detector showing recovery?
            # Recovery = valid, alarm NOT active, state not DROWSY
            td_recovering   = (
                td_valid
                and not td_r.alarm_active
                and td_r.state != _DROWSY
            )
            freq_recovering = (
                not need_freq_recovery           # freq not required -> counts as recovered
                or (freq_valid and not freq_r.alarm_active and freq_r.state != _DROWSY)
            )

            both_recovering = td_recovering and freq_recovering

            # Is there any valid evidence at all (to track recovery timer)?
            any_valid = td_valid or (freq_valid if need_freq_recovery else True)

            if both_recovering:
                # Start or continue recovery timer
                if self._recovery_start is None:
                    self._recovery_start = now
                    self._recovery_valid_s = 0.0

                dt = now - self._last_valid_recovery_time if hasattr(self, "_last_valid_recovery_time") and self._last_valid_recovery_time is not None else 0.0
                # Accumulate valid recovery time
                self._recovery_valid_s += dt
                self._last_valid_recovery_time = now

                if self._recovery_valid_s >= self._cfg.recovery_interval_s:
                    # Recovery complete
                    self._alarm_active = False
                    self._alarm_sources = set()
                    self._recovery_start = None
                    self._recovery_valid_s = 0.0
                    self._last_valid_recovery_time = None
                    return self._build_result(
                        state=_ALERT,
                        alarm_active=False,
                        reason_code=RC_RECOVERY_COMPLETED,
                        reason="Recovery interval completed; fused alarm cleared.",
                        td_state=td_state, freq_state=freq_state,
                        td_valid=bool(td_r is not None and td_r.valid),
                        freq_valid=bool(freq_r is not None and freq_r.valid),
                        td_age=td_age, freq_age=freq_age,
                        sources=[], now=now, meta=meta,
                    )
            else:
                # Not recovering
                if not any_valid:
                    # Signal loss during alarm (Rule 11)
                    meta["signal_loss"] = True
                    meta["signal_loss_at"] = datetime.datetime.now().isoformat()
                    # Pause or reset recovery timer
                    if not self._cfg.recovery_pause_on_invalid:
                        self._recovery_start = None
                        self._recovery_valid_s = 0.0
                    self._last_valid_recovery_time = None
                else:
                    # At least one detector still showing alarm or not yet recovered
                    # Reset recovery
                    self._recovery_start = None
                    self._recovery_valid_s = 0.0
                    self._last_valid_recovery_time = None

                return self._build_result(
                    state=_DROWSY,
                    alarm_active=True,
                    reason_code=RC_ALARM_LATCHED,
                    reason="Fused alarm is active; awaiting recovery from all relevant detectors.",
                    td_state=td_state, freq_state=freq_state,
                    td_valid=bool(td_r is not None and td_r.valid),
                    freq_valid=bool(freq_r is not None and freq_r.valid),
                    td_age=td_age, freq_age=freq_age,
                    sources=list(self._alarm_sources), now=now, meta=meta,
                )

            # Alarm still active, recovery in progress
            return self._build_result(
                state=_DROWSY,
                alarm_active=True,
                reason_code=RC_RECOVERY_IN_PROGRESS,
                reason=f"Recovery in progress ({self._recovery_valid_s:.1f} / {self._cfg.recovery_interval_s:.0f} s).",
                td_state=td_state, freq_state=freq_state,
                td_valid=bool(td_r is not None and td_r.valid),
                freq_valid=bool(freq_r is not None and freq_r.valid),
                td_age=td_age, freq_age=freq_age,
                sources=list(self._alarm_sources), now=now, meta=meta,
            )

        # =================================================================
        # Step 5: No active alarm — determine fused state
        # =================================================================
        # State hierarchy: DROWSY > SUSPECT > ALERT > NO_SIGNAL > INSUFFICIENT_DATA
        # (If we get here, _alarm_active is False.)

        # Collect states from valid detectors
        if td_valid and freq_valid:
            # Rule 6 promotion is handled above; here both are valid but no alarm
            # Pick the higher-severity state
            if _state_rank(td_r.state) >= _state_rank(freq_r.state):
                dominant_state = td_r.state
            else:
                dominant_state = freq_r.state

            if dominant_state == _DROWSY:
                # Should not happen here (alarms handled above), but handle defensively
                dominant_state = _SUSPECT

        elif td_valid:
            dominant_state = td_r.state
            if dominant_state == _DROWSY:
                dominant_state = _SUSPECT  # alarm was not activated (stale?) — downgrade

        elif freq_valid:
            dominant_state = freq_r.state
            if dominant_state == _DROWSY:
                dominant_state = _SUSPECT

        else:
            # Neither result is valid
            dominant_state = _INSUFFICIENT_DATA

        # Rule 7: NO_SIGNAL / INSUFFICIENT_DATA are NOT alert
        if dominant_state in _NON_ALERT_STATES and dominant_state != _NO_SIGNAL and dominant_state != _INSUFFICIENT_DATA:
            dominant_state = _INSUFFICIENT_DATA

        # Determine reason code
        if not td_valid and not freq_valid:
            reason_code = RC_INSUFFICIENT_VALID_DATA
            reason = "No valid detector result available."
        elif td_valid and not freq_valid:
            freq_stale = freq_r is not None and freq_age is not None and freq_age >= self._cfg.freq_result_freshness_s
            if freq_stale:
                reason_code = RC_STALE_FREQUENCY_RESULT
                reason = f"Frequency result is stale ({freq_age:.1f} s >= {self._cfg.freq_result_freshness_s} s freshness limit). Using time-domain only."
            elif freq_r is None:
                reason_code = RC_TIME_DOMAIN_ALARM if td_r.alarm_active else RC_NO_ACTIVE_ALARM
                reason = "No frequency-domain result yet; using time-domain only."
            else:
                reason_code = RC_FREQUENCY_DOMAIN_INVALID
                reason = f"Frequency-domain result invalid: {freq_r.reason}"
        elif not td_valid and freq_valid:
            reason_code = RC_FREQUENCY_DOMAIN_ALARM if freq_r.alarm_active else RC_NO_ACTIVE_ALARM
            reason = f"Time-domain result unavailable or invalid. Frequency: {freq_r.state}."
        else:
            # Both valid, no alarm
            if td_r.state == _SUSPECT:
                reason_code = RC_TIME_DOMAIN_SUSPECT
                reason = f"Time-domain SUSPECT; frequency {freq_r.state} — not promoting."
            else:
                reason_code = RC_NO_ACTIVE_ALARM
                reason = f"No alarm active. TD={td_r.state}, Freq={freq_r.state}."

        return self._build_result(
            state=dominant_state,
            alarm_active=False,
            reason_code=reason_code,
            reason=reason,
            td_state=td_state, freq_state=freq_state,
            td_valid=bool(td_r is not None and td_r.valid),
            freq_valid=bool(freq_r is not None and freq_r.valid),
            td_age=td_age, freq_age=freq_age,
            sources=[], now=now, meta=meta,
        )

    def reset(self) -> None:
        """Reset all fusion state at a session/recording boundary."""
        self._td_result             = None
        self._freq_result           = None
        self._alarm_active          = False
        self._alarm_sources         = set()
        self._recovery_start        = None
        self._recovery_valid_s      = 0.0
        self._last_evaluate_time    = None
        if hasattr(self, "_last_valid_recovery_time"):
            self._last_valid_recovery_time = None

    def get_status(self) -> dict:
        """Return the current fused state and diagnostic metadata."""
        return {
            "alarm_active":        self._alarm_active,
            "alarm_sources":       list(self._alarm_sources),
            "td_result_available": self._td_result   is not None,
            "freq_result_available": self._freq_result is not None,
            "recovery_start":      self._recovery_start,
            "recovery_valid_s":    self._recovery_valid_s,
            "last_evaluate_time":  self._last_evaluate_time,
            "config": {
                "freq_result_freshness_s":          self._cfg.freq_result_freshness_s,
                "recovery_interval_s":              self._cfg.recovery_interval_s,
                "recovery_pause_on_invalid":        self._cfg.recovery_pause_on_invalid,
                "require_fresh_result_for_new_alarm": self._cfg.require_fresh_result_for_new_alarm,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        *,
        state: str,
        alarm_active: bool,
        reason_code: str,
        reason: str,
        td_state: str,
        freq_state: str,
        td_valid: bool,
        freq_valid: bool,
        td_age: Optional[float],
        freq_age: Optional[float],
        sources: list,
        now: float,
        meta: dict,
    ) -> FusedResult:
        m = dict(meta)
        if self._cfg.debug:
            m["recovery_valid_s"] = self._recovery_valid_s
            m["alarm_sources"]    = list(self._alarm_sources)
            m["wall_clock"]       = datetime.datetime.now().isoformat()
        return FusedResult(
            state=state,
            decision_time=now,
            time_domain_state=td_state,
            frequency_domain_state=freq_state,
            time_domain_valid=td_valid,
            frequency_domain_valid=freq_valid,
            time_domain_result_age_s=td_age,
            frequency_domain_result_age_s=freq_age,
            active_alarm_sources=list(sources),
            reason_code=reason_code,
            reason=reason,
            alarm_active=alarm_active,
            metadata=m,
        )
