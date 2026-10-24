"""
tests/test_fusion_engine.py
===========================
Unit tests for the asymmetric fusion engine.

All 22 tests use synthetic DetectorResult objects; no PPG pipeline, hardware,
or real recordings are required.  Use now=BASE_TIME for all evaluate() calls
to make timestamps deterministic.
"""
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Ensure the ThresHoldModel package is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ThresHoldModel"))

from ThresHoldModel.fusion_engine import (
    FusionEngine,
    DetectorResult,
    FusedResult,
    RC_TIME_DOMAIN_ALARM,
    RC_FREQUENCY_DOMAIN_ALARM,
    RC_BOTH_DETECTORS_ALARM,
    RC_TIME_DOMAIN_SUSPECT,
    RC_STALE_FREQUENCY_RESULT,
    RC_FREQ_PROMOTES_SUSPECT_TO_DROWSY,
    RC_ALARM_LATCHED,
    RC_RECOVERY_IN_PROGRESS,
    RC_RECOVERY_COMPLETED,
    RC_INSUFFICIENT_VALID_DATA,
    RC_NO_ACTIVE_ALARM,
    RC_TIME_DOMAIN_INVALID,
    RC_FREQUENCY_DOMAIN_INVALID,
)
from ThresHoldModel.fusion_config import FusionConfig

BASE_TIME = 1000.0  # fixed monotonic base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_td_result(state, valid, alarm_active, age_s, evidence=None):
    """Build a synthetic time-domain DetectorResult with decision_time = BASE_TIME - age_s."""
    return DetectorResult(
        model="time_domain",
        state=state,
        valid=valid,
        window_start=0.0,
        window_end=10.0,
        decision_time=BASE_TIME - age_s,
        alarm_active=alarm_active,
        evidence_score=evidence,
        reason="synthetic",
        metadata={},
    )


def make_freq_result(state, valid, alarm_active, age_s, evidence=None):
    """Build a synthetic frequency-domain DetectorResult with decision_time = BASE_TIME - age_s."""
    return DetectorResult(
        model="frequency_domain",
        state=state,
        valid=valid,
        window_start=0.0,
        window_end=60.0,
        decision_time=BASE_TIME - age_s,
        alarm_active=alarm_active,
        evidence_score=evidence,
        reason="synthetic",
        metadata={},
    )


def fresh_engine(**cfg_kwargs) -> FusionEngine:
    """Return a new FusionEngine with optionally customized FusionConfig."""
    return FusionEngine(FusionConfig(**cfg_kwargs))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAsymmetricFusion:

    # ------------------------------------------------------------------
    # Test 1: Time-domain DROWSY overrides freq ALERT (Rules 1, 3)
    # ------------------------------------------------------------------
    def test_time_domain_drowsy_overrides_freq_alert(self):
        engine = fresh_engine()
        td   = make_td_result("DROWSY", valid=True, alarm_active=True,  age_s=1)
        freq = make_freq_result("ALERT", valid=True, alarm_active=False, age_s=1)
        engine.update_detector_result(td)
        engine.update_detector_result(freq)
        result = engine.evaluate(now=BASE_TIME)
        assert result.state == "DROWSY", f"Expected DROWSY, got {result.state}"
        assert result.alarm_active is True

    # ------------------------------------------------------------------
    # Test 2: Freq-domain DROWSY overrides time-domain ALERT (Rules 2, 4)
    # ------------------------------------------------------------------
    def test_freq_domain_drowsy_overrides_time_alert(self):
        engine = fresh_engine()
        td   = make_td_result("ALERT", valid=True, alarm_active=False, age_s=1)
        freq = make_freq_result("DROWSY", valid=True, alarm_active=True, age_s=1)
        engine.update_detector_result(td)
        engine.update_detector_result(freq)
        result = engine.evaluate(now=BASE_TIME)
        assert result.state == "DROWSY", f"Expected DROWSY, got {result.state}"
        assert result.alarm_active is True

    # ------------------------------------------------------------------
    # Test 3: Time-domain DROWSY with freq NO_SIGNAL (Rules 1, 8)
    # ------------------------------------------------------------------
    def test_time_drowsy_with_freq_no_signal(self):
        engine = fresh_engine()
        td   = make_td_result("DROWSY",    valid=True,  alarm_active=True,  age_s=1)
        freq = make_freq_result("NO_SIGNAL", valid=False, alarm_active=False, age_s=1)
        engine.update_detector_result(td)
        engine.update_detector_result(freq)
        result = engine.evaluate(now=BASE_TIME)
        assert result.state == "DROWSY", f"Expected DROWSY, got {result.state}"
        assert result.alarm_active is True

    # ------------------------------------------------------------------
    # Test 4: Time-domain DROWSY with stale freq ALERT (Rules 1, 7, 8)
    # ------------------------------------------------------------------
    def test_time_drowsy_with_stale_freq_alert(self):
        engine = fresh_engine(freq_result_freshness_s=15.0)
        td   = make_td_result("DROWSY", valid=True, alarm_active=True, age_s=1)
        freq = make_freq_result("ALERT", valid=True, alarm_active=False, age_s=20)  # stale
        engine.update_detector_result(td)
        engine.update_detector_result(freq)
        result = engine.evaluate(now=BASE_TIME)
        assert result.state == "DROWSY", f"Stale freq ALERT must not veto time-domain alarm; got {result.state}"
        assert result.alarm_active is True

    # ------------------------------------------------------------------
    # Test 5: Time-domain SUSPECT + freq ALERT stays SUSPECT (Rule 5)
    # ------------------------------------------------------------------
    def test_time_suspect_with_freq_alert_stays_suspect(self):
        engine = fresh_engine()
        td   = make_td_result("SUSPECT", valid=True, alarm_active=False, age_s=1)
        freq = make_freq_result("ALERT",   valid=True, alarm_active=False, age_s=1)
        engine.update_detector_result(td)
        engine.update_detector_result(freq)
        result = engine.evaluate(now=BASE_TIME)
        assert result.state == "SUSPECT", f"Expected SUSPECT, got {result.state}"
        assert result.alarm_active is False

    # ------------------------------------------------------------------
    # Test 6: Time-domain SUSPECT promoted by fresh freq DROWSY (Rule 6)
    # ------------------------------------------------------------------
    def test_time_suspect_promoted_by_freq_drowsy(self):
        engine = fresh_engine()
        td   = make_td_result("SUSPECT", valid=True, alarm_active=False, age_s=1)
        freq = make_freq_result("DROWSY",  valid=True, alarm_active=True,  age_s=1)
        engine.update_detector_result(td)
        engine.update_detector_result(freq)
        result = engine.evaluate(now=BASE_TIME)
        assert result.state == "DROWSY", f"Expected DROWSY, got {result.state}"
        assert result.reason_code == RC_FREQ_PROMOTES_SUSPECT_TO_DROWSY, \
            f"Expected RC_FREQ_PROMOTES_SUSPECT_TO_DROWSY, got {result.reason_code}"

    # ------------------------------------------------------------------
    # Test 7: Single ALERT cannot clear active fused alarm (Rule 9)
    # ------------------------------------------------------------------
    def test_single_alert_cannot_clear_active_fused_alarm(self):
        engine = fresh_engine()
        # Activate the alarm
        td_drowsy = make_td_result("DROWSY", valid=True, alarm_active=True, age_s=1)
        engine.update_detector_result(td_drowsy)
        fused1 = engine.evaluate(now=BASE_TIME)
        assert fused1.alarm_active is True, "Alarm should have activated"

        # Both detectors now show ALERT
        td_alert   = make_td_result("ALERT",   valid=True, alarm_active=False, age_s=1)
        freq_alert = make_freq_result("ALERT",  valid=True, alarm_active=False, age_s=1)
        engine.update_detector_result(td_alert)
        engine.update_detector_result(freq_alert)
        result = engine.evaluate(now=BASE_TIME)
        assert result.alarm_active is True, \
            "Single evaluate with both ALERT must not instantly clear the fused alarm"

    # ------------------------------------------------------------------
    # Test 8: Time-domain-only alarm not cleared by freq ALERT (Rules 9, 10)
    # ------------------------------------------------------------------
    def test_time_only_alarm_not_cleared_by_freq_alert(self):
        engine = fresh_engine()
        # Activate via TD only (no freq result)
        td_drowsy = make_td_result("DROWSY", valid=True, alarm_active=True, age_s=1)
        engine.update_detector_result(td_drowsy)
        fused1 = engine.evaluate(now=BASE_TIME)
        assert fused1.alarm_active is True

        # Now provide TD ALERT + Freq ALERT
        td_alert   = make_td_result("ALERT",  valid=True, alarm_active=False, age_s=1)
        freq_alert = make_freq_result("ALERT", valid=True, alarm_active=False, age_s=1)
        engine.update_detector_result(td_alert)
        engine.update_detector_result(freq_alert)
        result = engine.evaluate(now=BASE_TIME)
        assert result.alarm_active is True, \
            "Alarm must remain active immediately after both detectors return ALERT"

    # ------------------------------------------------------------------
    # Test 9: Freq alarm recovery requires both detectors (Rule 10)
    # ------------------------------------------------------------------
    def test_freq_alarm_recovery_requires_both_detectors(self):
        engine = fresh_engine(recovery_interval_s=60.0)
        # Activate with freq DROWSY + td SUSPECT
        freq_drowsy = make_freq_result("DROWSY",  valid=True, alarm_active=True,  age_s=1)
        td_suspect  = make_td_result("SUSPECT",   valid=True, alarm_active=False, age_s=1)
        engine.update_detector_result(freq_drowsy)
        engine.update_detector_result(td_suspect)
        fused1 = engine.evaluate(now=BASE_TIME)
        assert fused1.alarm_active is True, "Alarm should be active"

        # Both show recovery
        td_alert   = make_td_result("ALERT",  valid=True, alarm_active=False, age_s=1)
        freq_alert = make_freq_result("ALERT", valid=True, alarm_active=False, age_s=1)
        engine.update_detector_result(td_alert)
        engine.update_detector_result(freq_alert)
        # A single evaluate should not clear the alarm instantly
        result = engine.evaluate(now=BASE_TIME)
        assert result.alarm_active is True, \
            "Alarm must not clear instantly; requires recovery_interval_s of sustained recovery"

    # ------------------------------------------------------------------
    # Test 10: Missing result not treated as ALERT (Rule 7)
    # ------------------------------------------------------------------
    def test_missing_result_not_treated_as_alert(self):
        engine = fresh_engine()
        # Only TD ALERT; no freq result
        td = make_td_result("ALERT", valid=True, alarm_active=False, age_s=1)
        engine.update_detector_result(td)
        result = engine.evaluate(now=BASE_TIME)
        assert result.state == "ALERT"
        assert result.alarm_active is False

    # ------------------------------------------------------------------
    # Test 11: Invalid result not treated as ALERT (Rule 7)
    # ------------------------------------------------------------------
    def test_invalid_result_not_treated_as_alert(self):
        engine = fresh_engine()
        td   = make_td_result("DROWSY",            valid=False, alarm_active=False, age_s=1)
        freq = make_freq_result("INSUFFICIENT_DATA", valid=False, alarm_active=False, age_s=1)
        engine.update_detector_result(td)
        engine.update_detector_result(freq)
        result = engine.evaluate(now=BASE_TIME)
        assert result.state != "DROWSY", \
            f"Invalid results must not produce DROWSY; got {result.state}"
        assert result.alarm_active is False

    # ------------------------------------------------------------------
    # Test 12: NO_SIGNAL and INSUFFICIENT_DATA distinguished (Rule 7)
    # ------------------------------------------------------------------
    def test_no_signal_and_insufficient_data_distinguished(self):
        engine = fresh_engine()
        td   = make_td_result("NO_SIGNAL",          valid=False, alarm_active=False, age_s=1)
        freq = make_freq_result("INSUFFICIENT_DATA", valid=False, alarm_active=False, age_s=1)
        engine.update_detector_result(td)
        engine.update_detector_result(freq)
        result = engine.evaluate(now=BASE_TIME)
        assert result.alarm_active is False
        assert result.state not in ("DROWSY", "ALERT"), \
            f"NO_SIGNAL/INSUFFICIENT_DATA must not yield DROWSY or ALERT; got {result.state}"

    # ------------------------------------------------------------------
    # Test 13: Signal loss during alarm keeps alarm (Rule 11)
    # ------------------------------------------------------------------
    def test_signal_loss_during_alarm_keeps_alarm(self):
        engine = fresh_engine()
        # Activate alarm
        td_drowsy = make_td_result("DROWSY", valid=True, alarm_active=True, age_s=1)
        engine.update_detector_result(td_drowsy)
        engine.evaluate(now=BASE_TIME)

        # Signal loss from both detectors
        td_ns   = make_td_result("NO_SIGNAL",   valid=False, alarm_active=False, age_s=1)
        freq_ns = make_freq_result("NO_SIGNAL",  valid=False, alarm_active=False, age_s=1)
        engine.update_detector_result(td_ns)
        engine.update_detector_result(freq_ns)
        result = engine.evaluate(now=BASE_TIME)

        assert result.alarm_active is True, \
            "Signal loss must not clear an active fused alarm"
        # Metadata should contain a signal_loss indicator
        assert "signal_loss" in result.metadata, \
            "Metadata should contain 'signal_loss' key during signal loss with active alarm"

    # ------------------------------------------------------------------
    # Test 14: Recovery does not complete without valid evidence (Rule 10)
    # ------------------------------------------------------------------
    def test_recovery_does_not_complete_without_valid_evidence(self):
        engine = fresh_engine(recovery_interval_s=30.0)
        # Activate alarm
        td_drowsy = make_td_result("DROWSY", valid=True, alarm_active=True, age_s=1)
        engine.update_detector_result(td_drowsy)
        engine.evaluate(now=BASE_TIME)

        # Provide ALERT but with valid=False (invalid signal)
        td_inv   = make_td_result("ALERT", valid=False, alarm_active=False, age_s=1)
        freq_inv = make_freq_result("ALERT", valid=False, alarm_active=False, age_s=1)
        engine.update_detector_result(td_inv)
        engine.update_detector_result(freq_inv)

        # Evaluate many times with increasing now — recovery timer must not accumulate
        for i in range(10):
            result = engine.evaluate(now=BASE_TIME + 60.0 * (i + 1))
        assert result.alarm_active is True, \
            "Recovery must not complete when valid evidence is absent"

    # ------------------------------------------------------------------
    # Test 15: Stale positive does not activate new alarm (Rule 12)
    # ------------------------------------------------------------------
    def test_stale_positive_does_not_activate_new_alarm(self):
        engine = fresh_engine(freq_result_freshness_s=15.0, require_fresh_result_for_new_alarm=True)
        # TD DROWSY but age 20s (stale relative to... there's no freshness limit on TD,
        # but per the spec the freshness check uses decision_time vs now)
        # For this test, use FREQ DROWSY that is stale
        freq_stale = make_freq_result("DROWSY", valid=True, alarm_active=True, age_s=20)
        engine.update_detector_result(freq_stale)
        result = engine.evaluate(now=BASE_TIME)
        assert result.alarm_active is False, \
            "A stale positive result must not activate a new alarm"

    # ------------------------------------------------------------------
    # Test 16: Fresh positive from one model activates alarm (Rules 1, 2)
    # ------------------------------------------------------------------
    def test_fresh_positive_one_model_activates_alarm(self):
        engine = fresh_engine()
        td = make_td_result("DROWSY", valid=True, alarm_active=True, age_s=1)
        engine.update_detector_result(td)
        result = engine.evaluate(now=BASE_TIME)
        assert result.alarm_active is True, "Fresh positive must activate alarm"
        assert result.state == "DROWSY"

    # ------------------------------------------------------------------
    # Test 17: reset() clears all state
    # ------------------------------------------------------------------
    def test_reset_clears_all_state(self):
        engine = fresh_engine()
        # Activate an alarm first
        td_drowsy = make_td_result("DROWSY", valid=True, alarm_active=True, age_s=1)
        engine.update_detector_result(td_drowsy)
        engine.evaluate(now=BASE_TIME)
        assert engine._alarm_active is True

        engine.reset()

        assert engine._alarm_active is False, "reset() must clear _alarm_active"
        assert engine._td_result    is None,  "reset() must clear _td_result"
        assert engine._freq_result  is None,  "reset() must clear _freq_result"
        assert engine._recovery_start is None, "reset() must clear _recovery_start"

    # ------------------------------------------------------------------
    # Test 18: Duplicate result handled deterministically (Rule 13)
    # ------------------------------------------------------------------
    def test_duplicate_result_handled_deterministically(self):
        engine = fresh_engine()
        td = make_td_result("ALERT", valid=True, alarm_active=False, age_s=1)
        engine.update_detector_result(td)
        result1 = engine.evaluate(now=BASE_TIME)
        # Feed the exact same result again
        engine.update_detector_result(td)
        result2 = engine.evaluate(now=BASE_TIME)
        assert result1.state == result2.state, \
            "Feeding the same result twice must produce the same fused state"

    # ------------------------------------------------------------------
    # Test 19: Overlapping freq windows not counted as independent (Rule 13)
    # ------------------------------------------------------------------
    def test_overlapping_freq_windows_not_independent(self):
        engine = fresh_engine()
        freq1 = DetectorResult(
            model="frequency_domain", state="DROWSY", valid=True,
            window_start=0.0, window_end=60.0, decision_time=BASE_TIME - 2,
            alarm_active=True, evidence_score=5.0, reason="overlap1", metadata={},
        )
        freq2 = DetectorResult(
            model="frequency_domain", state="DROWSY", valid=True,
            window_start=10.0, window_end=70.0, decision_time=BASE_TIME - 1,
            alarm_active=True, evidence_score=5.5, reason="overlap2", metadata={},
        )
        engine.update_detector_result(freq1)
        engine.update_detector_result(freq2)

        # Only the most recent result per model should be stored
        assert engine._freq_result is freq2, \
            "Second update must replace the first (no additive overlap counting)"

    # ------------------------------------------------------------------
    # Test 20: Outputs arrive at different times (window alignment)
    # ------------------------------------------------------------------
    def test_outputs_arrive_at_different_times(self):
        engine = fresh_engine()
        # TD arrives at BASE_TIME (age=0 relative to BASE_TIME)
        td = DetectorResult(
            model="time_domain", state="ALERT", valid=True,
            window_start=0.0, window_end=10.0, decision_time=BASE_TIME,
            alarm_active=False, evidence_score=None, reason="td", metadata={},
        )
        # Freq arrives at BASE_TIME + 5 (age=0 relative to BASE_TIME+5)
        freq = DetectorResult(
            model="frequency_domain", state="ALERT", valid=True,
            window_start=0.0, window_end=60.0, decision_time=BASE_TIME + 5,
            alarm_active=False, evidence_score=None, reason="freq", metadata={},
        )
        engine.update_detector_result(td)
        engine.update_detector_result(freq)

        result = engine.evaluate(now=BASE_TIME + 5)
        # Both valid, both ALERT -> fused ALERT, no alarm
        assert result.alarm_active is False
        assert result.time_domain_valid is True
        assert result.frequency_domain_valid is True

    # ------------------------------------------------------------------
    # Test 21: Freshness boundary cases (exactly-at and slightly-under)
    # ------------------------------------------------------------------
    def test_freshness_boundary_cases(self):
        freshness = 15.0
        engine_at_boundary = fresh_engine(freq_result_freshness_s=freshness)
        engine_under       = fresh_engine(freq_result_freshness_s=freshness)

        # TD SUSPECT for both cases
        td = make_td_result("SUSPECT", valid=True, alarm_active=False, age_s=1)

        # Age exactly 15.0 -> stale (age >= freshness)
        freq_at = make_freq_result("DROWSY", valid=True, alarm_active=True, age_s=freshness)
        engine_at_boundary.update_detector_result(td)
        engine_at_boundary.update_detector_result(freq_at)
        result_at = engine_at_boundary.evaluate(now=BASE_TIME)
        assert result_at.alarm_active is False, \
            f"Age exactly at freshness limit must be treated as stale; alarm_active={result_at.alarm_active}"

        # Age 14.9 -> fresh (age < freshness)
        freq_under = make_freq_result("DROWSY", valid=True, alarm_active=True, age_s=14.9)
        engine_under.update_detector_result(td)
        engine_under.update_detector_result(freq_under)
        result_under = engine_under.evaluate(now=BASE_TIME)
        assert result_under.alarm_active is True, \
            f"Age slightly under freshness limit must be treated as fresh; alarm_active={result_under.alarm_active}"

    # ------------------------------------------------------------------
    # Test 22: Adapter smoke test with mock reports
    # ------------------------------------------------------------------
    def test_adapter_smoke_test_mock_report(self):
        from ThresHoldModel.detector_adapters import (
            adapt_time_domain_report,
            adapt_freq_domain_report,
        )

        class MockReport:
            def __init__(self):
                self.windows = pd.DataFrame([{
                    "t": 10.0, "valid": True, "state": "ALERT",
                    "S": 0.5, "alarm": False, "h_on": False,
                    "hr_bpm": 65.0, "b_hr": 65.0, "score": 0.8,
                    "z_hr": 0.1, "z_hrv": 0.2, "rmssd_ms": 45.0,
                }])
                self.events = pd.DataFrame()
                self.summary = {
                    "prediction": "ALERT",
                    "drowsy_detected": False,
                    "first_alarm_s": None,
                    "n_windows": 1,
                    "pct_valid": 100.0,
                    "state_pct": {"ALERT": 100.0},
                    "max_S": 0.5,
                    "final_alarm_level": 1.0,
                }
                self.info = {"channel": "IR", "method": "IVW_SSQI"}
                self.beats = {}
                self.quality = {}

        class MockFreqReport:
            def __init__(self):
                self.windows = pd.DataFrame([{
                    "start_s": 0.0, "end_s": 60.0, "valid": True,
                    "n": 50, "coverage": 0.95, "LF": 0.4, "HF": 0.3,
                    "lnLFHF": 0.3, "hr_bpm": 65.0, "z": 0.5, "S": 0.3,
                    "state": "ALERT", "alarm": False,
                }])
                self.events = pd.DataFrame()
                self.summary = {
                    "prediction": "ALERT",
                    "drowsy_detected": False,
                    "first_alarm_s": None,
                    "n_windows": 1,
                    "pct_valid": 100.0,
                    "state_pct": {"ALERT": 100.0},
                    "max_S": 0.3,
                    "baseline": 0.3,
                    "sigma": 0.1,
                }
                self.info = {"channel": "IR"}
                self.cfg = type(
                    "FreqConfig", (), {"window_s": 60, "hop_s": 10, "learn_s": 180}
                )()

        mock_td   = MockReport()
        mock_freq = MockFreqReport()

        td_result   = adapt_time_domain_report(mock_td,   now=BASE_TIME)
        freq_result = adapt_freq_domain_report(mock_freq, now=BASE_TIME)

        # Both must return a DetectorResult
        assert isinstance(td_result,   DetectorResult), "adapt_time_domain_report must return DetectorResult"
        assert isinstance(freq_result, DetectorResult), "adapt_freq_domain_report must return DetectorResult"

        # Check model field
        assert td_result.model   == "time_domain"
        assert freq_result.model == "frequency_domain"

        # State should be ALERT for both mock reports
        assert td_result.state   == "ALERT", f"Expected ALERT, got {td_result.state}"
        assert freq_result.state == "ALERT", f"Expected ALERT, got {freq_result.state}"

        # Valid = True (prediction ALERT + valid window exists)
        assert td_result.valid   is True, "TD mock ALERT with valid window must be valid"
        assert freq_result.valid is True, "Freq mock ALERT with post-LEARN window must be valid"

        # alarm_active = False (no active alarm in mock windows)
        assert td_result.alarm_active   is False
        assert freq_result.alarm_active is False

        # decision_time should match what we passed
        assert td_result.decision_time   == BASE_TIME
        assert freq_result.decision_time == BASE_TIME

        # metadata must be a dict
        assert isinstance(td_result.metadata,   dict)
        assert isinstance(freq_result.metadata, dict)
