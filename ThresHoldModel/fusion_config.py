"""
fusion_config.py
================
Configuration dataclass for the asymmetric fusion engine.
No imports from TImeDynamic_v2 or FreqDynamic_v2.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Configuration for the asymmetric fusion engine.

    Attributes
    ----------
    freq_result_freshness_s : float
        Maximum age (seconds) of a frequency-domain result for it to be
        considered fresh evidence. Default 15.0 (based on 10 s hop + latency).
    recovery_interval_s : float
        How long BOTH detectors must show continuous recovery before the
        fused alarm is cleared. Default 60.0 s.
    recovery_pause_on_invalid : bool
        True  = pause the recovery timer when valid evidence is absent.
        False = reset the recovery timer when valid evidence is lost.
        Default True.
    require_fresh_result_for_new_alarm : bool
        If True, a stale positive result cannot open a NEW alarm.
        Default True.
    auto_reset_on_session_start : bool
        If True, reset() is called automatically at session boundaries.
        Default True.
    debug : bool
        Emit extra diagnostic information in FusedResult.metadata.
        Default False.
    """
    freq_result_freshness_s: float = 15.0
    recovery_interval_s: float = 60.0
    recovery_pause_on_invalid: bool = True
    require_fresh_result_for_new_alarm: bool = True
    auto_reset_on_session_start: bool = True
    debug: bool = False

    def __post_init__(self):
        if self.freq_result_freshness_s <= 0:
            raise ValueError("freq_result_freshness_s must be > 0")
        if self.recovery_interval_s <= 0:
            raise ValueError("recovery_interval_s must be > 0")
