"""
run_all_fusion.py
=================
Runs TImeDynamic_v2 and FreqDynamic_v2 for every CSV in PPG_reading/,
feeds results to the FusionEngine, and saves fused outputs to fusion_output/<stem>/
"""
import sys
import time
import json
import datetime
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ThresHoldModel"))

from ThresHoldModel.TImeDynamic_v2_ssqi import analyze as td_analyze, PipelineError
from ThresHoldModel.FreqDynamic_v2 import analyze_fusion_csv as freq_analyze_fusion_csv
from ThresHoldModel.detector_adapters import adapt_time_domain_report, adapt_freq_domain_report
from ThresHoldModel.fusion_engine import FusionEngine, DetectorResult, FusedResult
from ThresHoldModel.fusion_config import FusionConfig

PPG_DIR = Path(os.environ.get("FUSION_PPG_DIR", ROOT / "DualReading"))
OUT_ROOT = Path(os.environ.get("FUSION_OUT_ROOT", ROOT / "fusion_output"))
OUT_ROOT.mkdir(parents=True, exist_ok=True)
INPUT_FILE = os.environ.get("FUSION_INPUT_FILE")
REUSE = os.environ.get("FUSION_REUSE", "1").lower() not in {"0", "false", "no"}

SKIP_DIRS = {"dual_reading"}

if INPUT_FILE:
    requested_csv = Path(INPUT_FILE)
    if not requested_csv.is_absolute():
        requested_csv = PPG_DIR / requested_csv
    csvs = [requested_csv] if requested_csv.is_file() else []
else:
    csvs = sorted(
        p for p in PPG_DIR.iterdir()
        if p.suffix.lower() == ".csv" and p.name not in SKIP_DIRS
    )

if not csvs:
    print("No CSV files found in DualReading/")
    sys.exit(1)

print(f"Found {len(csvs)} CSV files to process.\n{'='*60}")
results = []

for csv in csvs:
    stem   = csv.stem
    outdir = OUT_ROOT / stem
    outdir.mkdir(exist_ok=True)
    summary_log = outdir / "detection_summary.log"

    def display(message: str) -> None:
        timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
        print(message)
        with summary_log.open("a", encoding="utf-8") as log:
            log.write(f"{timestamp} {message}\n")

    print(f"\n[{stem}]")
    t0 = time.time()

    td_pred   = "ERROR"
    freq_pred = "ERROR"
    fused_state = "ERROR"
    reason = ""

    try:
        # Run time-domain detector
        td_report = td_analyze(
            raw_csv    = csv,
            workdir    = outdir,
            scripts_dir= ROOT,
            channel    = "auto",
            method     = "IVW_SSQI",
            reuse      = REUSE,
            plot       = str(outdir / f"{stem}_td_report.png"),
        )
        td_pred = td_report.summary["prediction"]
        td_window = td_report.windows.iloc[-1] if len(td_report.windows) else None
        if td_window is not None:
            display(
                f"DISPLAY: TD metrics | HR={td_window.get('hr_bpm', float('nan')):.1f} bpm | "
                f"evidence={td_window.get('S', float('nan')):.2f} | "
                f"score={td_window.get('score', float('nan')):.2f} | state={td_window.get('state', 'UNKNOWN')}"
            )

        # Run frequency detection directly on the fused output from the TD pass.
        fusion_csv = outdir / "ppg_fusion_ssqi_results" / "ppg_fusion_results.csv"
        if not fusion_csv.is_file():
            raise PipelineError(f"Time-domain pipeline did not produce {fusion_csv}")
        display("[5/5] frequency-domain detector (reusing generated fused data)")
        freq_report = freq_analyze_fusion_csv(
            fusion_csv=fusion_csv,
            scripts_dir=ROOT,
            channel="auto",
            method="IVW_SSQI",
            plot=str(outdir / f"{stem}_freq_report.png"),
        )
        freq_pred = freq_report.summary["prediction"]
        freq_reason = freq_report.summary.get("reason", "")
        freq_window = freq_report.windows.iloc[-1] if len(freq_report.windows) else None
        if freq_window is not None:
            display(
                f"DISPLAY: Freq metrics | HR={freq_window.get('hr_bpm', float('nan')):.1f} bpm | "
                f"evidence={freq_window.get('S', float('nan')):.2f} | "
                f"score={freq_window.get('z', float('nan')):.2f} | "
                f"state={freq_window.get('state', 'UNKNOWN')}"
            )
        print(f"  Frequency detail: {freq_pred}" +
              (f" ({freq_reason})" if freq_reason else ""))

        # Build fusion engine and feed results
        now = time.monotonic()
        engine = FusionEngine(FusionConfig())

        td_result = adapt_time_domain_report(td_report, now=now)
        freq_result = adapt_freq_domain_report(freq_report, now=now)

        if td_result:
            engine.update_detector_result(td_result)
        if freq_result:
            engine.update_detector_result(freq_result)

        fused = engine.evaluate(now=now)
        fused_state = fused.state
        display(
            f"DISPLAY: RESULTS | TD={td_pred} | Freq={freq_pred} | Fused={fused_state}"
        )

        # Save outputs
        summary_data = {
            "file": stem,
            "processed_at": datetime.datetime.now().isoformat(),
            "time_domain_prediction": td_pred,
            "frequency_domain_prediction": freq_pred,
            "fused_state": fused_state,
            "fused_alarm_active": fused.alarm_active,
            "reason_code": fused.reason_code,
            "reason": fused.reason,
            "active_alarm_sources": fused.active_alarm_sources,
            "metadata": fused.metadata,
        }
        (outdir / f"{stem}_fusion_summary.json").write_text(
            json.dumps(summary_data, indent=2, default=str)
        )

        print(f"  TD: {td_pred}  |  Freq: {freq_pred}  |  Fused: {fused_state}")
        if freq_reason:
            print(f"  Frequency reason: {freq_reason}")

        results.append(dict(
            file=stem,
            td_prediction=td_pred,
            freq_prediction=freq_pred,
            fused_state=fused_state,
            freq_reason=freq_reason,
            elapsed_s=round(time.time() - t0, 1),
        ))

    except PipelineError as e:
        msg = str(e)
        print(f"  PIPELINE ERROR: {msg}")
        results.append(dict(file=stem, td_prediction=td_pred, freq_prediction=freq_pred,
                            fused_state="ERROR", elapsed_s=round(time.time()-t0,1), error=msg))
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"  ERROR: {msg}")
        results.append(dict(file=stem, td_prediction=td_pred, freq_prediction=freq_pred,
                            fused_state="ERROR", elapsed_s=round(time.time()-t0,1), error=msg))

# Summary
print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
print(f"{'File':<45} {'TD':^12} {'Freq':^12} {'Fused':^12}")
print("-"*85)
for r in results:
    print(f"  {r['file']:<43} {r.get('td_prediction','?'):^12} {r.get('freq_prediction','?'):^12} {r.get('fused_state','?'):^12}")

(OUT_ROOT / "run_summary.json").write_text(json.dumps(
    {"run_at": datetime.datetime.now().isoformat(), "results": results},
    indent=2
))
