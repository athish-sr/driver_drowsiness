"""
run_all_freqdynamic.py
======================
Runs FreqDynamic_v2.py for every CSV in PPG_reading/ and writes
each recording's results into freq_output_v2/<stem>/
"""

import sys
import time
import json
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ThresHoldModel"))

from ThresHoldModel.FreqDynamic_v2 import analyze, PipelineError  # noqa: E402

PPG_DIR  = ROOT / "PPG_reading"
OUT_ROOT = ROOT / "freq_output_v2"
OUT_ROOT.mkdir(exist_ok=True)

SKIP_DIRS = {"dual_reading"}

csvs = sorted(
    p for p in PPG_DIR.iterdir()
    if p.suffix.lower() == ".csv" and p.name not in SKIP_DIRS
)

if not csvs:
    print("No CSV files found in PPG_reading/")
    sys.exit(1)

print(f"Found {len(csvs)} CSV files to process.\n{'='*60}")

results = []

for csv in csvs:
    stem   = csv.stem
    outdir = OUT_ROOT / stem
    outdir.mkdir(exist_ok=True)

    print(f"\n[{stem}]")
    print(f"  input  : {csv}")
    print(f"  outdir : {outdir}")
    t0 = time.time()

    try:
        rep = analyze(
            raw_csv     = csv,
            workdir     = outdir,
            scripts_dir = ROOT,
            channel     = "auto",
            method      = "IVW_SSQI",
            reuse       = False,
            plot        = str(outdir / f"{stem}_freq_report.png"),
        )

        elapsed = time.time() - t0
        pred    = rep.summary["prediction"]
        reason  = rep.summary.get("reason", "")

        print(f"  result : {pred}" + (f"  ({reason})" if reason else ""))
        print(f"  time   : {elapsed:.1f} s")

        # save outputs
        if len(rep.windows):
            rep.windows.to_csv(outdir / f"{stem}_freq_windows.csv", index=False)
            rep.events.to_csv(outdir / f"{stem}_freq_events.csv", index=False)
        (outdir / f"{stem}_freq_summary.json").write_text(json.dumps(
            {"summary": rep.summary, "info": rep.info, "config": rep.cfg.__dict__},
            indent=2, default=str
        ))

        results.append(dict(file=stem, prediction=pred, reason=reason,
                            elapsed_s=round(elapsed, 1)))

    except PipelineError as e:
        elapsed = time.time() - t0
        msg = str(e)
        print(f"  PIPELINE ERROR: {msg}")
        results.append(dict(file=stem, prediction="ERROR", reason=msg,
                            elapsed_s=round(elapsed, 1)))

    except Exception as e:
        elapsed = time.time() - t0
        msg = f"{type(e).__name__}: {e}"
        print(f"  UNEXPECTED ERROR: {msg}")
        results.append(dict(file=stem, prediction="ERROR", reason=msg,
                            elapsed_s=round(elapsed, 1)))

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'File':<50} {'Result':<22} {'Time (s)'}")
print("-"*85)
for r in results:
    print(f"  {r['file']:<48} {r['prediction']:<22} {r['elapsed_s']}")

summary_path = OUT_ROOT / "run_summary.json"
summary_path.write_text(json.dumps(
    {"run_at": datetime.datetime.now().isoformat(), "results": results},
    indent=2
))
print(f"\nSummary saved to: {summary_path}")
