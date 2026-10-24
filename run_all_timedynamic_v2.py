"""
run_all_timedynamic_v2.py
=========================
Runs TImeDynamic_v2.py for every CSV in PPG_reading/ and writes
each recording's results into outputs/<stem>/
"""

import sys
import time
from pathlib import Path

# Make sure the project root is on sys.path so TImeDynamic_v2 can find its peers
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ThresHoldModel.TImeDynamic_v2 import analyze, PipelineError  # noqa: E402

PPG_DIR  = ROOT / "PPG_reading"
OUT_ROOT = ROOT / "outputs_regular"
OUT_ROOT.mkdir(exist_ok=True)

SKIP_DIRS = {"dual_reading"}          # sub-dirs inside PPG_reading

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
            raw_csv    = csv,
            workdir    = outdir,
            scripts_dir= ROOT,          # dual_process.py, dual_ssqi.py etc. live in ROOT/DualPPG
            channel    = "auto",
            method     = "IVW_SSQI",
            reuse      = False,
            plot       = str(outdir / f"{stem}_drowsiness_report.png"),
        )

        elapsed = time.time() - t0
        pred    = rep.summary["prediction"]
        reason  = rep.summary.get("reason", "")

        print(f"  result : {pred}" + (f"  ({reason})" if reason else ""))
        print(f"  time   : {elapsed:.1f} s")

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
print(f"{'File':<45} {'Result':<22} {'Time (s)'}")
print("-"*80)
for r in results:
    print(f"  {r['file']:<43} {r['prediction']:<22} {r['elapsed_s']}")

import json, datetime
summary_path = OUT_ROOT / "run_summary.json"
summary_path.write_text(json.dumps(
    {"run_at": datetime.datetime.now().isoformat(), "results": results},
    indent=2
))
print(f"\nSummary saved to: {summary_path}")
