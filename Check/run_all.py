"""Run the Claude review steps in order. Usage: python Claude/run_all.py [first_step]
Each step writes its own log to Claude/outputs/cXX_log.txt; errors go to Claude/outputs/run_all_error.txt."""
import runpy
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STEPS = ["c01_build_data.py", "c02_peak_diagnostics.py", "c03_stress_model.py", "c04_growth.py",
         "c05_projection_and_compare.py"]
first = sys.argv[1] if len(sys.argv) > 1 else "c02"          # c01 is slow and already done
err = HERE / "outputs" / "run_all_error.txt"
if err.exists():
    err.unlink()
for s in STEPS:
    if s[:3] < first:
        continue
    print("=" * 20, s)
    try:
        runpy.run_path(str(HERE / s), run_name="__main__")
    except Exception:
        err.write_text(s + "\n" + traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc())
        break
(HERE / "outputs" / "run_all_status.txt").write_text("finished (check run_all_error.txt if present)", encoding="utf-8")
