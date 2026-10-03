"""Claude review - step 0: check the Python environment (writes Claude/outputs/c00_env.txt)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

lines = [f"python {sys.version}", f"executable {sys.executable}"]
for name in ["pandas", "numpy", "geopandas", "shapely", "lightgbm", "shap", "sklearn", "scipy", "openpyxl"]:
    try:
        mod = __import__(name)
        lines.append(f"{name} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:  # noqa
        lines.append(f"{name} MISSING ({e})")
(OUT / "c00_env.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print("DONE c00")
