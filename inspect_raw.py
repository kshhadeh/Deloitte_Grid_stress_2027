from pathlib import Path
import pandas as pd

RAW = Path("Data/raw")

for f in sorted(RAW.rglob("*")):
    if f.is_dir():
        continue
    size_mb = f.stat().st_size / 1_000_000
    print("=" * 70)
    print(f"{f.relative_to(RAW)}   ({size_mb:.1f} MB)")
    try:
        if f.suffix.lower() in (".csv", ".txt"):
            df = pd.read_csv(f, sep=None, engine="python", nrows=5)
        elif f.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(f, nrows=5)
        else:
            print("   (not a table: probably geodata or zip, check manually)")
            continue
        print("   columns:", df.columns.tolist())
        print(df.head(3).to_string())
    except Exception as e:
        print("   could not read:", e)