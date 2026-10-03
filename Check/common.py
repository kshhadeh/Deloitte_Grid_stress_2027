"""Shared paths and helpers for the Claude review scripts.

The review NEVER writes outside the Claude folder. Raw data is only read from ../Data/raw.
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent            # .../grid_stress_project/Claude
PROJECT = HERE.parent
RAW = PROJECT / "Data" / "raw"
USER_PROCESSED = PROJECT / "Data" / "processed"   # only READ, for the comparison step
DATA = HERE / "data"                              # Claude's own intermediate files
OUT = HERE / "outputs"                            # Claude's results (tables, text, figures)
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

LIANDER = RAW / "4 Liander" / "Raw Data"
CONS_DIR = LIANDER / "2. Consumption data for small-scale connections"
FEED_DIR = LIANDER / "3"
TRAFO_FILE = LIANDER / "1. Netbewust publiek laden Liander - Dataset knelpunt 2026-01.csv"
PV_FILE = LIANDER / "4 Generation data for small-scale connections" / "Liander_decentrale_opwek_kv_zon_20240409.csv"
KWB2024 = RAW / "2 CBS" / "1 kwb Raw" / "kwb2024.xlsx"
KNMI_XLSX = RAW / "3 knmi" / "All_Stations_Data.xlsx"
KNMI_TXT_DIR = RAW / "3 knmi" / "Stations Data Raw"
BUURT_GEO = RAW / "External" / "cbs_buurten_2023_simplified.geojson"
PC6_FILE = RAW / "External" / "pdok_pc6_centroids_liander.csv"


def log(msg, fh=None):
    print(msg)
    if fh is not None:
        fh.write(str(msg) + "\n")
        fh.flush()


def read_messy_csv(path):
    """Read a Liander CSV whatever its dialect (tab/semicolon, whole-line quotes, padded fields).
    Returns all columns as stripped text."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
    head = lines[0]
    sep = "\t" if head.count("\t") > head.count(";") else ";"
    if head.startswith('"') and head.endswith('"') and head.count('"') == 2:
        lines = [ln[1:-1].replace('""', '"') if (ln.startswith('"') and ln.endswith('"')) else ln
                 for ln in lines]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, dtype=str, quotechar='"')
    df.columns = [c.strip().strip('"').strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.strip('"').str.strip()
        df.loc[df[c].isin(["nan", "", "\\N"]), c] = np.nan
    return df


def to_num(s):
    """'1.234,5' is not used by Liander; values are either '36,00' or '36.0' -> 36.0"""
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def year_from_name(name):
    return int(re.search(r"(20\d{2})", name).group(1))
