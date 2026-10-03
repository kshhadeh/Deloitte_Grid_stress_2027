"""Claude review - step 1: build my own clean tables directly from the RAW files.

Independent re-implementation (does not read anything from Data/processed).
Differences from the original pipeline, on purpose:
  * Liander postcode ranges are split over ALL postcodes in the range (equal share) instead of
    being assigned to the buurt of the start postcode.
  * The 2025/2026 FEED-IN files (Terugleverdata KV) are joined row-by-row to consumption.
  * CBS drivers are taken from ONE year (kwb 2024, codes = 2023/2024 buurten) -> no back-filling.
  * Transformers keep their own capacity so a transformer-level model is possible.

Outputs (Claude/data/): pc6_buurt.csv, buurt_year.csv, transformers.csv, cbs2024.csv,
                        pv2024.csv, knmi_daily.csv, knmi_stations.csv
Log: Claude/outputs/c01_log.txt
"""
import numpy as np
import pandas as pd
import geopandas as gpd

from common import (DATA, OUT, CONS_DIR, FEED_DIR, TRAFO_FILE, PV_FILE, KWB2024, KNMI_XLSX,
                    KNMI_TXT_DIR, BUURT_GEO, PC6_FILE, read_messy_csv, to_num, year_from_name, log)

fh = open(OUT / "c01_log.txt", "w", encoding="utf-8")

# ------------------------------------------------------------------ 1. buurt shapes + pc6 -> buurt
geo = gpd.read_file(BUURT_GEO)
land = geo[(geo["water"] != "JA") & (geo["gemeentenaam"] != "Buitenland")][
    ["buurtcode", "buurtnaam", "gemeentecode", "gemeentenaam", "geometry"]].copy()
land_rd = land.to_crs("EPSG:28992")
log(f"land buurten: {len(land)}", fh)


def points_to_buurt(df, lon, lat, key):
    """within-join, then nearest buurt within 500 m for the rest."""
    pts = gpd.GeoDataFrame(df[[key, lon, lat]].copy(),
                           geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326").to_crs("EPSG:28992")
    j = gpd.sjoin(pts, land_rd[["buurtcode", "geometry"]], how="left", predicate="within")
    j = j.drop_duplicates(key)
    ok = j[j["buurtcode"].notna()][[key, "buurtcode"]].assign(link="within")
    miss = pts[pts[key].isin(j.loc[j["buurtcode"].isna(), key])]
    near = gpd.sjoin_nearest(miss, land_rd[["buurtcode", "geometry"]], how="left",
                             max_distance=500, distance_col="dist_m").drop_duplicates(key)
    near = near[near["buurtcode"].notna()][[key, "buurtcode"]].assign(link="nearest")
    return pd.concat([ok, near], ignore_index=True)


pc6 = pd.read_csv(PC6_FILE, dtype={"pc6": str})
pc6_b = points_to_buurt(pc6, "lon", "lat", "pc6")
pc6_b.to_csv(DATA / "pc6_buurt.csv", index=False)
log(f"pc6 linked: {len(pc6_b)} of {len(pc6)} ({pc6_b['link'].value_counts().to_dict()})", fh)

# ------------------------------------------------------------------ 2. consumption (+ feed-in) rows
rows = []
for f in sorted(CONS_DIR.glob("*.csv")):
    y = year_from_name(f.name)
    d = read_messy_csv(f)
    d = d[d["PRODUCTSOORT"] == "ELK"].copy()
    if "SJV_GEMIDDELD" in d.columns:                       # old format 2019-2024
        out = pd.DataFrame({
            "pc_from": d["POSTCODE_VAN"], "pc_to": d["POSTCODE_TOT"],
            "conn": to_num(d["AANSLUITINGEN_AANTAL"]),
            "kwh": to_num(d["AANSLUITINGEN_AANTAL"]) * to_num(d["SJV_GEMIDDELD"]),
            "conn_type": d["SOORT_AANSLUITING"],
            "active_pct": to_num(d["FYSIEKE_STATUS_PERC"]),
            "consume_only_pct": to_num(d["LEVERINGSRICHTING_PERC"]),
        })
        out["series"] = "old"
        out["feedin_kwh"] = np.nan
    else:                                                  # new format 2025-2026
        out = pd.DataFrame({
            "pc_from": d["POSTCODE"], "pc_to": d["POSTCODE_EIND"],
            "conn": to_num(d["AANSLUITINGEN_AANTAL"]), "kwh": to_num(d["TOT_E"]),
            "conn_type": d["TYPE_AANSLUITING_MEEST_VOORKOMEND"],
            "active_pct": np.nan, "consume_only_pct": np.nan,
        })
        out["series"] = "new"
        fi = read_messy_csv(FEED_DIR / f"Terugleverdata KV {y}.csv")
        fi = fi[fi["PRODUCTSOORT"] == "ELK"]
        fi = pd.DataFrame({"pc_from": fi["POSTCODE"], "pc_to": fi["POSTCODE_EIND"],
                           "feedin_kwh": to_num(fi["TOT_E_INV"])}).drop_duplicates(["pc_from", "pc_to"])
        out = out.merge(fi, on=["pc_from", "pc_to"], how="left")
        log(f"  {y}: feed-in matched to {out['feedin_kwh'].notna().mean():.4f} of consumption rows", fh)
    out["year"] = y
    rows.append(out)
    log(f"{f.name}: {len(out)} rows, {out['kwh'].sum() / 1e9:.2f} TWh, "
        f"feed-in {out['feedin_kwh'].sum() / 1e9:.2f} TWh", fh)
cons = pd.concat(rows, ignore_index=True)

# ------------------------------------------------------------------ 3. split each range over its postcodes
all_pc = np.sort(pc6_b["pc6"].unique())
lo = np.searchsorted(all_pc, cons["pc_from"].values, side="left")
hi = np.searchsorted(all_pc, cons["pc_to"].values, side="right")
n = (hi - lo).clip(min=0)
MAXN = 60                         # very long ranges -> fall back to the start postcode only
long_ = n > MAXN
log(f"ranges with 0 known postcodes: {(n == 0).mean():.4f}, longer than {MAXN}: {long_.mean():.4f}", fh)
tot = cons.groupby("year")["kwh"].sum()               # before dropping unknown ranges
cons["n_pc"] = np.where(long_, 1, n)
cons["lo"] = lo
cons = cons[cons["n_pc"] > 0].reset_index(drop=True)
rep = cons.index.repeat(cons["n_pc"].values)
offs = np.concatenate([np.arange(k) for k in cons["n_pc"].values])
ex = cons.loc[rep, ["year", "series", "conn", "kwh", "feedin_kwh", "conn_type", "active_pct",
                    "consume_only_pct", "n_pc"]].reset_index(drop=True)
ex["pc6"] = all_pc[cons.loc[rep, "lo"].values + offs]
w = 1.0 / ex["n_pc"]
for c in ["conn", "kwh", "feedin_kwh"]:
    ex[c] = ex[c] * w
ex = ex.merge(pc6_b[["pc6", "buurtcode"]], on="pc6", how="left")
ex["conn_3ph"] = np.where(ex["conn_type"].astype(str).str.startswith("3x"), ex["conn"], 0.0)
ex["w_active"] = ex["active_pct"] * ex["conn"]
ex["w_cons_only"] = ex["consume_only_pct"] * ex["conn"]

lin = ex[ex["buurtcode"].notna()].groupby("year")["kwh"].sum()
log("share of kWh linked to a buurt per year:\n" + (lin / tot).round(4).to_string(), fh)

by = ex[ex["buurtcode"].notna()].groupby(["buurtcode", "year", "series"]).agg(
    connections=("conn", "sum"), total_kwh=("kwh", "sum"),
    feedin_kwh=("feedin_kwh", lambda s: s.sum(min_count=1)),
    conn_3ph=("conn_3ph", "sum"), w_active=("w_active", "sum"), w_cons_only=("w_cons_only", "sum"),
).reset_index()
by["share_3phase"] = by["conn_3ph"] / by["connections"]
by["active_pct"] = np.where(by["series"] == "old", by["w_active"] / by["connections"], np.nan)
by["consume_only_pct"] = np.where(by["series"] == "old", by["w_cons_only"] / by["connections"], np.nan)
by = by.drop(columns=["conn_3ph", "w_active", "w_cons_only"])
by.to_csv(DATA / "buurt_year.csv", index=False)
log(f"buurt_year: {by.shape}, buurten per year:\n" + by.groupby("year")["buurtcode"].nunique().to_string(), fh)
log("area totals (TWh):\n" + (by.groupby("year")[["total_kwh", "feedin_kwh"]].sum() / 1e9).round(3).to_string(), fh)

# ------------------------------------------------------------------ 4. transformers
t = read_messy_csv(TRAFO_FILE)
tr = pd.DataFrame({
    "transformer_id": t["CONDUCTINGEQUIPMENT_NAME"],
    "kv": to_num(t["BASEVOLTAGE_NOMINALVOLTAGE"]),
    "buurt_reported": t["STREETDETAIL_CODE"],
    "lat": to_num(t["POSITIONPOINT_YPOSITION"]), "lon": to_num(t["POSITIONPOINT_XPOSITION"]),
    "peak_kw": to_num(t["ANALOGVALUE_VALUE"]),
    "peak_date": pd.to_datetime(t["ANALOGVALUE_TIMESTAMP"], dayfirst=True, errors="coerce"),
    "capacity_kw": to_num(t["ACTIVEPOWERLIMIT_VALUE"]),
    "nbl_kw": to_num(t["ACTIVEPOWERLIMIT_VALUE_NBL"]),
})
n0 = len(tr)
tr = tr[(tr["kv"] == 0.4) & (tr["capacity_kw"] > 0) & (tr["nbl_kw"] > 0) & (tr["peak_kw"] > 0)].copy()
log(f"transformers: {n0} raw -> {len(tr)} kept (0.4 kV, capacity/NBL/peak > 0)", fh)
tb = points_to_buurt(tr, "lon", "lat", "transformer_id")
tr = tr.merge(tb, on="transformer_id", how="left")
tr["gm"] = tr["buurtcode"].str[2:6]
tr["util"] = tr["peak_kw"] / tr["capacity_kw"]
tr["above_nbl"] = (tr["peak_kw"] > tr["nbl_kw"]).astype(int)
tr["atabove_nbl"] = (tr["peak_kw"] >= tr["nbl_kw"]).astype(int)
tr.to_csv(DATA / "transformers.csv", index=False)
log(f"transformers linked: {tr['buurtcode'].notna().sum()}, above NBL {tr['above_nbl'].sum()}, "
    f"at/above {tr['atabove_nbl'].sum()}", fh)

# ------------------------------------------------------------------ 5. CBS 2024 (one year, no filling)
k = pd.read_excel(KWB2024, dtype=str)
k.columns = [c.strip() for c in k.columns]
k = k[k["recs"].str.strip() == "Buurt"]
cbs_cols = ["a_inw", "a_hh", "g_hhgro", "bev_dich", "a_woning", "a_bedv", "a_lan_ha", "ste_mvs", "ste_oad",
            "g_wozbag", "p_1gezw", "p_mgezw", "p_koopw", "p_huurw", "p_bj_me10", "p_bj_mi10",
            "g_ele", "g_ele_tr", "g_gas", "p_won_z", "p_won_zs", "p_won_ev", "g_ink_pi",
            "a_pau", "g_pau_hh", "a_00_14", "a_65_oo"]
cbs = pd.DataFrame({"buurtcode": k["gwb_code"].str.strip()})
for c in cbs_cols:
    cbs[c] = to_num(k[c].str.strip().replace(".", np.nan))
cbs.to_csv(DATA / "cbs2024.csv", index=False)
log(f"cbs2024: {cbs.shape}, missing share:\n" + cbs[cbs_cols].isna().mean().round(3).to_string(), fh)

# ------------------------------------------------------------------ 6. Liander PV (April 2024)
pv = read_messy_csv(PV_FILE)
pv = pv[pv["CBS_BUURTCODE_2023"].str.startswith("BU", na=False)]
pvb = pd.DataFrame({"buurtcode": pv["CBS_BUURTCODE_2023"],
                    "pv_conn": to_num(pv["AANTAL_AANSLUITINGEN_MET_PV"]),
                    "pv_kw": to_num(pv["OPGESTELD_VERMOGEN_KW"])}).groupby("buurtcode").sum(min_count=1).reset_index()
pvb.to_csv(DATA / "pv2024.csv", index=False)
log(f"pv2024 buurten: {len(pvb)}", fh)

# ------------------------------------------------------------------ 7. KNMI daily + station coordinates
st = []
for f in sorted(KNMI_TXT_DIR.glob("*.txt")):
    with open(f, encoding="latin-1") as fhk:
        for ln in fhk:
            if not ln.startswith("#"):
                break
            p = ln[1:].split()
            if len(p) >= 5 and p[0].isdigit() and len(p[0]) == 3:
                try:
                    st.append({"stn": int(p[0]), "lon": float(p[1]), "lat": float(p[2]), "name": " ".join(p[4:])})
                except ValueError:
                    pass
st = pd.DataFrame(st).drop_duplicates("stn")
w = pd.read_excel(KNMI_XLSX)
w.columns = [c.strip() for c in w.columns]
w = w.rename(columns={"# STN": "stn", "Data": "date"})
w["date"] = pd.to_datetime(w["date"])
for c in ["TG", "TN", "TX", "Q", "SQ"]:
    w[c] = pd.to_numeric(w[c].astype(str).str.strip(), errors="coerce")
w = w[["stn", "date", "TG", "TN", "TX", "Q", "SQ"]]
w.to_csv(DATA / "knmi_daily.csv", index=False)
has_q = w.groupby("stn")["Q"].apply(lambda s: s.notna().mean() > 0.9)
st["has_q"] = st["stn"].map(has_q).fillna(False)
st.to_csv(DATA / "knmi_stations.csv", index=False)
log(f"KNMI: {len(w)} daily rows, stations with radiation: {int(st['has_q'].sum())}", fh)

log("DONE c01", fh)
fh.close()
