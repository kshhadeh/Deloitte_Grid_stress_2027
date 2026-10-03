# %% [markdown]
# # 09 (fixed) – Forecast, projected grid stress 2026 AND 2027, solar sensitivity
#
# Corrected copy of `notebooks/09_forecast_2027.ipynb`. The original notebook and `Data/processed` are NOT changed:
# this script only READS `Data/raw` and `Data/processed` and WRITES to `Claude/fixed/processed/`.
#
# **Fixes (see Claude/REVIEW.md, section 3)**
# 1. **Missing buurten** – the final table was built from consumption buurten only, so 179 transformers in 104 buurten
#    (5 at/above NBL today) disappeared. Now every transformer is kept. Buurten without small-connection consumption get
#    the national scenario growth (no local adjustment) and the flag `no_consumption_data`.
# 2. **Horizon** – the business question asks about the *coming year* (2027). Peaks were measured in 2025, so the
#    projection is now made for **1 year (use during 2026)** and **2 years (2027 = main answer)**.
# 3. **Measurement coverage** – no transformer peaks between 8 Dec 2024 and 3 Feb 2025 (and 13 Nov – 5 Dec): winter was
#    mostly not measured. The daily coverage table is saved for the dashboard and every transformer keeps its peak month.
# 4. **Solar feed-in** – the Terugleverdata KV 2025/2026 files are now used (same postcode→buurt rule as notebook 02:
#    start postcode, fallback end postcode). Feed-in growth is weather-corrected with KNMI radiation and shrunk for small
#    buurten. It is used as a **sensitivity** ("what if the measured peaks were solar feed-in"), not as the central case,
#    because buurt utilisation follows consumption per kW much more than feed-in per kW (Claude/outputs/c02_log.txt).
#
# Step 0 re-runs the original method unchanged and checks that it reproduces 671 / 693 / 812.

# %%
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # Claude/fixed
PROJECT = HERE.parent.parent
RAW = PROJECT / "Data" / "raw"
PROCESSED = PROJECT / "Data" / "processed"             # read only
OUTP = HERE / "processed"                              # our outputs
(OUTP / "dashboard").mkdir(parents=True, exist_ok=True)
LOG = open(OUTP / "f09_log.txt", "w", encoding="utf-8")


def log(*x):
    s = " ".join(str(v) for v in x)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# %% [markdown]
# ## Step 0 – original method (identical code to notebook 09)

# %%
a = pd.read_csv(PROCESSED / "model_a_table.csv")
d = a[a["target_log_growth"].notna() & a["train_ok"]].copy()
national = d.groupby("year")["target_log_growth"].median()
scenarios = {"low": national.min(), "central": national.median(), "high": national.max()}
log("Scenarios per year (%):", {k: round(100 * (np.exp(v) - 1), 2) for k, v in scenarios.items()})

not_features = ["buurtcode", "year", "series", "target_log_growth", "target_clipped", "train_ok",
                "smart_meter_pct", "consume_only_pct", "low_tariff_pct", "n_rows"]
features = [c for c in a.columns if c not in not_features]
d["target_rel"] = (d["target_log_growth"]
                   - d.groupby("year")["target_log_growth"].transform("median")).clip(-0.5, 0.5)
params = dict(objective="l1", n_estimators=40, learning_rate=0.03, num_leaves=31,
              min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, verbose=-1)
m_bt = lgb.LGBMRegressor(**params).fit(d[d["year"] <= 2022][features], d[d["year"] <= 2022]["target_rel"])
te = d[d["year"].isin([2023, 2025])].copy()
te["actual_rel"] = te["target_log_growth"] - te.groupby("year")["target_log_growth"].transform("median")
te["resid"] = te["actual_rel"] - m_bt.predict(te[features])
bands = [0, 100, 250, 500, 1000, np.inf]
band_labels = ["<100", "100-250", "250-500", "500-1000", "1000+"]
te["size_band"] = pd.cut(te["connections"], bands, labels=band_labels)
q = te.groupby("size_band", observed=True)["resid"].quantile([0.1, 0.9]).unstack()
q.columns = ["q10", "q90"]
m_local = lgb.LGBMRegressor(**params).fit(d[features], d["target_rel"])

base = a[a["year"] == 2026].copy()
base["size_band"] = pd.cut(base["connections"], bands, labels=band_labels)
base["q10"] = base["size_band"].map(q["q10"]).astype(float)
base["q90"] = base["size_band"].map(q["q90"]).astype(float)

# 2026 weather: actual Jan-Sep + daily normals (as in notebook 09)
w = pd.read_excel(RAW / "3 knmi" / "All_Stations_Data.xlsx")
w.columns = w.columns.str.strip()
w = w.rename(columns={"# STN": "stn", "Data": "date"})
w["date"] = pd.to_datetime(w["date"])
for c in ["TG", "TN", "Q"]:
    w[c] = pd.to_numeric(w[c].astype(str).str.strip(), errors="coerce")
w["temp"] = w["TG"] / 10
w["frost"] = (w["TN"] < 0).astype(float).where(w["TN"].notna())
w["year"] = w["date"].dt.year
w["md"] = w["date"].dt.strftime("%m-%d")
normal_day = w[w["year"].between(2019, 2025)].groupby(["stn", "md"])[["temp", "frost", "Q"]].mean().reset_index()
days = pd.DataFrame({"date": pd.date_range("2026-01-01", "2026-12-31")})
days["md"] = days["date"].dt.strftime("%m-%d")
full = (days.merge(pd.DataFrame({"stn": w["stn"].unique()}), how="cross")
            .merge(w[w["year"] == 2026][["stn", "date", "temp", "frost", "Q"]], on=["stn", "date"], how="left")
            .merge(normal_day, on=["stn", "md"], how="left", suffixes=("", "_normal")))
for c in ["temp", "frost", "Q"]:
    full[c] = full[c].fillna(full[c + "_normal"])
full["hdd"] = (18 - full["temp"]).clip(lower=0)
full["cdd"] = (full["temp"] - 18).clip(lower=0)
w26 = full.groupby("stn").agg(temp_mean_c=("temp", "mean"), hdd_year=("hdd", "sum"), cdd_year=("cdd", "sum"),
                              frost_days=("frost", "sum"), solar_kwh_m2_year=("Q", lambda s: s.sum() / 360))
link = pd.read_csv(PROCESSED / "buurt_knmi_station.csv")
for c in ["temp_mean_c", "hdd_year", "cdd_year", "frost_days"]:
    base[c + "_t"] = base["buurtcode"].map(link.set_index("buurtcode")["stn_temp"]).map(w26[c])
base["solar_kwh_m2_year_t"] = base["buurtcode"].map(link.set_index("buurtcode")["stn_sun"]).map(w26["solar_kwh_m2_year"])
base["hdd_change"] = base["hdd_year_t"] - base["hdd_year_tm1"]
base["local_adj"] = m_local.predict(base[features])

tr_all = pd.read_csv(PROCESSED / "transformers_buurt2023.csv", parse_dates=["peak_date"])
chk = tr_all.merge(base[["buurtcode", "local_adj"]], on="buurtcode", how="inner")
chk_now = int((chk["peak_kw"] >= chk["nbl_limit_kw"]).sum())
chk_c = int((chk["peak_kw"] * np.exp(scenarios["central"] + chk["local_adj"]) >= chk["nbl_limit_kw"]).sum())
chk_h = int((chk["peak_kw"] * np.exp(scenarios["high"] + chk["local_adj"]) >= chk["nbl_limit_kw"]).sum())
log(f"Step 0 reproduction (original: 671 / 693 / 812): now {chk_now} | central 2026 {chk_c} | high 2026 {chk_h}")

# %% [markdown]
# ## Fix 4 – solar feed-in per buurt (Terugleverdata KV 2025 / 2026)

# %%
import io


def read_new_liander(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    lines = [ln[1:-1].replace('""', '"') if ln.startswith('"') and ln.endswith('"') else ln for ln in lines]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", dtype=str)
    df.columns = df.columns.str.strip()
    return df


pc = pd.read_csv(PROCESSED / "pc6_to_buurt_2023.csv", usecols=["pc6", "buurtcode"]).set_index("pc6")["buurtcode"]
fparts = []
for yr in [2025, 2026]:
    f = read_new_liander(RAW / "4 Liander" / "Raw Data" / "3" / f"Terugleverdata KV {yr}.csv")
    f = f[f["PRODUCTSOORT"].str.strip() == "ELK"]
    f["feedin"] = pd.to_numeric(f["TOT_E_INV"], errors="coerce")
    f["buurtcode"] = f["POSTCODE"].str.strip().map(pc).fillna(f["POSTCODE_EIND"].str.strip().map(pc))
    fparts.append(f.groupby("buurtcode")["feedin"].sum().rename(f"feedin_kwh_{yr}"))
    log(f"feed-in {yr}: {f['feedin'].sum() / 1e9:.2f} TWh, linked to a buurt {f.loc[f['buurtcode'].notna(), 'feedin'].sum() / f['feedin'].sum():.4f}")
fb = pd.concat(fparts, axis=1)

# weather correction: yearly radiation 2025 / 2024 at the buurt's sunlight station
qy = w[w["year"].isin([2024, 2025])].groupby(["stn", "year"])["Q"].agg(["sum", "count"]).reset_index()
qy = qy[qy["count"] >= 330].pivot(index="stn", columns="year", values="sum")
q_ratio = (qy[2025] / qy[2024])
fb["q_ratio"] = fb.index.map(link.set_index("buurtcode")["stn_sun"]).map(q_ratio)
fb["q_ratio"] = fb["q_ratio"].fillna(q_ratio.median())
conn = base.set_index("buurtcode")["connections"]
fb["connections"] = conn.reindex(fb.index)
ok = (fb["feedin_kwh_2025"] > 0) & (fb["feedin_kwh_2026"] > 0)
fb["pv_growth_raw"] = np.where(ok, np.log(fb["feedin_kwh_2026"] / fb["feedin_kwh_2025"]), np.nan)
fb["pv_growth_wc"] = fb["pv_growth_raw"] - np.log(fb["q_ratio"])            # weather-corrected
area = np.log((fb["feedin_kwh_2026"].sum() / fb["feedin_kwh_2025"].sum()) / q_ratio.median())
wt = fb["connections"].fillna(0) / (fb["connections"].fillna(0) + 300)
fb["pv_growth"] = (wt * fb["pv_growth_wc"].fillna(area) + (1 - wt) * area).clip(-0.05, 0.40)
log(f"Area feed-in growth: raw {100 * (fb['feedin_kwh_2026'].sum() / fb['feedin_kwh_2025'].sum() - 1):+.1f}% | "
    f"radiation 2025/2024 {q_ratio.median():.3f} | weather-corrected PV growth {100 * (np.exp(area) - 1):+.1f}% per year")
fb.reset_index().rename(columns={"index": "buurtcode"}).to_csv(OUTP / "buurt_feedin.csv", index=False)

# %% [markdown]
# ## Fixes 1 + 2 – all transformers, 1-year and 2-year projection

# %%
tr = tr_all.merge(base[["buurtcode", "local_adj", "q90"]], on="buurtcode", how="left")
tr["no_consumption_data"] = tr["local_adj"].isna()
tr["local_adj"] = tr["local_adj"].fillna(0.0)                          # national growth only
tr["q90"] = tr["q90"].fillna(float(q["q90"].max()))                    # widest band when unknown
tr = tr.merge(fb[["pv_growth"]], left_on="buurtcode", right_index=True, how="left")
tr["pv_growth"] = tr["pv_growth"].fillna(area)
tr["headroom_pct"] = 100 * (tr["nbl_limit_kw"] - tr["peak_kw"]) / tr["nbl_limit_kw"]
tr["above_nbl_now"] = tr["peak_kw"] >= tr["nbl_limit_kw"]
tr["peak_month"] = tr["peak_date"].dt.month
now = int(tr["above_nbl_now"].sum())
log(f"\nTransformers: {len(tr)} (was {len(chk)} in the original) | without consumption forecast: {int(tr['no_consumption_data'].sum())}")
log(f"At/above NBL now: {now} (original: 671)")

rows = []
for h, yr in [(1, 2026), (2, 2027)]:
    for s in ["low", "central", "high"]:
        g = h * (scenarios[s] + tr["local_adj"])
        tr[f"above_nbl_{yr}_{s}"] = tr["peak_kw"] * np.exp(g) >= tr["nbl_limit_kw"]
    g_st = h * (scenarios["high"] + tr["local_adj"]) + np.sqrt(h) * tr["q90"]     # stress test
    tr[f"above_nbl_{yr}_stresstest"] = tr["peak_kw"] * np.exp(g_st) >= tr["nbl_limit_kw"]
    tr[f"above_nbl_{yr}_solar"] = tr["peak_kw"] * np.exp(h * tr["pv_growth"]) >= tr["nbl_limit_kw"]   # sensitivity
    for s in ["low", "central", "high", "stresstest", "solar"]:
        n = int(tr[f"above_nbl_{yr}_{s}"].sum())
        rows.append({"horizon": f"use during {yr}", "scenario": s, "transformers_at_or_above_nbl": n,
                     "newly": int((tr[f"above_nbl_{yr}_{s}"] & ~tr["above_nbl_now"]).sum()),
                     "change_vs_now_pct": round(100 * (n / now - 1), 1)})
scen_tab = pd.DataFrame(rows)
scen_tab.to_csv(OUTP / "scenario_table.csv", index=False)
log("\n" + scen_tab.to_string(index=False))

# %% [markdown]
# ## Buurt table and priority levels (main answer = 2027)

# %%
g = tr.groupby("buurtcode")
pb = pd.DataFrame({
    "n_transformers_proj": g.size(),
    "n_above_nbl_now": g["above_nbl_now"].sum(),
    "min_headroom_pct": g["headroom_pct"].min(),
    "n_new_2026_central": g.apply(lambda x: (x["above_nbl_2026_central"] & ~x["above_nbl_now"]).sum()),
    "n_new_2027_central": g.apply(lambda x: (x["above_nbl_2027_central"] & ~x["above_nbl_now"]).sum()),
    "n_new_2027_high": g.apply(lambda x: (x["above_nbl_2027_high"] & ~x["above_nbl_now"]).sum()),
    "n_new_2027_stresstest": g.apply(lambda x: (x["above_nbl_2027_stresstest"] & ~x["above_nbl_now"]).sum()),
    "n_new_2027_solar": g.apply(lambda x: (x["above_nbl_2027_solar"] & ~x["above_nbl_now"]).sum()),
    "tr_buurtnaam": g["buurtnaam"].first(), "tr_gemeentenaam": g["gemeentenaam"].first(),
}).reset_index()
pb["priority"] = np.select(
    [pb["n_above_nbl_now"] > 0, pb["n_new_2027_central"] > 0, pb["n_new_2027_high"] > 0, pb["n_new_2027_stresstest"] > 0],
    ["1 Stressed now", "2 Crosses by 2027 (central)", "3 Crosses by 2027 (high)", "4 Crosses in stress test only"],
    "5 No crossing expected")
pb["solar_sensitivity_flag"] = (pb["priority"].str[0].isin(["4", "5"])) & (pb["n_new_2027_solar"] > 0)

for h, yr in [(1, 2026), (2, 2027)]:
    for s in ["low", "central", "high"]:
        base[f"kwh_use{yr}_{s}"] = base["total_kwh"] * np.exp(h * (scenarios[s] + base["local_adj"]))
    base[f"central_{yr}_pct"] = 100 * (np.exp(h * (scenarios["central"] + base["local_adj"])) - 1)
    base[f"p10_{yr}_pct"] = 100 * (np.exp(h * (scenarios["central"] + base["local_adj"]) + np.sqrt(h) * base["q10"]) - 1)
    base[f"p90_{yr}_pct"] = 100 * (np.exp(h * (scenarios["central"] + base["local_adj"]) + np.sqrt(h) * base["q90"]) - 1)
base["kwh_use2025"] = base["total_kwh"]
base["low_reliability"] = base["connections"] < 50
fc = ["buurtcode", "connections", "kwh_use2025"] + [c for c in base.columns if c.startswith(("kwh_use202", "central_202", "p10_", "p90_"))] + ["local_adj", "low_reliability"]
fc = list(dict.fromkeys(fc))

stress = pd.read_csv(PROCESSED / "buurt_stress_predictions.csv")
names = pd.read_csv(PROCESSED / "pc6_to_buurt_2023.csv")[["buurtcode", "buurtnaam", "gemeentenaam"]].drop_duplicates("buurtcode")
final = (base[fc].merge(pb, on="buurtcode", how="outer")                  # FIX 1: outer join keeps every transformer buurt
                 .merge(names, on="buurtcode", how="left")
                 .merge(stress, on="buurtcode", how="left")
                 .merge(fb[["feedin_kwh_2025", "feedin_kwh_2026", "pv_growth"]], left_on="buurtcode", right_index=True, how="left"))
final["buurtnaam"] = final["buurtnaam"].fillna(final["tr_buurtnaam"])
final["gemeentenaam"] = final["gemeentenaam"].fillna(final["tr_gemeentenaam"])
final = final.drop(columns=["tr_buurtnaam", "tr_gemeentenaam"])
final["no_consumption_data"] = final["kwh_use2025"].isna()
final["priority"] = final["priority"].fillna("No transformer data")
final["solar_sensitivity_flag"] = final["solar_sensitivity_flag"].fillna(False).astype(bool)
final["feed_share_pct"] = 100 * final["feedin_kwh_2026"] / (final["feedin_kwh_2026"] + final["kwh_use2025"])
final["pv_growth_pct"] = 100 * (np.exp(final["pv_growth"]) - 1)
final.to_csv(OUTP / "final_buurt_risk_fixed.csv", index=False)
tr.to_csv(OUTP / "transformer_projection_fixed.csv", index=False)

log("\nBuurten per priority (main answer, 2027):\n" + final["priority"].value_counts().sort_index().to_string())
log(f"Buurten added by fix 1: {int(final['no_consumption_data'].sum())} (of which stressed now: "
    f"{int((final['no_consumption_data'] & final['priority'].str.startswith('1')).sum())})")
log(f"Solar-sensitivity flag (no crossing on consumption, but crossing if peaks were solar-driven): {int(final['solar_sensitivity_flag'].sum())} buurten")
show = ["buurtnaam", "gemeentenaam", "n_new_2027_central", "min_headroom_pct", "central_2027_pct", "driver_1"]
log("\nPriority 2 – crosses by 2027 (central):\n" + final[final["priority"].str.startswith("2")]
    .sort_values("min_headroom_pct")[show].round(2).to_string(index=False))

# %% [markdown]
# ## Fix 3 – measurement coverage table for the dashboard

# %%
win = w[(w["date"] >= tr["peak_date"].min()) & (w["date"] <= tr["peak_date"].max())]
cov = win.groupby("date").agg(radiation_kwh_m2=("Q", lambda s: s.mean() / 360), temp_c=("temp", "mean"))
cov["transformers_peaking"] = tr["peak_date"].value_counts().reindex(cov.index).fillna(0).astype(int)
cov.reset_index().to_csv(OUTP / "dashboard" / "peak_coverage.csv", index=False)
winter = cov.index.month.isin([11, 12, 1, 2])
log(f"\nPeaks in Nov-Feb: {100 * cov.loc[winter, 'transformers_peaking'].sum() / cov['transformers_peaking'].sum():.1f}% "
    f"(these months are {100 * winter.mean():.0f}% of the window); days with >=1 peak: {(cov['transformers_peaking'] > 0).sum()} of {len(cov)}")

# buurt trend 2019-2026 for the buurt profile (existing processed file, read only)
be = pd.read_csv(PROCESSED / "buurt_year_electricity.csv", usecols=["buurtcode", "year", "series", "total_kwh", "connections"])
be.to_csv(OUTP / "dashboard" / "buurt_trend.csv", index=False)

# shapes: the original geojson + the buurten added by fix 1
import geopandas as gpd
geo = gpd.read_file(RAW / "External" / "cbs_buurten_2023_simplified.geojson")
geo = geo[geo["buurtcode"].isin(final["buurtcode"])][["buurtcode", "geometry"]].to_crs("EPSG:28992")
geo["geometry"] = geo.simplify(25, preserve_topology=True)
geo.to_crs("EPSG:4326").to_file(OUTP / "dashboard" / "buurten_liander_fixed.geojson", driver="GeoJSON")
log(f"geojson buurten: {len(geo)} (final table {len(final)})")
log("DONE f09")
LOG.close()
