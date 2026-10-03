"""Claude review - step 4: growth of CONSUMPTION and of SOLAR FEED-IN per buurt.

(a) Consumption: re-test the original Model A finding (one-year local growth is hard to predict)
    with my own buurt allocation, using the same idea: LightGBM on growth relative to the year median,
    trained on t <= 2022, tested on 2023->24 (old series) and 2025->26 (new series).
(b) Feed-in: the original project never used the 'Terugleverdata KV 2025/2026' files. They show
    feed-in growth per buurt (use during 2024 -> 2025). Because peaks are measured on clear days,
    the relevant driver is installed PV, so I remove the weather part (sunnier 2025) using KNMI
    radiation at the buurt's nearest station, and shrink small-buurt noise toward the area median.

Output: data/buurt_growth.csv, outputs/c04_log.txt
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

from common import DATA, OUT, log

fh = open(OUT / "c04_log.txt", "w", encoding="utf-8")
by = pd.read_csv(DATA / "buurt_year.csv").sort_values(["buurtcode", "year"])
cbs = pd.read_csv(DATA / "cbs2024.csv")

# ---------------- (a) consumption growth backtest
g = by.groupby(["buurtcode", "series"])
by["next_kwh"], by["next_year"] = g["total_kwh"].shift(-1), g["year"].shift(-1)
by["prev_kwh"], by["prev_year"] = g["total_kwh"].shift(1), g["year"].shift(1)
by["prev_conn"] = g["connections"].shift(1)
by["y"] = np.where(by["next_year"] == by["year"] + 1, np.log(by["next_kwh"] / by["total_kwh"]), np.nan)
by["lag_g"] = np.where(by["prev_year"] == by["year"] - 1, np.log(by["total_kwh"] / by["prev_kwh"]), np.nan)
by["lag_conn_g"] = np.where(by["prev_year"] == by["year"] - 1, np.log(by["connections"] / by["prev_conn"]), np.nan)
by["kwh_per_conn"] = by["total_kwh"] / by["connections"]
d = by.merge(cbs, on="buurtcode", how="left")
d = d[d["y"].notna() & (d["connections"] >= 50)].replace([np.inf, -np.inf], np.nan)
nat = d.groupby("year")["y"].median()
log("Median buurt growth per year t->t+1 (%):\n" + (100 * (np.exp(nat) - 1)).round(2).to_string(), fh)
d["y_rel"] = (d["y"] - d["year"].map(nat)).clip(-0.5, 0.5)
feats = ["connections", "total_kwh", "kwh_per_conn", "share_3phase", "lag_g", "lag_conn_g"] + \
        [c for c in cbs.columns if c != "buurtcode"]
m = lgb.LGBMRegressor(objective="l1", n_estimators=150, learning_rate=0.03, num_leaves=15,
                      min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      random_state=42, verbose=-1).fit(d.loc[d["year"] <= 2022, feats], d.loc[d["year"] <= 2022, "y_rel"])
for yr in [2023, 2025]:
    t = d[d["year"] == yr]
    rel = t["y"] - t["y"].median()
    p = m.predict(t[feats])
    pers = (t["lag_g"] - t["lag_g"].median()).fillna(0)
    log(f"test {yr}->{yr + 1}: MAE LightGBM {mean_absolute_error(rel, p):.4f} | 'follow the country' "
        f"{mean_absolute_error(rel, 0 * p):.4f} | persistence {mean_absolute_error(rel, pers):.4f} | "
        f"Spearman {spearmanr(rel, p).correlation:.3f}", fh)
auto = d.pivot_table(index="buurtcode", columns="year", values="y_rel")
log("Year-to-year correlation of a buurt's RELATIVE growth (does local growth persist?):", fh)
for a, b in [(2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023)]:
    log(f"   {a} vs {b}: {auto[[a, b]].corr(method='spearman').iloc[0, 1]:.3f}", fh)

# local adjustment for the forecast base (2026 file = use during 2025), model on all years
m_all = lgb.LGBMRegressor(objective="l1", n_estimators=150, learning_rate=0.03, num_leaves=15,
                          min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=42, verbose=-1).fit(d[feats], d["y_rel"])
base = by[by["year"] == 2026].merge(cbs, on="buurtcode", how="left").replace([np.inf, -np.inf], np.nan)
base["cons_local_adj"] = m_all.predict(base[feats])

# ---------------- (b) feed-in growth 2025 -> 2026 file (= use during 2024 -> 2025)
f = by[by["year"].isin([2025, 2026])].pivot_table(index="buurtcode", columns="year", values="feedin_kwh")
c = by[by["year"] == 2026].set_index("buurtcode")["connections"]
fg = pd.DataFrame({"f25": f[2025], "f26": f[2026], "conn": c}).dropna()
fg = fg[(fg["f25"] > 0) & (fg["f26"] > 0)]
tot_g = fg["f26"].sum() / fg["f25"].sum() - 1
log(f"\nFeed-in, Liander area: {fg['f25'].sum() / 1e9:.2f} -> {fg['f26'].sum() / 1e9:.2f} TWh ({100 * tot_g:+.1f}%)", fh)

# weather correction: radiation at the nearest station, 2025 vs 2024 (full calendar years)
w = pd.read_csv(DATA / "knmi_daily.csv", parse_dates=["date"])
st = pd.read_csv(DATA / "knmi_stations.csv")
st = st[st["has_q"]]
w["year"] = w["date"].dt.year
qy = w[w["year"].isin([2024, 2025])].groupby(["stn", "year"])["Q"].agg(["sum", "count"]).reset_index()
qy = qy[qy["count"] >= 330].pivot(index="stn", columns="year", values="sum")
qy["ratio"] = qy[2025] / qy[2024]
st = st[st["stn"].isin(qy.index)]
log(f"Radiation 2025 / 2024 at stations: median {qy['ratio'].median():.3f} (min {qy['ratio'].min():.3f}, max {qy['ratio'].max():.3f})", fh)
pc = pd.read_csv(DATA / "pc6_buurt.csv", dtype={"pc6": str})
from common import PC6_FILE
pts = pd.read_csv(PC6_FILE, dtype={"pc6": str})[["pc6", "lon", "lat"]].merge(pc, on="pc6")
cent = pts.groupby("buurtcode")[["lon", "lat"]].mean()
lat0 = np.radians(52.2)
def xy(lon, lat):
    return np.c_[np.radians(lon) * np.cos(lat0), np.radians(lat)]
dd = ((xy(cent["lon"].values, cent["lat"].values)[:, None, :] - xy(st["lon"].values, st["lat"].values)[None]) ** 2).sum(-1)
cent["q_ratio"] = qy.loc[st["stn"].values[dd.argmin(1)], "ratio"].values
fg = fg.join(cent["q_ratio"], how="left")
fg["q_ratio"] = fg["q_ratio"].fillna(qy["ratio"].median())
fg["g_raw"] = np.log(fg["f26"] / fg["f25"])
fg["g_pv"] = fg["g_raw"] - np.log(fg["q_ratio"])              # growth in installed PV (weather removed)
K = 300.0                                                      # shrinkage strength (connections)
med = np.average(fg["g_pv"], weights=fg["f25"])
wgt = fg["conn"] / (fg["conn"] + K)
fg["g_pv_shrunk"] = (wgt * fg["g_pv"] + (1 - wgt) * med).clip(-0.05, 0.40)
log(f"Feed-in growth per buurt, raw median {100 * (np.exp(fg['g_raw'].median()) - 1):+.1f}% | "
    f"weather-corrected (PV) median {100 * (np.exp(fg['g_pv'].median()) - 1):+.1f}% | area (kWh-weighted) "
    f"{100 * (np.exp(med) - 1):+.1f}%", fh)
log("Shrunk PV growth per buurt (%):\n" + (100 * (np.exp(fg["g_pv_shrunk"]) - 1)).describe(
    percentiles=[.05, .25, .5, .75, .95]).round(2).to_string(), fh)
x = fg.join(cbs.set_index("buurtcode"), how="left")
log("Spearman of PV growth with drivers: " + ", ".join(
    f"{c} {spearmanr(x['g_pv'], x[c], nan_policy='omit').correlation:.2f}"
    for c in ["p_won_zs", "g_ele_tr", "p_1gezw", "p_koopw", "g_ink_pi", "g_wozbag", "p_bj_mi10"]), fh)

out = base[["buurtcode", "connections", "total_kwh", "feedin_kwh", "cons_local_adj"]].merge(
    fg[["g_raw", "g_pv", "g_pv_shrunk"]], left_on="buurtcode", right_index=True, how="left")
out["g_pv_shrunk"] = out["g_pv_shrunk"].fillna(med)
out["pv_area_growth"] = med
out.to_csv(DATA / "buurt_growth.csv", index=False)
log(f"saved buurt_growth.csv {out.shape}", fh)
log("DONE c04", fh)
fh.close()
