"""Claude review - step 5: project transformer peaks and compare with the original results.

Change ONE thing at a time versus the original notebook 09:
  M0  original method re-done with my data: every peak x consumption growth, 1 year (use during 2026)
  M1  same, 2 years ahead (the 'coming year' = 2027; peaks were measured in 2025)
  M2  peak driver by type: sunny-day peaks x PV growth, other peaks x consumption growth, 1 year
  M3  = M2, 2 years ahead (2027)
Main answer = M1 (consumption-driven, 2 years = 2027). M2/M3 are a SENSITIVITY ("what if the measured
peaks were solar feed-in"): c02 shows the peak days are the sunniest measured days, but buurt utilisation
follows consumption per kW more than feed-in per kW, so a feed-in-driven projection is not supported as
the central case.
Consumption scenarios = the original ones (low -3.41%, central +0.50%, high +2.93% per year) so that
any difference comes from the peak driver / horizon, not from a different demand forecast.
PV scenarios per buurt: central = weather-corrected, shrunk 2024->2025 growth; low = half; high = 1.5x.

Outputs: outputs/c05_log.txt, outputs/claude_buurt_priority.csv, outputs/claude_transformer_projection.csv,
         outputs/comparison_summary.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DATA, OUT, USER_PROCESSED, log

fh = open(OUT / "c05_log.txt", "w", encoding="utf-8")
tr = pd.read_csv(DATA / "transformer_oof.csv", parse_dates=["peak_date"])
gr = pd.read_csv(DATA / "buurt_growth.csv")
tr = tr.merge(gr[["buurtcode", "g_pv_shrunk", "cons_local_adj"]], on="buurtcode", how="left")
tr["g_pv_shrunk"] = tr["g_pv_shrunk"].fillna(gr["pv_area_growth"].iloc[0])
tr["cons_local_adj"] = tr["cons_local_adj"].fillna(0)

CONS = {"low": np.log(1 - 0.0341), "central": np.log(1 + 0.0050), "high": np.log(1 + 0.0293)}
PVMULT = {"low": 0.5, "central": 1.0, "high": 1.5}
tr["pv_type"] = tr["peak_type"].eq("sunny day (feed-in likely)") | (
    tr["peak_type"].eq("unclear") & tr["peak_date"].dt.month.between(3, 9))
now = int(tr["atabove_nbl"].sum())
log(f"Transformers: {len(tr)} | at/above NBL now: {now} | above now: {int(tr['above_nbl'].sum())} | "
    f"PV-type peaks: {tr['pv_type'].mean():.3f}", fh)
tr["headroom_pct"] = 100 * (tr["nbl_kw"] - tr["peak_kw"]) / tr["nbl_kw"]

rows = []
for s in ["low", "central", "high"]:
    g_cons = CONS[s] + tr["cons_local_adj"]
    g_pv = PVMULT[s] * tr["g_pv_shrunk"]
    for name, h, mixed in [("M0 consumption x1yr (original method)", 1, False),
                           ("M1 consumption x2yr", 2, False),
                           ("M2 peak-type x1yr", 1, True),
                           ("M3 peak-type x2yr (2027)", 2, True)]:
        g = np.where(tr["pv_type"], g_pv, g_cons) if mixed else g_cons
        proj = tr["peak_kw"] * np.exp(h * g)
        over = proj >= tr["nbl_kw"]
        col = f"{name.split()[0]}_{s}"
        tr[col] = over
        rows.append({"method": name, "scenario": s, "at_or_above_NBL": int(over.sum()),
                     "newly": int((over & (tr["atabove_nbl"] == 0)).sum()),
                     "change_vs_now_pct": round(100 * (over.sum() / now - 1), 1)})
tab = pd.DataFrame(rows)
log("\nProjected transformers at/above the NBL limit:\n" + tab.pivot(index="method", columns="scenario",
                                                                      values="at_or_above_NBL")[["low", "central", "high"]].to_string(), fh)
log("\nNewly crossing:\n" + tab.pivot(index="method", columns="scenario", values="newly")[["low", "central", "high"]].to_string(), fh)

# ---------------- buurt priority (my version)
g = tr.groupby("buurtcode")
bp = pd.DataFrame({"n_trafo": g.size(), "n_now": g["atabove_nbl"].sum(),
                   "n_2027_central": g["M1_central"].sum(), "n_2027_high": g["M1_high"].sum(),
                   "n_2027_if_pv_driven": g["M3_central"].sum(),
                   "min_headroom_pct": g["headroom_pct"].min(),
                   "p_buurt_model": g["p_above_nbl"].apply(lambda p: 1 - np.prod(1 - p)),
                   "share_pv_type": g["pv_type"].mean(),
                   "pv_growth_pct": 100 * (np.exp(g["g_pv_shrunk"].first()) - 1)})
bp["priority"] = np.select(
    [bp["n_now"] > 0, bp["n_2027_central"] > 0, bp["n_2027_high"] > 0, bp["n_2027_if_pv_driven"] > 0],
    ["1 Stressed now", "2 Crosses by 2027 (central)", "3 Crosses by 2027 (high)",
     "4 Crosses only if peaks are PV-driven"], "5 No crossing expected")
# 'watch' score: model risk for buurten with no crossing
bp = bp.reset_index()
names = pd.read_csv(USER_PROCESSED / "final_buurt_risk.csv", usecols=["buurtcode", "buurtnaam", "gemeentenaam", "priority",
                                                                     "risk_prob", "driver_1"])
names = names.rename(columns={"priority": "priority_original", "risk_prob": "risk_prob_original",
                              "driver_1": "driver_1_original"})
bp = bp.merge(names, on="buurtcode", how="left")
bp.sort_values(["priority", "min_headroom_pct"]).to_csv(OUT / "claude_buurt_priority.csv", index=False)
log("\nMy buurt priority levels:\n" + bp["priority"].value_counts().sort_index().to_string(), fh)
log("\nCross-table (rows = original priority, columns = my priority):\n"
    + pd.crosstab(bp["priority_original"].fillna("n/a"), bp["priority"]).to_string(), fh)
log(f"\nSpearman original risk_prob vs my transformer-model buurt risk: "
    f"{spearmanr(bp['risk_prob_original'], bp['p_buurt_model'], nan_policy='omit').correlation:.3f}", fh)
top_o = set(bp.nlargest(250, "risk_prob_original")["buurtcode"])
top_c = set(bp.nlargest(250, "p_buurt_model")["buurtcode"])
log(f"Overlap of the top-250 model-risk buurten: {len(top_o & top_c)} of 250", fh)
log("\nTop 20 buurten crossing by 2027 (central), least headroom first:\n" + bp[bp["priority"].str.startswith("2")]
    .sort_values("min_headroom_pct")[["buurtnaam", "gemeentenaam", "n_trafo", "n_2027_central", "min_headroom_pct",
                                      "p_buurt_model", "priority_original"]].head(20).round(2).to_string(index=False), fh)

tr.to_csv(OUT / "claude_transformer_projection.csv", index=False)

# ---------------- reproduce the original headline numbers from the original outputs
m0c = tab[tab["method"].str.startswith("M0") & (tab["scenario"] == "central")]["at_or_above_NBL"].iloc[0]
summ = [("M0 central (my data, original method)", int(m0c))]
try:
    ut = pd.read_csv(USER_PROCESSED / "transformer_projection_2026.csv")
    summ += [("original: at/above now", int(ut["above_nbl_now"].sum())),
             ("original: central 2026", int(ut["above_nbl_growth_central"].sum())),
             ("original: high 2026", int(ut["above_nbl_growth_high"].sum()))]
    j = ut[["transformer_id", "buurtcode"]].merge(tr[["transformer_id", "buurtcode"]], on="transformer_id",
                                                  suffixes=("_orig", "_claude"))
    log(f"\nSame buurt for the same transformer (original vs mine): {(j['buurtcode_orig'] == j['buurtcode_claude']).mean():.4f}", fh)
except FileNotFoundError:
    pass
summ += [("Claude: at/above now", now)] + [(f"Claude {r.method} {r.scenario}", r.at_or_above_NBL) for r in tab.itertuples()]
pd.DataFrame(summ, columns=["item", "transformers_at_or_above_NBL"]).to_csv(OUT / "comparison_summary.csv", index=False)

# ---------------- consumption per buurt: my allocation vs original start-postcode allocation
try:
    ue = pd.read_csv(USER_PROCESSED / "buurt_year_electricity.csv")
    me = pd.read_csv(DATA / "buurt_year.csv")
    k = ue.merge(me, on=["buurtcode", "year"], suffixes=("_orig", "_claude"))
    k = k[k["year"] == 2026]
    rel = (k["total_kwh_claude"] / k["total_kwh_orig"] - 1).abs()
    log(f"\n2026 buurt kWh, my range-splitting vs original start-postcode: median abs diff {100 * rel.median():.2f}%, "
        f"buurten with >10% difference: {(rel > 0.10).mean():.3f}, correlation {k[['total_kwh_orig', 'total_kwh_claude']].corr().iloc[0, 1]:.4f}", fh)
except FileNotFoundError:
    pass
log("DONE c05", fh)
fh.close()
