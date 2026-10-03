"""Claude review - step 2: WHAT causes the measured transformer peaks?

The original notebooks (03, 08) observed that ~98% of peaks fall in April-July and concluded this is
"likely a measurement artefact". Test: if peaks are solar feed-in (reverse power flow), they should
fall on the SUNNIEST days at the nearest KNMI station; if they are consumption peaks they should fall
on COLD / dark days (winter evenings, heat pumps) -- the measurement window Sep 2024 - Jul 2025
contains a full winter, so both are possible.

Outputs: outputs/c02_log.txt, outputs/c02_peak_days.csv, data/transformers_peaktype.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DATA, OUT, log

fh = open(OUT / "c02_log.txt", "w", encoding="utf-8")
tr = pd.read_csv(DATA / "transformers.csv", parse_dates=["peak_date"])
w = pd.read_csv(DATA / "knmi_daily.csv", parse_dates=["date"])
st = pd.read_csv(DATA / "knmi_stations.csv")
st = st[st["has_q"]].reset_index(drop=True)

WIN0, WIN1 = tr["peak_date"].min(), tr["peak_date"].max()
log(f"Peak window: {WIN0.date()} - {WIN1.date()}  ({(WIN1 - WIN0).days + 1} days)", fh)
win = w[(w["date"] >= WIN0) & (w["date"] <= WIN1)].copy()

# 1. national picture: peaks per day vs national radiation / temperature
nat = win.groupby("date")[["Q", "TG", "TX"]].mean()
nat["n_peaks"] = tr["peak_date"].value_counts().reindex(nat.index).fillna(0)
nat["Q_rank_pct"] = nat["Q"].rank(pct=True) * 100
nat["cold_rank_pct"] = (-nat["TG"]).rank(pct=True) * 100
top = nat.sort_values("n_peaks", ascending=False).head(15)
log("\n15 dates with the most transformer peaks (Q = radiation J/cm2, TG/TX in 0.1 C):\n"
    + top.round(1).to_string(), fh)
tot = nat["n_peaks"].sum()
for lab, mask in [("10% sunniest days", nat["Q_rank_pct"] >= 90),
                  ("10% coldest days", nat["cold_rank_pct"] >= 90),
                  ("Nov-Feb (winter)", nat.index.month.isin([11, 12, 1, 2]))]:
    log(f"share of all peaks on the {lab:18s}: {100 * nat.loc[mask, 'n_peaks'].sum() / tot:5.1f}%  "
        f"(these days are {100 * mask.mean():.0f}% of the window)", fh)
nat.to_csv(OUT / "c02_peak_days.csv")

# 1b. measurement coverage: very sunny days WITHOUT any peak point to periods that were not measured
spring = nat.loc["2025-03-01":]
log("\nSunniest 20 days from March 2025 and their number of peaks (0 on a top-sunny day = likely no data):\n"
    + spring.sort_values("Q", ascending=False).head(20)[["Q", "TX", "n_peaks"]].round(0).to_string(), fh)
zero = (nat["n_peaks"] == 0).astype(int)
runs, start, prev = [], None, 0
for day, z in zero.items():
    if z and not prev:
        start = day
    if not z and prev:
        runs.append((start, day - pd.Timedelta(days=1)))
    prev = z
if prev:
    runs.append((start, nat.index[-1]))
long_runs = [(a.date(), b.date(), (b - a).days + 1) for a, b in runs if (b - a).days + 1 >= 7]
log("Blocks of >= 7 consecutive days with ZERO peaks among 31k transformers:\n"
    + "\n".join(f"   {a} to {b} ({n} days)" for a, b, n in long_runs), fh)
days_with_peaks = nat[nat["n_peaks"] > 0]
log(f"Days with at least one peak: {len(days_with_peaks)} of {len(nat)}", fh)
q_in = days_with_peaks["Q"].rank(pct=True)
log(f"Inside the days that DO have peaks: Spearman(n_peaks, radiation) = "
    f"{spearmanr(days_with_peaks['n_peaks'], days_with_peaks['Q']).correlation:.3f}, "
    f"vs mean temperature = {spearmanr(days_with_peaks['n_peaks'], days_with_peaks['TG']).correlation:.3f}", fh)

# 2. per transformer: rank of its peak day at its NEAREST radiation station
lat0 = np.radians(52.2)
def xy(lon, lat):
    return np.c_[np.radians(lon) * np.cos(lat0) * 6371, np.radians(lat) * 6371]
d = ((xy(tr["lon"], tr["lat"])[:, None, :] - xy(st["lon"], st["lat"])[None, :, :]) ** 2).sum(-1)
tr["stn"] = st["stn"].values[d.argmin(1)]
win["q_rank_pct"] = win.groupby("stn")["Q"].rank(pct=True) * 100
win["cold_rank_pct"] = win.groupby("stn")["TG"].rank(ascending=False, pct=True) * 100
tr = tr.merge(win[["stn", "date", "Q", "TG", "q_rank_pct", "cold_rank_pct"]],
              left_on=["stn", "peak_date"], right_on=["stn", "date"], how="left").drop(columns="date")
log(f"\nTransformers with station weather on the peak day: {tr['q_rank_pct'].notna().mean():.3f}", fh)
log("Radiation rank of the peak day at the nearest station (100 = sunniest day of the window):\n"
    + tr["q_rank_pct"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string(), fh)
log(f"peak on a top-10% sunny day: {100 * (tr['q_rank_pct'] >= 90).mean():.1f}% of transformers "
    f"| on a top-10% cold day: {100 * (tr['cold_rank_pct'] >= 90).mean():.1f}%", fh)

sunny = (tr["q_rank_pct"] >= 75) & tr["peak_date"].dt.month.between(3, 9)
coldish = tr["peak_date"].dt.month.isin([10, 11, 12, 1, 2]) | (tr["cold_rank_pct"] >= 75)
tr["peak_type"] = np.select([sunny, coldish], ["sunny day (feed-in likely)", "cold/dark day (consumption likely)"],
                            "unclear")
log("\nPeak type (rule: sunny = spring/summer day in the station's top-25% radiation):\n"
    + tr["peak_type"].value_counts().to_string(), fh)
log("Share above NBL by peak type:\n"
    + tr.groupby("peak_type")[["above_nbl", "util"]].mean().round(3).to_string(), fh)
log("Weekday of peak (0=Mon):\n" + tr["peak_date"].dt.dayofweek.value_counts().sort_index().to_string(), fh)

# 3. buurt level: does utilisation follow feed-in or consumption, per kW of capacity?
by = pd.read_csv(DATA / "buurt_year.csv")
b26 = by[by["year"] == 2026].set_index("buurtcode")          # SJV/feed-in 2026 file = use during 2025
bt = tr.groupby("buurtcode").agg(cap=("capacity_kw", "sum"), peak=("peak_kw", "sum"),
                                 share_sunny=("peak_type", lambda s: (s == "sunny day (feed-in likely)").mean()))
bt = bt.join(b26[["total_kwh", "feedin_kwh", "connections"]], how="inner")
bt["util"] = bt["peak"] / bt["cap"]
bt["cons_kw_per_kw"] = bt["total_kwh"] / 8760 / bt["cap"]         # average consumption load per kW cap
bt["feed_kw_per_kw"] = bt["feedin_kwh"] / 8760 / bt["cap"]
bt["feed_share"] = bt["feedin_kwh"] / (bt["feedin_kwh"] + bt["total_kwh"])
bt = bt.replace([np.inf, -np.inf], np.nan).dropna()
r = bt[["util", "cons_kw_per_kw", "feed_kw_per_kw"]].rank()
r = (r - r.mean()) / r.std()
beta = np.linalg.lstsq(np.c_[np.ones(len(r)), r[["cons_kw_per_kw", "feed_kw_per_kw"]].values],
                       r["util"].values, rcond=None)[0]
log(f"\nBuurten: {len(bt)}", fh)
log(f"Spearman util ~ consumption per kW cap: {spearmanr(bt['util'], bt['cons_kw_per_kw']).correlation:.3f}", fh)
log(f"Spearman util ~ feed-in per kW cap:     {spearmanr(bt['util'], bt['feed_kw_per_kw']).correlation:.3f}", fh)
log(f"Rank regression (standardised) util ~ consumption + feed-in: cons {beta[1]:.3f}, feed-in {beta[2]:.3f}", fh)
bt["feed_q"] = pd.qcut(bt["feed_share"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
log("By feed-in share of the buurt (feed-in / (feed-in + consumption)):\n"
    + bt.groupby("feed_q", observed=True)[["feed_share", "util", "share_sunny"]].mean().round(3).to_string(), fh)

tr.to_csv(DATA / "transformers_peaktype.csv", index=False)
log("DONE c02", fh)
fh.close()
