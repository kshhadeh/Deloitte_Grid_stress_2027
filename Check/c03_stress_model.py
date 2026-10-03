"""Claude review - step 3: TRANSFORMER-level stress model (LightGBM + SHAP via pred_contrib).

Original approach (notebook 08): one row per buurt, label = ">=1 transformer above NBL".
That label grows with the number of transformers in the buurt (size effect).
Alternative here: one row per transformer (31k rows), label = this transformer above NBL,
plus a regression on utilisation. Buurt risk = 1 - prod(1 - p_transformer).
Validation: 5-fold GroupKFold by municipality (same idea as the original, so results are comparable).

Outputs: outputs/c03_log.txt, outputs/c03_shap_*.csv, data/transformer_oof.csv, data/buurt_risk_claude.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error
from scipy.stats import spearmanr

from common import DATA, OUT, USER_PROCESSED, log

fh = open(OUT / "c03_log.txt", "w", encoding="utf-8")
tr = pd.read_csv(DATA / "transformers_peaktype.csv", parse_dates=["peak_date"])
tr = tr[tr["buurtcode"].notna()].copy()
by = pd.read_csv(DATA / "buurt_year.csv")
cbs = pd.read_csv(DATA / "cbs2024.csv")
pv = pd.read_csv(DATA / "pv2024.csv")

# ---------------- buurt features (Liander 2026 file = use during 2025, overlaps the peak window)
b26 = by[by["year"] == 2026].set_index("buurtcode")
b25 = by[by["year"] == 2025].set_index("buurtcode")
b24 = by[by["year"] == 2024].set_index("buurtcode")
bf = pd.DataFrame(index=b26.index)
bf["b_connections"] = b26["connections"]
bf["b_kwh"] = b26["total_kwh"]
bf["b_feedin_kwh"] = b26["feedin_kwh"]
bf["b_kwh_per_conn"] = b26["total_kwh"] / b26["connections"]
bf["b_feedin_per_conn"] = b26["feedin_kwh"] / b26["connections"]
bf["b_feed_share"] = b26["feedin_kwh"] / (b26["feedin_kwh"] + b26["total_kwh"])
bf["b_share_3phase"] = b26["share_3phase"]
bf["b_cons_growth"] = np.log(b26["total_kwh"] / b25["total_kwh"])
bf["b_feed_growth"] = np.log(b26["feedin_kwh"] / b25["feedin_kwh"])
bf["b_consume_only_2024"] = b24["consume_only_pct"]
bf = bf.replace([np.inf, -np.inf], np.nan)

g = tr.groupby("buurtcode")
bt = pd.DataFrame({"b_n_trafo": g.size(), "b_cap_kw": g["capacity_kw"].sum()})
bf = bf.join(bt, how="outer")
bf["b_kwh_per_kw_cap"] = bf["b_kwh"] / bf["b_cap_kw"]
bf["b_feedin_per_kw_cap"] = bf["b_feedin_kwh"] / bf["b_cap_kw"]
bf["b_conn_per_kw_cap"] = bf["b_connections"] / bf["b_cap_kw"]
bf = bf.join(cbs.set_index("buurtcode"), how="left")
pv = pv.set_index("buurtcode")
bf["pv_kw_per_kw_cap"] = pv["pv_kw"].reindex(bf.index) / bf["b_cap_kw"]
bf["pv_conn_share"] = pv["pv_conn"].reindex(bf.index) / bf["b_connections"]

X = tr.merge(bf, left_on="buurtcode", right_index=True, how="left")
X["capacity_kw_f"] = X["capacity_kw"]
X["nbl_ratio"] = X["nbl_kw"] / X["capacity_kw"]
X["cap_share_in_buurt"] = X["capacity_kw"] / X["b_cap_kw"]
X["exp_kwh_this_trafo"] = X["b_kwh_per_kw_cap"] * X["capacity_kw"]        # buurt load spread by capacity
X["exp_feedin_this_trafo"] = X["b_feedin_per_kw_cap"] * X["capacity_kw"]

feat = ["capacity_kw_f", "nbl_ratio", "cap_share_in_buurt", "exp_kwh_this_trafo", "exp_feedin_this_trafo",
        "b_n_trafo", "b_cap_kw", "b_connections", "b_kwh", "b_feedin_kwh", "b_kwh_per_conn", "b_feedin_per_conn",
        "b_feed_share", "b_share_3phase", "b_cons_growth", "b_feed_growth", "b_consume_only_2024",
        "b_kwh_per_kw_cap", "b_feedin_per_kw_cap", "b_conn_per_kw_cap", "pv_kw_per_kw_cap", "pv_conn_share"] + \
       [c for c in cbs.columns if c != "buurtcode"]
log(f"transformers: {len(X)}, features: {len(feat)}, municipalities: {X['gm'].nunique()}", fh)
log(f"above NBL: {X['above_nbl'].mean():.4f}   (never used as features: peak, util, peak date/type)", fh)

P = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)
gkf = GroupKFold(n_splits=5)
X = X.reset_index(drop=True)
y_c = X["above_nbl"].values
y_r = X["util"].clip(upper=2.0).values


def oof(feats, kind):
    o = np.zeros(len(X))
    for a, b in gkf.split(X, y_c, groups=X["gm"]):
        if kind == "clf":
            m = lgb.LGBMClassifier(objective="binary", **P).fit(X.loc[a, feats], y_c[a])
            o[b] = m.predict_proba(X.loc[b, feats])[:, 1]
        else:
            m = lgb.LGBMRegressor(objective="regression", **P).fit(X.loc[a, feats], y_r[a])
            o[b] = m.predict(X.loc[b, feats])
    return o


res = {}
for name, feats in [("capacity only", ["capacity_kw_f", "nbl_ratio"]),
                    ("consumption-side only", ["capacity_kw_f", "nbl_ratio", "exp_kwh_this_trafo", "b_kwh_per_kw_cap",
                                               "b_conn_per_kw_cap", "b_kwh_per_conn", "b_share_3phase"]),
                    ("feed-in-side only", ["capacity_kw_f", "nbl_ratio", "exp_feedin_this_trafo", "b_feedin_per_kw_cap",
                                           "b_feedin_per_conn", "b_feed_share", "pv_kw_per_kw_cap"]),
                    ("all features", feat)]:
    pc, pr = oof(feats, "clf"), oof(feats, "reg")
    res[name] = (pc, pr)
    log(f"{name:24s} transformer ROC-AUC {roc_auc_score(y_c, pc):.3f}  PR-AUC {average_precision_score(y_c, pc):.3f}"
        f" | util R2 {r2_score(y_r, pr):.3f}  MAE {mean_absolute_error(y_r, pr):.4f}"
        f"  Spearman {spearmanr(y_r, pr).correlation:.3f}", fh)

X["p_above_nbl"], X["util_pred"] = res["all features"]
X["p_capacity_only"] = res["capacity only"][0]

# ---------------- aggregate to buurt and compare with the ORIGINAL buurt model on the same buurten
bk = X.groupby("buurtcode").agg(n_trafo=("transformer_id", "size"),
                                 stressed=("above_nbl", "max"),
                                 p_buurt=("p_above_nbl", lambda p: 1 - np.prod(1 - p)),
                                 exp_n_above=("p_above_nbl", "sum"),
                                 util_meas=("util", "mean"), util_pred=("util_pred", "mean")).reset_index()
bk.to_csv(DATA / "buurt_risk_claude.csv", index=False)
log(f"\nBuurt level (from transformer model): ROC-AUC {roc_auc_score(bk['stressed'], bk['p_buurt']):.3f} "
    f"PR-AUC {average_precision_score(bk['stressed'], bk['p_buurt']):.3f}  (buurten {len(bk)}, "
    f"stressed {bk['stressed'].mean():.3f})", fh)
try:
    u = pd.read_csv(USER_PROCESSED / "buurt_stress_predictions.csv")
    m = bk.merge(u[["buurtcode", "stress_nbl", "risk_prob"]], on="buurtcode", how="inner")
    log(f"Same {len(m)} buurten as the original Model B (original label stress_nbl):", fh)
    log(f"   original buurt model  ROC-AUC {roc_auc_score(m['stress_nbl'], m['risk_prob']):.3f}  "
        f"PR-AUC {average_precision_score(m['stress_nbl'], m['risk_prob']):.3f}", fh)
    log(f"   Claude transformer->buurt ROC-AUC {roc_auc_score(m['stress_nbl'], m['p_buurt']):.3f}  "
        f"PR-AUC {average_precision_score(m['stress_nbl'], m['p_buurt']):.3f}", fh)
    log(f"   Spearman between the two risk scores: {spearmanr(m['risk_prob'], m['p_buurt']).correlation:.3f}", fh)
    # which ranks stressed buurten better once buurt SIZE is controlled? -> AUC within n_trafo bands
    m["size_band"] = pd.cut(m["n_trafo"], [0, 2, 4, 7, 12, 999])
    rows = []
    for sb, d in m.groupby("size_band", observed=True):
        if d["stress_nbl"].nunique() == 2:
            rows.append({"n_trafo": str(sb), "buurten": len(d), "stressed": d["stress_nbl"].mean(),
                         "auc_original": roc_auc_score(d["stress_nbl"], d["risk_prob"]),
                         "auc_claude": roc_auc_score(d["stress_nbl"], d["p_buurt"])})
    log("AUC inside buurt-size bands (removes the 'more transformers = more chance' effect):\n"
        + pd.DataFrame(rows).round(3).to_string(index=False), fh)
except FileNotFoundError:
    log("original predictions not found - skipped comparison", fh)

# ---------------- tiers (calibration check)
X["tier"] = pd.qcut(X["p_above_nbl"].rank(method="first"), [0, .5, .8, .95, 1.0], labels=["Low", "Medium", "High", "Very high"])
log("\nTransformer tiers (out-of-fold): actual share above NBL\n"
    + X.groupby("tier", observed=True)["above_nbl"].agg(["size", "mean"]).round(3).to_string(), fh)

# ---------------- final models + SHAP (LightGBM native TreeSHAP = pred_contrib)
for kind, y in [("clf", y_c), ("reg", y_r)]:
    mdl = (lgb.LGBMClassifier(objective="binary", **P) if kind == "clf"
           else lgb.LGBMRegressor(objective="regression", **P)).fit(X[feat], y)
    sv = mdl.booster_.predict(X[feat], pred_contrib=True)[:, :-1]
    imp = pd.DataFrame({"feature": feat, "mean_abs_shap": np.abs(sv).mean(0)})
    imp["direction"] = [spearmanr(X[c], sv[:, i], nan_policy="omit").correlation for i, c in enumerate(feat)]
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(OUT / f"c03_shap_{kind}.csv", index=False)
    log(f"\nTop 15 SHAP ({'above NBL' if kind == 'clf' else 'utilisation'}):\n" + imp.head(15).round(4).to_string(index=False), fh)
    if kind == "reg":
        X["top_driver"] = [feat[i] for i in np.argmax(sv, axis=1)]

X[["transformer_id", "buurtcode", "gm", "capacity_kw", "nbl_kw", "peak_kw", "util", "above_nbl", "atabove_nbl",
   "peak_date", "peak_type", "p_above_nbl", "util_pred", "p_capacity_only", "top_driver"]].to_csv(DATA / "transformer_oof.csv", index=False)
log("DONE c03", fh)
fh.close()
