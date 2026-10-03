# Review of "Powering Ahead: Predicting Local Grid Stress" (Claude, 30 Sep 2026)

I rebuilt the core of the project on my own, starting from the **raw files**, and then compared my results with yours.
Nothing outside this `Claude/` folder was changed. My pipeline only *reads* `Data/raw` and, for the comparison step, `Data/processed`.

**Overall verdict: the project is in good shape.** Your data work reproduces exactly, and your models hold up when rebuilt
differently. The conclusions are honest and well documented (the decision logs are a real strength). There are **four
things I would fix before the final report**. None of them needs a rebuild, and two of them are quick.

---

## 1. What I did independently

| Step | Your approach | My independent approach (`Claude/c01…c05`) |
|---|---|---|
| Liander consumption → buurt | Postcode range assigned to the buurt of its **start** postcode | Range **split equally over all postcodes** in the range |
| Feed-in (Terugleverdata KV 2025/2026) | not used | joined row-by-row to consumption (100% match) |
| CBS drivers | 2019–2026 panel, gaps back/forward-filled | **kwb 2024 only**, no filling, so no leakage |
| Stress model | 1 row per **buurt**, label "≥1 transformer above NBL" | 1 row per **transformer** (31,304), then combined per buurt: 1 − ∏(1 − p) |
| Validation | 5-fold GroupKFold by municipality | same, so the numbers can be compared |
| Peak timing | "98% summer, likely measurement artefact" | tested against KNMI daily radiation/temperature at the nearest station |
| Projection | peak × consumption growth, 1 year | same method plus a 2-year horizon (2027) and a solar feed-in sensitivity |

All models are LightGBM, run in your own `.venv` (Python 3.9.13, lightgbm 4.6.0). SHAP comes from LightGBM's native TreeSHAP (`pred_contrib`).

---

## 2. What I could confirm ✅

**Data reproduces exactly.**

| Check | Yours | Mine |
|---|---|---|
| Liander TWh 2019 … 2024 | 14.39 / 14.56 / 14.72 / 14.98 / 14.67 / 14.33 | identical |
| Liander TWh 2025 / 2026 | 11.70 / 12.07 | identical |
| PC6 linked (within + nearest) | 188,952 + 110 | identical |
| Transformers kept (0.4 kV, valid) | 31,304 | 31,304 |
| Above NBL (strict >) | 651 | 651 |
| Same buurt for each transformer | – | 100.0% agreement |

(The only difference: 2021 has 144,621 ELK rows in my reader vs 144,620 in yours. One row, not material.)

**Model A conclusion confirmed.** One-year local demand growth is hard to predict. My version uses different
allocation, no back-filled CBS data and simpler features, and it lands in the same place:

| Test | LightGBM (relative) | "follow the country" | persistence |
|---|---|---|---|
| 2023→24 | MAE 0.0379 | 0.0392 | 0.0490 |
| 2025→26 | MAE 0.0357 | 0.0358 | 0.0358 |

A buurt's relative growth barely repeats from one year to the next (Spearman 0.01–0.22). Your decision to use
national scenarios plus a small local adjustment is right.

**Model B confirmed.** My transformer-level model, combined per buurt, scores the same as your buurt model on the same 5,423 buurten:

| | ROC-AUC | PR-AUC |
|---|---|---|
| Your buurt model | **0.753** | **0.271** |
| My transformer → buurt model | 0.750 | 0.257 |

- Within buurt-size bands (1–2, 3–4, 5–7, 8–12, 13+ transformers), both models are also similar (AUC 0.61–0.72). Your size-bias caveat is handled correctly.
- The two risk scores correlate at Spearman 0.78.
- At transformer level, utilisation is only weakly predictable (R² 0.28). Buurt-level data cannot say which transformer inside a buurt will peak. So your buurt-level choice is sensible.
- **The drivers agree:** demand per kW of capacity (+), small transformers / low capacity per connection (+), businesses (+), density (+), 3-phase and gas-free homes (−), and a higher feed-in share (−).

**Projection reproduced.** With your method on my data I get 708 transformers at/above NBL (central, 2026) vs your 693, and 826 (high) vs your 812.
Of your 17 "crosses in central scenario" buurten, 16 are also central crossers in my version.

---

## 3. Things I would fix before the final report ⚠️

### 3.1 Five transformers that are stressed today are missing from the dashboard (quick fix)
`final_buurt_risk.csv` is built from the **consumption** buurten (`base`), so buurten that have transformers but no
small-connection consumption drop out. That removes **179 transformers in 104 buurten, 5 of them at/above NBL now**:
Haart-kern (Aalten), Poort van Dronten, Larserplein (Lelystad), Franeker Industrieterrein Oost and
Recreatiepark De Korenmolen (Zuidplas).
This is also why your "stressed now" count is 671 while the full dataset has 676 (≥ NBL).
**Fix:** in notebook 09, outer-join `per_buurt` into `final`, and give these buurten priority 1 with "no forecast (no small connections)".

### 3.2 "Coming year" should be 2027, not "use during 2026"
Today is end of September 2026, the peaks were measured in 2025, and your forecast is "use during 2026". That is a year that is almost over.
The business question asks about the *coming year*, so the stress projection needs **two** years of growth after the peak measurement:

| Transformers at/above NBL | low | central | high |
|---|---|---|---|
| Your method, 1 year (use during 2026) | 538 | 708 | 826 |
| **Same method, 2 years (2027)** | 437 | **750** | **1,005** |

The central case moves a little (+6%). The high scenario moves a lot (+22%), which fits your point that many transformers have almost no headroom.
**Fix:** either add the 2027 horizon (just `exp(2 × growth)`), or state clearly in the report that the projection is for 2026 and why.

### 3.3 The "summer peak" question: winter was mostly not measured
You were right to be suspicious. The data shows why (figure: `outputs/fig_peak_days_vs_sun.png`):

- **No transformer has its peak between 8 Dec 2024 and 3 Feb 2025 (58 days in a row)**, or from 13 Nov to 5 Dec. Only 0.9% of peaks fall in Nov–Feb, while those months are 39% of the window. Several very sunny days (9–13 May 2025) also have zero peaks. The data was clearly collected in blocks.
- **Inside the measured days, peaks fall on the sunniest days** (Spearman 0.56 with radiation; 46% of transformers peak on a top-10% sunny day at their nearest KNMI station, and 0% on a top-10% cold day).
- At buurt level, utilisation follows **consumption per kW** (Spearman 0.60) much more than **feed-in per kW** (0.27). With both in one rank regression, feed-in even goes slightly negative (−0.17).

**What this means for the report:** replace "likely a measurement artefact" with *"the dataset mostly covers
spring/summer 2025; winter peaks (heat pumps, EV evening charging) are largely missing"*. So the stress levels are
probably **understated** for consumption-heavy transformers. That is a key limitation and a good question for Liander/Deloitte.

### 3.4 The feed-in files are not used, but they say something important
`Raw Data/3/Terugleverdata KV 2025/2026.csv` match the consumption files row for row. Feed-in grew from **2.74 → 3.27 TWh (+19%)**,
compared with +3% for consumption. After removing the sunnier 2025 weather (KNMI radiation +11%), installed-PV growth is about **+7% per year** (median buurt).
I tested whether this explains the 2024→2025 break in the series. It does not: the ratio new/old is flat across solar quintiles (0.81–0.85), so your "two series" decision stands.

As a **sensitivity only**: if the measured peaks were feed-in driven, projected stress would be much higher
(1,267 transformers in 2026, 2,251 in 2027, central). The buurt evidence in 3.3 does *not* support that as the central case, so I kept
consumption as the main driver, just as you did. I would still add one sentence and this range to the report, because it is the biggest uncertainty in the projection.

---

## 4. Smaller points

1. **Postcode ranges.** Assigning a range to its start postcode is fine in total (correlation 0.998 with my split). But **23% of buurten change by more than 10% in kWh**, and that feeds straight into your top driver, `kwh_per_kw_capacity`. Mention it as a limitation, or switch to splitting (see `c01_build_data.py`, step 3).
2. **Leftover files.** `Data/processed/final_buurt_risk_2027.csv` is not created by any current notebook (it is from the superseded first forecast), and there is also `Data/Processed_old_reference/`. Remove or archive them before handing in, so nobody picks up the wrong file.
3. **Superseded cells** in notebooks 08 and 09 are documented, which is good. For the final version I would delete them (keep them in git history) so the notebooks run cleanly from top to bottom.
4. **Dashboard vs brief.** The brief asks for *electricity demand trends over time*. The buurt profile only shows 2025 against the 2026 forecast. `buurt_year_electricity.csv` already has 2019–2026, so a small line chart per buurt (with the 2024/25 break marked) would cover this.
5. **README numbers.** "671 of ~31,000 transformers" counts the 31,125 that have a buurt forecast; the full set is 676 (≥ NBL) or 651 (> NBL). One clarifying sentence avoids questions.
6. **Scenarios.** Using the median of the yearly medians for the central scenario (+0.5%) and the min/max for low/high is reasonable. Just say explicitly that the low scenario is an energy-crisis year.

---

## 5. Files in this folder

| File | What it is |
|---|---|
| `common.py` | paths and a robust Liander CSV reader |
| `c01_build_data.py` | raw → my tables (`data/`) |
| `c02_peak_diagnostics.py` | peak dates vs KNMI weather, feed-in vs consumption |
| `c03_stress_model.py` | transformer-level LightGBM + SHAP, comparison with your Model B |
| `c04_growth.py` | consumption growth backtest, feed-in (PV) growth per buurt |
| `c05_projection_and_compare.py` | projections (1 vs 2 years, consumption vs PV), comparison tables |
| `run_all.py` | runs c02 → c05 (`python Claude/run_all.py c01` to also rebuild c01) |
| `outputs/*_log.txt` | full printed results of each step |
| `outputs/claude_buurt_priority.csv` | my buurt priority list next to your priority and risk score |
| `outputs/claude_transformer_projection.csv` | per-transformer projections for every method and scenario |
| `outputs/comparison_summary.csv` | headline counts, yours vs mine |
| `outputs/fig_peak_days_vs_sun.png` | peak dates vs daily radiation (evidence for 3.3) |

Note: `c01_log.txt` shows "share of kWh linked = 1.0" because it was run before a small logging fix. The real share is ~99.5%, the same as yours.
