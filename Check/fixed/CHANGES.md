# Fixes applied (Claude/fixed)

These are the four fixes from `Claude/REVIEW.md` §3.
**Nothing outside the `Claude` folder was changed.** Your notebooks, `app.py` and `Data/processed` are exactly as they were.
The fixed pipeline only *reads* `Data/raw` and `Data/processed` and writes to `Claude/fixed/processed/`.

| File | What it is |
|---|---|
| `f09_forecast_fixed.py` | corrected copy of notebook 09 (VS Code shows `# %%` cells, so it runs cell by cell like a notebook) |
| `app_fixed.py` | corrected copy of `app.py`, reading the fixed outputs |
| `run_dashboard_fixed.bat` | double-click to start the fixed dashboard with your `.venv` |
| `processed/final_buurt_risk_fixed.csv` | final buurt table (5,724 buurten, was 5,620) |
| `processed/transformer_projection_fixed.csv` | all 31,304 transformers (was 31,125), 2026 + 2027 + solar sensitivity |
| `processed/scenario_table.csv` | counts per horizon and scenario |
| `processed/buurt_feedin.csv` | solar feed-in 2025/2026 per buurt + weather-corrected PV growth |
| `processed/dashboard/…` | map shapes (incl. added buurten), buurt trend 2019–2026, peak-coverage table |
| `processed/f09_log.txt` | full printed output |

**Reproduction check:** step 0 of `f09_forecast_fixed.py` re-runs your original method unchanged and gives
**671 / 693 / 812**, exactly your numbers. So every difference below comes from the fixes, not from a different model.

---

## Fix 1: stressed transformers no longer drop out
- The final table is now an **outer join** of consumption buurten and transformer buurten.
- **+104 buurten, +179 transformers.** Five of them are at/above NBL today and are now priority 1: Haart-kern (Aalten), Poort van Dronten, Larserplein (Lelystad), Recreatiepark De Korenmolen (Zuidplas) and Franeker Industrieterrein Oost.
- These buurten have no small-connection consumption (mostly business or recreation areas). They get national scenario growth without a local adjustment, the flag `no_consumption_data`, and no stress-model score.
- Transformers at/above NBL now: **676** (was 671).

## Fix 2: horizon is now the coming year, 2027
Growth is applied for 1 year (use during 2026) and for 2 years (2027, the main answer). The 80% range and the stress test widen with √2.

| Transformers at/above NBL | low | central | high | stress test |
|---|---|---|---|---|
| Today | 676 | 676 | 676 | 676 |
| Use during 2026 | 530 | 698 | 817 | 1,187 |
| **2027 (main answer)** | 431 | **733 (+8%)** | **1,003 (+48%)** | 1,701 |

Priorities are now based on 2027:

| Priority | Buurten (was) |
|---|---|
| 1 Stressed now | 582 (577) |
| 2 Crosses by 2027 (central) | 42 (17, for 2026) |
| 3 Crosses by 2027 (high) | 201 (92) |
| 4 Crosses in stress test only | 516 (286) |
| 5 No crossing expected | 4,182 (4,447) |
| No transformer data | 201 (201) |

## Fix 3: measurement coverage is shown and explained
- There is a warning banner at the top of the dashboard.
- The "National context" tab has a new chart, **Measurement coverage**: peaks per day next to KNMI daily radiation. Only 0.9% of peaks fall in Nov–Feb, those months are 39% of the window, and peaks were recorded on just 105 of 311 days.
- The cautions text in the Drivers tab is updated.
- The transformer list in the buurt profile now shows `peak_date`.

## Fix 4: solar feed-in data used
- `Terugleverdata KV 2025/2026` are linked to buurten with your own notebook 02 rule (start postcode, fallback end postcode). 100% of feed-in is linked.
- Feed-in went from 2.74 to 3.27 TWh (+19.3%). Radiation 2025/2024 was ×1.114, so weather-corrected **PV growth is +7.2% per year** (buurt median +7.1%).
- In the dashboard:
  - two new map layers: "Solar feed-in growth" and "Solar share of electricity"
  - a solar line in the buurt profile
  - a sidebar filter for the **solar sensitivity flag** (643 buurten would cross by 2027 only if the peaks were solar-driven)
  - a "Sensitivity: peaks solar-driven" row in the scenario table (1,279 in 2026, 2,294 in 2027)
- This is labelled as an upper-bound sensitivity, not a forecast. Buurt utilisation follows consumption per kW (Spearman 0.60) far more than feed-in per kW (0.27), so consumption stays the central case.

## Also added (from the review's smaller points)
- **Buurt profile:** now shows the demand history 2019–2026 (both Liander series, with the definition break marked), plus the 2026 and 2027 forecast and the 80% range. This covers the brief's "electricity demand trends over time".

---

## Text you can paste into README / final report

**Key results (replace the "Projected stress" bullet):**
> **Projected stress for the coming year (2027):** central scenario **733 transformers at/above their NBL limit (+8% vs 676 today)**, high-demand scenario **1,003 (+48%)**, low scenario 431. 42 buurten cross in the central scenario and 201 more in the high scenario. The projection is made two years ahead because the peaks were measured in 2025.

**Limitations (replace the "Transformer peaks are a single snapshot…" bullet):**
> Transformer peaks come from one measurement campaign (Sep 2024 – Jul 2025) that mostly covers spring and summer 2025: no transformer has its peak between 8 Dec 2024 and 3 Feb 2025, and only 0.9% of peaks fall in November–February. Winter evening peaks (heat pumps, EV charging) are therefore largely missing, and stress is probably understated for consumption-heavy transformers. Within the measured days, peaks fall on the sunniest days. Still, buurt utilisation follows consumption per kW of capacity (Spearman 0.60) much more than solar feed-in per kW (0.27). So the projection uses consumption growth, and solar feed-in growth (+7%/yr after weather correction, from Liander's feed-in files) is shown as an upper-bound sensitivity (2,294 transformers in 2027).

**Data sources (add a row):**
> | Liander open data | Small-connection feed-in (teruglevering) 2025–2026 | Postcode range → buurt |

---

## How to run
1. In VS Code open `Claude/fixed/f09_forecast_fixed.py` and choose Run (about 2 minutes).
2. Double-click `Claude/fixed/run_dashboard_fixed.bat`. From a terminal in the project folder you can instead run
   `streamlit run Claude/fixed/app_fixed.py`.

If you want these fixes in the real project later, copy the logic from `f09_forecast_fixed.py` into notebook 09 and the changes from `app_fixed.py` into `app.py`. The originals were left untouched on purpose.
