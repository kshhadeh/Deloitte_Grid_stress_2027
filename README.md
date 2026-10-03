# Powering Ahead: Predicting Local Grid Stress in the Netherlands

Deloitte capstone project (2026). A predictive model and an exploratory dashboard that show **where** local
electricity grid stress is expected to increase in the Liander service area, and **why**.

## Business Question
> Which neighbourhoods in the Liander area are most at risk of grid stress in the coming year, because their
> expected electricity demand growth meets transformers that already have little remaining capacity, and which
> local drivers (solar, electrification, housing, demographics, weather) explain this risk?

## Key results
- **Stress today:** 671 of ~31,000 neighbourhood transformers are at or above their NBL limit (Liander, 2024–25).
- **Projected stress (use during 2026):** central scenario **693 (+3.3%)**, high-demand scenario **812 (+21%)**,
  low scenario 526 (−22%). 260 transformers are within 5% of their limit → stress is very sensitive to demand growth.
- **Main driver:** demand **relative to transformer capacity** (kWh per kW of capacity, capacity per connection),
  highest in dense, older, compact buurten with many businesses.
- **Demand forecast:** one-year local growth is dominated by the national trend (energy crisis −3.4% in 2022,
  rebound +2.9% in 2025); local differences are small and hard to predict → national scenarios + small local
  adjustment with uncertainty ranges.
- **Stress model:** grid utilisation R² 0.50 and stress label ROC-AUC 0.75, validated on municipalities the
  model never saw.

## Data sources
| Source | Content | Level |
|---|---|---|
| Liander open data | Small-connection consumption 2019–2026, transformer peak load vs capacity (NBL), solar PV | Postcode / transformer / buurt |
| CBS Kerncijfers wijken en buurten | Population, housing, energy, income, solar | Buurt, 2019–2026 |
| KNMI | Daily weather (temperature, heating degree days, sunlight) | Weather station |
| ENTSO-E | National load (15-minute) | Netherlands |
| CBS / RDW | Passenger cars by fuel type | Netherlands |
| PDOK / CBS | Buurt boundaries 2023, postcode centroids | Buurt / postcode |

Raw data is not included in this repository (size and licences). See the notebooks for the exact files used.

## Project structure
```
grid_stress_project/
├── notebooks/
│   ├── 01_liander_consumption.ipynb   clean Liander consumption files (2019–2026)
│   ├── 02_postcode_to_buurt.ipynb     link postcodes to 2023 buurten (spatial join)
│   ├── 03_transformer_stress.ipynb    transformer peak load → stress per buurt
│   ├── 04_cbs_drivers.ipynb           CBS drivers panel + first exploration
│   ├── 05_knmi_weather.ipynb          yearly weather per buurt
│   ├── 06_model_table.ipynb           model tables A (demand) and B (stress)
│   ├── 07_model_a_demand.ipynb        LightGBM demand growth model
│   ├── 08_model_b_stress.ipynb        LightGBM stress model + SHAP + solar test
│   ├── 09_forecast_2027.ipynb         scenarios, forecast of use during 2026, projected stress
│   └── 10_dashboard_prep.ipynb        small files for the dashboard (map, ENTSO-E, EVs)
├── Data/
│   ├── raw/                           downloaded source files (not in the repository)
│   └── processed/                     outputs of the notebooks
├── app.py                             Streamlit dashboard
└── requirements.txt
```
Every notebook starts with a **decision log**: goal, inputs, outputs, findings, decisions and limitations.

## How to run
Requires Python 3.9 or later.
```
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
1. Put the raw data in `Data/raw/` (same folder structure as used in the notebooks).
2. Run the notebooks **in order, 01 → 10** (each one uses the outputs of the previous ones).
3. Start the dashboard:
```
streamlit run app.py
```

## Dashboard
- **Risk map:** buurten coloured by priority, stress risk, predicted utilisation or expected growth; filters and CSV download.
- **Buurt profile:** key figures, forecast with 80% range, drivers, comparison with the median buurt, transformer list.
- **Drivers:** SHAP importance for both models, demand-vs-capacity relationship, most common drivers.
- **National context:** ENTSO-E demand, peaks and seasonal pattern; EV trend; scenario table.

## Methods in short
- **Model A (demand):** LightGBM on year-to-year growth within a consistent series (Liander changed its
  definition in 2025 → no 2024→2025 pair); time-based split; compared with simple baselines.
- **Model B (stress):** LightGBM classification (≥1 transformer above NBL) and regression (utilisation);
  5-fold cross-validation **grouped by municipality**; SHAP for explanations.
- **Projection:** measured transformer peak × buurt growth factor, per scenario.

## Main limitations
- Transformer peaks are a single snapshot (Sep 2024 – Jul 2025); ~98% fall in spring/summer in all buurten,
  likely a measurement effect. Peaks mix consumption and solar feed-in; the projection is consumption-based.
- Liander's SJV is an *expected* consumption based on the previous year, not measured use.
- The stress label partly reflects buurt size (more transformers = more chances that one is over the limit).
- CBS values are filled across years where buurt codes changed; 26% of buurten miss solar data (CBS privacy).
- No local EV data (national only). SHAP shows associations, not causes.