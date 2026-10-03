# Dashboard2

Same layout and colours as the original `app.py`, showing the updated results:
2027 as the coming year, all 31,304 transformers (incl. 5 stressed ones that were missing),
the measurement-coverage warning and chart, and the solar-feed-in sensitivity.

* Start: double-click `run_dashboard.bat` (or `streamlit run Dashboard2/app.py` from the project folder).
* Self-contained: all data it needs is in `Dashboard2/data/` (copies; the originals are unchanged).
* Data sources: `Check/fixed/processed/` (fixed forecast) and `Data/processed/` (models, SHAP, ENTSO-E, cars).
