import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Grid stress – Liander area", layout="wide")

DATA = Path("Data/processed")
DASH = DATA / "dashboard"


@st.cache_data
def load_data():
    final = pd.read_csv(DATA / "final_buurt_risk.csv")
    tr = pd.read_csv(DATA / "transformer_projection_2026.csv")
    with open(DASH / "buurten_liander.geojson", encoding="utf-8") as f:
        geo = json.load(f)
    b = pd.read_csv(DATA / "model_b_table.csv")
    imp_u = pd.read_csv(DATA / "shap_importance_util.csv")
    imp_s = pd.read_csv(DATA / "shap_importance_stress.csv")
    return final, tr, geo, b, imp_u, imp_s

@st.cache_data
def load_national():
    ey = pd.read_csv(DASH / "entsoe_yearly.csv")
    em = pd.read_csv(DASH / "entsoe_monthly.csv")
    ep = pd.read_csv(DASH / "entsoe_profile_month_hour.csv")
    cars = pd.read_csv(DASH / "cars_fuel_national.csv")
    return ey, em, ep, cars
final, tr, geo, b, imp_u, imp_s = load_data()

st.title("Powering Ahead: local grid stress in the Liander area")
st.caption("Deloitte capstone prototype · data: Liander, CBS, KNMI, ENTSO-E · "
           "forecast = use during 2026 (SJV 1 Jan 2027)")

# ---------- key numbers ----------
now = int(tr["above_nbl_now"].sum())
central = int(tr["above_nbl_growth_central"].sum())
high = int(tr["above_nbl_growth_high"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transformers at/above NBL limit now", f"{now:,}")
c2.metric("Projected 2026 (central)", f"{central:,}", f"{100 * (central / now - 1):+.1f}%",
          delta_color="inverse")
c3.metric("Projected 2026 (high demand)", f"{high:,}", f"{100 * (high / now - 1):+.1f}%",
          delta_color="inverse")
c4.metric("Buurten stressed now or crossing (central)",
          f"{final['priority'].str[0].isin(['1', '2']).sum():,}")

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
gemeenten = sorted(final["gemeentenaam"].dropna().unique())
sel_gm = st.sidebar.multiselect("Municipality (empty = all)", gemeenten)
prios = sorted(final["priority"].unique())
sel_prio = st.sidebar.multiselect("Priority level", prios, default=prios)

view = final.copy()
if sel_gm:
    view = view[view["gemeentenaam"].isin(sel_gm)]
view = view[view["priority"].isin(sel_prio)]
st.sidebar.write(f"**{len(view):,}** buurten selected")

tab_map, tab_buurt, tab_drivers, tab_national = st.tabs(
    ["Risk map", "Buurt profile", "Drivers", "National context"])

# ---------- tab 1: risk map ----------
with tab_map:
    measure = st.radio(
        "Colour the map by",
        ["Priority", "Stress risk (model)", "Predicted utilisation (model)", "Expected growth 2026 (%)"],
        horizontal=True,
    )
    color_col = {"Priority": "priority",
                 "Stress risk (model)": "risk_prob",
                 "Predicted utilisation (model)": "util_pred",
                 "Expected growth 2026 (%)": "central_pct"}[measure]

    prio_colors = {
        "1 Stressed now": "#b2182b",
        "2 Crosses in central scenario": "#ef8a62",
        "3 Crosses in high scenario": "#fddbc7",
        "4 Crosses in stress test only": "#d1e5f0",
        "5 No crossing expected": "#e0e0e0",
        "No transformer data": "#ffffff",
    }
    map_args = dict(
        geojson=geo, locations="buurtcode", featureidkey="properties.buurtcode",
        color=color_col,
        hover_name="buurtnaam",
        hover_data={"buurtcode": False, "gemeentenaam": True, "priority": True,
                    "risk_prob": ":.2f", "util_pred": ":.2f", "central_pct": ":.2f",
                    "driver_1": True, "driver_2": True, "driver_3": True},
        center={"lat": 52.45, "lon": 5.6}, zoom=7.8, opacity=0.75, height=650,    )
    if color_col == "priority":
        map_args.update(color_discrete_map=prio_colors,
                        category_orders={"priority": list(prio_colors)})
    else:
        map_args.update(color_continuous_scale="Reds")

    if hasattr(px, "choropleth_map"):          # newer Plotly
        fig = px.choropleth_map(view, map_style="carto-positron", **map_args)
    else:                                       # older Plotly
        fig = px.choropleth_mapbox(view, mapbox_style="carto-positron", **map_args)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.update_traces(marker_line_width=0.3, marker_line_color="white")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Selected buurten")
    table_cols = ["buurtnaam", "gemeentenaam", "priority", "n_above_nbl_now", "n_new_central",
                  "min_headroom_pct", "risk_prob", "risk_tier", "central_pct",
                  "p10_pct", "p90_pct", "driver_1", "driver_2", "driver_3"]
    table = view.sort_values(["priority", "risk_prob"], ascending=[True, False])[table_cols]
    st.dataframe(table.round(2), use_container_width=True, hide_index=True)
    st.download_button("Download selection (CSV)", table.to_csv(index=False),
                        file_name="grid_stress_selection.csv")

with tab_buurt:
    opts = view.sort_values(["priority", "risk_prob"], ascending=[True, False]).reset_index(drop=True)
    if opts.empty:
        st.warning("No buurten match the filters.")
    else:
        names_list = (opts["buurtnaam"].fillna(opts["buurtcode"]) + " (" +
                      opts["gemeentenaam"].fillna("?") + ")").tolist()
        choice = st.selectbox("Choose a buurt (follows the filters, highest priority first)",
                              range(len(opts)), format_func=lambda i: names_list[i])
        r = opts.iloc[choice]

        st.subheader(f"{r['buurtnaam']} – {r['gemeentenaam']}")
        if r["low_reliability"]:
            st.warning("Fewer than 50 connections: the forecast for this buurt has low reliability.")
        if r["priority"] == "1 Stressed now" and r.get("stress_nbl") == 0:
            st.info("A transformer here is exactly AT its NBL limit. The priority levels count 'at or above' "
                    "the limit as stressed; the stress model was trained on 'above' only.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Priority", r["priority"])
        m2.metric("Stress risk (model)",
                  f"{r['risk_prob']:.0%} ({r['risk_tier']})" if pd.notna(r["risk_prob"]) else "n/a")
        m3.metric("Grid utilisation (measured)",
                  f"{r['util_weighted']:.0%}" if pd.notna(r["util_weighted"]) else "n/a")
        m4.metric("Transformers at/above NBL",
                  f"{int(r['n_above_nbl_now'])} of {int(r['n_transformers_proj'])}"
                  if pd.notna(r["n_transformers_proj"]) else "n/a")

        left, right = st.columns(2)

        # forecast with 80% range
        with left:
            st.markdown("**Electricity use (MWh)**")
            use25 = r["kwh_use2025"] / 1000
            cen = r["kwh_use2026_central"] / 1000
            lo = use25 * (1 + r["p10_pct"] / 100)
            hi = use25 * (1 + r["p90_pct"] / 100)
            fig_f = go.Figure()
            fig_f.add_bar(x=["Use during 2025"], y=[use25], marker_color="#9e9e9e", name="2025")
            fig_f.add_bar(x=["Use during 2026 (forecast)"], y=[cen], marker_color="#ef8a62", name="2026",
                          error_y=dict(type="data", symmetric=False, array=[hi - cen], arrayminus=[cen - lo]))
            fig_f.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_f, use_container_width=True)
            st.caption(f"Central {r['central_pct']:+.1f}% · **80% range (normal year)** "
                       f"{r['p10_pct']:+.1f}% to {r['p90_pct']:+.1f}% · "
                       f"scenario range (weak/strong national year) {r['scen_low_pct']:+.1f}% to "
                       f"{r['scen_high_pct']:+.1f}% · stress test {r['stresstest_low_pct']:+.1f}% to "
                       f"{r['stresstest_high_pct']:+.1f}%")
        # drivers and comparison with the median buurt
        with right:
            st.markdown("**Main drivers of grid utilisation here**")
            for k in ["driver_1", "driver_2", "driver_3"]:
                if pd.notna(r.get(k)):
                    st.write(f"• {r[k]}")
            comp = {"kwh_per_kw_capacity": "Demand per kW of capacity (kWh/kW)",
                    "kw_capacity_per_conn": "Capacity per connection (kW)",
                    "conn_per_trafo": "Connections per transformer",
                    "bev_dich": "Population density (per km²)",
                    "p_won_zs": "Homes with solar panels (%)",
                    "share_3phase_pct": "3-phase connections (%)",
                    "a_bedv": "Number of businesses"}
            rb = b[b["buurtcode"] == r["buurtcode"]]
            if len(rb):
                st.dataframe(pd.DataFrame({
                    "Measure": list(comp.values()),
                    "This buurt": [rb.iloc[0][c] for c in comp],
                    "Median of all buurten": [b[c].median() for c in comp],
                }).round(1), hide_index=True, use_container_width=True)
            st.caption("Drivers are associations found by the model (SHAP), not proven causes.")

        # transformers in this buurt
        st.markdown("**Transformers in this buurt**")
        t_b = tr[tr["buurtcode"] == r["buurtcode"]]
        if t_b.empty:
            st.write("No transformer data for this buurt.")
        else:
            st.dataframe(t_b[["transformer_id", "peak_kw", "capacity_kw", "nbl_limit_kw", "utilisation",
                              "headroom_pct", "above_nbl_now", "above_nbl_growth_central"]]
                         .rename(columns={"above_nbl_now": "at/above NBL now",
                                          "above_nbl_growth_central": "at/above NBL 2026 (central)"})
                         .sort_values("headroom_pct").round(2),
                         hide_index=True, use_container_width=True)
with tab_drivers:
    nice = {
        "kwh_per_kw_capacity": "Demand per kW of capacity", "kw_capacity_per_conn": "Capacity per connection",
        "conn_per_trafo": "Connections per transformer", "total_kwh": "Total electricity demand",
        "a_bedv": "Number of businesses", "n_transformers": "Number of transformers",
        "g_ele": "Electricity use per home", "p_won_z": "Share of gas-free homes",
        "solar_homes_per_trafo": "Solar homes per transformer", "bev_dich": "Population density",
        "g_ink_pi": "Income per inhabitant", "share_3phase_pct": "Share of 3-phase connections",
        "kwh_per_conn": "Demand per connection", "ste_oad": "Address density",
        "p_bjj2k": "Homes built before 2000", "feedin_kwh_per_kw_cap": "Solar feed-in per kW of capacity",
        "a_opp_ha": "Buurt area", "a_lan_ha": "Buurt land area", "p_bj_me10": "Homes older than 10 years",
        "temp_mean_c": "Average temperature", "capacity_kw_total": "Total transformer capacity",
        "connections": "Number of connections", "a_inw": "Number of inhabitants",
        "consume_only_pct": "Connections without feed-in", "p_won_zs": "Homes with solar panels",
        "g_ele_tr": "Solar feed-in per home", "woz": "Home value",
    }
    colors = {"higher value → more stress": "#b2182b", "higher value → less stress": "#2166ac"}

    st.markdown("### What drives grid stress?")
    col1, col2 = st.columns(2)
    for col, imp, title in [(col1, imp_u, "Grid utilisation (size-independent) · R² 0.50"),
                            (col2, imp_s, "Stress label: ≥1 transformer above NBL · ROC-AUC 0.75")]:
        top = imp.head(12).copy()
        top["name"] = top["feature"].map(nice).fillna(top["feature"])
        top["effect"] = np.where(top["direction"] > 0, "higher value → more stress", "higher value → less stress")
        fig_i = px.bar(top.sort_values("mean_abs_shap"), x="mean_abs_shap", y="name", color="effect",
                       orientation="h", color_discrete_map=colors,
                       labels={"mean_abs_shap": "Average impact on the prediction (mean |SHAP|)", "name": ""})
        fig_i.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), title=title,
                            legend=dict(orientation="h", y=-0.2, title=""),
                            yaxis={"categoryorder": "total ascending"})
        col.plotly_chart(fig_i, use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Demand relative to capacity is the key relationship**")
        sc = b[["kwh_per_kw_capacity", "util_weighted", "stress_nbl"]].dropna().copy()
        sc["Stress"] = np.where(sc["stress_nbl"] == 1, "≥1 transformer above NBL", "no transformer above NBL")
        fig_s = px.scatter(sc, x="kwh_per_kw_capacity", y="util_weighted", color="Stress", opacity=0.45,
                           log_x=True, color_discrete_map={"≥1 transformer above NBL": "#b2182b",
                                                           "no transformer above NBL": "#bdbdbd"},
                           labels={"kwh_per_kw_capacity": "Yearly demand per kW of transformer capacity (kWh/kW, log)",
                                   "util_weighted": "Grid utilisation (peak / capacity)"})
        fig_s.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                            legend=dict(orientation="h", y=-0.25, title=""))
        st.plotly_chart(fig_s, use_container_width=True)

    with col4:
        st.markdown("**Most common #1 driver per buurt**")
        d1 = final["driver_1"].dropna().value_counts().head(10).reset_index()
        d1.columns = ["driver", "buurten"]
        fig_d = px.bar(d1.sort_values("buurten"), x="buurten", y="driver", orientation="h",
                       color_discrete_sequence=["#ef8a62"], labels={"driver": "", "buurten": "Number of buurten"})
        fig_d.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_d, use_container_width=True)

    st.divider()
    st.markdown("""
**Key findings**
- Utilisation rises most with **demand per kW of transformer capacity** and falls with **capacity per connection**:
  what matters is demand *relative to* the grid, not demand alone.
- Stress is highest in **dense, older, compact buurten with many businesses**; a single overloaded transformer can
  make a large, otherwise well-equipped buurt "stressed" (see Markt, Arnhem).
- Gas-free homes, 3-phase connections and solar homes per transformer go with **less** stress, likely because
  these are newer areas with a modern, well-sized grid.

**Cautions**
- SHAP shows **associations, not causes**. Correlated features share credit, so read groups of drivers, not single signs.
- The stress label partly reflects **buurt size** (more transformers = more chance that one is over the limit).
- Two definitions: the stress model uses peak **above** the NBL limit (555 buurten); the priority levels use
  **at or above** (577 buurten). The 22 extra buurten have a transformer exactly at its limit.
- Measured peaks mix consumption and solar feed-in; ~98% fall in spring/summer in all buurten, which is likely a
  measurement effect (to be verified with Liander).
""")
with tab_national:
    ey, em, ep, cars = load_national()
    st.markdown("### National context: demand, peaks and electrification")

    n1, n2 = st.columns(2)

    # 1. yearly energy and peak
    with n1:
        st.markdown("**National electricity demand and peak (ENTSO-E)**")
        fig_y = make_subplots(specs=[[{"secondary_y": True}]])
        full_years = ey[ey["complete_year"]]
        fig_y.add_bar(x=full_years["year"], y=full_years["energy_twh"], name="Yearly demand (TWh)",
                      marker_color="#bdbdbd", secondary_y=False)
        fig_y.add_scatter(x=ey["year"], y=ey["peak_mw"] / 1000, name="Highest peak (GW)",
                          mode="lines+markers", line=dict(color="#b2182b"), secondary_y=True)
        fig_y.update_yaxes(title_text="TWh", secondary_y=False, rangemode="tozero")
        fig_y.update_yaxes(title_text="GW", secondary_y=True)
        fig_y.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                            legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_y, use_container_width=True)
        st.caption("Demand dropped in 2022 (energy crisis) and recovered by 2025. The peak keeps rising "
                   "(record 20.7 GW on 3 Feb 2026; 2026 is an incomplete year) → peaks grow faster than demand.")

    # 2. seasonal pattern: average load per month and hour
    with n2:
        st.markdown("**Seasonal pattern: average national load (MW), 2019–2025**")
        heat = ep.pivot(index="month", columns="hour", values="load_mw")
        heat.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig_h = px.imshow(heat, aspect="auto", color_continuous_scale="Reds",
                          labels={"x": "Hour of the day", "y": "", "color": "MW"})
        fig_h.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption("National peaks occur on winter evenings (17:00–18:00). Local Liander transformer peaks "
                   "were measured mostly in spring/summer: local stress behaves differently.")

    n3, n4 = st.columns(2)

    # 3. EV trend
    with n3:
        st.markdown("**Electric cars in the Netherlands (CBS/RDW, 1 January)**")
        ev = cars.copy()
        ev["Fully electric (%)"] = 100 * ev["Volledig elektrisch (FEV)"] / ev["Totaal"]
        ev["Plug-in hybrid (%)"] = 100 * ev["Plug-in hybirde (PHEV)"] / ev["Totaal"]
        ev_long = ev.melt(id_vars="year", value_vars=["Fully electric (%)", "Plug-in hybrid (%)"],
                          var_name="type", value_name="share")
        fig_e = px.line(ev_long, x="year", y="share", color="type", markers=True,
                        color_discrete_sequence=["#b2182b", "#ef8a62"],
                        labels={"share": "% of all passenger cars", "year": "", "type": ""})
        fig_e.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_e, use_container_width=True)
        st.caption("National only: no local EV data at buurt level in this dataset.")

    # 4. scenarios for the Liander area
    with n4:
        st.markdown("**Scenarios for the Liander area (use during 2026)**")
        base_twh = final["kwh_use2025"].sum() / 1e9
        rows = []
        for label, col, trc in [("Low (like 2022, −3.4%)", "kwh_use2026_low", "above_nbl_growth_low"),
                                ("Central (+0.5%)", "kwh_use2026_central", "above_nbl_growth_central"),
                                ("High (like 2025, +2.9%)", "kwh_use2026_high", "above_nbl_growth_high")]:
            twh = final[col].sum() / 1e9
            n_tr = int(tr[trc].sum())
            rows.append({"Scenario": label, "Demand (TWh)": round(twh, 2),
                         "vs 2025": f"{100 * (twh / base_twh - 1):+.1f}%",
                         "Transformers at/above NBL": n_tr,
                         "vs today": f"{100 * (n_tr / now - 1):+.1f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Use during 2025: {base_twh:.2f} TWh · transformers at/above NBL today: {now:,}. "
                   "Stress projection is consumption-based (peaks also include solar feed-in).")
        st.markdown("**How close are transformers to their NBL limit?**")
        hb = tr.copy()
        hb["band"] = pd.cut(hb["headroom_pct"], bins=[-np.inf, 0, 1, 5, 10, np.inf],
                            labels=["At/above limit now", "0–1% below", "1–5% below", "5–10% below", "> 10% below"])
        newly = lambda s: (~hb["above_nbl_now"] & hb[s])
        band_tab = hb.groupby("band", observed=False).agg(transformers=("transformer_id", "size")).reset_index()
        for s, name in [("above_nbl_growth_central", "cross: central"),
                        ("above_nbl_growth_high", "cross: high"),
                        ("above_nbl_range_high", "cross: stress test")]:
            band_tab[name] = hb[newly(s)].groupby("band", observed=False).size().values
        st.dataframe(band_tab.rename(columns={"band": "Headroom band"}), hide_index=True, use_container_width=True)
        st.caption("Peaks and limits are whole kW: < 1% headroom is 1–4 kW on a typical 384 kW limit, within "
                   "measurement rounding. Read the crossings as 'transformers with almost no buffer', "
                   "not as a precise forecast.")