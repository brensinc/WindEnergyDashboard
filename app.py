import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from windlib import (
    load_projects_from_disk,
    save_projects_to_disk,
    DEFAULT_PROJECT,
    get_tz_from_latlon,
    get_wind_data,
    add_data_type,
    add_power_output,
    add_prices_and_revenue,
    ISO_CHOICES,
    LMP_MARKETS,
    PRICE_MODELS,  # if you have it; otherwise define locally
    monthly_capacity_factor, 
    seasonal_capacity_factor,
    exceedance_curve, 
    downtime_risk_by_month,
    annual_energy_bootstrap, 
    interannual_variability,
    save_projects_to_disk, 
    ISO_CHOICES, 
    LMP_MARKETS,
    tz_mismatch_message
)

st.set_page_config(page_title="Wind Energy Dashboard", layout="wide")

# -----------------------------
# Session state init
# -----------------------------
if "projects" not in st.session_state:
    projects = load_projects_from_disk()
    if "demo" not in projects:
        projects["demo"] = DEFAULT_PROJECT
        save_projects_to_disk(projects)
    st.session_state.projects = projects

if "selected_project_id" not in st.session_state:
    st.session_state.selected_project_id = "demo"

if "df" not in st.session_state:
    st.session_state.df = None

if "last_query_key" not in st.session_state:
    st.session_state.last_query_key = None

if "query_params" not in st.session_state:
    st.session_state.query_params = {}

# -----------------------------
# Sidebar: project selection
# -----------------------------
st.sidebar.header("Projects")

project_ids = list(st.session_state.projects.keys())
prev_project_id = st.session_state.selected_project_id

selected_project_id = st.sidebar.selectbox(
    "Select a project",
    options=project_ids,
    index=project_ids.index(st.session_state.selected_project_id) if st.session_state.selected_project_id in project_ids else 0,
)
st.session_state.selected_project_id = selected_project_id
p = st.session_state.projects[selected_project_id]

# If project changed, invalidate loaded df so tabs reflect the new project
if selected_project_id != prev_project_id:
    st.session_state.df = None
    st.session_state.last_query_key = None
    # optional: immediately rerun so UI reflects reset state
    st.rerun()

# -----------------------------
# Sidebar: create/delete projects
# -----------------------------
with st.sidebar.expander("Create a new project", expanded=False):
    # If you don't have PRICE_MODELS exported from windlib, use:
    # PRICE_MODELS = ["Fixed price ($/MWh)", "Market price (gridstatus)"]
    pricing_model_ui = st.selectbox("Pricing model", PRICE_MODELS, index=0)

    with st.form("create_project_form", clear_on_submit=False):
        name = st.text_input("Project name", value="New Wind Project")
        lat = st.number_input("Latitude", value=float(p.get("latitude", 37.75)), format="%.6f")
        lon = st.number_input("Longitude", value=float(p.get("longitude", -122.45)), format="%.6f")

        fixed_price = None
        iso_name = None
        lmp_market = None
        location_type = None

        if pricing_model_ui.startswith("Fixed"):
            fixed_price = st.number_input("Fixed price ($/MWh)", value=50.0, step=1.0)
        else:
            iso_name = st.selectbox("ISO / Market", ISO_CHOICES, index=0)
            lmp_market = st.selectbox("LMP market", LMP_MARKETS, index=0)
            location_type = st.selectbox("Location type", ["Hub"], index=0)

        st.info("Wind reference height is fixed at 100m.")
        cut_in = st.number_input("Cut-in speed (m/s)", value=3.0, step=0.1)
        rated_speed = st.number_input("Rated speed (m/s)", value=12.0, step=0.1)
        cut_out = st.number_input("Cut-out speed (m/s)", value=25.0, step=0.1)
        rated_power = st.number_input("Rated power (MW)", value=3.0, step=0.1)

        new_id = st.text_input("Project ID (unique)", value=f"proj_{len(project_ids)+1}")
        persist = st.checkbox("Save to projects.json (recommended)", value=True)

        submitted = st.form_submit_button("Create project")

    if submitted:
        if new_id in st.session_state.projects:
            st.error("That Project ID already exists. Choose another.")
        else:
            st.session_state.projects[new_id] = {
                "project_id": new_id,
                "name": name,
                "latitude": float(lat),
                "longitude": float(lon),
                "hub_height_m": 100,
                "cut_in_mps": float(cut_in),
                "rated_speed_mps": float(rated_speed),
                "cut_out_mps": float(cut_out),
                "rated_power_mw": float(rated_power),
                "pricing_model": "fixed" if pricing_model_ui.startswith("Fixed") else "market",
                "fixed_price_usd_mwh": float(fixed_price) if fixed_price is not None else None,
                "iso_name": iso_name,
                "lmp_market": lmp_market,
                "location_type": location_type,
            }
            if persist:
                save_projects_to_disk(st.session_state.projects)

            # select newly created project and invalidate df
            st.session_state.selected_project_id = new_id
            st.session_state.df = None
            st.session_state.last_query_key = None
            st.success(f"Created project: {new_id}")
            st.rerun()

with st.sidebar.expander("Delete a project", expanded=False):
    deletable_ids = [k for k in st.session_state.projects.keys() if k != "demo"]
    if not deletable_ids:
        st.caption("No deletable projects yet.")
    else:
        delete_id = st.selectbox("Choose project to delete", options=deletable_ids)
        if st.button("Delete this project", type="secondary"):
            st.session_state.projects.pop(delete_id, None)
            save_projects_to_disk(st.session_state.projects)

            # If you deleted the selected project, revert to demo
            if st.session_state.selected_project_id == delete_id:
                st.session_state.selected_project_id = "demo"

            st.session_state.df = None
            st.session_state.last_query_key = None
            st.cache_data.clear()
            st.rerun()

st.sidebar.divider()
if st.sidebar.button("Clear cached data"):
    st.cache_data.clear()
    st.session_state.df = None
    st.session_state.last_query_key = None
    st.sidebar.success("Cache cleared.")
    st.rerun()

# -----------------------------
# Sidebar: query controls
# -----------------------------
st.sidebar.divider()
st.sidebar.subheader("Query")

date_start = st.sidebar.date_input("Start date", value=pd.to_datetime("2024-01-01").date())
date_end   = st.sidebar.date_input("End date",   value=pd.to_datetime("2024-01-07").date())

include_prices = st.sidebar.checkbox("Pull price data", value=True)

# Pricing model controls (only meaningful if include_prices)
pricing_model = st.sidebar.selectbox(
    "Pricing model",
    options=["fixed", "market"],
    index=0 if p.get("pricing_model", "fixed") == "fixed" else 1,
    disabled=not include_prices
)

fixed_price = None
iso_name = None
lmp_market = None
location_type = None

if include_prices:
    if pricing_model == "fixed":
        fixed_price = st.sidebar.number_input(
            "Fixed price ($/MWh)",
            value=float(p.get("fixed_price_usd_mwh", 50.0) or 50.0),
            step=1.0
        )
    else:
        iso_name = st.sidebar.selectbox("ISO", ISO_CHOICES, index=ISO_CHOICES.index(p.get("iso_name", "CAISO")))
        lmp_market = st.sidebar.selectbox("LMP market", LMP_MARKETS, index=LMP_MARKETS.index(p.get("lmp_market", "DAY_AHEAD_HOURLY")))
        location_type = st.sidebar.selectbox("Location type", ["Hub"], index=0)

# Commit pricing edits back to project in-memory (and optionally persist later)
p["pricing_model"] = pricing_model
if fixed_price is not None:
    p["fixed_price_usd_mwh"] = float(fixed_price)
if iso_name is not None:
    p["iso_name"] = iso_name
if lmp_market is not None:
    p["lmp_market"] = lmp_market
if location_type is not None:
    p["location_type"] = location_type

do_query = st.sidebar.button("Requery", type="primary")

# Query key: if anything important changes, df is considered stale
query_key = (
    st.session_state.selected_project_id,
    str(date_start), str(date_end),
    include_prices,
    p.get("pricing_model"),
    p.get("fixed_price_usd_mwh"),
    p.get("iso_name"),
    p.get("lmp_market"),
    p.get("location_type"),
    p.get("cut_in_mps"), p.get("rated_speed_mps"), p.get("cut_out_mps"), p.get("rated_power_mw"),
)

# Fetch only when user clicks Requery OR df missing OR query changed + user clicked Requery
if do_query or (st.session_state.df is None):
    local_tz = get_tz_from_latlon(p["latitude"], p["longitude"])

    with st.spinner("Fetching wind (and optional price) data..."):
        df = get_wind_data(p["latitude"], p["longitude"], str(date_start), str(date_end))
        df = add_data_type(df, local_tz)
        df = add_power_output(df, p)
        df = add_prices_and_revenue(df, p, local_tz, include_prices=include_prices)

    st.session_state.df = df
    st.session_state.last_query_key = query_key
    st.session_state.query_params = {
        "date_start": date_start,
        "date_end": date_end,
        "include_prices": include_prices,
        "local_tz": str(local_tz.key),
    }

# If query changed but user hasn't clicked requery yet, show a gentle indicator
query_is_stale = (st.session_state.last_query_key is not None) and (st.session_state.last_query_key != query_key)

# -----------------------------
# Main page
# -----------------------------
st.title("Wind Energy Dashboard")
st.caption("Use the sidebar controls to select a project and click **Requery** to refresh the data.")

if query_is_stale:
    st.warning("Sidebar settings changed. Click **Requery** to refresh results.")

if st.session_state.df is None or len(st.session_state.df) == 0:
    st.info("No data loaded yet. Select a range and click **Requery**.")
    st.stop()

df = st.session_state.df
local_tz = get_tz_from_latlon(p["latitude"], p["longitude"])
st.success(f"Loaded {len(df):,} rows | {df['timestamp'].min()} → {df['timestamp'].max()} ({local_tz.key})")


tz_info = tz_mismatch_message(
    project_tz=local_tz,
    iso_name=p.get("iso_name"),
)

if tz_info and st.session_state.query_params.get("include_prices", False):
    st.info(
        f"""
**Timezone notice**

Your wind site is in **{tz_info['project_tz']}**,  
but market prices for **{p['iso_name']}** are published in **{tz_info['iso_tz']}**.

As a result:
- Market prices may be misaligned from generation data

All prices are shown in the **project's local timezone** for consistency.
""",
        icon="⏰",
    )

# -----------------------------
# Project Metadata
# -----------------------------

# Layout
colA, colB = st.columns([1, 2]) # Modulates width of each column

with colA:
    st.subheader("Project")
    st.write(f"**{p['name']}** (`{p['project_id']}`)")
    st.write(f"Lat/Lon: **{p['latitude']:.4f}, {p['longitude']:.4f}**")
    st.write(f"ISO Name: **{iso_name}**")
    st.write(f"lmp_market: **{lmp_market}**")

    st.write(f"Hub height used: **{int(df['hub_height_m'].iloc[0])} m**")
    st.write(f"Rated power: **{p['rated_power_mw']} MW**")
    st.write(f"Cut-in / Rated / Cut-out: **{p['cut_in_mps']} / {p['rated_speed_mps']} / {p['cut_out_mps']} m/s**")

    # map_df = pd.DataFrame({"lat": [p["latitude"]], "lon": [p["longitude"]]})
    # st.map(map_df, zoom=6)
    dot_color = [255, 120, 0]

# -----------------------------
# Map
# -----------------------------

with colB:
    map_df = pd.DataFrame({
        "lat": [p["latitude"]],
        "lon": [p["longitude"]],
        "label": [f"{p['name']}\n({p['latitude']:.4f}, {p['longitude']:.4f})"],
    })

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius=2000,   # adjust for zoom/scale
        pickable=True, 
        get_fill_color=dot_color,

    )

    tooltip = {
        "text": "{label}"
    }

    view_state = pdk.ViewState(
        latitude=p["latitude"],
        longitude=p["longitude"],
        zoom=7,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='light'
    ))


# -----------------------------
# Tabs (Option 1)
# -----------------------------
tab_overview, tab_ts, tab_wind, tab_rev, tab_settings = st.tabs(
    ["Overview", "Time Series", "Wind Analytics", "Revenue Analytics", "Settings"]
)

# ---------- Overview ----------
with tab_overview:
    # st.subheader("Project Summary")
    # c1, c2, c3, c4 = st.columns(4)
    # c1.metric("Project", p.get("name", "—"))
    # c2.metric("Lat/Lon", f"{p['latitude']:.3f}, {p['longitude']:.3f}")
    # c3.metric("Rated Power (MW)", f"{p.get('rated_power_mw', np.nan)}")
    # c4.metric("Pricing", "Included" if st.session_state.query_params.get("include_prices", True) else "Not pulled")

    # st.divider()

    # A couple of quick KPIs (feel free to expand)
    st.subheader("Quick KPIs (loaded range)")
    if "energy_mwh" in df.columns:
        total_energy = float(np.nansum(df["energy_mwh"]))
        st.metric("Total Energy (MWh)", f"{total_energy:,.1f}")

    if "price_usd_mwh" in df.columns and df["price_usd_mwh"].notna().any():
        st.metric("Avg Price ($/MWh)", f"{df['price_usd_mwh'].mean():.2f}")

    if "revenue_usd" in df.columns and df["revenue_usd"].notna().any():
        st.metric("Total Revenue ($)", f"{df['revenue_usd'].sum():,.0f}")

    st.caption("Use the tabs above to drill into time series and analytics.")

# ---------- Time Series ----------
with tab_ts:
    # st.set_page_config(page_title="Time Series", layout="wide")
    st.title("Time Series")

    if st.session_state.get("df") is None or len(st.session_state.df) == 0:
        st.warning("No data loaded. Go to the main page and click **Requery**.")
        st.stop()

    df = st.session_state.df

    st.subheader("Wind speed (m/s)")
    st.plotly_chart(px.line(df, x="timestamp", y="wind_speed_mps", color="data_type"), use_container_width=True, key="ts_wind_speed")

    st.subheader("Power (MW) and Price ($/MWh)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["power_mw"],
            name=f"Power (MW)",
            mode="lines",
        ),
        secondary_y=False,
    )

    if df["price_usd_mwh"].notna().any():
        for dtype, dfg in df.groupby("data_type"):
            fig.add_trace(
                go.Scatter(
                    x=dfg["timestamp"],
                    y=dfg["price_usd_mwh"],
                    name=f"Price ({dtype})",
                    mode="lines",
                    line=dict(dash="dot"),
                ),
                secondary_y=True,
            )
    else:
        st.info("Price data is disabled. Enable **Pull price data** on the main page and click **Requery** to see price/revenue charts.")
    
    # ---- Axis labels ----
    fig.update_yaxes(title_text="Power (MW)", secondary_y=False)
    fig.update_yaxes(title_text="Price ($/MWh)", secondary_y=True)


    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, key="ts_power_price")

    st.subheader("Revenue ($/hr)")
    st.plotly_chart(px.line(df, x="timestamp", y="revenue_usd", color="data_type"), use_container_width=True)

# ---------- Wind Analytics ----------
with tab_wind:
    # st.set_page_config(page_title="Wind Analytics", layout="wide")
    st.title("Wind Resource Analytics")

    if st.session_state.get("df") is None or len(st.session_state.df) == 0:
        st.warning("No data loaded. Go to the main page and click **Requery**.")
        st.stop()

    df = st.session_state.df
    p = st.session_state.projects[st.session_state.selected_project_id]

    df_hist = df[df["data_type"] == "historical"].copy()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Monthly capacity factor profile")
        m = monthly_capacity_factor(df_hist, p["rated_power_mw"])
        st.plotly_chart(px.line(m, x="month", y="capacity_factor"), use_container_width=True)

    with c2:
        st.subheader("Seasonal capacity factor")
        s = seasonal_capacity_factor(df_hist, p["rated_power_mw"])
        st.plotly_chart(px.bar(s, x="season", y="capacity_factor"), use_container_width=True)

    st.divider()

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Exceedance curve")
        ex = exceedance_curve(df_hist, threshold_mps=p["rated_speed_mps"])
        fig = px.line(ex, x="wind_speed_mps", y="pct_hours_ge")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"% hours ≥ rated speed ({p['rated_speed_mps']} m/s): {ex.attrs.get('threshold_pct', float('nan')):.2f}%")

    with c4:
        st.subheader("Downtime risk by month")
        d = downtime_risk_by_month(df_hist, p["cut_in_mps"], p["cut_out_mps"])
        fig = px.line(d, x="month", y=["pct_below_cut_in", "pct_above_cut_out"])
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Interannual variability")
    iav = interannual_variability(df_hist)
    st.plotly_chart(px.bar(iav, x="year", y="total_energy"), use_container_width=True)
    st.dataframe(iav, use_container_width=True)

    st.subheader("P50/P90 Annual Energy (bootstrap over years)")
    boot = annual_energy_bootstrap(df_hist, n_boot=2000)
    st.write(f"Years available: **{boot['years']}**")
    st.write(f"P50 AEP (MWh): **{boot['p50']:.0f}**")
    st.write(f"P90 AEP (MWh): **{boot['p90']:.0f}**")
    st.write(f"P10 AEP (MWh): **{boot['p10']:.0f}**")
    st.dataframe(boot["annual_table"], use_container_width=True)


# ---------- Revenue Analytics ----------
with tab_rev:
    # st.set_page_config(page_title="Revenue Analytics", layout="wide")
    st.title("Revenue & Market Analytics")

    if st.session_state.get("df") is None or len(st.session_state.df) == 0:
        st.warning("No data loaded. Go to the main page and click **Requery**.")
        st.stop()

    df = st.session_state.df
    p = st.session_state.projects[st.session_state.selected_project_id]

    if not df["price_usd_mwh"].notna().any():
        st.info("Price data is disabled. Enable **Pull price data** on the main page and click **Requery**.")
        st.stop()

    df_hist = df[df["data_type"] == "historical"].copy()
    df_hist["month"] = df_hist["timestamp"].dt.to_period("M").dt.to_timestamp()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Revenue time series")
        st.plotly_chart(px.line(df, x="timestamp", y="revenue_usd", color="data_type"), use_container_width=True, key="ts_revenue")

    with c2:
        st.subheader("Price distribution by month")
        box = px.box(df_hist, x="month", y="price_usd_mwh")
        box.update_xaxes(tickformat="%Y-%m")
        st.plotly_chart(box, use_container_width=True)

    st.divider()

    st.subheader("Monthly revenue")
    monthly = df_hist.groupby("month", as_index=False)["revenue_usd"].sum()
    st.plotly_chart(px.bar(monthly, x="month", y="revenue_usd"), use_container_width=True, key="rev_revenue_ts")
    st.dataframe(monthly, use_container_width=True)


# ---------- Settings ----------
with tab_settings:
    # st.set_page_config(page_title="Settings", layout="wide")
    st.title("Project & Settings")

    pid = st.session_state.selected_project_id
    p = st.session_state.projects[pid]

    st.subheader("Turbine parameters")
    p["cut_in_mps"] = st.number_input("Cut-in (m/s)", value=float(p.get("cut_in_mps", 3.0)), step=0.1)
    p["rated_speed_mps"] = st.number_input("Rated speed (m/s)", value=float(p.get("rated_speed_mps", 12.0)), step=0.1)
    p["cut_out_mps"] = st.number_input("Cut-out (m/s)", value=float(p.get("cut_out_mps", 25.0)), step=0.1)
    p["rated_power_mw"] = st.number_input("Rated power (MW)", value=float(p.get("rated_power_mw", 3.0)), step=0.1)

    st.subheader("Pricing defaults (used when prices enabled)")
    p["pricing_model"] = st.selectbox("Pricing model", ["fixed", "market"], index=0 if p.get("pricing_model")=="fixed" else 1)
    if p["pricing_model"] == "fixed":
        p["fixed_price_usd_mwh"] = st.number_input("Fixed price ($/MWh)", value=float(p.get("fixed_price_usd_mwh", 50.0)), step=1.0)
    else:
        p["iso_name"] = st.selectbox("ISO", ISO_CHOICES, index=ISO_CHOICES.index(p.get("iso_name","CAISO")))
        p["lmp_market"] = st.selectbox("LMP market", LMP_MARKETS, index=LMP_MARKETS.index(p.get("lmp_market","DAY_AHEAD_HOURLY")))
        p["location_type"] = st.selectbox("Location type", ["Hub"], index=0)

    st.divider()
    if st.button("Save projects.json"):
        save_projects_to_disk(st.session_state.projects)
        st.success("Saved.")
