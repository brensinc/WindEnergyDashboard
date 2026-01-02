import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import pydeck as pdk
import inspect
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo 
import gridstatus

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Wind Energy Dashboard", layout="wide")

PROJECTS_FILE = Path("projects.json")  # optional persistence

DEFAULT_PROJECT = {
    "project_id": "demo",
    "name": "Demo Wind Project",
    "latitude": 37.75,
    "longitude": -122.45,
    "hub_height_m": 80,
    "cut_in_mps": 3.0,
    "rated_speed_mps": 12.0,
    "cut_out_mps": 25.0,
    "rated_power_mw": 3.0,
}

# Pricing options
PRICE_MODELS = ["Fixed price ($/MWh)", "Market price (gridstatus)"]

# ISO choices (keep this explicit & stable)
ISO_CHOICES = ["CAISO", "ERCOT", "ISONE", "MISO", "NYISO", "PJM", "SPP"]

# Market choices (these strings are used by gridstatus across ISOs for LMP queries)
LMP_MARKETS = ["DAY_AHEAD_HOURLY", "REAL_TIME_HOURLY", "REAL_TIME_5_MIN"]


# -----------------------------
# Persistence helpers
# -----------------------------
def load_projects_from_disk() -> dict:
    if PROJECTS_FILE.exists():
        try:
            return json.loads(PROJECTS_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_projects_to_disk(projects: dict) -> None:
    PROJECTS_FILE.write_text(json.dumps(projects, indent=2, sort_keys=True))

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_prices_gridstatus(
    iso_name: str,
    start_date: str,
    end_date: str,
    market: str = "DAY_AHEAD_HOURLY",
    location_type: str = "Hub",
) -> pd.DataFrame:
    import gridstatus

    iso_class = getattr(gridstatus, iso_name, None)
    if iso_class is None:
        raise ValueError(f"Unsupported ISO class: {iso_name}")

    iso = iso_class()
    

    # Build kwargs safely based on supported signature
    sig = inspect.signature(iso.get_lmp)
    kwargs = {
        "start": pd.Timestamp(start_date),
        "end": pd.Timestamp(end_date),
        "market": market,
    }
    if "location_type" in sig.parameters:
        kwargs["location_type"] = location_type

    df = iso.get_lmp(**kwargs)

    st.write(df)

    if df is None or len(df) == 0:
        raise ValueError("No LMP data returned from gridstatus for this query.")

    # Normalize time column name: gridstatus commonly returns 'Time' (capital T)
    # We’ll handle a few common variants.
    time_col = None
    for c in ["Time", "time", "timestamp", "Datetime", "DATETIME"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"Could not find a time column in LMP data. Columns: {list(df.columns)}")

    # Normalize price column: often 'LMP'
    price_col = None
    for c in ["LMP", "lmp", "price", "Price", "LBMP"]:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError(f"Could not find a price column in LMP data. Columns: {list(df.columns)}")

    out = df.copy()
    out = out.rename(columns={time_col: "timestamp", price_col: "price_usd_mwh"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)

    # MVP: reduce to a single “proxy” price per timestamp
    # (mean over hubs returned). You can replace with hub selection later.
    out = (
        out.groupby("timestamp", as_index=False)["price_usd_mwh"]
        .mean()
        .sort_values("timestamp")
    )

    st.write("DEBUG 1")
    st.write(out)

    return out

def get_tz_from_latlon(lat: float, lon: float) -> ZoneInfo:
    _tf = TimezoneFinder()
    tzname = _tf.timezone_at(lat=lat, lng=lon)
    if tzname is None:
        return ZoneInfo("UTC")
    return ZoneInfo(tzname)

def add_prices_and_revenue(df: pd.DataFrame, project: dict) -> pd.DataFrame:
    """
    Adds price_usd_mwh and revenue_usd columns to df (wind/power dataframe).
    """
    out = df.copy()

    if project.get("pricing_model") == "fixed":
        fixed = float(project.get("fixed_price_usd_mwh", 0.0))
        out["price_usd_mwh"] = fixed

    elif project.get("pricing_model") == "market":
        iso_name = project.get("iso_name")
        market = project.get("lmp_market", "DAY_AHEAD_HOURLY")
        location_type = project.get("location_type", "Hub")

        prices = get_market_prices_gridstatus(
            iso_name=iso_name,
            start_date=str(out["timestamp"].min().date()),
            end_date=str(out["timestamp"].max().date()),
            market=market,
            location_type=location_type,
        )

        # Merge prices onto wind timestamps
        out = out.merge(prices, on="timestamp", how="left")

        # If the price series is sparse (e.g., DA hourly but your df is hourly too),
        # forward-fill within the time range
        out["price_usd_mwh"] = out["price_usd_mwh"].ffill()

    else:
        out["price_usd_mwh"] = np.nan

    # Revenue in $ per interval
    out["revenue_usd"] = out["energy_mwh"] * out["price_usd_mwh"]
    out["revenue_usd"] = out["revenue_usd"].astype(float)

    return out

def get_wind_data(lat: float, lon: float, date_start: str, date_end: str, hub_height_m: int) -> pd.DataFrame:
    """
    Fetch wind speed + direction from Open-Meteo. Cached with TTL to avoid repeated requests.
    Returns UTC-aware timestamps and wind speed in m/s.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    # url = "https://api.open-meteo.com/v1/archive"


    # Open-Meteo uses variables like wind_speed_80m, wind_speed_100m, etc.
    # We'll request the closest supported height by snapping to common heights.
    # You can expand this later.
    supported_heights = [10, 80, 120, 180]
    height = min(supported_heights, key=lambda h: abs(h - hub_height_m))

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [f"wind_speed_{height}m", f"wind_direction_{height}m"],
        "wind_speed_unit": "ms",      # IMPORTANT: get m/s directly
        "timezone": "UTC",
        "start_date": date_start,
        "end_date": date_end,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    time = hourly.get("time", [])
    ws = hourly.get(f"wind_speed_{height}m", [])
    wd = hourly.get(f"wind_direction_{height}m", [])

    if not time or not ws:
        raise ValueError("No hourly wind data returned for this query.")

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(time, utc=True),
        "wind_speed_mps": ws,
        "wind_direction_deg": wd if wd else [np.nan] * len(ws),
        "hub_height_m": height,
    })

    return df

def add_data_type(df: pd.DataFrame) -> pd.DataFrame:
    now_utc = datetime.now(timezone.utc)

    # now = datetime.now(timezone.utc)
    out = df.copy()

    out["data_type"] = np.where(out["timestamp"] <= now_utc, "historical", "forecast")
    
    # st.write(out)
    
    return out

# -----------------------------
# Power curve + power output
# -----------------------------
def wind_to_power(
    wind_speed_mps,
    cut_in_mps=3.0,
    rated_speed_mps=12.0,
    cut_out_mps=25.0,
    rated_power_mw=3.0
):
    """
    Idealized wind turbine power curve:
      - 0 below cut-in
      - linear ramp cut-in -> rated speed
      - rated power rated speed -> cut-out
      - 0 above cut-out
    """
    ws = np.asarray(wind_speed_mps, dtype=float)
    power = np.zeros_like(ws, dtype=float)

    ramp = (ws >= cut_in_mps) & (ws < rated_speed_mps)
    power[ramp] = rated_power_mw * (ws[ramp] - cut_in_mps) / (rated_speed_mps - cut_in_mps)

    rated = (ws >= rated_speed_mps) & (ws <= cut_out_mps)
    power[rated] = rated_power_mw

    return power

def add_power_output(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    out = df.copy()
    out["power_mw"] = wind_to_power(
        out["wind_speed_mps"].values,
        cut_in_mps=p["cut_in_mps"],
        rated_speed_mps=p["rated_speed_mps"],
        cut_out_mps=p["cut_out_mps"],
        rated_power_mw=p["rated_power_mw"],
    )
    # hourly data => energy per interval (MWh) is power(MW) * 1 hour
    out["energy_mwh"] = out["power_mw"] * 1.0
    return out

# -----------------------------
# Generation stats you requested
# -----------------------------
def compute_generation_stats(df: pd.DataFrame, p: dict) -> dict:
    """
    Stats computed on the full dataframe (historical + forecast).
    You can compute separately by data_type if desired.
    """
    ws = df["wind_speed_mps"].to_numpy()
    power = df["power_mw"].to_numpy()
    energy = df["energy_mwh"].to_numpy()

    below_cut_in = ws < p["cut_in_mps"]
    above_cut_out = ws > p["cut_out_mps"]
    operating = (~below_cut_in) & (~above_cut_out)

    # % time (of available timestamps)
    denom = max(len(ws), 1)
    pct_below = 100.0 * below_cut_in.sum() / denom
    pct_above = 100.0 * above_cut_out.sum() / denom
    pct_operating = 100.0 * operating.sum() / denom

    total_energy_mwh = float(np.nansum(energy))
    mean_power_mw = float(np.nanmean(power))
    p50_power_mw = float(np.nanpercentile(power, 50))
    p90_power_mw = float(np.nanpercentile(power, 90))
    max_power_mw = float(np.nanmax(power))

    # capacity factor over the period (using mean power / rated power)
    capacity_factor_pct = 100.0 * mean_power_mw / p["rated_power_mw"] if p["rated_power_mw"] > 0 else np.nan

    return {
        "pct_time_below_cut_in": pct_below,
        "pct_time_above_cut_out": pct_above,
        "pct_time_operating": pct_operating,
        "total_energy_mwh": total_energy_mwh,
        "mean_power_mw": mean_power_mw,
        "p50_power_mw": p50_power_mw,
        "p90_power_mw": p90_power_mw,
        "max_power_mw": max_power_mw,
        "capacity_factor_pct": capacity_factor_pct,
    }

# -----------------------------
# Initialize state
# -----------------------------
if "projects" not in st.session_state:
    disk_projects = load_projects_from_disk()
    # Ensure at least demo project exists
    if "demo" not in disk_projects:
        disk_projects["demo"] = DEFAULT_PROJECT
        save_projects_to_disk(disk_projects)
    st.session_state.projects = disk_projects

if "selected_project_id" not in st.session_state:
    st.session_state.selected_project_id = "demo"

# -----------------------------
# Sidebar: create/select project
# -----------------------------
st.sidebar.header("Projects")

project_ids = list(st.session_state.projects.keys())
selected = st.sidebar.selectbox(
    "Select a project",
    options=project_ids,
    index=project_ids.index(st.session_state.selected_project_id) if st.session_state.selected_project_id in project_ids else 0,
)
st.session_state.selected_project_id = selected
# st.sidebar.subheader("Manage")

# if st.sidebar.button("Delete selected project", type="secondary"):
#     pid = st.session_state.selected_project_id

#     # Optional: prevent deleting the demo project
#     if pid == "demo":
#         st.sidebar.error("Can't delete the demo project.")
#     else:
#         # Delete
#         st.session_state.projects.pop(pid, None)

#         # Persist
#         save_projects_to_disk(st.session_state.projects)

#         # Pick a new selection safely
#         remaining = list(st.session_state.projects.keys())
#         st.session_state.selected_project_id = remaining[0] if remaining else "demo"

#         # Clear cache (optional but recommended so UI updates immediately)
#         st.cache_data.clear()

#         st.rerun()


with st.sidebar.expander("Create a new project", expanded=False):
    pricing_model = st.selectbox("Pricing model", PRICE_MODELS, index=0)
    
    with st.form("create_project_form"):
        name = st.text_input("Project name", value="New Wind Project")
        lat = st.number_input("Latitude", value=37.75, format="%.6f")
        lon = st.number_input("Longitude", value=-122.45, format="%.6f")

        # st.markdown("### Pricing")

        fixed_price = None
        iso_name = None
        lmp_market = None
        location_type = None

        if pricing_model == "Fixed price ($/MWh)":
            fixed_price = st.number_input("Fixed price ($/MWh)", value=50.0, step=1.0)
        else:
            iso_name = st.selectbox("ISO / Market", ISO_CHOICES, index=0)
            lmp_market = st.selectbox("LMP market", LMP_MARKETS, index=0)
            # For an MVP, use hubs (much smaller than nodal/bus data)
            location_type = st.selectbox("Location type", ["Hub"], index=0)


        supported_heights = [10, 80, 120, 180]
        # hub_height_m = st.number_input("Hub height (m)", value=80, step=10)
        hub_height_m = st.selectbox(
            "Hub height (m)",
            options=supported_heights,
            index=supported_heights.index(80)  # default to 80
    )        

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
                "hub_height_m": int(hub_height_m),
                "cut_in_mps": float(cut_in),
                "rated_speed_mps": float(rated_speed),
                "cut_out_mps": float(cut_out),
                "rated_power_mw": float(rated_power),
                "pricing_model": "fixed" if pricing_model.startswith("Fixed") else "market",
                "fixed_price_usd_mwh": float(fixed_price) if fixed_price is not None else None,
                "iso_name": iso_name,
                "lmp_market": lmp_market,
                "location_type": location_type,

            }
            if persist:
                save_projects_to_disk(st.session_state.projects)
            st.session_state.selected_project_id = new_id
            st.success(f"Created project: {new_id}")

with st.sidebar.expander("Delete a project"):
    deletable_ids = [k for k in st.session_state.projects.keys() if k != "demo"]
    delete_id = st.selectbox("Choose project to delete", options=deletable_ids)
    if st.button("Delete this project", type="secondary"):
        st.session_state.projects.pop(delete_id, None)
        save_projects_to_disk(st.session_state.projects)
        if st.session_state.selected_project_id == delete_id:
            st.session_state.selected_project_id = "demo"
        st.cache_data.clear()
        st.rerun()

st.sidebar.divider()
if st.sidebar.button("Clear cached data"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared.")

# -----------------------------
# Main page: dashboard
# -----------------------------
p = st.session_state.projects[st.session_state.selected_project_id]
st.title("Wind Energy Dashboard")
st.caption("Historical vs forecast wind + estimated generation using an idealized turbine power curve.")

# Date range controls
with st.sidebar:
    st.subheader("Data Range")
    date_start = st.date_input("Start date", value=pd.to_datetime("2025-11-20").date())
    date_end = st.date_input("End date", value=pd.to_datetime("2025-12-10").date())

# Calculate local time
local_tz = get_tz_from_latlon(p["latitude"], p["longitude"])

now_utc = datetime.now(timezone.utc)
now_local = now_utc.astimezone(local_tz)

st.write(f"Local time: {now_local.strftime('%Y-%m-%d %H:%M %Z')}")


# Fetch + compute
try:
    with st.spinner("Loading wind data..."):
        df = get_wind_data(p["latitude"], p["longitude"], str(date_start), str(date_end), p["hub_height_m"])
        df = add_data_type(df)
        df = add_power_output(df, p)
        df = add_prices_and_revenue(df, p)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()
    
# Update dataframe to include local time
df["timestamp_local"] = df["timestamp"].dt.tz_convert(local_tz)



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


# with colB:
st.subheader("Wind Speed (m/s)")
fig_ws = px.line(
    df,
    x="timestamp_local",
    y="wind_speed_mps",
    color="data_type",
    line_dash="data_type",
    hover_data=["wind_direction_deg", "hub_height_m"],
    labels={"wind_speed_mps": "Wind speed (m/s)", "timestamp": "Time"},
)
fig_ws.add_hline(y=p["cut_in_mps"], line_dash="dot", annotation_text="Cut-in")
fig_ws.add_hline(y=p["cut_out_mps"], line_dash="dot", annotation_text="Cut-out")

fig_ws.update_traces(
    selector=dict(name="historical"),
    line=dict(dash="solid")
)

fig_ws.update_traces(
    selector=dict(name="forecast"),
    line=dict(dash="dot")
)

fig_ws.update_layout(
    height=350
)


st.plotly_chart(fig_ws, use_container_width=True)

st.subheader("Estimated Power Output (MW)")
fig_p = px.line(
    df,
    x="timestamp_local",
    y="power_mw",
    color="data_type",
    line_dash="data_type",
    labels={"power_mw": "Power (MW)", "timestamp": "Time"},
)

fig_p.update_traces(
    selector=dict(name="historical"),
    line=dict(dash="solid")
)

fig_p.update_traces(
    selector=dict(name="forecast"),
    line=dict(dash="dot")
)

fig_p.update_layout(
    height=350
)


st.plotly_chart(fig_p, use_container_width=True)


st.subheader("Energy Price Time Series ($/MWh)")

fig_price = px.line(
    df,
    x="timestamp_local",
    y="price_usd_mwh",
    color="data_type",
    line_dash="data_type",
    labels={"price_usd_mwh": "Price ($/MWh)", "timestamp": "Time"},
)

# Optional: match your dash styling
fig_price.update_traces(selector=dict(name="historical"), line=dict(dash="solid"))
fig_price.update_traces(selector=dict(name="forecast"), line=dict(dash="dot"))

fig_price.update_layout(height=300)

st.plotly_chart(fig_price, width="stretch")



st.subheader("Revenue Time Series")

fig_r = px.line(
    df,
    x="timestamp_local",
    y="revenue_usd",
    color="data_type",
    line_dash="data_type",
    labels={"revenue_usd": "Revenue ($/hr)", "timestamp": "Time"},
)
st.plotly_chart(fig_r, use_container_width=True)

    # st.subheader("Cumulative Revenue")

    # df_sorted = df.sort_values("timestamp").copy()
    # df_sorted["cum_revenue_usd"] = df_sorted["revenue_usd"].cumsum()

    # fig_cr = px.line(
    #     df_sorted,
    #     x="timestamp",
    #     y="cum_revenue_usd",
    #     color="data_type",
    #     line_dash="data_type",
    #     labels={"cum_revenue_usd": "Cumulative revenue ($)", "timestamp": "Time"},
    # )
    # st.plotly_chart(fig_cr, use_container_width=True)



# Stats section
st.subheader("Generation Stats")
stats_all = compute_generation_stats(df, p)

c1, c2, c3, c4 = st.columns(4)
c1.metric("% time below cut-in", f"{stats_all['pct_time_below_cut_in']:.1f}%")
c2.metric("% time above cut-out", f"{stats_all['pct_time_above_cut_out']:.1f}%")
c3.metric("% time operating", f"{stats_all['pct_time_operating']:.1f}%")
c4.metric("Capacity factor (period)", f"{stats_all['capacity_factor_pct']:.1f}%")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Total energy (MWh)", f"{stats_all['total_energy_mwh']:.1f}")
d2.metric("Mean power (MW)", f"{stats_all['mean_power_mw']:.2f}")
d3.metric("P50 power (MW)", f"{stats_all['p50_power_mw']:.2f}")
d4.metric("P90 power (MW)", f"{stats_all['p90_power_mw']:.2f}")


st.subheader("Revenue Stats")

hist = df[df["data_type"] == "historical"].copy()
fcst = df[df["data_type"] == "forecast"].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg price ($/MWh)", f"{hist['price_usd_mwh'].mean():.2f}" if len(hist) else "—")
col2.metric("Historical revenue ($)", f"{hist['revenue_usd'].sum():,.0f}" if len(hist) else "—")
col3.metric("Forecast revenue ($)", f"{fcst['revenue_usd'].sum():,.0f}" if len(fcst) else "—")
col4.metric("Total revenue ($)", f"{df['revenue_usd'].sum():,.0f}")


st.caption("Note: Energy is computed from hourly power estimates (power * 1 hour). The power curve is idealized; add an OEM curve later for realism.")
