from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import sys
import os
import json
import base64
from pathlib import Path
import io

# Add parent directory to path to import windlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from app.config import get_settings

router = APIRouter(prefix="/api/reports", tags=["reports"])


class ReportRequest(BaseModel):
    project_id: str
    project_name: str
    user_id: str
    latitude: float
    longitude: float
    hub_height_m: int = 100
    rotor_diameter_m: int = 100
    rated_power_kw: int = 3000
    start_date: str
    end_date: str
    pricing_mode: str = "market"
    fixed_price: float = 50.0
    iso_override: Optional[str] = None
    save_to_account: bool = False


class ReportResponse(BaseModel):
    report_id: Optional[str]
    html_content: str
    generated_at: str
    saved: bool


def generate_chart_image(fig_data: dict, fig_layout: dict) -> str:
    """
    Generate a base64-encoded PNG image from Plotly figure data.
    Returns empty string if generation fails.
    """
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=fig_data, layout=fig_layout)
        img_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"Chart generation failed: {e}")
        return ""


def generate_simple_bar_svg(values: list, labels: list, title: str, color: str = "#3b82f6") -> str:
    """Generate a simple SVG bar chart."""
    if not values:
        return ""
    
    max_val = max(values) if values else 1
    width = 600
    height = 300
    padding = 60
    bar_width = (width - 2 * padding) / len(values) * 0.8
    
    svg_parts = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>',
    ]
    
    for i, (val, label) in enumerate(zip(values, labels)):
        bar_height = (val / max_val) * (height - 2 * padding - 40) if max_val > 0 else 0
        x = padding + i * (width - 2 * padding) / len(values)
        y = height - padding - bar_height
        
        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" />'
        )
        # Label
        svg_parts.append(
            f'<text x="{x + bar_width/2}" y="{height - padding + 15}" text-anchor="middle" font-size="10">{label}</text>'
        )
        # Value
        svg_parts.append(
            f'<text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" font-size="9">{val:.0f}</text>'
        )
    
    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


@router.post("", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    Generate an HTML report for a wind energy project.
    The report includes all analytics sections with embedded charts.
    """
    try:
        import numpy as np
        import pandas as pd
        import windlib
        from datetime import date
        
        # Import analytics functions
        from app.routers.analytics import (
            load_hourly_prices, 
            load_iso_annual_prices,
            HOURLY_PRICES_DIR,
        )
        
        # Parse dates
        start = date.fromisoformat(request.start_date)
        end = date.fromisoformat(request.end_date)
        
        # Fetch wind data
        df = windlib.get_wind_data(
            lat=request.latitude,
            lon=request.longitude,
            date_start=str(start),
            date_end=str(end),
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No wind data available")
        
        # Determine ISO region
        if request.iso_override:
            iso_region = request.iso_override
        else:
            iso_region = windlib.get_iso_for_location(request.latitude, request.longitude)
            if not iso_region:
                iso_prices = load_iso_annual_prices()
                _, _, price_metadata = windlib.get_price_for_location(
                    request.latitude, request.longitude, iso_prices
                )
                iso_region = price_metadata.get("nearest_iso", "MISO")
        
        # Calculate power
        rated_power_mw = request.rated_power_kw / 1000.0
        cut_in_mps = 3.0
        rated_speed_mps = 12.0
        cut_out_mps = 25.0
        
        df["power_mw"] = windlib.wind_to_power(
            df["wind_speed_mps"].values,
            cut_in_mps=cut_in_mps,
            rated_speed_mps=rated_speed_mps,
            cut_out_mps=cut_out_mps,
            rated_power_mw=rated_power_mw,
        )
        df["power_kw"] = df["power_mw"] * 1000
        df["energy_mwh"] = df["power_mw"] * 1.0
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Apply pricing
        if request.pricing_mode == "fixed":
            df["price_usd_mwh"] = request.fixed_price
        else:
            prices_df = load_hourly_prices(iso_region)
            if prices_df is not None and not prices_df.empty:
                if df["timestamp"].dt.tz is not None:
                    df["timestamp_utc"] = df["timestamp"].dt.tz_convert("UTC")
                else:
                    df["timestamp_utc"] = df["timestamp"]
                
                if prices_df["timestamp"].dt.tz is not None:
                    prices_df["timestamp_utc"] = prices_df["timestamp"].dt.tz_convert("UTC")
                else:
                    prices_df["timestamp_utc"] = prices_df["timestamp"].dt.tz_localize("UTC")
                
                prices_df = prices_df[["timestamp_utc", "price_usd_mwh"]].drop_duplicates(subset=["timestamp_utc"])
                df = df.sort_values("timestamp_utc")
                prices_df = prices_df.sort_values("timestamp_utc")
                
                df = pd.merge_asof(
                    df, prices_df, on="timestamp_utc",
                    direction="nearest", tolerance=pd.Timedelta("2h")
                )
                
                iso_prices = load_iso_annual_prices()
                default_price = iso_prices.get(iso_region, 35.0)
                df["price_usd_mwh"] = df["price_usd_mwh"].fillna(default_price)
                df = df.drop(columns=["timestamp_utc"])
            else:
                iso_prices = load_iso_annual_prices()
                df["price_usd_mwh"] = iso_prices.get(iso_region, 35.0)
        
        df["revenue_usd"] = df["energy_mwh"] * df["price_usd_mwh"]
        
        # Calculate summary stats
        total_energy_mwh = float(df["energy_mwh"].sum())
        total_revenue = float(df["revenue_usd"].sum())
        capacity_factor = float(df["power_mw"].mean() / rated_power_mw) if rated_power_mw > 0 else 0
        avg_wind_speed = float(df["wind_speed_mps"].mean())
        avg_price = float(df["price_usd_mwh"].mean())
        peak_power_kw = float(df["power_kw"].max())
        hours_at_rated = int((df["power_mw"] >= rated_power_mw * 0.99).sum())
        hours_below_cutin = int((df["wind_speed_mps"] < cut_in_mps).sum())
        data_period_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
        
        # Monthly aggregation
        df["month"] = df["timestamp"].dt.to_period("M")
        monthly = df.groupby("month").agg({
            "energy_mwh": "sum",
            "revenue_usd": "sum",
            "power_mw": "mean",
            "wind_speed_mps": "mean",
            "price_usd_mwh": "mean",
        }).reset_index()
        monthly["capacity_factor"] = monthly["power_mw"] / rated_power_mw
        
        # Seasonal aggregation
        def get_season(month):
            if month in [12, 1, 2]: return "Winter"
            elif month in [3, 4, 5]: return "Spring"
            elif month in [6, 7, 8]: return "Summer"
            else: return "Fall"
        
        df["season"] = df["timestamp"].dt.month.map(get_season)
        seasonal = df.groupby("season").agg({
            "energy_mwh": "sum",
            "power_mw": "mean",
            "wind_speed_mps": "mean",
        }).reset_index()
        seasonal["capacity_factor"] = seasonal["power_mw"] / rated_power_mw
        
        # Hourly pattern
        df["hour"] = df["timestamp"].dt.hour
        hourly = df.groupby("hour").agg({
            "power_kw": "mean",
            "wind_speed_mps": "mean",
            "price_usd_mwh": "mean",
        }).reset_index()
        
        # Generate SVG charts
        monthly_revenue_svg = generate_simple_bar_svg(
            monthly["revenue_usd"].tolist(),
            [str(m) for m in monthly["month"]],
            "Monthly Revenue ($)",
            "#3b82f6"
        )
        
        monthly_energy_svg = generate_simple_bar_svg(
            monthly["energy_mwh"].tolist(),
            [str(m) for m in monthly["month"]],
            "Monthly Energy (MWh)",
            "#10b981"
        )
        
        seasonal_cf_svg = generate_simple_bar_svg(
            (seasonal["capacity_factor"] * 100).tolist(),
            seasonal["season"].tolist(),
            "Seasonal Capacity Factor (%)",
            "#f59e0b"
        )
        
        # Generate HTML report
        generated_at = datetime.now().isoformat()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wind Energy Report - {request.project_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        h1 {{ font-size: 28px; margin-bottom: 10px; color: #111827; }}
        h2 {{ font-size: 20px; margin: 30px 0 15px; color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
        h3 {{ font-size: 16px; margin: 20px 0 10px; color: #4b5563; }}
        .header {{ margin-bottom: 30px; }}
        .subtitle {{ color: #6b7280; font-size: 14px; }}
        .meta {{ display: flex; gap: 30px; margin-top: 15px; flex-wrap: wrap; }}
        .meta-item {{ }}
        .meta-label {{ font-size: 12px; color: #9ca3af; text-transform: uppercase; }}
        .meta-value {{ font-size: 16px; font-weight: 600; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .kpi-card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; }}
        .kpi-label {{ font-size: 12px; color: #6b7280; }}
        .kpi-value {{ font-size: 24px; font-weight: 700; color: #111827; }}
        .kpi-subtext {{ font-size: 11px; color: #9ca3af; }}
        .chart-container {{ margin: 20px 0; text-align: center; }}
        .chart-container svg {{ max-width: 100%; height: auto; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f9fafb; font-weight: 600; }}
        .section {{ margin-bottom: 40px; page-break-inside: avoid; }}
        .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #9ca3af; text-align: center; }}
        @media print {{
            body {{ padding: 20px; }}
            .section {{ page-break-inside: avoid; }}
        }}
        @media (max-width: 768px) {{
            .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .meta {{ flex-direction: column; gap: 10px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{request.project_name}</h1>
        <p class="subtitle">Wind Energy Analysis Report</p>
        <div class="meta">
            <div class="meta-item">
                <div class="meta-label">Location</div>
                <div class="meta-value">{request.latitude:.4f}°N, {abs(request.longitude):.4f}°W</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Analysis Period</div>
                <div class="meta-value">{request.start_date} to {request.end_date}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">ISO Region</div>
                <div class="meta-value">{iso_region or 'N/A'}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Pricing Mode</div>
                <div class="meta-value">{request.pricing_mode.capitalize()}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Project Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Hub Height</td><td>{request.hub_height_m} m</td></tr>
            <tr><td>Rotor Diameter</td><td>{request.rotor_diameter_m} m</td></tr>
            <tr><td>Rated Power</td><td>{request.rated_power_kw} kW ({rated_power_mw} MW)</td></tr>
            <tr><td>Cut-in Wind Speed</td><td>{cut_in_mps} m/s</td></tr>
            <tr><td>Rated Wind Speed</td><td>{rated_speed_mps} m/s</td></tr>
            <tr><td>Cut-out Wind Speed</td><td>{cut_out_mps} m/s</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Energy</div>
                <div class="kpi-value">{total_energy_mwh/1000:.1f} GWh</div>
                <div class="kpi-subtext">{data_period_days} days</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-value">${total_revenue/1000:.0f}K</div>
                <div class="kpi-subtext">Avg ${avg_price:.2f}/MWh</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Capacity Factor</div>
                <div class="kpi-value">{capacity_factor*100:.1f}%</div>
                <div class="kpi-subtext">{hours_at_rated} hrs at rated</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Avg Wind Speed</div>
                <div class="kpi-value">{avg_wind_speed:.1f} m/s</div>
                <div class="kpi-subtext">{hours_below_cutin} hrs below cut-in</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Monthly Performance</h2>
        <div class="chart-container">
            {monthly_revenue_svg}
        </div>
        <div class="chart-container">
            {monthly_energy_svg}
        </div>
        <table>
            <tr>
                <th>Month</th>
                <th>Energy (MWh)</th>
                <th>Revenue ($)</th>
                <th>Capacity Factor</th>
                <th>Avg Wind (m/s)</th>
                <th>Avg Price ($/MWh)</th>
            </tr>
            {''.join(f'''<tr>
                <td>{m}</td>
                <td>{e:.1f}</td>
                <td>${r:,.0f}</td>
                <td>{cf*100:.1f}%</td>
                <td>{w:.1f}</td>
                <td>${p:.2f}</td>
            </tr>''' for m, e, r, cf, w, p in zip(
                [str(x) for x in monthly["month"]],
                monthly["energy_mwh"],
                monthly["revenue_usd"],
                monthly["capacity_factor"],
                monthly["wind_speed_mps"],
                monthly["price_usd_mwh"]
            ))}
        </table>
    </div>

    <div class="section">
        <h2>Seasonal Analysis</h2>
        <div class="chart-container">
            {seasonal_cf_svg}
        </div>
        <table>
            <tr>
                <th>Season</th>
                <th>Energy (MWh)</th>
                <th>Capacity Factor</th>
                <th>Avg Wind Speed (m/s)</th>
            </tr>
            {''.join(f'''<tr>
                <td>{s}</td>
                <td>{e:.1f}</td>
                <td>{cf*100:.1f}%</td>
                <td>{w:.1f}</td>
            </tr>''' for s, e, cf, w in zip(
                seasonal["season"],
                seasonal["energy_mwh"],
                seasonal["capacity_factor"],
                seasonal["wind_speed_mps"]
            ))}
        </table>
    </div>

    <div class="section">
        <h2>Hourly Patterns</h2>
        <table>
            <tr>
                <th>Hour</th>
                <th>Avg Power (kW)</th>
                <th>Avg Wind (m/s)</th>
                <th>Avg Price ($/MWh)</th>
            </tr>
            {''.join(f'''<tr>
                <td>{h:02d}:00</td>
                <td>{p:.0f}</td>
                <td>{w:.1f}</td>
                <td>${pr:.2f}</td>
            </tr>''' for h, p, w, pr in zip(
                hourly["hour"],
                hourly["power_kw"],
                hourly["wind_speed_mps"],
                hourly["price_usd_mwh"]
            ))}
        </table>
    </div>

    <div class="footer">
        <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        <p>Wind Energy Dashboard - Analysis Report</p>
    </div>
</body>
</html>
"""
        
        # Save to Supabase if requested
        report_id = None
        saved = False
        
        if request.save_to_account and request.user_id:
            try:
                settings = get_settings()
                from supabase import create_client
                
                supabase = create_client(settings.supabase_url, settings.supabase_service_key)
                
                # Insert report record
                report_data = {
                    "user_id": request.user_id,
                    "project_id": request.project_id,
                    "report_type": "analysis",
                    "html_content": html_content,
                    "parameters": {
                        "start_date": request.start_date,
                        "end_date": request.end_date,
                        "pricing_mode": request.pricing_mode,
                        "iso_region": iso_region,
                    },
                    "summary": {
                        "total_energy_mwh": total_energy_mwh,
                        "total_revenue": total_revenue,
                        "capacity_factor": capacity_factor,
                        "avg_price": avg_price,
                    },
                }
                
                result = supabase.table("reports").insert(report_data).execute()
                if result.data:
                    report_id = result.data[0].get("id")
                    saved = True
            except Exception as e:
                print(f"Failed to save report to database: {e}")
        
        return ReportResponse(
            report_id=report_id,
            html_content=html_content,
            generated_at=generated_at,
            saved=saved,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@router.get("/html/{report_id}")
async def get_report_html(report_id: str):
    """Get a saved report by ID as HTML."""
    try:
        settings = get_settings()
        from supabase import create_client
        
        supabase = create_client(settings.supabase_url, settings.supabase_service_key)
        
        result = supabase.table("reports").select("html_content").eq("id", report_id).single().execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Report not found")
        
        return HTMLResponse(content=result.data["html_content"])
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching report: {str(e)}")


@router.get("/user/{user_id}")
async def get_user_reports(user_id: str):
    """Get all reports for a user."""
    try:
        settings = get_settings()
        from supabase import create_client
        
        supabase = create_client(settings.supabase_url, settings.supabase_service_key)
        
        result = supabase.table("reports")\
            .select("id, project_id, report_type, created_at, parameters, summary")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .execute()
        
        return result.data or []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching reports: {str(e)}")


@router.delete("/{report_id}")
async def delete_report(report_id: str, user_id: str = Query(...)):
    """Delete a report."""
    try:
        settings = get_settings()
        from supabase import create_client
        
        supabase = create_client(settings.supabase_url, settings.supabase_service_key)
        
        # Verify ownership
        check = supabase.table("reports")\
            .select("id")\
            .eq("id", report_id)\
            .eq("user_id", user_id)\
            .single()\
            .execute()
        
        if not check.data:
            raise HTTPException(status_code=404, detail="Report not found or access denied")
        
        supabase.table("reports").delete().eq("id", report_id).execute()
        
        return {"deleted": True}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting report: {str(e)}")
