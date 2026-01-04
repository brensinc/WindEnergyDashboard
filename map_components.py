"""
Folium map components for wind energy dashboard.

This module provides functions for:
- Creating interactive Folium maps
- Adding project markers
- Adding heatmap layers
- Handling map interactions
"""

import folium
from folium.plugins import HeatMap
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Constants
# -----------------------------

# US center coordinates for default map view
US_CENTER = [39.8283, -98.5795]
DEFAULT_ZOOM = 4

# Project marker colors
PROJECT_MARKER_COLOR = 'red'
SELECTED_MARKER_COLOR = 'green'

# Heatmap gradient (revenue: low to high)
REVENUE_GRADIENT = {
    0.0: '#3288bd',   # Blue (low revenue)
    0.25: '#66c2a5',  # Teal
    0.5: '#abdda4',   # Light green
    0.75: '#fdae61',  # Orange
    1.0: '#d53e4f',   # Red (high revenue)
}


# -----------------------------
# Map Creation
# -----------------------------
def create_base_map(
    center: List[float] = None,
    zoom: int = DEFAULT_ZOOM,
    tiles: str = 'OpenStreetMap'
) -> folium.Map:
    """
    Create a US-centered Folium map.
    
    Args:
        center: [lat, lon] center coordinates (defaults to US center)
        zoom: Initial zoom level
        tiles: Base tile layer ('OpenStreetMap', 'CartoDB positron', etc.)
        
    Returns:
        Folium Map object
    """
    if center is None:
        center = US_CENTER
    
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=tiles,
        control_scale=True,
    )
    
    return m


def create_project_map(
    projects: Dict,
    selected_project_id: str = None,
    center: List[float] = None,
    zoom: int = DEFAULT_ZOOM
) -> folium.Map:
    """
    Create a map with all project markers.
    
    Args:
        projects: Dictionary of project data
        selected_project_id: ID of currently selected project (highlighted)
        center: Map center coordinates
        zoom: Initial zoom level
        
    Returns:
        Folium Map object with project markers
    """
    m = create_base_map(center=center, zoom=zoom)
    add_project_markers(m, projects, selected_project_id)
    return m


# -----------------------------
# Project Markers
# -----------------------------
def add_project_markers(
    m: folium.Map,
    projects: Dict,
    selected_project_id: str = None
) -> None:
    """
    Add project location markers to a Folium map.
    
    Args:
        m: Folium Map object
        projects: Dictionary of project data
        selected_project_id: ID of currently selected project (will be green)
    """
    for proj_id, proj in projects.items():
        lat = proj.get("latitude")
        lon = proj.get("longitude")
        name = proj.get("name", proj_id)
        
        if lat is None or lon is None:
            continue
        
        # Determine marker color
        is_selected = (proj_id == selected_project_id)
        color = SELECTED_MARKER_COLOR if is_selected else PROJECT_MARKER_COLOR
        
        # Create popup content
        popup_html = f"""
        <div style="min-width: 150px;">
            <b>{name}</b><br>
            <small>ID: {proj_id}</small><br>
            <hr style="margin: 5px 0;">
            Lat: {lat:.4f}<br>
            Lon: {lon:.4f}<br>
            Rated Power: {proj.get('rated_power_mw', 'N/A')} MW
        </div>
        """
        
        # Add marker
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{name} ({proj_id})",
            icon=folium.Icon(
                color=color,
                icon='bolt',
                prefix='fa'
            )
        ).add_to(m)


def add_single_marker(
    m: folium.Map,
    lat: float,
    lon: float,
    name: str = "Selected Location",
    color: str = 'blue'
) -> None:
    """
    Add a single marker to the map.
    
    Args:
        m: Folium Map object
        lat: Latitude
        lon: Longitude
        name: Marker name/tooltip
        color: Marker color
    """
    folium.Marker(
        location=[lat, lon],
        popup=f"{name}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}",
        tooltip=name,
        icon=folium.Icon(color=color, icon='crosshairs', prefix='fa')
    ).add_to(m)


# -----------------------------
# Heatmap Layer
# -----------------------------
def add_heatmap_layer(
    m: folium.Map,
    heat_data: List[List[float]],
    name: str = "Annual Revenue Heatmap",
    radius: int = 15,
    blur: int = 10,
    min_opacity: float = 0.4,
    gradient: Dict = None
) -> None:
    """
    Add a revenue heatmap layer to the map.
    
    Args:
        m: Folium Map object
        heat_data: List of [lat, lon, intensity] values
        name: Layer name for layer control
        radius: Heatmap point radius
        blur: Heatmap blur amount
        min_opacity: Minimum opacity for heatmap
        gradient: Color gradient dictionary
    """
    if not heat_data:
        return
    
    if gradient is None:
        gradient = REVENUE_GRADIENT
    
    # Create heatmap layer
    heatmap = HeatMap(
        data=heat_data,
        name=name,
        radius=radius,
        blur=blur,
        min_opacity=min_opacity,
        gradient=gradient,
        show=True,  # Show by default
    )
    
    heatmap.add_to(m)


def add_heatmap_from_cache(
    m: folium.Map,
    heatmap_data: Dict,
    name: str = "Annual Revenue Heatmap"
) -> None:
    """
    Add heatmap layer from cached heatmap data.
    
    Args:
        m: Folium Map object
        heatmap_data: Dictionary from load_heatmap_cache()
        name: Layer name
    """
    from heatmap_utils import get_heatmap_data_for_folium
    
    heat_data = get_heatmap_data_for_folium(heatmap_data)
    if heat_data:
        add_heatmap_layer(m, heat_data, name=name)


# -----------------------------
# Layer Control
# -----------------------------
def add_layer_control(m: folium.Map) -> None:
    """
    Add layer control to the map for toggling layers.
    
    Args:
        m: Folium Map object
    """
    folium.LayerControl(
        position='topright',
        collapsed=False
    ).add_to(m)


# -----------------------------
# Click Handling Support
# -----------------------------
def get_click_coordinates(map_data: Dict) -> Optional[Tuple[float, float]]:
    """
    Extract click coordinates from st_folium return data.
    
    Args:
        map_data: Dictionary returned by st_folium()
        
    Returns:
        Tuple of (lat, lon) rounded to 4 decimal places, or None if no click
    """
    if map_data is None:
        return None
    
    last_clicked = map_data.get('last_clicked')
    if last_clicked is None:
        return None
    
    lat = last_clicked.get('lat')
    lng = last_clicked.get('lng')
    
    if lat is None or lng is None:
        return None
    
    # Round to 4 decimal places (~11m precision)
    return (round(lat, 4), round(lng, 4))


def format_coordinates(lat: float, lon: float) -> str:
    """
    Format coordinates for display.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Formatted string like "37.7749, -122.4194"
    """
    return f"{lat:.4f}, {lon:.4f}"


# -----------------------------
# Legend Creation
# -----------------------------
def create_revenue_legend_html(
    min_revenue: float,
    max_revenue: float,
    currency: str = "$"
) -> str:
    """
    Create HTML for a revenue legend.
    
    Args:
        min_revenue: Minimum revenue value
        max_revenue: Maximum revenue value
        currency: Currency symbol
        
    Returns:
        HTML string for the legend
    """
    return f"""
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 1000;
        background: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        font-size: 12px;
    ">
        <b>Annual Revenue</b><br>
        <div style="
            display: flex;
            align-items: center;
            margin-top: 5px;
        ">
            <div style="
                width: 100px;
                height: 15px;
                background: linear-gradient(to right, #3288bd, #66c2a5, #abdda4, #fdae61, #d53e4f);
            "></div>
        </div>
        <div style="
            display: flex;
            justify-content: space-between;
            width: 100px;
        ">
            <span>{currency}{min_revenue:,.0f}</span>
            <span>{currency}{max_revenue:,.0f}</span>
        </div>
        <small style="color: #666;">$/year (est.)</small>
    </div>
    """


def add_legend_to_map(
    m: folium.Map,
    min_revenue: float,
    max_revenue: float
) -> None:
    """
    Add a revenue legend to the map.
    
    Args:
        m: Folium Map object
        min_revenue: Minimum revenue value
        max_revenue: Maximum revenue value
    """
    legend_html = create_revenue_legend_html(min_revenue, max_revenue)
    m.get_root().html.add_child(folium.Element(legend_html))
