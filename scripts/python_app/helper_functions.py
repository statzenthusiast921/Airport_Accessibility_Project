"""Shared UI and analytics helpers for the Dash app."""

import math
import re

import numpy as np
import pandas as pd
from dash import html
import dash_bootstrap_components as dbc

import helper_data

# Legacy master_air display_name: "City (IATA), Country" → "City, Country (IATA)"
AIRPORT_DISPLAY_RE = re.compile(r"^(.+)\s+\(([A-Z0-9]{3})\)\s*,\s*(.+)$")


def format_airport_label(display_name=None, *, city=None, country=None, iata=None):
    """Format an airport for display as City, Country (IATA)."""
    if display_name and not (city and iata):
        text = str(display_name).strip()
        match = AIRPORT_DISPLAY_RE.match(text)
        if match:
            city = match.group(1).strip()
            iata = match.group(2).strip()
            country = match.group(3).strip()
        elif city is None and iata is None:
            return text
    if city and country and iata:
        return f"{city}, {country} ({iata})"
    if city and iata:
        return f"{city} ({iata})"
    if display_name:
        return str(display_name).strip()
    if iata:
        return str(iata)
    return "—"


def format_airport_label_from_iata(iata):
    meta = helper_data.AIRPORT_IATA_META.get(iata) or {}
    return format_airport_label(
        display_name=meta.get("display_name"),
        country=meta.get("country"),
        iata=iata,
    )


# ------ Define some functions for later use
def great_circle_points(lat1, lon1, lat2, lon2, n=None):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d = 2 * math.asin(math.sqrt(
        math.sin((lat2 - lat1) / 2) ** 2 +
        math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    ))
    if d == 0:
        return [math.degrees(lat1)], [math.degrees(lon1)]
    if n is None:
        # Use more sample points for longer routes so the curve stays smooth.
        n = max(50, min(180, int(math.degrees(d) * 1.5)))
    lats, lons = [], []
    for t in np.linspace(0, 1, n):
        A = math.sin((1 - t) * d) / math.sin(d)
        B = math.sin(t * d) / math.sin(d)
        x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
        y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
        z = A * math.sin(lat1) + B * math.sin(lat2)
        lats.append(math.degrees(math.atan2(z, math.sqrt(x**2 + y**2))))
        lons.append(math.degrees(math.atan2(y, x)))
    return lats, lons


def split_antimeridian_segments(lats, lons):
    segmented_lats = []
    segmented_lons = []

    for idx, (lat, lon) in enumerate(zip(lats, lons)):
        if idx > 0 and abs(lon - lons[idx - 1]) > 180:
            segmented_lats.append(None)
            segmented_lons.append(None)
        segmented_lats.append(lat)
        segmented_lons.append(lon)

    return segmented_lats, segmented_lons


def extract_airline_names(carriers):
    if not isinstance(carriers, (list, tuple, set, np.ndarray)):
        return []
    return list(
        {
            carrier.get("name")
            for carrier in carriers
            if isinstance(carrier, dict) and carrier.get("name")
        }
    )


METRIC_CARD_STYLE = {
    "width": "100%",
    "height": "100%",
    "border": "none",
    "borderRadius": "18px",
    "background": "linear-gradient(135deg, #2E91E5 0%, #1B5FC1 100%)",
    "boxShadow": "0 10px 24px rgba(46, 145, 229, 0.35)",
    "textAlign": "left"
}

METRIC_CARD_TITLE_STYLE = {
    "margin": "0 0 8px 0",
    "fontSize": "0.8rem",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "1px",
    "color": "rgba(255,255,255,0.8)",
}

LABEL_STYLE_WHITE = {"color": "#f4f6fb", "fontWeight": "600"}

# Airport Metrics instructions modal: black panel, accent code for formulas
INSTRUCTIONS_MODAL_BODY_STYLE = {
    "paddingTop": "1.1rem",
    "paddingBottom": "1.35rem",
    "backgroundColor": "#000000",
    "color": "#e8ecf4",
}
INSTRUCTIONS_INTRO_STYLE = {
    "lineHeight": 1.55,
    "marginBottom": "18px",
    "fontWeight": "500",
    "color": "#f4f6fb",
}
INSTRUCTIONS_CODE_STYLE = {
    "color": "#f0a8d8",
    "backgroundColor": "rgba(232, 121, 200, 0.16)",
    "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    "fontSize": "0.95rem",
    "padding": "2px 8px",
    "borderRadius": "5px",
    "border": "1px solid rgba(240, 168, 216, 0.38)",
}


def inst_code(text):
    return html.Code(text, style=INSTRUCTIONS_CODE_STYLE)


def build_metric_card_body(title, value, font_size="1.8rem"):
    return dbc.CardBody([
        html.P(
            title,
            style=METRIC_CARD_TITLE_STYLE,
        ),
        html.H2(
            str(value),
            style={
                "margin": "0",
                "fontSize": font_size,
                "fontWeight": "700",
                "lineHeight": "1.15",
                "color": "white",
                "overflowWrap": "anywhere",
                "minHeight": "56px",
                "display": "flex",
                "alignItems": "center"
            }
        )
    ], style={
        "padding": "0.75rem 0.75rem",
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "justifyContent": "space-between"
    })


def dominant_airline_share_at_iata(iata):
    """Dominant carrier by frequency in airport carrier lists (airline_stats tab logic)."""
    ap_rows = helper_data.airport_df[helper_data.airport_df["iata"] == iata]
    if ap_rows.empty:
        return None, None
    names = ap_rows["carriers"].apply(extract_airline_names).explode().dropna()
    if names.empty:
        return None, None
    counts = names.value_counts()
    return counts.index[0], float(counts.iloc[0] / counts.sum())

# Airline Metrics layout (must be defined before app.layout).
AIRLINE_MAP_CONTROLS_BASE_STYLE = {
    "display": "grid",
    "gap": "12px",
    "alignItems": "end",
    "padding": "10px 14px",
    "borderRadius": "14px",
    "backgroundColor": "rgba(255,255,255,0.08)",
    "marginBottom": "8px",
}


def airline_map_controls_row_style(grid_template_columns):
    style = dict(AIRLINE_MAP_CONTROLS_BASE_STYLE)
    style["gridTemplateColumns"] = grid_template_columns
    return style


AIRLINE_MAP_MODE_PICKER_LABEL_STYLE = {
    "marginRight": 0,
    "padding": "6px 12px",
    "backgroundColor": "rgba(255,255,255,0.12)",
    "borderRadius": "999px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "width": "100%",
    "boxSizing": "border-box",
}
AIRLINE_CHARTS_ROW_STYLE = {
    "display": "flex",
    "alignItems": "stretch",
}
AIRLINE_MAP_COLUMN_STYLE = {
    "display": "flex",
    "flexDirection": "column",
    "minWidth": 0,
}
AIRLINE_TREEMAP_COLUMN_STYLE = {
    "display": "flex",
    "flexDirection": "column",
    "minWidth": 0,
}
AIRLINE_MAP_GRAPH_WRAP_STYLE = {
    "flex": "1",
    "minHeight": "480px",
    "display": "flex",
    "flexDirection": "column",
}
AIRLINE_TREEMAP_GRAPH_WRAP_STYLE = {
    "flex": "1",
    "minHeight": "560px",
    "display": "flex",
    "flexDirection": "column",
}
AIRLINE_MAP_FIGURE_HEIGHT = 520
AIRLINE_TREEMAP_FIGURE_HEIGHT = 592

AIRLINE_MAP_COLORS = [
    "#FF5A36", "#FFB000", "#00C2A8", "#7ED957", "#C449A0",
    "#FF7A00", "#5B4CFF", "#FF4F87", "#0077FF", "#00D1B2",
]


def prepare_country_airline_data(selected_country):
    country_key = str(selected_country)
    country_filtered = helper_data.airport_df[
        helper_data.airport_df["country"].astype(str) == country_key
    ].copy()
    country_filtered["airline_names"] = country_filtered["carriers"].apply(
        extract_airline_names
    )

    airport_exploded = country_filtered.explode("airline_names").dropna(subset=["airline_names"])
    airport_exploded = airport_exploded[
        ['city_name', 'country', 'display_name', 'iata', 'airline_names',
         'latitude', 'longitude']
    ]
    if airport_exploded.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    city_counts = (
        airport_exploded.groupby(["city_name", "airline_names"], observed=True)
        .size()
        .reset_index(name="flight_count")
    )
    city_counts["city_total"] = city_counts.groupby("city_name", observed=True)["flight_count"].transform("sum")
    city_counts["pct_share"] = city_counts["flight_count"] / city_counts["city_total"]

    dominant_airline = (
        city_counts.loc[city_counts.groupby("city_name", observed=True)["pct_share"].idxmax()]
        .reset_index(drop=True)
        .rename(columns={"airline_names": "dominant_airline"})
    )
    dominant_airline = dominant_airline[['city_name', 'dominant_airline', 'pct_share']]

    country_airline_mapping_df = pd.merge(
        country_filtered,
        dominant_airline,
        on='city_name',
        how='left'
    )
    country_airline_mapping_df = country_airline_mapping_df[
        ['city_name', 'display_name', 'iata',
         'latitude', 'longitude', 'dominant_airline', 'pct_share']
    ]
    country_airline_mapping_df = country_airline_mapping_df.drop_duplicates().rename(
        columns={"dominant_airline": "dominant_airline_actual"}
    )
    country_airline_mapping_df["latitude"] = pd.to_numeric(
        country_airline_mapping_df["latitude"], errors="coerce"
    )
    country_airline_mapping_df["longitude"] = pd.to_numeric(
        country_airline_mapping_df["longitude"], errors="coerce"
    )
    country_airline_mapping_df["pct_share"] = pd.to_numeric(
        country_airline_mapping_df["pct_share"], errors="coerce"
    )
    country_airline_mapping_df = country_airline_mapping_df.dropna(
        subset=["latitude", "longitude", "dominant_airline_actual", "pct_share"]
    )
    if country_airline_mapping_df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    top_airlines = list(
        country_airline_mapping_df["dominant_airline_actual"]
        .value_counts()
        .head(10)
        .index
    )
    country_airline_mapping_df["dominant_airline_group"] = np.where(
        country_airline_mapping_df["dominant_airline_actual"].isin(top_airlines),
        country_airline_mapping_df["dominant_airline_actual"],
        "Other",
    )
    country_airline_mapping_df["marker_size_value"] = np.where(
        country_airline_mapping_df["dominant_airline_group"] == "Other",
        8.0,
        (country_airline_mapping_df["pct_share"] * 18).clip(lower=10.0),
    )
    for col in ("dominant_airline_group", "dominant_airline_actual", "city_name", "iata"):
        country_airline_mapping_df[col] = country_airline_mapping_df[col].astype(str)

    airport_airline_share_df = (
        airport_exploded.groupby(
            ["display_name", "city_name", "iata", "latitude", "longitude", "airline_names"],
            observed=True,
        )
        .size()
        .reset_index(name="flight_count")
    )
    airport_airline_share_df["airport_total"] = airport_airline_share_df.groupby(
        ["display_name", "iata"], observed=True
    )["flight_count"].transform("sum")
    airport_airline_share_df["pct_share"] = (
        airport_airline_share_df["flight_count"] / airport_airline_share_df["airport_total"]
    )
    airport_airline_share_df = airport_airline_share_df.rename(
        columns={"airline_names": "selected_airline"}
    )
    airport_airline_share_df["latitude"] = pd.to_numeric(
        airport_airline_share_df["latitude"], errors="coerce"
    )
    airport_airline_share_df["longitude"] = pd.to_numeric(
        airport_airline_share_df["longitude"], errors="coerce"
    )
    airport_airline_share_df["pct_share"] = pd.to_numeric(
        airport_airline_share_df["pct_share"], errors="coerce"
    )
    airport_airline_share_df = airport_airline_share_df.dropna(
        subset=["latitude", "longitude", "pct_share"]
    )
    for col in ("selected_airline", "city_name", "iata", "display_name"):
        if col in airport_airline_share_df.columns:
            airport_airline_share_df[col] = airport_airline_share_df[col].astype(str)

    top_airlines = [str(a) for a in top_airlines]
    return country_airline_mapping_df, airport_airline_share_df, top_airlines


def get_airline_color_map(top_airlines):
    color_map = {
        airline: color
        for airline, color in zip(top_airlines, AIRLINE_MAP_COLORS)
    }
    color_map["Other"] = "#9E9E9E"
    return color_map


def get_country_view(df):
    lat_min = df["latitude"].min()
    lat_max = df["latitude"].max()
    lon_min = df["longitude"].min()
    lon_max = df["longitude"].max()
    max_span = max(lat_max - lat_min, lon_max - lon_min, 1)

    if max_span > 60:
        zoom = 2.0
    elif max_span > 30:
        zoom = 2.5
    elif max_span > 15:
        zoom = 3.0
    elif max_span > 8:
        zoom = 3.5
    else:
        zoom = 4.5

    center = {"lat": (lat_min + lat_max) / 2, "lon": (lon_min + lon_max) / 2}
    return center, zoom


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    red, green, blue = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def build_heat_colorscale(hex_color):
    return [
        [0.0, hex_to_rgba(hex_color, 0.0)],
        [0.08, hex_to_rgba(hex_color, 0.4)],
        [0.3, hex_to_rgba(hex_color, 0.75)],
        [0.65, hex_to_rgba(hex_color, 0.92)],
        [1.0, hex_to_rgba(hex_color, 1.0)],
    ]
