"""
Connections-tab network charts and metrics.
"""
import pandas as pd
import visdcc
from dash import html
import dash_bootstrap_components as dbc

import helper_data
import numpy as np

from helper_functions import (
    METRIC_CARD_STYLE,
    build_metric_card_body,
    dominant_airline_share_at_iata,
    format_airport_label_from_iata,
)


NETWORK_NODE_FONT = {
    "size": 13,
    "multi": True,
    "color": "#ffffff",
    "strokeWidth": 2,
    "strokeColor": "#080a0d",
    "face": "system-ui, -apple-system, 'Segoe UI', sans-serif",
}

# Vis-network canvas height for Connections tab (shorter than default 900px to cut vertical scroll).
NETWORK_VIS_HEIGHT = "520px"

NETWORK_CHART_SURFACE = {
    "backgroundColor": "#12151c",
    "borderRadius": "12px",
    "padding": "6px",
    "border": "1px solid rgba(255, 255, 255, 0.12)",
    "minHeight": "532px",
}

NETWORK_EMPTY_STYLE = {"color": "#e8ecf4", "padding": "20px", "fontSize": "1rem"}

# In-country airports read as “central”; peers use a distinct hue per connection mode.
NETWORK_CENTRAL_NODE_COLOR = "#FACC15"
NETWORK_PEER_NODE_COLOR = {
    "carriers_airline": "#38BDF8",
    "similarity": "#6366F1",
    "proximity": "#22C55E",
    "shared_raw": "#C026D3",
    "shared_cosine": "#FB7185",
}

CONN_AIRPORT_ALL = "__ALL__"
MAX_NETWORK_EDGES = 100


def airport_node_sizes_and_color(iata, selected_country, peer_hex):
    """Larger yellow node if airport is in the selected country; otherwise peer color."""
    meta_ct = (helper_data.AIRPORT_IATA_META.get(iata) or {}).get("country")
    if meta_ct == selected_country:
        return NETWORK_CENTRAL_NODE_COLOR, 15
    return peer_hex, 12


def filter_airport_edges(df, focus_iata):
    """Keep only edges incident to focus_iata (both columns must be source/target)."""
    if not focus_iata or focus_iata == CONN_AIRPORT_ALL:
        return df
    return df[(df["source"] == focus_iata) | (df["target"] == focus_iata)]


def countries_in_node_set(node_ids):
    countries = set()
    for iata in node_ids:
        ct = (helper_data.AIRPORT_IATA_META.get(iata) or {}).get("country")
        if ct:
            countries.add(ct)
    return len(countries)


def strongest_edge_pair_iata(filtered):
    if filtered.empty:
        return "—"
    row = filtered.iloc[0]
    return f"{row['source']}–{row['target']}"


def metric_capped_if_needed(n):
    n = int(n)
    return f"{n} (Capped)" if n >= MAX_NETWORK_EDGES else str(n)


def strongest_weight_count_pair(filtered):
    if filtered.empty:
        return "—"
    max_w = filtered["weight"].max()
    top = filtered.loc[filtered["weight"] == max_w].sort_values(["source", "target"])
    row = top.iloc[0]
    return f"{int(row['weight'])} ({row['source']}–{row['target']})"


def strongest_adjusted_score_label(filtered):
    if filtered.empty:
        return "—"
    max_w = filtered["weight"].max()
    top = filtered.loc[filtered["weight"] == max_w].sort_values(["source", "target"])
    row = top.iloc[0]
    return f"{float(row['weight']):.3f} ({row['source']}–{row['target']})"


def strongest_similarity_label(filtered):
    if filtered.empty:
        return "—"
    max_w = filtered["weight"].max()
    top = filtered.loc[filtered["weight"] == max_w].sort_values(["source", "target"])
    row = top.iloc[0]
    return f"{round(float(row['weight']), 3)} ({row['source']}-{row['target']})"


def similarity_feature_distance(iata_a, iata_b):
    """Equal-weight mean of |Δz| for connectivity, log(1+dests), redundancy, and elevation."""
    pa = helper_data.SIMILARITY_AIRPORT_PROFILE.get(iata_a)
    pb = helper_data.SIMILARITY_AIRPORT_PROFILE.get(iata_b)
    if not pa or not pb:
        return np.nan
    diffs = [abs(pa[k] - pb[k]) for k in helper_data.SIMILARITY_Z_FEATURE_KEYS]
    return float(sum(diffs) / len(diffs))


def apply_similarity_scores(edges_df):
    """Replace edge weights with 0-1 similarity (1 = most alike in this edge set)."""
    if edges_df.empty:
        return edges_df
    out = edges_df.copy()
    distances = [
        similarity_feature_distance(s, t)
        for s, t in zip(out["source"], out["target"])
    ]
    out["_sim_dist"] = distances
    valid = out["_sim_dist"].notna()
    if not valid.any():
        return out.drop(columns=["_sim_dist"], errors="ignore")
    d = out.loc[valid, "_sim_dist"].astype(float)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max == d_min:
        out.loc[valid, "weight"] = 1.0
    else:
        out.loc[valid, "weight"] = 1.0 - (d - d_min) / (d_max - d_min)
    return out.drop(columns=["_sim_dist"])


def similarity_pct_diff_vs_focus(clicked_val, focus_val):
    """Percent difference of clicked airport metric relative to the focus airport."""
    if focus_val == 0:
        return "+0.0% vs focus" if clicked_val == 0 else None
    pct = (clicked_val - focus_val) / focus_val * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}% vs focus"


def build_statistical_similarity_detail_body(info, focus_iata=None):
    profile = helper_data.SIMILARITY_AIRPORT_PROFILE.get(info.get("iata"))
    if not profile:
        return [
            html.P("Profile data unavailable for this airport.", style={"color": "#e8ecf4"}),
        ]

    body = []
    if focus_iata and focus_iata != CONN_AIRPORT_ALL:
        focus_profile = helper_data.SIMILARITY_AIRPORT_PROFILE.get(focus_iata)
        if focus_profile:
            focus_meta = helper_data.AIRPORT_IATA_META.get(focus_iata) or {}
            focus_name = format_airport_label_from_iata(focus_iata)
            body.append(html.P(
                f"Compared to focus airport: {focus_name}",
                style={
                    "marginBottom": "8px",
                    "fontSize": "0.9rem",
                    "color": "rgba(255,255,255,0.82)",
                },
            ))

    metric_specs = (
        ("Connectivity Index", "connectivity_index", False),
        ("Redundancy Index", "redundancy_score", False),
        ("# Destinations", "num_dests", True),
        ("Elevation (ft)", "elevation", True),
    )
    focus_profile = (
        helper_data.SIMILARITY_AIRPORT_PROFILE.get(focus_iata)
        if focus_iata and focus_iata != CONN_AIRPORT_ALL
        else None
    )

    lines = []
    for label, key, as_int in metric_specs:
        val = profile[key]
        text = f"{label}: {int(val)}" if as_int else f"{label}: {val:.2f}"
        if focus_profile:
            diff = similarity_pct_diff_vs_focus(val, focus_profile[key])
            if diff is not None:
                text = f"{text} ({diff})"
        lines.append(html.Li(text))

    body.append(html.Ul(
        lines,
        style={"paddingLeft": "20px", "color": "#e8ecf4", "lineHeight": 1.6},
    ))
    return body


def wrap_network_chart(net, legend_items=None):
    """Place the network graph in a styled container; optionally add a color legend."""
    surface_style = dict(NETWORK_CHART_SURFACE)
    surface_style["position"] = "relative"

    chart_children = [net]

    if legend_items:
        legend_rows = []
        for label, color in legend_items:
            color_dot = html.Span(
                style={
                    "display": "inline-block",
                    "width": "12px",
                    "height": "12px",
                    "borderRadius": "50%",
                    "backgroundColor": color,
                    "marginRight": "8px",
                    "verticalAlign": "middle",
                }
            )
            label_text = html.Span(label, style={"verticalAlign": "middle"})
            legend_rows.append(
                html.Div(
                    [color_dot, label_text],
                    style={"marginBottom": "4px"},
                )
            )

        legend_box = html.Div(
            legend_rows,
            style={
                "position": "absolute",
                "top": "10px",
                "right": "12px",
                "backgroundColor": "rgba(18, 21, 28, 0.92)",
                "border": "1px solid rgba(255,255,255,0.15)",
                "borderRadius": "8px",
                "padding": "8px 12px",
                "zIndex": 10,
                "fontSize": "0.8rem",
                "color": "#e8ecf4",
                "lineHeight": 1.4,
            },
        )
        chart_children.append(legend_box)

    return html.Div(chart_children, style=surface_style)


def airport_network_display_label(iata):
    return format_airport_label_from_iata(iata)


def carrier_network_physics_options():
    return {
        "height": NETWORK_VIS_HEIGHT,
        "width": "100%",
        "layout": {"hierarchical": {"enabled": False}},
        "physics": {
            "enabled": True,
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -28000,
                "centralGravity": 0.25,
                "springLength": 240,
                "springConstant": 0.012,
                "damping": 0.62,
                "avoidOverlap": 0.55,
            },
            "maxVelocity": 22,
            "minVelocity": 0.4,
            "stabilization": {"enabled": True, "iterations": 220, "updateInterval": 30},
        },
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "tooltipDelay": 100,
        },
        "nodes": {
            "font": NETWORK_NODE_FONT,
        },
        "edges": {
            "color": {"color": "#9dabc4", "opacity": 0.85},
            "smooth": {"type": "continuous", "roundness": 0.35},
        },
    }

def build_connection_types_guide(active_connection_type):
    """Short definitions for the Connections tab sidebar."""
    blocks = []
    for title, blurb in helper_data.CONNECTION_TYPE_DEFINITIONS:
        is_active = title == active_connection_type
        blocks.append(
            html.Div(
                [
                    html.P(
                        title,
                        style={
                            "margin": "0 0 4px 0",
                            "fontWeight": "700",
                            "fontSize": "0.82rem",
                            "color": "#ffffff" if is_active else "rgba(255,255,255,0.95)",
                            "letterSpacing": "0.02em",
                        },
                    ),
                    html.P(
                        blurb,
                        style={
                            "margin": 0,
                            "fontSize": "0.8rem",
                            "lineHeight": 1.45,
                            "color": "rgba(232,236,244,0.92)" if is_active else "rgba(232,236,244,0.78)",
                        },
                    ),
                ],
                style={
                    "marginBottom": "12px",
                    "paddingLeft": "10px",
                    "borderLeft": f"3px solid {'#FACC15' if is_active else 'rgba(255,255,255,0.15)'}",
                },
            )
        )
    return html.Div(
        [
            html.P(
                "What each connection type shows",
                style={
                    "margin": "0 0 10px 0",
                    "fontWeight": "700",
                    "fontSize": "0.9rem",
                    "color": "#ffffff",
                },
            ),
            html.Div(
                blocks,
                style={
                    "maxHeight": "260px",
                    "overflowY": "auto",
                    "paddingRight": "6px",
                    "marginBottom": "12px",
                },
            ),
        ]
    )


def _empty_metric_cards():
    return dbc.Row([
        dbc.Col(dbc.Card(build_metric_card_body("Connection view", "—"), style=METRIC_CARD_STYLE), width=3),
        dbc.Col(dbc.Card(build_metric_card_body("Nodes", "—"), style=METRIC_CARD_STYLE), width=3),
        dbc.Col(dbc.Card(build_metric_card_body("Links", "—"), style=METRIC_CARD_STYLE), width=3),
        dbc.Col(dbc.Card(build_metric_card_body("Detail", "—"), style=METRIC_CARD_STYLE), width=3),
    ])


def _no_data(message):
    return (
        html.Div(message, style=NETWORK_EMPTY_STYLE),
        {},
        _empty_metric_cards(),
    )


def _metric_row_two(title_a, value_a, title_b, value_b):
    return dbc.Row([
        dbc.Col(dbc.Card(build_metric_card_body(title_a, value_a), style=METRIC_CARD_STYLE), width=6),
        dbc.Col(dbc.Card(build_metric_card_body(title_b, value_b), style=METRIC_CARD_STYLE), width=6)
    ])


def _metric_row_four(title_a, value_a, title_b, value_b, title_c, value_c, title_d, value_d):
    return dbc.Row([
        dbc.Col(dbc.Card(build_metric_card_body(title_a, value_a), style=METRIC_CARD_STYLE), width=3),
        dbc.Col(dbc.Card(build_metric_card_body(title_b, value_b), style=METRIC_CARD_STYLE), width=3),
        dbc.Col(dbc.Card(build_metric_card_body(title_c, value_c), style=METRIC_CARD_STYLE), width=3),
        dbc.Col(dbc.Card(build_metric_card_body(title_d, value_d), style=METRIC_CARD_STYLE), width=3)
    ])


def _standard_vis_options(gravitational_constant=-20000, spring_length=180):
    return {
        "height": NETWORK_VIS_HEIGHT,
        "width": "100%",
        "layout": {"hierarchical": {"enabled": False}},
        "physics": {
            "enabled": True,
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": gravitational_constant,
                "springLength": spring_length,
                "springConstant": 0.02,
                "damping": 0.58,
            },
            "stabilization": {"iterations": 200},
        },
        "nodes": {"font": NETWORK_NODE_FONT},
        "edges": {"smooth": {"type": "continuous", "roundness": 0.35}},
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "tooltipDelay": 100
        }
    }


def _node_ids_from_edges(edges_df):
    all_codes = edges_df["source"].tolist() + edges_df["target"].tolist()
    return set(all_codes)


def _filter_edges_by_focus(edges_df, focus_airport):
    if focus_airport == CONN_AIRPORT_ALL:
        return edges_df
    in_focus = (edges_df["source"] == focus_airport) | (edges_df["target"] == focus_airport)
    return edges_df.loc[in_focus].copy()


def _airport_meta(iata):
    meta = helper_data.AIRPORT_IATA_META.get(iata) or {}
    country = meta.get("country") or "—"
    name = format_airport_label_from_iata(iata)
    return name, country


def _build_airport_vis_nodes(node_ids, selected_country, peer_color, node_meta_extra, title_suffix):
    nodes = []
    node_meta = {}
    for iata in node_ids:
        name, country = _airport_meta(iata)
        color, size = airport_node_sizes_and_color(iata, selected_country, peer_color)
        meta = {
            "kind": "airport",
            "name": name,
            "iata": iata,
            "country": country,
            "airport_count": None,
            "airports": [],
            "airline_count": None,
            "airlines": [],
        }
        meta.update(node_meta_extra)
        node_meta[iata] = meta
        nodes.append({
            "id": iata,
            "label": airport_network_display_label(iata),
            "title": f"{name}\n{title_suffix}",
            "shape": "dot",
            "size": size,
            "color": color,
        })
    return nodes, node_meta


def _peer_lines_from_edges(edges_df, line_formatter, sort_descending=True):
    """Build per-node peer description lines from an edge table (source, target, weight)."""
    peer_map = {iata: [] for iata in _node_ids_from_edges(edges_df)}
    for source, target, weight in zip(
        edges_df["source"].tolist(),
        edges_df["target"].tolist(),
        edges_df["weight"].tolist(),
    ):
        target_name, _ = _airport_meta(target)
        source_name, _ = _airport_meta(source)
        peer_map[source].append((target, float(weight), target_name))
        peer_map[target].append((source, float(weight), source_name))

    lines_by_node = {}
    for iata, peers in peer_map.items():
        peers_sorted = sorted(peers, key=lambda item: item[1], reverse=sort_descending)
        lines_by_node[iata] = []
        for peer_iata, weight, peer_name in peers_sorted[:50]:
            lines_by_node[iata].append(line_formatter(peer_name, peer_iata, weight))
    return lines_by_node


# --------------------#
# ----- Carriers -----#
# --------------------#

def build_carriers(selected_country, focus_airport):
    routes = helper_data.graph1_merged.loc[helper_data.graph1_merged["country"] == selected_country].copy()
    if routes.empty:
        return _no_data("No data available for this selection")

    top_airline_list = routes["airline"].value_counts().head(15).index.tolist()
    routes = routes.loc[routes["airline"].isin(top_airline_list)].copy()

    if focus_airport != CONN_AIRPORT_ALL:
        routes = routes.loc[routes["airport"] == focus_airport].copy()
        if routes.empty:
            return _no_data(
                "No carrier routes for this airport in the current top-airline view."
            )

    airline_list = routes["airline"].unique().tolist()
    airport_list = routes["airport"].unique().tolist()

    nodes = []
    edges = []
    node_meta = {}

    for airline in airline_list:
        legal_name = helper_data.AIRLINE_IATA_TO_NAME.get(airline)
        label = legal_name if legal_name else airline
        airport_codes = routes.loc[routes["airline"] == airline, "airport"].unique().tolist()
        airport_codes.sort()
        airport_lines = []
        for code in airport_codes[:40]:
            airport_lines.append(format_airport_label_from_iata(code))
        node_meta[airline] = {
            "kind": "airline",
            "name": legal_name or "—",
            "iata": airline,
            "airport_count": len(airport_codes),
            "airports": airport_lines,
        }
        nodes.append({
            "id": airline,
            "label": label,
            "title": (
                f"{legal_name or 'Unknown name'}\n"
                f"IATA airline code: {airline}\n"
                "Yellow nodes are airports in the selected country."
            ),
            "color": NETWORK_PEER_NODE_COLOR["carriers_airline"],
            "shape": "dot",
            "size": 20,
            "group": "airline",
        })

    for airport in airport_list:
        ap_name, ap_country = _airport_meta(airport)
        airline_codes = routes.loc[routes["airport"] == airport, "airline"].unique().tolist()
        airline_codes.sort()
        airline_labels = []
        for code in airline_codes:
            nm = helper_data.AIRLINE_IATA_TO_NAME.get(code)
            airline_labels.append(f"{nm} ({code})" if nm else str(code))
        node_meta[airport] = {
            "kind": "airport",
            "name": ap_name,
            "iata": airport,
            "country": ap_country or selected_country,
            "airline_count": len(airline_codes),
            "airlines": airline_labels,
        }
        color, size = airport_node_sizes_and_color(
            airport, selected_country, NETWORK_CENTRAL_NODE_COLOR
        )
        nodes.append({
            "id": airport,
            "label": airport_network_display_label(airport),
            "title": format_airport_label_from_iata(airport),
            "color": color,
            "shape": "dot",
            "size": size,
            "group": "airport",
        })

    for airline, airport in zip(routes["airline"].tolist(), routes["airport"].tolist()):
        edges.append({
            "from": airline,
            "to": airport,
            "title": f"{airline} → {airport}",
        })

    n_airlines = len(airline_list)
    n_airports = len(airport_list)
    n_links = len(edges)

    hub_counts = routes.groupby("airport")["airline"].nunique().sort_values(ascending=False)
    if len(hub_counts) > 0:
        hub_iata = hub_counts.index[0]
        hub_n = int(hub_counts.iloc[0])
        hub_name, _ = _airport_meta(hub_iata)
        most_airlines_val = f"{hub_name} ({hub_n} airlines)"
    else:
        most_airlines_val = "—"

    if focus_airport != CONN_AIRPORT_ALL:
        dom_name, dom_pct = dominant_airline_share_at_iata(focus_airport)
        if dom_name and dom_pct is not None:
            dominant_val = f"{dom_name} ({dom_pct:.1%})"
        else:
            dominant_val = "—"
        metric_cards = _metric_row_two(
            f"Carriers · {selected_country}",
            f"{n_airlines} airlines (top)",
            "Dominant Airline",
            dominant_val,
        )
    else:
        metric_cards = _metric_row_four(
            f"Carriers · {selected_country}",
            f"{n_airlines} airlines (top)",
            "Airports in view",
            n_airports,
            "Airport–airline links",
            n_links,
            "Airport with most airlines",
            most_airlines_val,
        )

    net = visdcc.Network(
        id="network",
        selection={"nodes": [], "edges": []},
        data={"nodes": nodes, "edges": edges},
        options=carrier_network_physics_options(),
    )
    chart = wrap_network_chart(
        net,
        [
            ("Airports in selected country", NETWORK_CENTRAL_NODE_COLOR),
            ("Airlines", NETWORK_PEER_NODE_COLOR["carriers_airline"]),
        ],
    )
    return chart, node_meta, metric_cards


# ---------------------------------- #
# ----- Statistical similarity ----- #
# ---------------------------------- #

def build_similarity(selected_country, focus_airport):
    edges = helper_data.graph2_merged.loc[helper_data.graph2_merged["source_country"] == selected_country].copy()
    edges = _filter_edges_by_focus(edges, focus_airport)
    if edges.empty:
        return _no_data(
            "No similarity data for this selection (try another airport or “All airports”)."
        )

    edges = apply_similarity_scores(edges)
    if focus_airport == CONN_AIRPORT_ALL:
        cutoff = edges["weight"].quantile(0.75)
        edges = edges.loc[edges["weight"] > cutoff].copy()
    edges = edges.groupby(["source", "target"], as_index=False)["weight"].max()
    edges = edges.sort_values("weight", ascending=False).head(MAX_NETWORK_EDGES)

    node_ids = _node_ids_from_edges(edges)
    peer_color = NETWORK_PEER_NODE_COLOR["similarity"]
    nodes, node_meta = _build_airport_vis_nodes(
        node_ids,
        selected_country,
        peer_color,
        {"similarity_airport": True},
        "Statistical similarity (0-1 score on link)",
    )

    max_weight = float(edges["weight"].max()) if len(edges) else 1.0
    vis_edges = []
    for source, target, weight in zip(
        edges["source"].tolist(),
        edges["target"].tolist(),
        edges["weight"].tolist(),
    ):
        vis_edges.append({
            "from": source,
            "to": target,
            "width": max(1, float(weight) / max_weight * 10),
            "title": f"Similarity score: {float(weight):.3f}",
            "color": {"color": "#818CF8", "opacity": 0.82},
        })

    mean_score = round(float(edges["weight"].mean()), 3) if len(edges) else "—"
    links_val = metric_capped_if_needed(len(vis_edges))
    strongest = strongest_similarity_label(edges)
    metric_cards = _metric_row_four(
        "Similarity links (shown)",
        links_val,
        "# countries in view",
        countries_in_node_set(node_ids),
        "Mean similarity score",
        mean_score,
        "Strongest link",
        strongest,
    )

    net = visdcc.Network(
        id="network",
        selection={"nodes": [], "edges": []},
        data={"nodes": nodes, "edges": vis_edges},
        options=_standard_vis_options(),
    )
    chart = wrap_network_chart(
        net,
        [
            ("Airports in selected country", NETWORK_CENTRAL_NODE_COLOR),
            ("Airports in other countries", peer_color),
        ],
    )
    return chart, node_meta, metric_cards


# --------------------- #
# ----- Proximity ----- #
# --------------------- #

def build_proximity(selected_country, focus_airport):
    edges = helper_data.graph3_merged.loc[helper_data.graph3_merged["source_country"] == selected_country].copy()
    edges = _filter_edges_by_focus(edges, focus_airport)
    if edges.empty:
        return _no_data(
            "No proximity data for this selection (try another airport or “All airports”)."
        )

    if focus_airport == CONN_AIRPORT_ALL:
        cutoff = edges["weight"].quantile(0.40)
        edges = edges.loc[edges["weight"] <= cutoff].copy()
    edges = edges.groupby(["source", "target"], as_index=False)["weight"].min()
    edges = edges.sort_values("weight", ascending=True).head(MAX_NETWORK_EDGES)

    node_ids = _node_ids_from_edges(edges)
    peer_color = NETWORK_PEER_NODE_COLOR["proximity"]
    neighbor_lines = _peer_lines_from_edges(
        edges,
        lambda name, code, miles: f"{name}: {miles:.1f} mi",
        sort_descending=False,
    )

    nodes = []
    node_meta = {}
    for iata in node_ids:
        name, country = _airport_meta(iata)
        color, size = airport_node_sizes_and_color(iata, selected_country, peer_color)
        node_meta[iata] = {
            "kind": "airport",
            "name": name,
            "iata": iata,
            "country": country,
            "proximity_neighbors": neighbor_lines.get(iata, []),
        }
        nodes.append({
            "id": iata,
            "label": airport_network_display_label(iata),
            "title": (
                f"{name}\n"
                f"Edges: within {helper_data.PROXIMITY_EDGE_MAX_MILES:.0f} mi (data cap)"
            ),
            "color": color,
            "shape": "dot",
            "size": size,
        })

    max_mi = float(edges["weight"].max()) if len(edges) else 1.0
    min_mi = float(edges["weight"].min()) if len(edges) else 1.0
    span = max(max_mi - min_mi, 1e-6)
    vis_edges = []
    for source, target, miles in zip(
        edges["source"].tolist(),
        edges["target"].tolist(),
        edges["weight"].tolist(),
    ):
        miles = float(miles)
        width = max(1.0, (max_mi - miles) / span * 9 + 1)
        vis_edges.append({
            "from": source,
            "to": target,
            "width": width,
            "title": f"Distance: {miles:.1f} mi (≤{helper_data.PROXIMITY_EDGE_MAX_MILES:.0f} mi layer)",
            "color": {"color": "#4ADE80", "opacity": 0.78},
        })

    mean_mi = round(float(edges["weight"].mean()), 1) if len(edges) else "—"
    links_val = metric_capped_if_needed(len(vis_edges))
    if len(edges) > 0:
        first = edges.iloc[0]
        shortest = (
            f"{round(float(first['weight']), 1)} "
            f"({first['source']}–{first['target']})"
        )
    else:
        shortest = "—"

    metric_cards = _metric_row_four(
        "Proximity links (shown)",
        links_val,
        "# countries in view",
        countries_in_node_set(node_ids),
        "Mean link distance (mi)",
        mean_mi,
        "Shortest shown link (mi)",
        shortest,
    )

    net = visdcc.Network(
        id="network",
        selection={"nodes": [], "edges": []},
        data={"nodes": nodes, "edges": vis_edges},
        options=_standard_vis_options(-22000, 160),
    )
    chart = wrap_network_chart(
        net,
        [
            ("Airports in selected country", NETWORK_CENTRAL_NODE_COLOR),
            ("Airports in other countries", peer_color),
        ],
    )
    return chart, node_meta, metric_cards


# ------------------------------------------------------ #
# ----- Shared destinations (raw and hub-adjusted) ----- #
# ------------------------------------------------------ #

def _build_shared_destinations(
    selected_country,
    focus_airport,
    load_edges_fn,
    peer_color_key,
    node_meta_key,
    edge_title_fn,
    edge_color,
    title_suffix,
    mean_label,
    strongest_label,
    strongest_fn,
    mean_decimals,
):
    edges = load_edges_fn()
    edges = edges.loc[edges["source_country"] == selected_country].copy()
    edges = _filter_edges_by_focus(edges, focus_airport)
    if edges.empty:
        return None

    if focus_airport == CONN_AIRPORT_ALL:
        cutoff = edges["weight"].quantile(0.75)
        edges = edges.loc[edges["weight"] > cutoff].copy()
    edges = edges.groupby(["source", "target"], as_index=False)["weight"].max()
    edges = edges.sort_values("weight", ascending=False).head(MAX_NETWORK_EDGES)

    node_ids = _node_ids_from_edges(edges)
    peer_color = NETWORK_PEER_NODE_COLOR[peer_color_key]

    if node_meta_key == "shared_dest_peers":
        line_fn = lambda name, code, w: (
            f"{name}: {int(w)}"
        )
    else:
        line_fn = lambda name, code, w: f"{name}: {float(w):.3f}"

    peer_lines = _peer_lines_from_edges(edges, line_fn, sort_descending=True)

    nodes = []
    node_meta = {}
    for iata in node_ids:
        name, country = _airport_meta(iata)
        color, size = airport_node_sizes_and_color(iata, selected_country, peer_color)
        meta = {
            "kind": "airport",
            "name": name,
            "iata": iata,
            "country": country,
        }
        meta[node_meta_key] = peer_lines.get(iata, [])
        node_meta[iata] = meta
        nodes.append({
            "id": iata,
            "label": airport_network_display_label(iata),
            "title": f"{name}\n{title_suffix}",
            "color": color,
            "shape": "dot",
            "size": size,
        })

    max_w = float(edges["weight"].max()) if len(edges) else 1.0
    vis_edges = []
    for source, target, weight in zip(
        edges["source"].tolist(),
        edges["target"].tolist(),
        edges["weight"].tolist(),
    ):
        w = float(weight)
        vis_edges.append({
            "from": source,
            "to": target,
            "width": max(1, w / max_w * 10),
            "title": edge_title_fn(w),
            "color": edge_color,
        })

    if mean_decimals == 1:
        mean_val = round(float(edges["weight"].mean()), 1) if len(edges) else "—"
    else:
        mean_val = round(float(edges["weight"].mean()), 3) if len(edges) else "—"

    links_val = metric_capped_if_needed(len(vis_edges))
    strongest = strongest_fn(edges)
    metric_cards = _metric_row_four(
        "Links (shown)",
        links_val,
        "# countries in view",
        countries_in_node_set(node_ids),
        mean_label,
        mean_val,
        strongest_label,
        strongest,
    )

    net = visdcc.Network(
        id="network",
        selection={"nodes": [], "edges": []},
        data={"nodes": nodes, "edges": vis_edges},
        options=_standard_vis_options(),
    )
    chart = wrap_network_chart(
        net,
        [
            ("Airports in selected country", NETWORK_CENTRAL_NODE_COLOR),
            ("Airports in other countries", peer_color),
        ],
    )
    return chart, node_meta, metric_cards


def build_shared_destinations_raw(selected_country, focus_airport):
    result = _build_shared_destinations(
        selected_country,
        focus_airport,
        helper_data.load_merged_shared_destinations_edges,
        "shared_raw",
        "shared_dest_peers",
        lambda w: f"Raw overlap: {int(w)} shared destinations",
        {"color": "#E879F9", "opacity": 0.85},
        "Edges: raw count of destinations served by both airports",
        "Mean shared count",
        "Strongest pair (count)",
        strongest_weight_count_pair,
        mean_decimals=1,
    )
    if result is None:
        return _no_data(
            "No shared-destination data for this selection (try another airport or “All airports”)."
        )
    return result


def build_shared_destinations_adjusted(selected_country, focus_airport):
    result = _build_shared_destinations(
        selected_country,
        focus_airport,
        helper_data.load_merged_shared_destinations_cosine_edges,
        "shared_cosine",
        "shared_cosine_peers",
        lambda w: f"Hub-adjusted destination overlap: {float(w):.3f}",
        {"color": "#FDA4AF", "opacity": 0.85},
        "Cosine similarity of destination sets (size-adjusted)",
        "Mean adjusted score",
        "Strongest score",
        strongest_adjusted_score_label,
        mean_decimals=3,
    )
    if result is None:
        return _no_data(
            "No hub-adjusted destination overlap for this selection "
            "(try another airport or “All airports”)."
        )
    return result


# Entry point

def build_network_connection(connection_type, selected_country, focus_airport):
    focus_airport = focus_airport or CONN_AIRPORT_ALL

    if connection_type == "Carriers":
        return build_carriers(selected_country, focus_airport)
    if connection_type == "Statistical Similarity":
        return build_similarity(selected_country, focus_airport)
    if connection_type == "Proximity":
        return build_proximity(selected_country, focus_airport)
    if connection_type == "Shared Destinations":
        return build_shared_destinations_raw(selected_country, focus_airport)
    if connection_type == "Shared Destinations (Hub-Adjusted)":
        return build_shared_destinations_adjusted(selected_country, focus_airport)

    return (
        html.Div("Select a valid connection type", style=NETWORK_EMPTY_STYLE),
        {},
        _empty_metric_cards(),
    )
