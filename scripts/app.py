import pandas as pd
import numpy as np
import os
import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dash_table
import plotly.graph_objects as go
import math
import visdcc

#-----Read in and set up data
airport_df = pd.read_parquet('https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/master_air.parquet')

#-----One lookup for IATA to country (avoid huge merge joins on edge tables with 1000s of rows)
IATA_TO_COUNTRY = (
    airport_df.drop_duplicates(subset=["iata"])
    .set_index("iata")["country"]
)


def attach_country_columns_to_edges(edges_df):
    out = edges_df.copy()
    out["source_country"] = out["source"].map(IATA_TO_COUNTRY)
    out["target_country"] = out["target"].map(IATA_TO_COUNTRY)
    return out


graph1 = pd.read_parquet('https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/edges_airline_airport.parquet')
airport_df1 = airport_df[['iata','country','display_name']].drop_duplicates()
airport_df1.rename(columns={'iata':'airport'}, inplace=True)
graph1_merged = pd.merge(graph1, airport_df1, on ='airport')


def build_airline_iata_to_name(airport_df):
    mapping = {}
    for carriers in airport_df["carriers"].dropna():
        if not isinstance(carriers, (list, tuple, np.ndarray)):
            continue
        for c in carriers:
            if not isinstance(c, dict):
                continue
            code = c.get("iata") or c.get("IATA") or c.get("airline_iata") or c.get("code")
            name = c.get("name")
            if code and name:
                code = str(code).strip().upper()
                if len(code) == 2:
                    mapping[code] = name
    return mapping


AIRLINE_IATA_TO_NAME = build_airline_iata_to_name(airport_df)
AIRPORT_IATA_META = (
    airport_df.drop_duplicates(subset=["iata"])
    .set_index("iata")[["display_name", "country"]]
    .to_dict("index")
)

def build_similarity_airport_profiles():
    ap = airport_df.drop_duplicates(subset=["iata"]).copy()
    ap["num_dests"] = pd.to_numeric(ap["num_dests"], errors="coerce").fillna(0).astype(int)
    ap["connectivity_index"] = pd.to_numeric(ap["connectivity_index"], errors="coerce").fillna(0.0)
    ap["redundancy_score"] = pd.to_numeric(ap["redundancy_score"], errors="coerce").fillna(0.0)
    ap["log1p_num_dests"] = np.log1p(ap["num_dests"].to_numpy(dtype=float))
    z_cols = ["connectivity_index", "log1p_num_dests", "redundancy_score"]
    mat = ap[z_cols].to_numpy(dtype=float)
    mu = mat.mean(axis=0)
    sig = mat.std(axis=0)
    sig[sig == 0] = 1.0
    z_mat = (mat - mu) / sig
    profiles = {}
    for i, iata in enumerate(ap["iata"].astype(str)):
        profiles[iata] = {
            "connectivity_index": float(mat[i, 0]),
            "redundancy_score": float(mat[i, 2]),
            "num_dests": int(ap["num_dests"].iloc[i]),
            "z_connectivity_index": float(z_mat[i, 0]),
            "z_log1p_num_dests": float(z_mat[i, 1]),
            "z_redundancy_score": float(z_mat[i, 2]),
        }
    return profiles


SIMILARITY_AIRPORT_PROFILE = build_similarity_airport_profiles()

SIMILARITY_Z_FEATURE_KEYS = (
    "z_connectivity_index",
    "z_log1p_num_dests",
    "z_redundancy_score",
)


graph2 = pd.read_parquet("https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/edges_feature_similarity.parquet")
graph2_merged = attach_country_columns_to_edges(graph2)

graph3 = pd.read_parquet("https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/edges_proximity.parquet")
graph3_merged = attach_country_columns_to_edges(graph3)

# Upper bound used when building edges_proximity.parquet (great-circle miles).
PROXIMITY_EDGE_MAX_MILES = 250.0

SHARED_DESTINATIONS_PARQUET_URL = (
    "https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/"
    "edges_shared_destinations.parquet"
)

SHARED_DESTINATIONS_COSINE_PARQUET_URL = (
    "https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/"
    "edges_shared_destinations_cosine.parquet"
)

#-----Loaded on first use only; keeps app startup responsive
merged_shared_destinations_edges = None
merged_shared_destinations_cosine_edges = None


def load_merged_shared_destinations_edges():
    """Load shared-destination edges once; subsequent calls reuse the same DataFrame."""
    global merged_shared_destinations_edges
    if merged_shared_destinations_edges is None:
        raw = pd.read_parquet(SHARED_DESTINATIONS_PARQUET_URL)
        merged_shared_destinations_edges = attach_country_columns_to_edges(raw)
    return merged_shared_destinations_edges


def load_merged_shared_destinations_cosine_edges():
    """Load cosine-weighted shared-destination edges once; subsequent calls reuse the same DataFrame."""
    global merged_shared_destinations_cosine_edges
    if merged_shared_destinations_cosine_edges is None:
        raw = pd.read_parquet(SHARED_DESTINATIONS_COSINE_PARQUET_URL)
        merged_shared_destinations_cosine_edges = attach_country_columns_to_edges(raw)
    return merged_shared_destinations_cosine_edges


dest_tbl = (
    airport_df[["display_name", "dest_name",'dest_iata', "connectivity_index", "redundancy_score"]].rename(
        columns={
            "dest_name": "Destination",
            "connectivity_index": "Connectivity Index",
            "redundancy_score": "Redundancy Index",
        }
    )
)

# ----- Set up choices for dropdown menus
country_choices = sorted(airport_df['country'].unique())
airport_choices = sorted(airport_df['display_name'].unique())
connection_type_choices = [
    "Carriers",
    "Statistical Similarity",
    "Proximity",
    "Shared Destinations",
    "Shared Destinations (Hub-Adjusted)",
]

#----- Label defintions for connection types
CONNECTION_TYPE_DEFINITIONS = [
    (
        "Carriers",
        "Airlines linked to airports they fly to within the selected country. "
        "Useful for seeing which carriers cluster at which hubs (after the top-carrier filter).",
    ),
    (
        "Statistical Similarity",
        "Airports linked when connectivity index, redundancy score, and destination count "
        "are alike (z-scored, then compared). Link scores are 0–1 (higher = more similar).",
    ),
    (
        "Proximity",
        "Airports linked by short distance (within the proximity edge cap in the dataset). "
        "Shows geographic neighbors, not airline overlap.",
    ),
    (
        "Shared Destinations",
        "Airports linked by how many destination cities they both serve (same destination IATA in both route lists). "
        "The weight is the raw count — large hubs usually share more destinations simply because they fly everywhere.",
    ),
    (
        "Shared Destinations (Hub-Adjusted)",
        "Same underlying overlap as the raw count, but each link is scaled down when either airport serves "
        "a very large number of destinations (cosine-style adjustment). "
        "Makes overlap between a mega-hub and a mid-size airport easier to interpret fairly.",
    ),
]

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


def build_conn_metric_cards_row(*card_bodies):
    if not card_bodies:
        return html.Div()
    col_width = 12 // len(card_bodies)
    return dbc.Row([
        dbc.Col(dbc.Card(body, style=METRIC_CARD_STYLE), width=col_width)
        for body in card_bodies
    ])


def dominant_airline_share_at_iata(iata):
    """Dominant carrier by frequency in airport carrier lists (airline_stats tab logic)."""
    ap_rows = airport_df[airport_df["iata"] == iata]
    if ap_rows.empty:
        return None, None
    names = ap_rows["carriers"].apply(extract_airline_names).explode().dropna()
    if names.empty:
        return None, None
    counts = names.value_counts()
    return counts.index[0], float(counts.iloc[0] / counts.sum())

# ----- Define Hover columns to be used in map over dots
HOVER_COLS = ['latitude', 'longitude', 'display_name', 'num_dests', 'redundancy_score', 'connectivity_index']


#----- Counry --> Airport Dictionary
df_for_dict = airport_df[['country','display_name']]
df_for_dict = df_for_dict.drop_duplicates(subset='display_name',keep='first')
country_airport_dict = df_for_dict.groupby('country')['display_name'].apply(list).to_dict()


#----- Define style for different pages in app
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'color':'white',
    'backgroundColor': '#222222'

}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#626ffb',
    'color': 'white',
    'padding': '6px'
}

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


app = dash.Dash(__name__,assets_folder=os.path.join(os.curdir,"assets"))
server = app.server
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Welcome',value='tab-1',style=tab_style, selected_style=tab_selected_style,
               children=[
                   html.Div([
                       html.H1(dcc.Markdown('''**Welcome to my Airport Accessibility Dashboard!**''')),
                       html.Br()
                   ]),
                   
                   html.Div([
                        html.P(dcc.Markdown('''**What is the purpose of this dashboard?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   html.Div([
                       html.P("This dashboard served two purposes for me:",style={'color':'white'}),
                       html.P("1.) Revisit and expand upon an analysis I originally conducted six years ago",style={'color':'white'}),
                       html.P("2.) Gain hands-on experience building and analyzing graph-structured datasets",style={'color':'white'}),
                       html.Br()
                   ]),
                   html.Div([
                       html.P(dcc.Markdown('''**What data is being used for this analysis?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   
                   html.Div([
                       html.P(["The data for this analysis was pulled from this ",html.A('Github repository',href='https://github.com/Jonty/airline-route-data'), "."],style={'color':'white'}),
                       html.Br()
                   ]),
                   html.Div([
                       html.P(dcc.Markdown('''**What are the limitations of this data?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   html.Div([
                       html.P("1.) While this dataset is very extensive, it is not an exhaustive list of every commercial airport in the world.",style={'color':'white'}),
                       html.P("2.) The primary metrics used in this dashboard, connectivity and redundancy scores, were developed by me for this analysis and should be viewed as exploratory measures.",style={'color':'white'}),

                   ])


               ]),
        dcc.Tab(label='Airport Metrics',value='tab-2',style=tab_style, selected_style=tab_selected_style,
            children=[
                #----- Modal Instructions 1
                html.Div([
                    dbc.Button(
                        "Click Here for Instructions",
                        id="open1",
                        color="secondary",
                        className="w-100",
                        style={"fontSize": 18},
                    ),
                    dbc.Modal([
                        dbc.ModalHeader(
                            html.Span("Instructions"),
                            className="text-white border-secondary",
                            style={"backgroundColor": "#000000"},
                            close_button=False,
                        ),
                        dbc.ModalBody(
                            style=INSTRUCTIONS_MODAL_BODY_STYLE,
                            children=[
                                html.P(["Below this button, use the dropdown menu on the left to select a country, then the dropdown menu on the right to select an airport.  The map showcases all the destinations one can fly to from the selected airport.  Below the map is a table showcasing information on those destinations.  The following is a description of the 2 primary metrics in this dashboard:"],style=INSTRUCTIONS_INTRO_STYLE,),
                                html.H4("Connectivity index (0-100)", style={"marginBottom": "8px", "marginTop": "4px", "color": "#ffffff"}),
                                html.P(html.Strong("What goes into it"), style={"marginBottom": "6px"}),
                                html.Ul([
                                    html.Li("The number of distinct destinations this airport serves."),
                                    html.Li([
                                        "A tier from that count, using the same cutoffs for every airport: ",
                                        inst_code("<10 destinations: tier 1; 10-24: 2; 25-49: 3; 50-99: 4; 100+: 5"),
                                        ".",
                                    ]),
                                    html.Li([
                                        "An airport-side measure: ",
                                        inst_code("(destinations served) X (this airport's tier)"),
                                        ".",
                                    ]),
                                    html.Li("A destination-side total: for each destination this airport serves, take that destination's own number of destinations served, multiply by its tier using the same cutoffs, then add those products across all destinations on this airport's list."),
                                    html.Li([
                                        "A weighted logarithmic combination: ",
                                        inst_code("0.75 X log(1 + airport-side measure) + 0.25 X log(1 + destination-side total)"),
                                        ". The logarithm limits the influence of very large counts.",
                                    ]),
                                ],style={"marginBottom": "14px", "paddingLeft": "20px", "lineHeight": 1.55}),
                                html.P(html.Strong("How it is calculated"), style={"marginBottom": "6px"}),
                                html.Ol([
                                    html.Li("Assign the tier from the airport's destination count, multiply that count by the tier, and record the product as the airport-side measure."),
                                    html.Li("For each destination on the airport's route list, assign the tier from that destination's destination count, multiply count by tier, sum across destinations to obtain the destination-side total."),
                                    html.Li([
                                        "Evaluate ",
                                        inst_code("0.75 X log(1 + airport-side measure) + 0.25 X log(1 + destination-side total)"),
                                        ".",
                                    ]),
                                    html.Li([
                                        "Apply ",
                                        inst_code("min-max"),
                                        " scaling across all airports so the result is standardized as a ",
                                        inst_code("0-100"),
                                        " connectivity index.",
                                    ]),
                                ],style={"marginBottom": "24px", "paddingLeft": "20px", "lineHeight": 1.55}),
                                html.H4(
                                    "Redundancy score (0-100)",
                                    style={"marginBottom": "8px", "color": "#ffffff"},
                                ),
                                html.P(html.Strong("What goes into it"), style={"marginBottom": "6px"}),
                                html.Ul([
                                    html.Li("Distance in miles to the nearest commercial airport (closer alternates receive a higher contribution after scaling)."),
                                    html.Li([
                                        "Counts of other airports within 10, 25, 50, 100, 250, and 500 miles, combined as ",
                                        inst_code("4X(within 10 mi) + 3X(within 25) + 2X(within 50) + 1X(within 100) + 0.5X(within 250) + 0.25X(within 500)"),
                                        ", then take the natural logarithm of one plus that total.",
                                    ]),
                                    html.Li("Average route length in miles from this airport (shorter averages receive a higher contribution after inverse scaling)."),
                                    html.Li([
                                        "This airport's ",
                                        inst_code("connectivity index (0-100)"),
                                        ".",
                                    ]),
                                ], style={"marginBottom": "14px", "paddingLeft": "20px", "lineHeight": 1.55}),
                                html.P(html.Strong("How it is calculated"), style={"marginBottom": "6px"}),
                                html.Ol([
                                    html.Li("For each airport, evaluate the four quantities listed above independently."),
                                    html.Li([
                                        "Rescale each quantity separately to ",
                                        inst_code("[0, 1]"),
                                        " using ",
                                        inst_code("min-max"),
                                        " scaling across all airports.",
                                    ]),
                                    html.Li([
                                        "Form a weighted mean of the four rescaled values: ",
                                        inst_code("0.25X(nearest airport) + 0.30X(nearby-airport density) + 0.20X(average route length) + 0.25X(connectivity index)"),
                                        ".",
                                    ]),
                                    html.Li([
                                        "Apply ",
                                        inst_code("min-max"),
                                        " scaling to that weighted mean across all airports so the result is reported as a ",
                                        inst_code("0-100"),
                                        " redundancy score.",
                                    ]),
                                ],style={"marginBottom": "0", "paddingLeft": "20px", "lineHeight": 1.55}),
                            ],
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close1", className="ml-auto"),
                            className="border-secondary",
                            style={"backgroundColor": "#000000"},
                        )
                    ], id="modal1", size="xl", scrollable=True, content_style={"backgroundColor": "#000000"})
                ], className="w-100"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose a country:'),
                        dcc.Dropdown(
                            id='dropdown1',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in country_choices],
                            value=country_choices[0]
                        )
                    ], width = 6),
                    dbc.Col([
                        dbc.Label('Choose an airport:'),
                        dcc.Dropdown(
                            id='dropdown2',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in airport_choices],
                            value=airport_choices[0]
                        )
                    ], width = 6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(id='card1', style=METRIC_CARD_STYLE)
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card2', style=METRIC_CARD_STYLE)
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card3', style=METRIC_CARD_STYLE)
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card4', style=METRIC_CARD_STYLE)
                    ],width=3),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='destination_map')
                    ], width = 12),
                ]),
                dbc.Row([
                    dbc.Col([
                        dash_table.DataTable(
                            id="dest_tbl",
                            page_action="native",
                            page_size=20,
                            style_table={
                                "overflowX": "auto",
                                "overflowY": "auto"
                            }
                        )
                    ], width = 12)
                ])
            ]
        ),
        dcc.Tab(label='Airline Metrics',value='tab-3',style=tab_style, selected_style=tab_selected_style,
            children=[
                #----- Modal Instructions 2
                html.Div([
                    dbc.Button(
                        "Click Here for Instructions",
                        id="open2",
                        color="secondary",
                        className="w-100",
                        style={"fontSize": 18},
                    ),
                    dbc.Modal([
                        dbc.ModalHeader(
                            html.Span("Instructions"),
                            className="text-white border-secondary",
                            style={"backgroundColor": "#000000"},
                            close_button=False,
                        ),
                        dbc.ModalBody(
                            style=INSTRUCTIONS_MODAL_BODY_STYLE,
                            children=[
                                html.P("Below this button, use the dropdown menu on the left to select a country, then the dropdown menu on the right to select an airport.", style=INSTRUCTIONS_INTRO_STYLE),
                                html.P("The chart on the left showcases the dominant airlines by city for the selected country (defined as the majority of flights are operated from this airline).  The map has a default view of showing the dominant airline for all cities in the selected country.  Choose 'Airline % Share' to get an airline specific view of the airline usage per city.  You can also change which airline frequency is displayed on the map.", style=INSTRUCTIONS_INTRO_STYLE),
                                html.P("The chart on the right showcases the available airlines in the selected airport.", style=INSTRUCTIONS_INTRO_STYLE),
                            ]
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close2", className="ml-auto"),
                            className="border-secondary",
                            style={"backgroundColor": "#000000"},
                        )
                    ], id="modal2", size="xl", scrollable=True, content_style={"backgroundColor": "#000000"})
                ], className="w-100"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose a country:'),
                        dcc.Dropdown(
                            id='dropdown3',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in country_choices],
                            value=country_choices[0]
                        )
                    ], width = 6),
                    dbc.Col([
                        dbc.Label('Choose an airport:'),
                        dcc.Dropdown(
                            id='dropdown4',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in airport_choices],
                            value=airport_choices[0]
                        )
                    ], width = 6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            id='card5',
                            style=METRIC_CARD_STYLE
                        )
                    ],width=3),
                    dbc.Col([
                        dbc.Card(
                            id='card6',
                            style=METRIC_CARD_STYLE
                        )
                    ],width=3),
                    dbc.Col([
                        dbc.Card(
                            id='card7',
                            style=METRIC_CARD_STYLE
                        )
                    ],width=3),
                    dbc.Col([
                        dbc.Card(
                            id='card8',
                            style=METRIC_CARD_STYLE
                        )
                    ],width=3),
                ]),
                dcc.Store(id='airline_map_mode', data='All Airlines'),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div([
                                dbc.Label(
                                    'Map view:',
                                    style={'marginBottom': '6px', 'fontWeight': '600'}
                                ),
                                dcc.RadioItems(
                                    id='airline_map_mode_all',
                                    options=[
                                        {'label': 'All Airlines', 'value': 'All Airlines'},
                                    ],
                                    value='All Airlines',
                                    inputStyle={'marginRight': '6px'},
                                    labelStyle=AIRLINE_MAP_MODE_PICKER_LABEL_STYLE,
                                ),
                            ], style={'minWidth': 0}),
                            html.Div([
                                dbc.Label(
                                    '\u00a0',
                                    style={'marginBottom': '6px', 'fontWeight': '600'}
                                ),
                                dcc.RadioItems(
                                    id='airline_map_mode_share',
                                    options=[
                                        {'label': 'Airline % Share', 'value': 'Airline % Share'},
                                    ],
                                    value=None,
                                    inputStyle={'marginRight': '6px'},
                                    labelStyle=AIRLINE_MAP_MODE_PICKER_LABEL_STYLE,
                                ),
                            ], style={'minWidth': 0}),
                            html.Div([
                                dbc.Label(
                                    'Heat map airline:',
                                    style={'marginBottom': '6px', 'fontWeight': '600'}
                                ),
                                dcc.Dropdown(
                                    id='dropdown5',
                                    style={'color': 'black'},
                                    clearable=False
                                )
                            ], id='heatmap_airline_control', style={'display': 'none', 'minWidth': 0}),
                        ], id='airline_map_controls_row', style=airline_map_controls_row_style('1fr 1fr')),
                        html.Div(
                            dcc.Graph(
                                id='dominant_airline_by_country_map',
                                style={'height': '100%', 'width': '100%'},
                            ),
                            style=AIRLINE_MAP_GRAPH_WRAP_STYLE,
                        ),
                    ], width=6, style=AIRLINE_MAP_COLUMN_STYLE),
                    dbc.Col([
                        html.Div(
                            dcc.Graph(
                                id='airline_treemap',
                                style={'height': '100%', 'width': '100%'},
                            ),
                            style=AIRLINE_TREEMAP_GRAPH_WRAP_STYLE,
                        ),
                    ], width=6, style=AIRLINE_TREEMAP_COLUMN_STYLE),
                ], style=AIRLINE_CHARTS_ROW_STYLE)
            ]
        ),
        dcc.Tab(label='Connections',value='tab-4',style=tab_style, selected_style=tab_selected_style,
            children=[
                dcc.Store(id='network-node-meta', data={}),
                #----- Modal Instructions 3
                html.Div([
                    dbc.Button(
                        "Click Here for Instructions",
                        id="open3",
                        color="secondary",
                        className="w-100",
                        style={"fontSize": 18},
                    ),
                    dbc.Modal([
                        dbc.ModalHeader(
                            html.Span("Instructions"),
                            className="text-white border-secondary",
                            style={"backgroundColor": "#000000"},
                            close_button=False,
                        ),
                        dbc.ModalBody(
                            style=INSTRUCTIONS_MODAL_BODY_STYLE,
                            children=[
                                html.P(["The network charts below show relationships between airports and airlines which are altered by the selections from the 3 dropdown menus.  Click a node to read details of the connection in the panel to the right of the graph. "],style=INSTRUCTIONS_INTRO_STYLE),
                                html.P(["Below the button, use the dropdown menu on the far left to choose a graph connection type.  Use the dropdown menu in the middle to select a country.  If desired, use the dropdown menu on the far right to select an airport, otherwise the default value for this dropdown showcases all airports for the selected country."]), 
                                html.P(["Use the panel on the right for a description of what each connection type represents.  Each connection type has a series of metrics that populate upon the dropdown selection.  Here is a summary of those metrics:"]),
                                html.H4("Carriers", style={"marginBottom": "8px", "marginTop": "4px", "color": "#ffffff"}),
                                html.Ul([
                                    html.Li("Metric 1: How many top airlines appear in the chart (capped at 15)"),
                                    html.Li("Metric 2: How many airports in the country are shown"),
                                    html.Li("Metric 3: How many airline-airport links connect airlines to airports in the chart"),
                                    html.Li("Metric 4: The airport in this view served by the most airlines (name and airline count)"),
                                    html.Li(
                                        "When a focus airport is selected, only 2 metrics are shown: "
                                        "Metric 1 stays the same; Metric 2 changes to Dominant Airline "
                                        "(airline with the highest carrier-list share at that airport)."
                                    ),
                                ],style={"marginBottom": "14px", "paddingLeft": "20px", "lineHeight": 1.55}),
                                html.H4("Statistical Similarity", style={"marginBottom": "8px", "marginTop": "14px", "color": "#ffffff"}),
                                html.Ul([
                                    html.Li("Metric 1: How many similarity links are shown in the chart"),
                                    html.Li("Metric 2: How many countries are represented among the airports in the chart"),
                                    html.Li("Metric 3: Average similarity score on the connections shown"),
                                    html.Li("Metric 4: Strongest similarity score among the connections shown"),
                                    html.Li(
                                        "When a focus airport is selected, Metric 4 changes to: "
                                        "strongest similarity score with the airport pair in parentheses"
                                    ),
                                ],style={"marginBottom": "14px", "paddingLeft": "20px", "lineHeight": 1.55}),
                                html.H4("Proximity",style={"marginBottom": "8px", "marginTop": "14px", "color": "#ffffff"}),
                                html.Ul([
                                    html.Li("Metric 1: How many proximity links are shown in the chart"),
                                    html.Li("Metric 2: How many countries are represented among the airports in the chart"),
                                    html.Li("Metric 3: Average distance in miles on the connections shown"),
                                    html.Li("Metric 4: Shortest distance among the connections shown, with the airport pair in parentheses"),
                                ],style={"marginBottom": "14px", "paddingLeft": "20px", "lineHeight": 1.55}),
                                html.H4("Shared Destinations", style={"marginBottom": "8px", "marginTop": "14px", "color": "#ffffff"}),
                                html.Ul([
                                    html.Li("Metric 1: How many shared-destination links are shown in the chart"),
                                    html.Li("Metric 2: How many countries are represented among the airports in the chart"),
                                    html.Li("Metric 3: Average raw shared-destination count on the connections shown"),
                                    html.Li("Metric 4: Highest raw overlap count, with the airport pair in parentheses"),
                                ], style={"marginBottom": "14px", "paddingLeft": "20px", "lineHeight": 1.55}),
                                html.H4("Shared Destinations (Hub-Adjusted)", style={"marginBottom": "8px", "marginTop": "14px", "color": "#ffffff"}),
                                html.Ul([
                                    html.Li("Metric 1: How many hub-adjusted links are shown in the chart"),
                                    html.Li("Metric 2: How many countries are represented among the airports in the chart"),
                                    html.Li("Metric 3: Average hub-adjusted score on the connections shown"),
                                    html.Li("Metric 4: Highest hub-adjusted score, with the airport pair in parentheses"),
                                ], style={"marginBottom": "14px", "paddingLeft": "20px", "lineHeight": 1.55})
                            ],
                        ),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close3", className="ml-auto"),
                            className="border-secondary",
                            style={"backgroundColor": "#000000"},
                        )
                    ], id="modal3", size="xl", scrollable=True, content_style={"backgroundColor": "#000000"})
                ], className="w-100"),

             
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Choose a connection type:", style=LABEL_STYLE_WHITE),
                        dcc.Dropdown(
                            id='dropdown6',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in connection_type_choices],
                            value=connection_type_choices[0]
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Choose a country:", style=LABEL_STYLE_WHITE),
                        dcc.Dropdown(
                            id='dropdown7',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in country_choices],
                            value=country_choices[0]
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Focus airport (optional):", style=LABEL_STYLE_WHITE),
                        dcc.Dropdown(
                            id='dropdown_conn_airport',
                            style={'color': 'black'},
                            placeholder="All airports in country",
                            clearable=False,
                        )
                    ], width=4),
                ]),
                html.Div(id='conn-metric-cards'),
                dbc.Row([
                    dbc.Col([
                        html.P(
                            "Tip: use the network toolbar to zoom and fit. "
                            "Yellow nodes are airports in the selected country; other colors are connection peers. "
                            "The hub-adjusted destination view scales overlap so mega-hubs do not dominate every link.",
                            style={
                                "color": "rgba(244,246,251,0.82)",
                                "fontSize": "0.88rem",
                                "marginTop": "4px",
                                "marginBottom": "12px",
                            },
                        )
                    ], width=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='network_chart')
                    ], width=8),
                    dbc.Col([
                        html.Div(
                            id='network-node-detail',
                            style={
                                "background": "rgba(255,255,255,0.09)",
                                "borderRadius": "12px",
                                "padding": "16px 18px",
                                "minHeight": "320px",
                                "maxHeight": "720px",
                                "overflowY": "auto",
                                "color": "#f4f6fb",
                                "border": "1px solid rgba(255,255,255,0.12)",
                                "lineHeight": "1.45",
                            }
                        )
                    ], width=4),
                ]),
            ]
        )
    ])
])

# ------------------------------------------- #
# ------ Tab #2: Route Map by Airports ------ #
# ------------------------------------------- #

@app.callback(
    Output('dropdown2', 'options'), #--> filter airports
    Output('dropdown2', 'value'),
    Input('dropdown1', 'value') #--> choose counry
)
def set_airport_options(selected_country):
    return [{'label': i, 'value': i} for i in country_airport_dict[selected_country]], country_airport_dict[selected_country][0],


@app.callback(
    Output('card1', 'children'),
    Output('card2', 'children'),
    Output('card3', 'children'),
    Output('card4', 'children'),
    Input('dropdown2', 'value')

)

def airport_selection_stats(selected_airport):
    filtered = dest_tbl[dest_tbl["display_name"] == selected_airport]
    filtered_again = airport_df[airport_df['display_name'] == selected_airport]

    #----- Grab first 2 metrics from airport_df
    metric1 = filtered_again['num_dests'].unique()[0]

    diff_dests = filtered['dest_iata'].unique()
    filtering_to_only_dests = airport_df[airport_df['iata'].isin(diff_dests)]
    metric2 = len(filtering_to_only_dests['country'].unique())

    #----- Grab last 2 metrics from destination table
    metric3 = round(filtered['Connectivity Index'].unique()[0],2)
    metric4 = round(filtered['Redundancy Index'].unique()[0],2)

    airport_code = filtered_again['iata'].unique()[0]

    card1 = build_metric_card_body(f"Destinations from {airport_code}", metric1)
    card2 = build_metric_card_body(f"# Countries Accessible from {airport_code}", metric2)
    card3 = build_metric_card_body(f"Connectivity Index for {airport_code}", metric3)
    card4 = build_metric_card_body(f"Redundancy Index for {airport_code}", metric4)
    return card1, card2, card3, card4


@app.callback(
    Output('destination_map', 'figure'),
    Input('dropdown2', 'value'),
)
def destination_map(selected_airport):
    available = [c for c in HOVER_COLS if c in airport_df.columns]
    coord_lookup = (
        airport_df.drop_duplicates(subset='iata')
        .set_index('iata')[available]
        .assign(
            latitude=lambda x: pd.to_numeric(x['latitude'], errors='coerce'),
            longitude=lambda x: pd.to_numeric(x['longitude'], errors='coerce'),
            num_dests=lambda x: pd.to_numeric(x['num_dests'], errors='coerce').fillna(0).astype(int),
            redundancy_score=lambda x: pd.to_numeric(x['redundancy_score'], errors='coerce').round(2).fillna(0),
            connectivity_index=lambda x: pd.to_numeric(x['connectivity_index'], errors='coerce').round(2).fillna(0),
        )
    )

    origin_rows = airport_df[airport_df['display_name'] == selected_airport]
    if origin_rows.empty:
        return go.Figure()

    origin_iata = origin_rows['iata'].iloc[0]
    origin_lat  = float(origin_rows['latitude'].iloc[0])
    origin_lon  = float(origin_rows['longitude'].iloc[0])

    dest_iatas = airport_df[airport_df['iata'] == origin_iata]['dest_iata'].dropna().unique()
    dest_info  = coord_lookup[coord_lookup.index.isin(dest_iatas)].reset_index()

    def make_customdata(df):
        return list(zip(
            df['display_name'],
            df['iata'],
            df['num_dests'],
            df['connectivity_index'],
            df['redundancy_score'],
        ))

    fig = go.Figure()

    # Curved great-circle lines
    for _, dest in dest_info.iterrows():
        lats, lons = great_circle_points(origin_lat, origin_lon, dest['latitude'], dest['longitude'])
        lats, lons = split_antimeridian_segments(lats, lons)
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons,
            mode='lines',
            line=dict(width=1, color='blue'),
            hoverinfo='none',
            showlegend=False,
        ))

    # Destination markers (red)
    fig.add_trace(
        go.Scattermapbox(
            lat=dest_info['latitude'],
            lon=dest_info['longitude'],
            mode='markers',
            marker=dict(size=8, color='red'),
            customdata=make_customdata(dest_info),
            hovertemplate=
                '<b>%{customdata[0]}</b><br>'
                'IATA: %{customdata[1]}<br>'
                '# Destinations: %{customdata[2]}<br>'
                'Connectivity Score: %{customdata[3]}<br>'
                'Redundancy Score: %{customdata[4]}'
                '<extra></extra>',
            name='Destinations'
        )
    )

    # Origin marker (yellow)
    origin_info = coord_lookup[coord_lookup.index == origin_iata].reset_index()
    fig.add_trace(
        go.Scattermapbox(
            lat=[origin_lat],
            lon=[origin_lon],
            mode='markers',
            marker=dict(size=14, color='yellow'),
            customdata=make_customdata(origin_info) if not origin_info.empty else [[selected_airport, origin_iata, 0, 0, 0]],
            hovertemplate=
                '<b>%{customdata[0]}</b><br>'
                'IATA: %{customdata[1]}<br>'
                '# Destinations: %{customdata[2]}<br>'
                'Connectivity Score: %{customdata[3]}<br>'
                'Redundancy Score: %{customdata[4]}'
                '<extra></extra>',
            name='Origin',
    ))

    fig.update_layout(
        height = 360,
        mapbox_style='carto-darkmatter',
        mapbox=dict(center=dict(lat=origin_lat, lon=origin_lon), zoom=2),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
        )
    )
    return fig



@app.callback(
    Output("dest_tbl", "data"),
    Output("dest_tbl", "columns"),
    Output("dest_tbl", "page_current"),
    Input("dropdown2", "value"),
)
def update_dest_table(selected_airport):
    filtered = dest_tbl[dest_tbl["display_name"] == selected_airport]
    dest_metrics = airport_df[
        ["display_name", "iata", "num_dests","redundancy_score", "connectivity_index"]
    ].drop_duplicates(subset=["display_name"])

    joined = filtered.merge(
        dest_metrics,
        left_on="dest_iata",
        right_on="iata", 
        how="left"
    )
    joined["Connectivity Index"] = joined["connectivity_index"].combine_first(joined["Connectivity Index"])
    joined["Redundancy Index"] = joined["redundancy_score"].combine_first(joined["Redundancy Index"])
    joined = joined[[
        'display_name_y',
        'Destination',
        'num_dests',
        'Connectivity Index',
        'Redundancy Index'
    ]]
  
    joined = joined.rename(
        columns={
            'num_dests':'# Destinations',
            'display_name_y':'City (IATA), Country',
            'Destination':'Airport Name'
            }
        )
    joined = joined.sort_values(by = '# Destinations',ascending=False)
    joined["Connectivity Index"] = pd.to_numeric(
        joined["Connectivity Index"], errors="coerce"
    ).round(1)
    joined["Redundancy Index"] = pd.to_numeric(
        joined["Redundancy Index"], errors="coerce"
    ).round(1)

    cols = [{"name": c, "id": c} for c in joined.columns]
    return joined.to_dict("records"), cols, 0

# ----------------------------------- #
# ----- Tab #3: Airline Metrics ----- #
# ----------------------------------- #

@app.callback(
    Output('dropdown4', 'options'), #--> filter airports
    Output('dropdown4', 'value'),
    Input('dropdown3', 'value') #--> choose counry
)
def set_airport_options(selected_country):
    return [{'label': i, 'value': i} for i in country_airport_dict[selected_country]], country_airport_dict[selected_country][0],


@app.callback(
    Output('airline_treemap', 'figure'),
    Input('dropdown4', 'value'),
)
def airline_treemap_chart(selected_airport):

    airport = airport_df[airport_df["display_name"] == selected_airport][["iata", "dest_iata", "carriers"]]
    airport["airline_names"] = airport["carriers"].apply(
        lambda x: list({carrier["name"] for carrier in x})
    )

    airport_exploded = airport.explode("airline_names")
    airport_exploded = airport.explode("airline_names").dropna(subset=["airline_names"])

    fig = px.treemap(
        airport_exploded,
        path=["airline_names", "dest_iata"],
        custom_data=["airline_names", "dest_iata"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Airline: %{customdata[0]}<br>"
            "IATA: %{customdata[1]}<br>"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        height=AIRLINE_TREEMAP_FIGURE_HEIGHT,
        title=dict(
            text=f"Airlines Available at {selected_airport}",
            font=dict(color="white", size=18),
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
    )
    return fig

AIRLINE_MAP_COLORS = [
    "#FF5A36", "#FFB000", "#00C2A8", "#7ED957", "#C449A0",
    "#FF7A00", "#5B4CFF", "#FF4F87", "#0077FF", "#00D1B2",
]


def prepare_country_airline_data(selected_country):
    country_filtered = airport_df[airport_df['country'] == selected_country].copy()
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
        airport_exploded.groupby(["city_name", "airline_names"])
        .size()
        .reset_index(name="flight_count")
    )
    city_counts["city_total"] = city_counts.groupby("city_name")["flight_count"].transform("sum")
    city_counts["pct_share"] = city_counts["flight_count"] / city_counts["city_total"]

    dominant_airline = (
        city_counts.loc[city_counts.groupby("city_name")["pct_share"].idxmax()]
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
        0.18,
        country_airline_mapping_df["pct_share"],
    )

    airport_airline_share_df = (
        airport_exploded.groupby(
            ["display_name", "city_name", "iata", "latitude", "longitude", "airline_names"]
        )
        .size()
        .reset_index(name="flight_count")
    )
    airport_airline_share_df["airport_total"] = airport_airline_share_df.groupby(
        ["display_name", "iata"]
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

    return country_airline_mapping_df, airport_airline_share_df, top_airlines


@app.callback(
    Output('card5', 'children'),
    Output('card6', 'children'),
    Output('card7', 'children'),
    Output('card8', 'children'),
    Input('dropdown3', 'value'),
    Input('dropdown4', 'value')

)

def airline_stats(selected_country, selected_airport):
    country_filtered = airport_df[airport_df['country'] == selected_country].copy()
    country_filtered["airline_names"] = country_filtered["carriers"].apply(extract_airline_names)
    country_airlines = country_filtered["airline_names"].explode().dropna()

    metric1 = country_airlines.nunique()
    country_counts = country_airlines.value_counts()
    if country_counts.empty:
        metric2 = "N/A"
    else:
        top_country_airline = country_counts.index[0]
        top_country_pct = country_counts.iloc[0] / country_counts.sum()
        metric2 = f"{top_country_airline} ({top_country_pct:.1%})"

    airport_filtered = airport_df[airport_df['display_name'] == selected_airport].copy()
    airport_filtered["airline_names"] = airport_filtered["carriers"].apply(extract_airline_names)
    airport_airlines = airport_filtered["airline_names"].explode().dropna()
    airport_code = (
        airport_filtered['iata'].iloc[0]
        if not airport_filtered.empty and 'iata' in airport_filtered.columns
        else selected_airport
    )
    metric3 = airport_airlines.nunique()
    airport_counts = airport_airlines.value_counts()
    if airport_counts.empty:
        metric4 = "N/A"
    else:
        top_airport_airline = airport_counts.index[0]
        top_airport_pct = airport_counts.iloc[0] / airport_counts.sum()
        metric4 = f"{top_airport_airline} ({top_airport_pct:.1%})"

    card5 = build_metric_card_body(f"Airlines available in {selected_country}", metric1)
    card6 = build_metric_card_body(
        f"Airline with the highest % share in {selected_country}",
        metric2
    )
    card7 = build_metric_card_body(f"Airlines in {airport_code}", metric3)
    card8 = build_metric_card_body(
        f"Airline with highest % share in {airport_code}",
        metric4
    )

    return card5, card6, card7, card8


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


@app.callback(
    Output('airline_map_mode', 'data'),
    Output('airline_map_mode_all', 'value'),
    Output('airline_map_mode_share', 'value'),
    Input('airline_map_mode_all', 'value'),
    Input('airline_map_mode_share', 'value'),
)
def sync_airline_map_mode(all_val, share_val):
    triggered = dash.callback_context.triggered_id
    if triggered == 'airline_map_mode_share' and share_val:
        return 'Airline % Share', None, 'Airline % Share'
    return 'All Airlines', 'All Airlines', None


@app.callback(
    Output('dropdown5', 'options'),
    Output('dropdown5', 'value'),
    Output('heatmap_airline_control', 'style'),
    Output('airline_map_controls_row', 'style'),
    Input('dropdown3', 'value'),
    Input('airline_map_mode', 'data'),
    State('dropdown5', 'value'),
)
def update_heatmap_airline_control(selected_country, map_mode, current_airline):
    prepared_country_data = prepare_country_airline_data(selected_country)
    top_airlines = prepared_country_data[2]
    options = [{'label': airline, 'value': airline} for airline in top_airlines]
    value = current_airline if current_airline in top_airlines else (top_airlines[0] if top_airlines else None)
    if map_mode == 'Airline % Share':
        heat_style = {'display': 'block', 'minWidth': 0}
        row_style = airline_map_controls_row_style('1fr 1fr 1fr')
    else:
        heat_style = {'display': 'none', 'minWidth': 0}
        row_style = airline_map_controls_row_style('1fr 1fr')
    return options, value, heat_style, row_style


@app.callback(
    Output('dominant_airline_by_country_map', 'figure'),
    Input('dropdown3', 'value'),
    Input('airline_map_mode', 'data'),
    Input('dropdown5', 'value'),
)
def dominant_airline_by_country_map(selected_country, map_mode, selected_airline):
    country_airline_mapping_df, airport_airline_share_df, top_airlines = prepare_country_airline_data(selected_country)
    if country_airline_mapping_df.empty:
        return go.Figure()

    legend_order = list(top_airlines) + ["Other"]
    color_map = get_airline_color_map(top_airlines)

    if map_mode == 'Airline % Share' and selected_airline:
        heat_df = airport_airline_share_df[
            airport_airline_share_df["selected_airline"] == selected_airline
        ].copy()
        if heat_df.empty:
            return go.Figure()

        center, zoom = get_country_view(country_airline_mapping_df)
        airline_color = color_map.get(selected_airline, AIRLINE_MAP_COLORS[0])
        max_share = max(float(heat_df["pct_share"].max()), 0.01)
        visible_max_share = max_share * 0.85 if max_share > 0.05 else max_share

        fig = px.density_mapbox(
            heat_df,
            lat="latitude",
            lon="longitude",
            z="pct_share",
            radius=28,
            center=center,
            zoom=zoom,
            height=AIRLINE_MAP_FIGURE_HEIGHT,
            custom_data=["selected_airline", "pct_share", "city_name", "iata"],
            color_continuous_scale=build_heat_colorscale(airline_color),
            range_color=(0, visible_max_share),
        )
        fig.update_traces(
            hovertemplate=(
                "Selected Airline: %{customdata[0]}<br>"
                "Pct Share: %{customdata[1]:.1%}<br>"
                "City Name: %{customdata[2]}<br>"
                "IATA: %{customdata[3]}"
                "<extra></extra>"
            )
        )
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            paper_bgcolor="black",
            plot_bgcolor="black",
            coloraxis_colorbar=dict(
                title=dict(
                    text = "% Share",
                    font=dict(color="white")
 
                ),
                tickformat=".0%",
                bgcolor="black",
                bordercolor="rgba(0,0,0,0)",
                tickfont=dict(color="white"),
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        return fig

    fig = px.scatter_mapbox(
        country_airline_mapping_df,
        lat="latitude",
        lon="longitude",
        custom_data=[
            "dominant_airline_group",
            "dominant_airline_actual",
            "pct_share",
            "city_name",
            "iata",
        ],
        color="dominant_airline_group",
        size="marker_size_value",
        category_orders={"dominant_airline_group": legend_order},
        color_discrete_map=color_map,
        zoom=2,
        height=AIRLINE_MAP_FIGURE_HEIGHT,
        size_max=18,
    )
    fig.update_traces(
        marker=dict(opacity=0.72),
        hovertemplate=(
            "Dominant Airline Group: %{customdata[0]}<br>"
            "Dominant Airline Actual: %{customdata[1]}<br>"
            "Pct Share: %{customdata[2]:.1%}<br>"
            "City Name: %{customdata[3]}<br>"
            "IATA: %{customdata[4]}"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        legend_title_text="Dominant Airline",
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# ----------------------------------- #
# ------- Tab #4: Connections ------- #
# ----------------------------------- #


@app.callback(
    Output('dropdown_conn_airport', 'options'),
    Output('dropdown_conn_airport', 'value'),
    Input('dropdown7', 'value'),
)
def set_connections_airport_dropdown(selected_country):
    sub = (
        airport_df[airport_df["country"] == selected_country][["iata", "name", "country"]]
        .drop_duplicates(subset=["iata"])
        .sort_values("name")
    )
    opts = [{"label": "All airports in country", "value": CONN_AIRPORT_ALL}]
    for _, row in sub.iterrows():
        opts.append({
            "label": f"{row['name']}, {row['country']} ({row['iata']})",
            "value": row["iata"],
        })
    return opts, CONN_AIRPORT_ALL





# vis-network defaults use gray node labels; white + subtle stroke reads better on dark surfaces.
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
    meta_ct = (AIRPORT_IATA_META.get(iata) or {}).get("country")
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
        ct = (AIRPORT_IATA_META.get(iata) or {}).get("country")
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
    return f"{round(float(row['weight']), 3)} ({row['source']}–{row['target']})"


def similarity_feature_distance(iata_a, iata_b):
    """Equal-weight linear combo of |Δz| for connectivity, log(1+dests), and redundancy."""
    pa = SIMILARITY_AIRPORT_PROFILE.get(iata_a)
    pb = SIMILARITY_AIRPORT_PROFILE.get(iata_b)
    if not pa or not pb:
        return np.nan
    diffs = [abs(pa[k] - pb[k]) for k in SIMILARITY_Z_FEATURE_KEYS]
    return float(sum(diffs) / len(diffs))


def apply_similarity_scores(edges_df):
    """Replace edge weights with 0–1 similarity (1 = most alike in this edge set)."""
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
    profile = SIMILARITY_AIRPORT_PROFILE.get(info.get("iata"))
    if not profile:
        return [
            html.P("Profile data unavailable for this airport.", style={"color": "#e8ecf4"}),
        ]

    body = []
    if focus_iata and focus_iata != CONN_AIRPORT_ALL:
        focus_profile = SIMILARITY_AIRPORT_PROFILE.get(focus_iata)
        if focus_profile:
            focus_meta = AIRPORT_IATA_META.get(focus_iata) or {}
            focus_name = focus_meta.get("display_name") or focus_iata
            body.append(html.P(
                f"Compared to focus airport: {focus_name} ({focus_iata})",
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
    )
    focus_profile = (
        SIMILARITY_AIRPORT_PROFILE.get(focus_iata)
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
    surface_style = dict(NETWORK_CHART_SURFACE)
    surface_style["position"] = "relative"
    children = [net]
    if legend_items:
        children.append(
            html.Div(
                [
                    *[
                        html.Div(
                            [
                                html.Span(
                                    style={
                                        "display": "inline-block",
                                        "width": "12px",
                                        "height": "12px",
                                        "borderRadius": "50%",
                                        "backgroundColor": color,
                                        "marginRight": "8px",
                                        "verticalAlign": "middle",
                                    }
                                ),
                                html.Span(label, style={"verticalAlign": "middle"}),
                            ],
                            style={"marginBottom": "4px"},
                        )
                        for label, color in legend_items
                    ],
                ],
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
        )
    return html.Div(children, style=surface_style)


def airport_network_display_label(iata):
    meta = AIRPORT_IATA_META.get(iata) or {}
    name = meta.get("display_name") or iata
    return name if name == iata else f"{name}\n({iata})"


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


@app.callback(
    Output('network_chart', 'children'),
    Output('network-node-meta', 'data'),
    Output('conn-metric-cards', 'children'),
    Input('dropdown6', 'value'),
    Input('dropdown7', 'value'),
    Input('dropdown_conn_airport', 'value'),
)
def network_connections(connection_type, selected_country, focus_airport):
    empty_metric_bodies = (
        build_metric_card_body("Connection view", "—"),
        build_metric_card_body("Nodes", "—"),
        build_metric_card_body("Links", "—"),
        build_metric_card_body("Detail", "—"),
    )
    empty_cards_row = build_conn_metric_cards_row(*empty_metric_bodies)

    focus_airport = focus_airport or CONN_AIRPORT_ALL

    if connection_type == "Carriers":

        filtered_graph = graph1_merged[graph1_merged['country'] == selected_country]

        if filtered_graph.empty:
            return (
                html.Div("No data available for this selection", style=NETWORK_EMPTY_STYLE),
                {},
                empty_cards_row,
            )

        # ---- Optional: reduce noise (top airlines only)
        top_airlines = (
            filtered_graph['airline']
            .value_counts()
            .head(15)
            .index
        )
        filtered_graph = filtered_graph[filtered_graph['airline'].isin(top_airlines)]

        if focus_airport != CONN_AIRPORT_ALL:
            filtered_graph = filtered_graph[filtered_graph["airport"] == focus_airport]
            if filtered_graph.empty:
                return (
                    html.Div(
                        "No carrier routes for this airport in the current top-airline view.",
                        style=NETWORK_EMPTY_STYLE,
                    ),
                    {},
                    empty_cards_row,
                )

        airlines = filtered_graph['airline'].unique()
        airports = filtered_graph['airport'].unique()

        nodes = []
        edges = []
        node_meta = {}

        # Airline nodes (blue): show airline name when known; FG/FZ are IATA airline codes.
        for airline in airlines:
            legal_name = AIRLINE_IATA_TO_NAME.get(airline)
            label = legal_name if legal_name else airline
            hover_title = (
                f"{legal_name or 'Unknown name'}\n"
                f"IATA airline code: {airline}\n"
                "Yellow nodes are airports in the selected country."
            )
            ap_codes = sorted(filtered_graph.loc[filtered_graph["airline"] == airline, "airport"].unique())
            ap_lines = []
            for code in ap_codes[:40]:
                m = AIRPORT_IATA_META.get(code) or {}
                disp = m.get("display_name") or code
                ap_lines.append(f"{disp} ({code})")
            node_meta[airline] = {
                "kind": "airline",
                "name": legal_name or "—",
                "iata": airline,
                "airport_count": len(ap_codes),
                "airports": ap_lines,
            }
            nodes.append({
                "id": airline,
                "label": label,
                "title": hover_title,
                "color": NETWORK_PEER_NODE_COLOR["carriers_airline"],
                "shape": "dot",
                "size": 20,
                "group": "airline",
            })

        # Airport nodes (yellow when in selected country — all rows here are in-country)
        for airport in airports:
            ap_meta = AIRPORT_IATA_META.get(airport) or {}
            disp = ap_meta.get("display_name") or airport
            al_codes = sorted(filtered_graph.loc[filtered_graph["airport"] == airport, "airline"].unique())
            al_labels = []
            for ac in al_codes:
                nm = AIRLINE_IATA_TO_NAME.get(ac)
                al_labels.append(f"{nm} ({ac})" if nm else str(ac))
            node_meta[airport] = {
                "kind": "airport",
                "name": disp,
                "iata": airport,
                "country": ap_meta.get("country") or selected_country,
                "airline_count": len(al_codes),
                "airlines": al_labels,
            }
            ap_col, ap_sz = airport_node_sizes_and_color(
                airport, selected_country, NETWORK_CENTRAL_NODE_COLOR
            )
            nodes.append({
                "id": airport,
                "label": airport_network_display_label(airport),
                "title": (
                    f"{disp}\n"
                    f"IATA: {airport}\n"
                    f"{ap_meta.get('country') or selected_country}"
                ),
                "color": ap_col,
                "shape": "dot",
                "size": ap_sz,
                "group": "airport",
            })

        for _, row in filtered_graph.iterrows():
            edges.append({
                "from": row["airline"],
                "to": row["airport"],
                "title": f"{row['airline']} → {row['airport']}",
            })

        n_airlines = len(airlines)
        n_airports = len(airports)
        n_edges = len(edges)
        hub_airline_counts = (
            filtered_graph.groupby("airport")["airline"].nunique().sort_values(ascending=False)
            if n_edges else pd.Series(dtype=int)
        )
        if len(hub_airline_counts):
            hub_iata = hub_airline_counts.index[0]
            hub_n_airlines = int(hub_airline_counts.iloc[0])
            hub_name = (AIRPORT_IATA_META.get(hub_iata) or {}).get("display_name") or hub_iata
            most_airlines_val = f"{hub_name} ({hub_n_airlines} airlines)"
        else:
            most_airlines_val = "—"

        if focus_airport != CONN_AIRPORT_ALL:
            dom_name, dom_pct = dominant_airline_share_at_iata(focus_airport)
            if dom_name and dom_pct is not None:
                dominant_val = f"{dom_name} ({dom_pct:.1%})"
            else:
                dominant_val = "—"
            cards = (
                build_metric_card_body(f"Carriers · {selected_country}", f"{n_airlines} airlines (top)"),
                build_metric_card_body("Dominant Airline", dominant_val),
            )
        else:
            cards = (
                build_metric_card_body(f"Carriers · {selected_country}", f"{n_airlines} airlines (top)"),
                build_metric_card_body("Airports in view", n_airports),
                build_metric_card_body("Airport–airline links", n_edges),
                build_metric_card_body("Airport with most airlines", most_airlines_val),
            )

        net = visdcc.Network(
            id='network',
            selection={"nodes": [], "edges": []},
            data={"nodes": nodes, "edges": edges},
            options=carrier_network_physics_options(),
        )

        return wrap_network_chart(
            net,
            [
                ("Airports in selected country", NETWORK_CENTRAL_NODE_COLOR),
                ("Airlines", NETWORK_PEER_NODE_COLOR["carriers_airline"]),
            ],
        ), node_meta, build_conn_metric_cards_row(*cards)

    elif connection_type == "Statistical Similarity":

        filtered = graph2_merged[graph2_merged['source_country'] == selected_country]
        filtered = filter_airport_edges(filtered, focus_airport)

        if filtered.empty:
            return (
                html.Div(
                    "No similarity data for this selection (try another airport or “All airports”).",
                    style=NETWORK_EMPTY_STYLE,
                ),
                {},
                empty_cards_row,
            )

        filtered = apply_similarity_scores(filtered)

        # ---- Filter weak edges (VERY important for readability)
        if focus_airport == CONN_AIRPORT_ALL:
            filtered = filtered[filtered["weight"] > filtered["weight"].quantile(0.75)]
        filtered = filtered.groupby(["source", "target"], as_index=False)["weight"].max()

        filtered = filtered.sort_values("weight", ascending=False).head(MAX_NETWORK_EDGES)

        node_ids = set(filtered["source"]).union(set(filtered["target"]))

        nodes = []
        edges = []
        node_meta = {}
        peer_c = NETWORK_PEER_NODE_COLOR["similarity"]

        for node in node_ids:
            m = AIRPORT_IATA_META.get(node) or {}
            disp = m.get("display_name") or node
            ct = m.get("country", "—")
            node_meta[node] = {
                "kind": "airport",
                "name": disp,
                "iata": node,
                "country": ct,
                "airport_count": None,
                "airports": [],
                "airline_count": None,
                "airlines": [],
                "similarity_airport": True,
            }
            ncol, nsz = airport_node_sizes_and_color(node, selected_country, peer_c)
            nodes.append({
                "id": node,
                "label": airport_network_display_label(node),
                "title": f"{disp}\nIATA: {node}\n{ct}",
                "shape": "dot",
                "size": nsz,
                "color": ncol,
            })

        max_w = float(filtered["weight"].max()) if len(filtered) else 1.0
        for _, row in filtered.iterrows():
            edges.append({
                "from": row["source"],
                "to": row["target"],
                "width": max(1, row["weight"] / max_w * 10),
                "title": f"Similarity score: {row['weight']:.3f}",
                "color": {
                    "color": "#818CF8",
                    "opacity": 0.82
                }
            })

        mean_w = round(float(filtered["weight"].mean()), 3) if len(filtered) else "—"
        links_val = metric_capped_if_needed(len(edges))
        strongest_sim = strongest_similarity_label(filtered)
        cards = (
            build_metric_card_body("Similarity links (shown)", links_val),
            build_metric_card_body("# countries in view", countries_in_node_set(node_ids)),
            build_metric_card_body("Mean similarity score", mean_w),
            build_metric_card_body("Strongest link", strongest_sim),
        )

        net = visdcc.Network(
            id='network',
            selection={"nodes": [], "edges": []},
            data={"nodes": nodes, "edges": edges},
            options={
                "height": NETWORK_VIS_HEIGHT,
                "width": "100%",
                "layout": {"hierarchical": {"enabled": False}},
                "physics": {
                    "enabled": True,
                    "solver": "barnesHut",
                    "barnesHut": {
                        "gravitationalConstant": -20000,
                        "springLength": 180,
                        "springConstant": 0.02,
                        "damping": 0.58,
                    },
                    "stabilization": {"iterations": 200},
                },
                "nodes": {
                    "font": NETWORK_NODE_FONT,
                },
                "edges": {
                    "smooth": {"type": "continuous", "roundness": 0.35}
                },
                "interaction": {
                    "hover": True,
                    "navigationButtons": True,
                    "dragNodes": True,
                    "dragView": True,
                    "zoomView": True,
                    "tooltipDelay": 100,
                }
            }
        )

        return wrap_network_chart(
            net,
            [
                ("Airports in selected country", NETWORK_CENTRAL_NODE_COLOR),
                ("Airports in other countries", NETWORK_PEER_NODE_COLOR["similarity"]),
            ],
        ), node_meta, build_conn_metric_cards_row(*cards)

    elif connection_type == "Proximity":
    
        filtered = graph3_merged[graph3_merged["source_country"] == selected_country]
        filtered = filter_airport_edges(filtered, focus_airport)

        if filtered.empty:
            return (
                html.Div(
                    "No proximity data for this selection (try another airport or “All airports”).",
                    style=NETWORK_EMPTY_STYLE,
                ),
                {},
                empty_cards_row,
            )

        # Keep nearer pairs first (weight = distance in miles; smaller = closer).
        if focus_airport == CONN_AIRPORT_ALL:
            filtered = filtered[filtered["weight"] <= filtered["weight"].quantile(0.40)]
        filtered = filtered.groupby(["source", "target"], as_index=False)["weight"].min()

        filtered = filtered.sort_values("weight", ascending=True).head(MAX_NETWORK_EDGES)

        node_ids = set(filtered["source"]).union(set(filtered["target"]))

        nodes = []
        edges = []
        node_meta = {}

        neighbor_map = {nid: [] for nid in node_ids}
        peer_c = NETWORK_PEER_NODE_COLOR["proximity"]
        for _, row in filtered.iterrows():
            s, t, w = row["source"], row["target"], float(row["weight"])
            ms = AIRPORT_IATA_META.get(s) or {}
            mt = AIRPORT_IATA_META.get(t) or {}
            ds = ms.get("display_name") or s
            dt = mt.get("display_name") or t
            neighbor_map[s].append((t, w, dt))
            neighbor_map[t].append((s, w, ds))

        for node in node_ids:
            m = AIRPORT_IATA_META.get(node) or {}
            disp = m.get("display_name") or node
            ct = m.get("country", "—")
            neigh_rows = sorted(neighbor_map.get(node, []), key=lambda x: x[1])[:50]
            prox_list = [f"{name} ({code}) — {mi:.1f} mi" for code, mi, name in neigh_rows]
            node_meta[node] = {
                "kind": "airport",
                "name": disp,
                "iata": node,
                "country": ct,
                "airport_count": None,
                "airports": [],
                "airline_count": None,
                "airlines": [],
                "proximity_neighbors": prox_list,
            }
            ncol, nsz = airport_node_sizes_and_color(node, selected_country, peer_c)
            nodes.append({
                "id": node,
                "label": airport_network_display_label(node),
                "title": (
                    f"{disp}\nIATA: {node}\n{ct}\n"
                    f"Edges: within {PROXIMITY_EDGE_MAX_MILES:.0f} mi (data cap)"
                ),
                "shape": "dot",
                "size": nsz,
                "color": ncol,
            })

        max_mi = float(filtered["weight"].max()) if len(filtered) else 1.0
        min_mi = float(filtered["weight"].min()) if len(filtered) else 1.0
        for _, row in filtered.iterrows():
            w = float(row["weight"])
            # Shorter links draw slightly thicker (easier to read).
            width = max(1.0, (max_mi - w) / max(max_mi - min_mi, 1e-6) * 9 + 1)
            edges.append({
                "from": row["source"],
                "to": row["target"],
                "width": width,
                "title": f"Distance: {w:.1f} mi (≤{PROXIMITY_EDGE_MAX_MILES:.0f} mi layer)",
                "color": {
                    "color": "#4ADE80",
                    "opacity": 0.78,
                },
            })

        mean_mi = round(float(filtered["weight"].mean()), 1) if len(filtered) else "—"
        links_val = metric_capped_if_needed(len(edges))
        if len(filtered):
            shortest_row = filtered.iloc[0]
            shortest_val = f"{round(float(shortest_row['weight']), 1)} ({shortest_row['source']}–{shortest_row['target']})"
        else:
            shortest_val = "—"
        if focus_airport != CONN_AIRPORT_ALL:
            cards = (
                build_metric_card_body("Proximity links (shown)", links_val),
                build_metric_card_body("# countries in view", countries_in_node_set(node_ids)),
                build_metric_card_body("Mean link distance (mi)", mean_mi),
                build_metric_card_body("Shortest shown link (mi)", shortest_val),
            )
        else:
            cards = (
                build_metric_card_body("Proximity links (shown)", links_val),
                build_metric_card_body("# countries in view", countries_in_node_set(node_ids)),
                build_metric_card_body("Mean link distance (mi)", mean_mi),
                build_metric_card_body("Shortest shown link (mi)", shortest_val),
            )

        net = visdcc.Network(
            id="network",
            selection={"nodes": [], "edges": []},
            data={"nodes": nodes, "edges": edges},
            options={
                "height": NETWORK_VIS_HEIGHT,
                "width": "100%",
                "layout": {"hierarchical": {"enabled": False}},
                "physics": {
                    "enabled": True,
                    "solver": "barnesHut",
                    "barnesHut": {
                        "gravitationalConstant": -22000,
                        "springLength": 160,
                        "springConstant": 0.018,
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
                    "tooltipDelay": 100,
                },
            },
        )

        return wrap_network_chart(
            net,
            [
                ("Airports in selected country", NETWORK_CENTRAL_NODE_COLOR),
                ("Airports in other countries", NETWORK_PEER_NODE_COLOR["proximity"]),
            ],
        ), node_meta, build_conn_metric_cards_row(*cards)

    elif connection_type == "Shared Destinations":

        filtered = load_merged_shared_destinations_edges()
        filtered = filtered[filtered["source_country"] == selected_country]
        filtered = filter_airport_edges(filtered, focus_airport)

        if filtered.empty:
            return (
                html.Div(
                    "No shared-destination data for this selection (try another airport or “All airports”).",
                    style=NETWORK_EMPTY_STYLE,
                ),
                {},
                empty_cards_row,
            )

        # Keep pairs with many overlapping destinations (weight = count of shared dests).
        if focus_airport == CONN_AIRPORT_ALL:
            filtered = filtered[filtered["weight"] > filtered["weight"].quantile(0.75)]
        filtered = filtered.groupby(["source", "target"], as_index=False)["weight"].max()

        filtered = filtered.sort_values("weight", ascending=False).head(MAX_NETWORK_EDGES)

        node_ids = set(filtered["source"]).union(set(filtered["target"]))

        nodes = []
        edges = []
        node_meta = {}

        peer_map = {nid: [] for nid in node_ids}
        peer_c = NETWORK_PEER_NODE_COLOR["shared_raw"]
        for _, row in filtered.iterrows():
            s, t, w = row["source"], row["target"], float(row["weight"])
            mt = AIRPORT_IATA_META.get(t) or {}
            ms = AIRPORT_IATA_META.get(s) or {}
            dt = mt.get("display_name") or t
            ds = ms.get("display_name") or s
            peer_map[s].append((t, w, dt))
            peer_map[t].append((s, w, ds))

        for node in node_ids:
            m = AIRPORT_IATA_META.get(node) or {}
            disp = m.get("display_name") or node
            ct = m.get("country", "—")
            peer_rows = sorted(peer_map.get(node, []), key=lambda x: -x[1])[:50]
            peer_lines = [
                f"{name} ({code}) — {int(cnt)} overlapping destinations (raw count)"
                for code, cnt, name in peer_rows
            ]
            node_meta[node] = {
                "kind": "airport",
                "name": disp,
                "iata": node,
                "country": ct,
                "airport_count": None,
                "airports": [],
                "airline_count": None,
                "airlines": [],
                "shared_dest_peers": peer_lines,
            }
            ncol, nsz = airport_node_sizes_and_color(node, selected_country, peer_c)
            nodes.append({
                "id": node,
                "label": airport_network_display_label(node),
                "title": (
                    f"{disp}\nIATA: {node}\n{ct}\n"
                    "Edges: raw count of destinations served by both airports"
                ),
                "shape": "dot",
                "size": nsz,
                "color": ncol,
            })

        max_w = float(filtered["weight"].max()) if len(filtered) else 1.0
        for _, row in filtered.iterrows():
            w = float(row["weight"])
            edges.append({
                "from": row["source"],
                "to": row["target"],
                "width": max(1, w / max_w * 10),
                "title": f"Raw overlap: {int(w)} shared destinations",
                "color": {
                    "color": "#E879F9",
                    "opacity": 0.85,
                },
            })

        mean_w = round(float(filtered["weight"].mean()), 1) if len(filtered) else "—"
        strongest_pair = strongest_weight_count_pair(filtered)
        links_val = metric_capped_if_needed(len(edges))
        if focus_airport != CONN_AIRPORT_ALL:
            cards = (
                build_metric_card_body("Links (shown)", links_val),
                build_metric_card_body("# countries in view", countries_in_node_set(node_ids)),
                build_metric_card_body("Mean shared count", mean_w),
                build_metric_card_body("Strongest pair (count)", strongest_pair),
            )
        else:
            cards = (
                build_metric_card_body("Links (shown)", links_val),
                build_metric_card_body("# countries in view", countries_in_node_set(node_ids)),
                build_metric_card_body("Mean shared count", mean_w),
                build_metric_card_body("Strongest pair (count)", strongest_pair),
            )

        net = visdcc.Network(
            id="network",
            selection={"nodes": [], "edges": []},
            data={"nodes": nodes, "edges": edges},
            options={
                "height": NETWORK_VIS_HEIGHT,
                "width": "100%",
                "layout": {"hierarchical": {"enabled": False}},
                "physics": {
                    "enabled": True,
                    "solver": "barnesHut",
                    "barnesHut": {
                        "gravitationalConstant": -20000,
                        "springLength": 180,
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
                    "tooltipDelay": 100,
                },
            },
        )

        return wrap_network_chart(
            net,
            [
                ("Airports in selected country", NETWORK_CENTRAL_NODE_COLOR),
                ("Airports in other countries", NETWORK_PEER_NODE_COLOR["shared_raw"]),
            ],
        ), node_meta, build_conn_metric_cards_row(*cards)

    elif connection_type == "Shared Destinations (Hub-Adjusted)":

        filtered = load_merged_shared_destinations_cosine_edges()
        filtered = filtered[filtered["source_country"] == selected_country]
        filtered = filter_airport_edges(filtered, focus_airport)

        if filtered.empty:
            return (
                html.Div(
                    "No hub-adjusted destination overlap for this selection (try another airport or “All airports”).",
                    style=NETWORK_EMPTY_STYLE,
                ),
                {},
                empty_cards_row,
            )

        if focus_airport == CONN_AIRPORT_ALL:
            filtered = filtered[filtered["weight"] > filtered["weight"].quantile(0.75)]
        filtered = filtered.groupby(["source", "target"], as_index=False)["weight"].max()

        filtered = filtered.sort_values("weight", ascending=False).head(MAX_NETWORK_EDGES)

        node_ids = set(filtered["source"]).union(set(filtered["target"]))

        nodes = []
        edges = []
        node_meta = {}

        peer_map = {nid: [] for nid in node_ids}
        peer_c = NETWORK_PEER_NODE_COLOR["shared_cosine"]
        for _, row in filtered.iterrows():
            s, t, w = row["source"], row["target"], float(row["weight"])
            mt = AIRPORT_IATA_META.get(t) or {}
            ms = AIRPORT_IATA_META.get(s) or {}
            dt = mt.get("display_name") or t
            ds = ms.get("display_name") or s
            peer_map[s].append((t, w, dt))
            peer_map[t].append((s, w, ds))

        for node in node_ids:
            m = AIRPORT_IATA_META.get(node) or {}
            disp = m.get("display_name") or node
            ct = m.get("country", "—")
            peer_rows = sorted(peer_map.get(node, []), key=lambda x: -x[1])[:50]
            peer_lines = [
                f"{name} ({code}) — hub-adjusted score {sc:.3f}"
                for code, sc, name in peer_rows
            ]
            node_meta[node] = {
                "kind": "airport",
                "name": disp,
                "iata": node,
                "country": ct,
                "airport_count": None,
                "airports": [],
                "airline_count": None,
                "airlines": [],
                "shared_cosine_peers": peer_lines,
            }
            ncol, nsz = airport_node_sizes_and_color(node, selected_country, peer_c)
            nodes.append({
                "id": node,
                "label": airport_network_display_label(node),
                "title": (
                    f"{disp}\nIATA: {node}\n{ct}\n"
                    "Cosine similarity of destination sets (size-adjusted)"
                ),
                "shape": "dot",
                "size": nsz,
                "color": ncol,
            })

        max_w = float(filtered["weight"].max()) if len(filtered) else 1.0
        for _, row in filtered.iterrows():
            w = float(row["weight"])
            edges.append({
                "from": row["source"],
                "to": row["target"],
                "width": max(1, w / max_w * 10),
                "title": f"Hub-adjusted destination overlap: {w:.3f}",
                "color": {
                    "color": "#FDA4AF",
                    "opacity": 0.85,
                },
            })

        mean_w = round(float(filtered["weight"].mean()), 3) if len(filtered) else "—"
        strongest_pair = strongest_adjusted_score_label(filtered)
        links_val = metric_capped_if_needed(len(edges))
        if focus_airport != CONN_AIRPORT_ALL:
            cards = (
                build_metric_card_body("Links (shown)", links_val),
                build_metric_card_body("# countries in view", countries_in_node_set(node_ids)),
                build_metric_card_body("Mean adjusted score", mean_w),
                build_metric_card_body("Strongest score", strongest_pair),
            )
        else:
            cards = (
                build_metric_card_body("Links (shown)", links_val),
                build_metric_card_body("# countries in view", countries_in_node_set(node_ids)),
                build_metric_card_body("Mean adjusted score", mean_w),
                build_metric_card_body("Strongest score", strongest_pair),
            )

        net = visdcc.Network(
            id="network",
            selection={"nodes": [], "edges": []},
            data={"nodes": nodes, "edges": edges},
            options={
                "height": NETWORK_VIS_HEIGHT,
                "width": "100%",
                "layout": {"hierarchical": {"enabled": False}},
                "physics": {
                    "enabled": True,
                    "solver": "barnesHut",
                    "barnesHut": {
                        "gravitationalConstant": -20000,
                        "springLength": 180,
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
                    "tooltipDelay": 100,
                },
            },
        )

        return wrap_network_chart(
            net,
            [
                ("Airports in selected country", NETWORK_CENTRAL_NODE_COLOR),
                ("Airports in other countries", NETWORK_PEER_NODE_COLOR["shared_cosine"]),
            ],
        ), node_meta, build_conn_metric_cards_row(*cards)

    # -----------------------------
    # 3. FALLBACK
    # -----------------------------
    return (
        html.Div("Select a valid connection type", style=NETWORK_EMPTY_STYLE),
        {},
        empty_cards_row,
    )


def build_connection_types_guide(active_connection_type):
    """Short definitions for the Connections tab sidebar."""
    blocks = []
    for title, blurb in CONNECTION_TYPE_DEFINITIONS:
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


@app.callback(
    Output('network-node-detail', 'children'),
    Input('network', 'selection'),
    Input('dropdown6', 'value'),
    Input('dropdown_conn_airport', 'value'),
    State('network-node-meta', 'data'),
)
def network_node_detail_panel(selection, connection_type, focus_airport, meta):
    meta = meta or {}
    guide = build_connection_types_guide(connection_type)
    divider = html.Hr(
        style={
            "border": "none",
            "borderTop": "1px solid rgba(255,255,255,0.15)",
            "margin": "4px 0 14px 0",
        }
    )
    hint = html.P(
        "Click a node in the graph for full details below.",
        style={"color": "rgba(255,255,255,0.78)", "margin": 0, "lineHeight": 1.5},
    )
    if not selection or not selection.get("nodes"):
        return html.Div([guide, divider, hint])

    node_id = selection["nodes"][0]
    info = meta.get(node_id)
    if not info:
        return html.Div([
            guide,
            divider,
            html.H4(node_id, style={"marginBottom": "12px", "color": "#ffffff"}),
            hint,
        ])

    if info["kind"] == "airline":
        body = [
            html.H4(
                info.get("name") or info.get("iata"),
                style={"marginBottom": "8px", "color": "#ffffff"},
            ),
            html.P(
                [
                    html.Strong("IATA airline code: "),
                    info.get("iata", "—"),
                ],
                style={"marginBottom": "8px", "color": "#e8ecf4"},
            ),
            html.P(
                f"Serves {info.get('airport_count', 0)} airports in this view (top carriers / selected country).",
                style={"marginBottom": "12px", "color": "#e8ecf4"},
            ),
            html.P(html.Strong("Airports"), style={"marginBottom": "6px", "color": "#ffffff"}),
            html.Ul(
                [html.Li(a) for a in info.get("airports") or []],
                style={"maxHeight": "360px", "overflowY": "auto", "paddingLeft": "20px", "color": "#e8ecf4"},
            ),
        ]
        return html.Div([guide, divider] + body)

    # airport (carriers, similarity, proximity, shared-dest, cosine, …)
    body = [
        html.H4(
            info.get("name") or info.get("iata"),
            style={"marginBottom": "8px", "color": "#ffffff"},
        ),
        html.P(
            [
                html.Strong("IATA: "),
            f"{info.get('iata', '—')} · {info.get('country', '—')}",
            ],
            style={"marginBottom": "12px", "color": "#e8ecf4"},
        ),
    ]
    if info.get("airline_count"):
        body.append(html.P(
            f"Connected to {info['airline_count']} airlines in this carrier graph.",
            style={"marginBottom": "8px", "color": "#e8ecf4"},
        ))
        body.append(html.P(html.Strong("Airlines"), style={"marginBottom": "6px", "color": "#ffffff"}))
        body.append(html.Ul(
            [html.Li(a) for a in info.get("airlines") or []],
            style={"maxHeight": "280px", "overflowY": "auto", "paddingLeft": "20px", "color": "#e8ecf4"},
        ))
    elif info.get("similarity_airport"):
        body.extend(build_statistical_similarity_detail_body(info, focus_airport))
    elif info.get("proximity_neighbors"):
        body.append(html.P(
            (
                f"Nearby airports in this view (links capped at "
                f"{PROXIMITY_EDGE_MAX_MILES:.0f} mi in source data; chart keeps shorter links first)."
            ),
            style={"marginBottom": "8px", "fontSize": "0.95rem", "color": "rgba(255,255,255,0.92)"},
        ))
        body.append(html.P(html.Strong("Nearby airports (shown links)"), style={"marginBottom": "6px", "color": "#fff"}))
        body.append(html.Ul(
            [html.Li(a) for a in info.get("proximity_neighbors") or []],
            style={"maxHeight": "320px", "overflowY": "auto", "paddingLeft": "20px", "color": "#e8ecf4"},
        ))
    elif info.get("shared_dest_peers"):
        body.append(html.P(
            "Raw overlap: each link counts destinations that appear in both airports’ outbound route lists "
            "(same destination IATA from each hub). This is the unadjusted count — large hubs tend to "
            "score higher simply because they serve more cities.",
            style={"marginBottom": "8px", "fontSize": "0.95rem", "color": "rgba(255,255,255,0.92)"},
        ))
        body.append(html.P(html.Strong("Peers — raw overlap counts"), style={"marginBottom": "6px", "color": "#fff"}))
        body.append(html.Ul(
            [html.Li(a) for a in info.get("shared_dest_peers") or []],
            style={"maxHeight": "320px", "overflowY": "auto", "paddingLeft": "20px", "color": "#e8ecf4"},
        ))
    elif info.get("shared_cosine_peers"):
        body.append(html.P(
            "Hub-adjusted overlap: the same shared-destination idea as the raw count, but scaled by each airport’s "
            "destination-count footprint so mega-hubs do not automatically dominate every comparison.",
            style={"marginBottom": "8px", "fontSize": "0.95rem", "color": "rgba(255,255,255,0.92)"},
        ))
        body.append(html.P(html.Strong("Peers — hub-adjusted scores"), style={"marginBottom": "6px", "color": "#fff"}))
        body.append(html.Ul(
            [html.Li(a) for a in info.get("shared_cosine_peers") or []],
            style={"maxHeight": "320px", "overflowY": "auto", "paddingLeft": "20px", "color": "#e8ecf4"},
        ))
    else:
        body.append(html.P(
            "This graph links airports whose derived attribute profiles are statistically similar "
            "(stronger edges = more alike in the source feature metrics).",
            style={"color": "rgba(255,255,255,0.75)", "fontSize": "0.95rem"},
        ))
    return html.Div([guide, divider] + body)

#----- Modal Callbacks
@app.callback(
    Output("modal1", "is_open"),
    [Input("open1", "n_clicks"), 
    Input("close1", "n_clicks")],
    [State("modal1", "is_open")],
)

def toggle_modal1(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output("modal2", "is_open"),
    [Input("open2", "n_clicks"), 
    Input("close2", "n_clicks")],
    [State("modal2", "is_open")],
)

def toggle_modal2(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output("modal3", "is_open"),
    [Input("open3", "n_clicks"), 
    Input("close3", "n_clicks")],
    [State("modal3", "is_open")],
)

def toggle_modal3(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__=='__main__':
	#app.run()    
    app.run(port=8051)