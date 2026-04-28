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
#-----Read in and set up data
airport_df = pd.read_parquet('https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/master_air.parquet')

graph1 = pd.read_parquet('https://raw.githubusercontent.com/statzenthusiast921/Airport_Accessibility_Project/main/data/edges_airline_airport.parquet')
airport_df_join = airport_df[['iata','country','display_name']].drop_duplicates()
airport_df_join.rename(columns={'iata':'airport'}, inplace=True)
graph1_merged = pd.merge(graph1, airport_df_join, on ='airport')

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
connection_type_choices = ['Similarity','Carriers','Proximity','etc']

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


def build_metric_card_body(title, value, font_size="1.8rem"):
    return dbc.CardBody([
        html.P(
            title,
            style={
                "margin": "0 0 8px 0",
                "fontSize": "0.8rem",
                "fontWeight": "600",
                "textTransform": "uppercase",
                "letterSpacing": "1px",
                "color": "rgba(255,255,255,0.8)"
            }
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
                       html.P("This dashboard was created to blah.",style={'color':'white'}),
                       html.Br()
                   ]),
                   html.Div([
                       html.P(dcc.Markdown('''**What data is being used for this analysis?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   
                   html.Div([
                       html.P(["The data utilized was pulled from ",html.A('here',href='https://www.railpassengers.org/resources/ridership-statistics/')],style={'color':'white'}),
                       html.Br()
                   ]),
                   html.Div([
                       html.P(dcc.Markdown('''**What are the limitations of this data?**'''),style={'color':'white'}),
                   ],style={'text-decoration': 'underline'}),
                   html.Div([
                       html.P("1.) Thing.",style={'color':'white'}),
                       html.P("2.) Something.",style={'color':'white'}),

                   ])


               ]),
        dcc.Tab(label='Airport Metrics',value='tab-2',style=tab_style, selected_style=tab_selected_style,
            children=[
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
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            html.Div([
                                dbc.Label(
                                    'Map view:',
                                    style={'marginBottom': '6px', 'fontWeight': '600'}
                                ),
                                dcc.RadioItems(
                                    id='airline_map_mode',
                                    options=[
                                        {'label': 'All Airlines', 'value': 'All Airlines'},
                                        {'label': 'Airline % Share', 'value': 'Airline % Share'},
                                    ],
                                    value='All Airlines',
                                    inline=True,
                                    inputStyle={'marginRight': '6px'},
                                    labelStyle={
                                        'marginRight': '12px',
                                        'padding': '6px 12px',
                                        'backgroundColor': 'rgba(255,255,255,0.12)',
                                        'borderRadius': '999px',
                                        'display': 'inline-flex',
                                        'alignItems': 'center',
                                    }
                                ),
                            ], style={'minWidth': '260px'}),
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
                            ], id='heatmap_airline_control', style={'display': 'none', 'minWidth': '300px', 'flex': '1'})
                        ], style={
                            'display': 'flex',
                            'gap': '16px',
                            'alignItems': 'flex-end',
                            'flexWrap': 'wrap',
                            'padding': '10px 14px',
                            'borderRadius': '14px',
                            'backgroundColor': 'rgba(255,255,255,0.08)',
                            'marginBottom': '8px'
                        }),
                        width=12
                    )
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='dominant_airline_by_country_map')
                    ], width = 6),
                    dbc.Col([
                        dcc.Graph(id='airline_treemap')
                    ], width = 6)
                ])
            ]
        ),
        dcc.Tab(label='Connections',value='tab-4',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Choose a connection type:'),
                        dcc.Dropdown(
                            id='dropdown6',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in connection_type_choices],
                            value=connection_type_choices[0]
                        )
                    ], width =6),
                    dbc.Col([
                        dbc.Label('Choose a country:'),
                        dcc.Dropdown(
                            id='dropdown7',
                            style={'color':'black'},
                            options=[{'label': i, 'value': i} for i in country_choices],
                            value=country_choices[0]
                        )
                    ], width =6),
                    dbc.Col([
                        dcc.Graph(id = 'network_chart')
                    ], width = 12),
               
            
                    

                ])
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
    Output('dropdown5', 'options'),
    Output('dropdown5', 'value'),
    Output('heatmap_airline_control', 'style'),
    Input('dropdown3', 'value'),
    Input('airline_map_mode', 'value'),
    State('dropdown5', 'value'),
)
def update_heatmap_airline_control(selected_country, map_mode, current_airline):
    prepared_country_data = prepare_country_airline_data(selected_country)
    top_airlines = prepared_country_data[2]
    options = [{'label': airline, 'value': airline} for airline in top_airlines]
    value = current_airline if current_airline in top_airlines else (top_airlines[0] if top_airlines else None)
    style = (
        {'display': 'block', 'minWidth': '300px', 'flex': '1'}
        if map_mode == 'Airline % Share'
        else {'display': 'none', 'minWidth': '300px', 'flex': '1'}
    )
    return options, value, style


@app.callback(
    Output('dominant_airline_by_country_map', 'figure'),
    Input('dropdown3', 'value'),
    Input('airline_map_mode', 'value'),
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
            height=500,
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
            coloraxis_colorbar=dict(
                title="% Share",
                tickformat=".0%",
                bgcolor="black",
                bordercolor="rgba(0,0,0,0)",
                tickfont=dict(color="white"),
                #titlefont=dict(color="white"),
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
        height=500,
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

if __name__=='__main__':
	app.run()