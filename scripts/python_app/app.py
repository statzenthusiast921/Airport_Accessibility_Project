import gc
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
import helper_data
from helper_data import (
    HOVER_COLS,
    PROXIMITY_EDGE_MAX_MILES,
    connection_type_choices,
    MASTER_AIR_PARQUET_URL,
    dest_table_for_airport,
    initialize_core_state,
    optimize_airport_df_memory,
)
from helper_functions import (
    great_circle_points,
    split_antimeridian_segments,
    extract_airline_names,
    METRIC_CARD_STYLE,
    LABEL_STYLE_WHITE,
    INSTRUCTIONS_MODAL_BODY_STYLE,
    INSTRUCTIONS_INTRO_STYLE,
    INSTRUCTIONS_CODE_STYLE,
    inst_code,
    build_metric_card_body,
    airline_map_controls_row_style,
    AIRLINE_MAP_MODE_PICKER_LABEL_STYLE,
    AIRLINE_CHARTS_ROW_STYLE,
    AIRLINE_MAP_COLUMN_STYLE,
    AIRLINE_TREEMAP_COLUMN_STYLE,
    AIRLINE_MAP_GRAPH_WRAP_STYLE,
    AIRLINE_TREEMAP_GRAPH_WRAP_STYLE,
    AIRLINE_MAP_FIGURE_HEIGHT,
    AIRLINE_TREEMAP_FIGURE_HEIGHT,
    AIRLINE_MAP_COLORS,
    prepare_country_airline_data,
    get_airline_color_map,
    get_country_view,
    hex_to_rgba,
    build_heat_colorscale,
    format_airport_label,
    format_airport_label_from_iata,
)
from helper_network import (
    build_network_connection,
    build_connection_types_guide,
    build_statistical_similarity_detail_body,
    CONN_AIRPORT_ALL,
)






# ----- Startup: master_air only; pre-merged edge parquets load on first Connections use
helper_data.airport_df = pd.read_parquet(MASTER_AIR_PARQUET_URL)
optimize_airport_df_memory(helper_data.airport_df)
print("airport_df MB:", helper_data.airport_df.memory_usage(deep=True).sum() / 1024**2)

initialize_core_state()
gc.collect()

airport_df = helper_data.airport_df
country_choices = helper_data.country_choices
airport_choices = helper_data.airport_choices
country_airport_dict = helper_data.country_airport_dict

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
                        "Click Here for More Information",
                        id="open1",
                        color="secondary",
                        className="w-100",
                        style={"fontSize": 18},
                    ),
                    dbc.Modal([
                        dbc.ModalHeader(
                            html.Span("More Information"),
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
                        "Click Here for More Information",
                        id="open2",
                        color="secondary",
                        className="w-100",
                        style={"fontSize": 18},
                    ),
                    dbc.Modal([
                        dbc.ModalHeader(
                            html.Span("More Information"),
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
                        "Click Here for More Information",
                        id="open3",
                        color="secondary",
                        className="w-100",
                        style={"fontSize": 18},
                    ),
                    dbc.Modal([
                        dbc.ModalHeader(
                            html.Span("More Information"),
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
                            [
                                html.Strong("Note: "),
                                "Shared Destinations uses the raw count: how many cities both airports fly to.",
                                " Hub-Adjusted uses the same count",
                                ", but divides by square root(destinations at airport A X destinations at airport B) so overlap is compared fairly when one airport serves many more cities.",
                            ],
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
    airports = country_airport_dict[selected_country]
    options = [{'label': format_airport_label(i), 'value': i} for i in airports]
    return options, airports[0]


@app.callback(
    Output('card1', 'children'),
    Output('card2', 'children'),
    Output('card3', 'children'),
    Output('card4', 'children'),
    Input('dropdown2', 'value')

)

def airport_selection_stats(selected_airport):
    filtered = dest_table_for_airport(selected_airport)
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
            df['display_name'].map(format_airport_label),
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
        fig.add_trace(go.Scattermap(
            lat=lats, lon=lons,
            mode='lines',
            line=dict(width=1, color='blue'),
            hoverinfo='none',
            showlegend=False,
        ))

    # Destination markers (red)
    fig.add_trace(
        go.Scattermap(
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
        go.Scattermap(
            lat=[origin_lat],
            lon=[origin_lon],
            mode='markers',
            marker=dict(size=14, color='yellow'),
            customdata=(
                make_customdata(origin_info)
                if not origin_info.empty
                else [[format_airport_label(selected_airport), origin_iata, 0, 0, 0]]
            ),
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
        height=360,
        map_style='carto-darkmatter',
        map_center=dict(lat=origin_lat, lon=origin_lon),
        map_zoom=2,
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
    filtered = dest_table_for_airport(selected_airport)
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
  
    joined["display_name_y"] = joined["display_name_y"].map(format_airport_label)
    joined = joined.rename(
        columns={
            'num_dests':'# Destinations',
            'display_name_y':'City, Country (IATA)',
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
def set_airline_airport_options(selected_country):
    airports = country_airport_dict[selected_country]
    options = [{'label': format_airport_label(i), 'value': i} for i in airports]
    return options, airports[0]


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


@app.callback(
    Output('card5', 'children'),
    Output('card6', 'children'),
    Output('card7', 'children'),
    Output('card8', 'children'),
    Input('dropdown3', 'value'),
    Input('dropdown4', 'value'),
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
        metric2,
    )
    card7 = build_metric_card_body(f"Airlines in {airport_code}", metric3)
    card8 = build_metric_card_body(
        f"Airline with highest % share in {airport_code}",
        metric4,
    )
    return card5, card6, card7, card8


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


def _empty_map_figure(message):
    fig = go.Figure()
    fig.update_layout(
        height=AIRLINE_MAP_FIGURE_HEIGHT,
        paper_bgcolor="black",
        plot_bgcolor="black",
        map_style="carto-darkmatter",
        annotations=[{
            "text": message,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5,
            "showarrow": False,
            "font": {"color": "white", "size": 14},
        }],
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


@app.callback(
    Output('dominant_airline_by_country_map', 'figure'),
    Input('dropdown3', 'value'),
    Input('airline_map_mode', 'data'),
    Input('dropdown5', 'value'),
)
def dominant_airline_by_country_map(selected_country, map_mode, selected_airline):
    country_airline_mapping_df, airport_airline_share_df, top_airlines = prepare_country_airline_data(selected_country)
    if country_airline_mapping_df.empty:
        return _empty_map_figure(f"No airline map data for {selected_country}.")

    legend_order = list(top_airlines) + ["Other"]
    color_map = get_airline_color_map(top_airlines)

    if map_mode == 'Airline % Share' and selected_airline:
        heat_df = airport_airline_share_df[
            airport_airline_share_df["selected_airline"] == selected_airline
        ].copy()
        if heat_df.empty:
            return _empty_map_figure(f"No share data for {selected_airline} in {selected_country}.")

        center, zoom = get_country_view(country_airline_mapping_df)
        airline_color = color_map.get(selected_airline, AIRLINE_MAP_COLORS[0])
        max_share = max(float(heat_df["pct_share"].max()), 0.01)
        visible_max_share = max_share * 0.85 if max_share > 0.05 else max_share

        fig = px.density_map(
            heat_df,
            lat="latitude",
            lon="longitude",
            z="pct_share",
            radius=28,
            center=center,
            zoom=zoom,
            map_style="carto-darkmatter",
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

    center, zoom = get_country_view(country_airline_mapping_df)
    fig = px.scatter_map(
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
        center=center,
        zoom=zoom,
        map_style="carto-darkmatter",
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
        margin=dict(l=0, r=0, t=0, b=0),
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
    country_airports = (
        airport_df.loc[airport_df["country"] == selected_country, ["iata", "name", "country"]]
        .drop_duplicates(subset=["iata"])
        .sort_values("name")
    )
    options = [{"label": "All airports in country", "value": CONN_AIRPORT_ALL}]
    for iata, name, country in zip(
        country_airports["iata"],
        country_airports["name"],
        country_airports["country"],
    ):
        options.append({
            "label": format_airport_label(city=name, country=country, iata=iata),
            "value": iata,
        })
    return options, CONN_AIRPORT_ALL





# vis-network defaults use gray node labels; white + subtle stroke reads better on dark surfaces.
@app.callback(
    Output('network_chart', 'children'),
    Output('network-node-meta', 'data'),
    Output('conn-metric-cards', 'children'),
    Input('dropdown6', 'value'),
    Input('dropdown7', 'value'),
    Input('dropdown_conn_airport', 'value'),
)
def network_connections(connection_type, selected_country, focus_airport):
    return build_network_connection(connection_type, selected_country, focus_airport)

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
            html.P([
                html.Strong("IATA airline code: "),
                info.get("iata", "—"),
            ], style={"marginBottom": "8px", "color": "#e8ecf4"}),
            html.P(
                f"Serves {info.get('airport_count', 0)} airports in this view (top carriers / selected country).",
                style={"marginBottom": "12px", "color": "#e8ecf4"},
            ),
            html.P(html.Strong("Airports"), style={"marginBottom": "6px", "color": "#ffffff"}),
            html.Ul([
                html.Li(a) for a in info.get("airports") or []],
                style={"maxHeight": "360px", "overflowY": "auto", "paddingLeft": "20px", "color": "#e8ecf4"},
            )
        ]
        return html.Div([guide, divider] + body)

    # airport (carriers, similarity, proximity, shared-dest, cosine, …)
    airport_title = info.get("name")
    if not airport_title and info.get("iata"):
        airport_title = format_airport_label_from_iata(info["iata"])
    body = [
        html.H4(
            airport_title or info.get("iata", "—"),
            style={"marginBottom": "12px", "color": "#ffffff"},
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
                f"Nearby airport links in this view are capped at "
                f"{PROXIMITY_EDGE_MAX_MILES:.0f} mi."
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
            "Raw overlap: each link counts destinations that appear in both airports' outbound route lists "
            "This is the unadjusted count; large hubs tend to score higher because they serve more cities.",
            style={"marginBottom": "8px", "fontSize": "0.95rem", "color": "rgba(255,255,255,0.92)"},
        ))
        body.append(html.P(html.Strong("Peers (Raw Overlap Counts)"), style={"marginBottom": "6px", "color": "#fff"}))
        body.append(html.Ul(
            [html.Li(a) for a in info.get("shared_dest_peers") or []],
            style={"maxHeight": "320px", "overflowY": "auto", "paddingLeft": "20px", "color": "#e8ecf4"},
        ))
    elif info.get("shared_cosine_peers"):
        body.append(html.P(
            "Hub-adjusted overlap: the same shared destination raw count, but scaled by the destination counts for each airport;"
            " so mega-hubs do not automatically dominate every comparison.",
            style={"marginBottom": "8px", "fontSize": "0.95rem", "color": "rgba(255,255,255,0.92)"},
        ))
        body.append(html.P(html.Strong("Peers (Hub-Adjusted Scores)"), style={"marginBottom": "6px", "color": "#fff"}))
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
    app.run(port=8052)