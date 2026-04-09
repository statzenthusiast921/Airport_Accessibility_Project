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


dest_tbl = (
    airport_df[["display_name", "dest_name",'dest_iata', "connectivity_index", "redundancy_score"]].rename(
        columns={
            "dest_name": "Destination",
            "connectivity_index": "Connectivity Index",
            "redundancy_score": "Redundancy Index",
        }
    )
)

# #-----Set up choices for dropdown menus
country_choices = sorted(airport_df['country'].unique())
airport_choices = sorted(airport_df['display_name'].unique())
# pr_choices = sorted(amtrak_df['business_line'].unique())


def great_circle_points(lat1, lon1, lat2, lon2, n=50):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d = 2 * math.asin(math.sqrt(
        math.sin((lat2 - lat1) / 2) ** 2 +
        math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    ))
    if d == 0:
        return [math.degrees(lat1)], [math.degrees(lon1)]
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
                        dbc.Card(id='card1')
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card2')
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card3')
                    ],width=3),
                    dbc.Col([
                        dbc.Card(id='card4')
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
        dcc.Tab(label='something else on tab 2',value='tab-3',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
        
      
                ])
            ]
        ),
        dcc.Tab(label='3rd tab',value='tab-4',style=tab_style, selected_style=tab_selected_style,
            children=[
                dbc.Row([
                    dbc.Col([
                
                    ], width =6),
                    dbc.Col([
                
                    ], width =6),
                    dbc.Col([
             
                    ], width = 12),
               
            
                    

                ])
            ]
        )
    ])
])

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

    card1 = dbc.Card(
    dbc.CardBody([
        html.P(
            f"Destinations from {airport_code}",
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
            f"{metric1}",
            style={
                "margin": "0",
                "fontSize": "2.4rem",
                "fontWeight": "700",
                "lineHeight": "1",
                "color": "white"
            }
        )
    ],
    style={
        "padding": "0.5rem 0.25rem"
    }),
    style={
        "width": "100%",
        "border": "none",
        "borderRadius": "18px",
        "background": "linear-gradient(135deg, #2E91E5 0%, #1B5FC1 100%)",
        "boxShadow": "0 10px 24px rgba(46, 145, 229, 0.35)",
        "textAlign": "left"
    }
    )
    card2 = dbc.Card(
    dbc.CardBody([
        html.P(
            f"# Countries Accessible from {airport_code}",
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
            f"{metric2}",
            style={
                "margin": "0",
                "fontSize": "2.4rem",
                "fontWeight": "700",
                "lineHeight": "1",
                "color": "white"
            }
        )
    ],
    style={
        "padding": "0.5rem 0.25rem"
    }),
    style={
        "width": "100%",
        "border": "none",
        "borderRadius": "18px",
        "background": "linear-gradient(135deg, #2E91E5 0%, #1B5FC1 100%)",
        "boxShadow": "0 10px 24px rgba(46, 145, 229, 0.35)",
        "textAlign": "left"
    }
    )
    card3 = dbc.Card(
    dbc.CardBody([
        html.P(
            f"Connectivity Index for {airport_code}",
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
            f"{metric3}",
            style={
                "margin": "0",
                "fontSize": "2.4rem",
                "fontWeight": "700",
                "lineHeight": "1",
                "color": "white"
            }
        )
    ],
    style={
        "padding": "0.5rem 0.25rem"
    }),
    style={
        "width": "100%",
        "border": "none",
        "borderRadius": "18px",
        "background": "linear-gradient(135deg, #2E91E5 0%, #1B5FC1 100%)",
        "boxShadow": "0 10px 24px rgba(46, 145, 229, 0.35)",
        "textAlign": "left"
    }
    )
    card4 = dbc.Card(
    dbc.CardBody([
        html.P(
            f"Redundancy Index for {airport_code}",
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
            f"{metric4}",
            style={
                "margin": "0",
                "fontSize": "2.4rem",
                "fontWeight": "700",
                "lineHeight": "1",
                "color": "white"
            }
        )
    ],
    style={
        "padding": "0.5rem 0.25rem"
    }),
    style={
        "width": "100%",
        "border": "none",
        "borderRadius": "18px",
        "background": "linear-gradient(135deg, #2E91E5 0%, #1B5FC1 100%)",
        "boxShadow": "0 10px 24px rgba(46, 145, 229, 0.35)",
        "textAlign": "left"
    }
    )
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
        mapbox_style='open-street-map',
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


if __name__=='__main__':
	app.run()