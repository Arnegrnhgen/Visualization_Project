from typing import List
from uuid import uuid4

import numpy as np
import plotly.graph_objects as go
from dash import MATCH, Input, Output, State, dcc, html, no_update
from dash_extensions.enrich import (
    DashProxy,
    Serverside,
    ServersideOutputTransform,
    Trigger,
    TriggerTransform,
)

x = np.arange(2_000_000)
noisy_sin = (3 + np.sin(x / 200) + np.random.randn(len(x)) / 10) * x / 1_000


from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB

from data import get_doocs_properties, load_parquet_data
from pathlib import Path

import base64


# logo
with open("lbsync.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

header = html.Div([
    html.Img(src=f"data:image/png;base64,{encoded_image}",
             style={'height': '100px', 'width': 'auto', 'margin-right': '150px'}),
    html.H1("LbSync Dashboard", style={'textAlign': 'center', 'line-height': '50px'})
], style={'display': 'flex', 'align-items': 'center'})

# Define properties
doocs_properties = get_doocs_properties(Path("C:/Users/Arne/Sources/Ml_data/daqdata/sorted"))

# Data storage
online_data = {}

# Create a Dash app with server-side output transformation
app = DashProxy(__name__, transforms=[ServersideOutputTransform(), TriggerTransform()])

header = html.Div([
    html.Img(src=f"data:image/png;base64,{encoded_image}", style={'height': '100px', 'width': 'auto', 'margin-right': '150px'}),
    html.H1("LbSync Dashboard", style={'textAlign': 'center', 'line-height': '50px'})
], style={'display': 'flex', 'align-items': 'center'})

body = html.Div([
    dcc.Tabs(
        children=[
            dcc.Tab(label="History plotter", children=[
                html.Label('Laser Properties'), dcc.Dropdown([i for i in doocs_properties.values() if "XFEL.SYNC/LASER" in i], multi=True, id="dcc_dropdown-laser"),
                html.Br(),
                html.Label('Link Properties'), dcc.Dropdown([i for i in doocs_properties.values() if "XFEL.SYNC/LINK" in i], multi=True, id="dcc_dropdown-link"),
                html.Br(),
                html.Label('Climate Properties'), dcc.Dropdown([i for i in doocs_properties.values() if "CLIMATE" in i], multi=True, id="dcc_dropdown-climate"),
                html.Br(),
                dcc.Checklist(
                    id="dcc_checklist-separate",
                    options=[{"label": "Plot properties separately", "value": True}],
                    value=[],
                ),
                html.Button("Load Data and Plot", id="dcc_button-loadplot"),
                html.Br(),
                html.Div(id="container", children=[]),
            ]),
            dcc.Tab(label="data analysis")
        ]
    )
])


## callbacks

# This method adds the needed components to the front-end, but does not yet contain the
# FigureResampler graph construction logic.
@app.callback(
    Output("container", "children"),
    Input("dcc_button-loadplot", "n_clicks"),
    State("dcc_dropdown-laser", "value"),
    State("dcc_dropdown-link", "value"),
    State("dcc_dropdown-climate", "value"),
    State("dcc_checklist-separate", "value"),
    State("container", "children"),
    prevent_initial_call=True,
)
def add_graph_div(n_clicks: int, laser_values:List[str], link_values:List[str], climate_values:List[str], seperate:List, div_children: List[html.Div]):

    if seperate == []:
        # plot all in one

        uid = str(uuid4())

        new_child = html.Div(
            children=[
                dcc.Graph(id={"type": "dynamic-graph", "index": uid, "name": "all"}, figure=go.Figure()),
                dcc.Loading(dcc.Store(id={"type": "store", "index": uid, "name": "all"})),
                dcc.Interval(id={"type": "interval", "index": uid, "name": "all"}, max_intervals=1, interval=1),
            ]
        )
        div_children = [new_child]
        return div_children


    elif seperate[0]:
        
        if laser_values is None:
            laser_values = []
        if link_values is None:
            link_values = []
        if climate_values is None:
            climate_values = []
        properties = laser_values + link_values + climate_values

    else:
        raise ValueError("check implemetation of plot seperately button")


@app.callback(
        
    State("dcc_dropdown-laser", "value"),
    State("dcc_dropdown-link", "value"),
    State("dcc_dropdown-climate", "value"),
    State({"type": "dynamic-graph", "index": MATCH, "name":MATCH}, "id"),

    Output({"type": "dynamic-graph", "index": MATCH, "name":MATCH}, "figure"),
    Output({"type": "store", "index": MATCH, "name":MATCH}, "data"),
    Trigger({"type": "interval", "index": MATCH, "name":MATCH}, "n_intervals"),
    prevent_initial_call=True,
)
def construct_display_graph(laser_values, link_values, climate_values, graph_id) -> FigureResampler:
    fig = FigureResampler(
        go.Figure(),
        default_n_shown_samples=2_000,
        default_downsampler=MinMaxLTTB(parallel=True),
    )

    if graph_id["name"] == 'all':
        sigma = 1e-6
        fig.add_trace(dict(name="log"), hf_x=x, hf_y=noisy_sin * (1 - sigma) ** x)
        fig.add_trace(dict(name="exp"), hf_x=x, hf_y=noisy_sin * (1 + sigma) ** x)
        fig.update_layout(title=f"<b>graph - {1}</b>", title_x=0.5)

    return fig, Serverside(fig)


# The plotly-resampler callback to update the graph after a relayout event (= zoom/pan)
# As we use the figure again as output, we need to set: allow_duplicate=True
@app.callback(
    Output({"type": "dynamic-graph", "index": MATCH, "name":MATCH}, "figure", allow_duplicate=True),
    Input({"type": "dynamic-graph", "index": MATCH, "name":MATCH}, "relayoutData"),
    State({"type": "store", "index": MATCH, "name":MATCH}, "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata: dict, fig: FigureResampler):
    if fig is not None:
        return fig.construct_update_data_patch(relayoutdata)
    return no_update


if __name__ == "__main__":
    app.layout = html.Div([header,body])
    app.run_server(debug=True, port=9023)