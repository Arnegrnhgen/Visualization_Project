from typing import List
from uuid import uuid4

import pandas as pd
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

from datetime import datetime
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB

from data import get_doocs_properties, load_parquet_data
from pathlib import Path
import base64


def ffill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    ffilled = arr[idx]  # Use the index array to forward fill the original array

    i=0
    while np.isnan(ffilled[0]) and i<len(ffilled):
        if not np.isnan(ffilled[i]):
            filler=ffilled[i]
            for j in range(i):
                ffilled[j] = filler
        i += 1
    return ffilled


def process_data(np_array, outlier_removal=False):
    if outlier_removal:
        median = np.median(np_array)
        stdev = np.std(np_array)
        lower_threshold = median - 4*stdev 
        upper_threshold = median + 4*stdev
        clean_idc = np.logical_and(upper_threshold >= np_array, lower_threshold <= np_array)
        np_array[~clean_idc] = np.nan
        np_array = ffill(np_array)

    return np_array


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
doocs_properties_inversed = {}
for k, v in doocs_properties.items():
    doocs_properties_inversed[v] = k

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
                    id="dcc_checklist-config",
                    options=[
                        "separate plots",
                        "remove outliers (1% - 99%)"
                    ],
                    value=[],
                ),
                html.Br(),
                html.Button("Load Data and Plot", id="dcc_button-loadplot"),
                html.Br(),
                html.Div(id="container", children=[]),
            ]),
            dcc.Tab(label="data analysis", children=[


                html.Label('Laser Properties'), dcc.Dropdown([i for i in doocs_properties.values() if "XFEL.SYNC/LASER" in i], multi=True, id="dcc_dropdown-laser2"),
                html.Br(),
                html.Label('Link Properties'), dcc.Dropdown([i for i in doocs_properties.values() if "XFEL.SYNC/LINK" in i], multi=True, id="dcc_dropdown-link2"),
                html.Br(),
                html.Label('Climate Properties'), dcc.Dropdown([i for i in doocs_properties.values() if "CLIMATE" in i], multi=True, id="dcc_dropdown-climate2"),
                html.Br(),
                html.Button("correlations", id="dcc_button-correlation"),
                html.Br(),
                html.Div(id="container2", children=[]),
            ]
            )
        ]
    )
])
## callbacks
@app.callback(
    Output("container2", "children"),
    Input("dcc_button-correlation", "n_clicks"),
    State("dcc_dropdown-laser2", "value"),
    State("dcc_dropdown-link2", "value"),
    State("dcc_dropdown-climate2", "value"),
    prevent_initial_call=True,
)
def correlation_analysis(n_clicks: int, laser_values:List[str], link_values:List[str], climate_values:List[str]):
    if laser_values is None:
            laser_values = []
    if link_values is None:
        link_values = []
    if climate_values is None:
        climate_values = []
    properties = laser_values + link_values + climate_values
    
    merged_df = pd.DataFrame({"bunchID":[]})
    pq_data_set = load_parquet_data([Path(doocs_properties_inversed[p]) for p in properties], datetime(2023, 10, 15, 17, 30), datetime(2023, 11, 15, 17, 30))
    for name, pq_df in pq_data_set.items():
        tmp_df = pq_df.to_pandas()[["bunchID", "data"]].rename(columns={"data":doocs_properties[str(name)]})
        merged_df = pd.merge(merged_df, tmp_df, on="bunchID", how="outer")
    merged_df.ffill(inplace=True)
    merged_df.dropna(inplace=True)
    correlation_matrix = merged_df.corr()

    fig_heatmap = go.Figure(
        data=go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.columns,
                        colorscale='Viridis', colorbar=dict(title='Correlation'), ))

    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            annotations.append(
                dict(x=correlation_matrix.columns[j], y=correlation_matrix.columns[i], text=str(round(value, 2)),
                     xref='x', yref='y',
                     font=dict(color='white'),
                     showarrow=False)
            )

    fig_heatmap.update_layout(title='Correlation Matrix', xaxis_title='Variables', yaxis_title='Variables',
                              annotations=annotations)
    correlation_heatmap = dcc.Graph(figure=fig_heatmap)

    return [correlation_heatmap]


# This method adds the needed components to the front-end, but does not yet contain the
# FigureResampler graph construction logic.
@app.callback(
    Output("container", "children", allow_duplicate=True),
    Input("dcc_button-loadplot", "n_clicks"),
    State("dcc_dropdown-laser", "value"),
    State("dcc_dropdown-link", "value"),
    State("dcc_dropdown-climate", "value"),
    State("dcc_checklist-config", "value"),
    State("container", "children"),
    prevent_initial_call=True,
)
def add_graph_div(n_clicks: int, laser_values:List[str], link_values:List[str], climate_values:List[str], plot_config:List, div_children: List[html.Div]):
    if not ('separate plots' in plot_config):
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

    elif 'separate plots' in plot_config:
        div_children = []
        
        if laser_values is None:
            laser_values = []
        if link_values is None:
            link_values = []
        if climate_values is None:
            climate_values = []
        properties = laser_values + link_values + climate_values
        for p in properties:
            uid = str(uuid4())

            new_child = html.Div(
                children=[
                    dcc.Graph(id={"type": "dynamic-graph", "index": uid, "name": str(p)}, figure=go.Figure()),
                    dcc.Loading(dcc.Store(id={"type": "store", "index": uid, "name": str(p)})),
                    dcc.Interval(id={"type": "interval", "index": uid, "name": str(p)}, max_intervals=1, interval=1),
                ]
            )
            div_children.append(new_child)
        
    return div_children


@app.callback(
    State("dcc_dropdown-laser", "value"),
    State("dcc_dropdown-link", "value"),
    State("dcc_dropdown-climate", "value"),
    State("dcc_checklist-config", "value"),
    State({"type": "dynamic-graph", "index": MATCH, "name":MATCH}, "id"),
    Output({"type": "dynamic-graph", "index": MATCH, "name":MATCH}, "figure"),
    Output({"type": "store", "index": MATCH, "name":MATCH}, "data"),
    Trigger({"type": "interval", "index": MATCH, "name":MATCH}, "n_intervals"),
    prevent_initial_call=True,
)
def construct_display_graph(laser_values, link_values, climate_values,plot_config:List, graph_id) -> FigureResampler:
    fig = FigureResampler(
        go.Figure(),
        default_n_shown_samples=2_000,
        default_downsampler=MinMaxLTTB(parallel=True),
    )
    if laser_values is None:
        laser_values = []
    if link_values is None:
        link_values = []
    if climate_values is None:
        climate_values = []
    properties = laser_values + link_values + climate_values
    remove_outliers = 'remove outliers (1% - 99%)' in plot_config
    
    if graph_id["name"] == 'all':
        pq_data_set = load_parquet_data([Path(doocs_properties_inversed[p]) for p in properties], datetime(2023, 10, 15, 17, 30), datetime(2023, 11, 15, 17, 30))

        for pq in pq_data_set:
            x_values = pq_data_set[pq]["datetime"].to_numpy()
            y_values = pq_data_set[pq]["data"].to_numpy()
            fig.add_trace(dict(name=doocs_properties[str(pq)]), hf_x=x_values, hf_y=process_data(y_values, remove_outliers))
        if len(pq_data_set) == 1:
            fig.update_layout(title=f"<b>History: {doocs_properties[str(pq)]}</b>", title_x=0.5)
        else:
            fig.update_layout(title=f"<b>Histories</b>", title_x=0.5)
    else:

        p = graph_id["name"]
        pq_data_set = load_parquet_data([Path(doocs_properties_inversed[p])], datetime(2023, 10, 15, 17, 30), datetime(2023, 11, 15, 17, 30))
        
        if len(pq_data_set) != 1:
            raise ValueError("Check separate implementation")
        for pq in pq_data_set:
            x_values = pq_data_set[pq]["datetime"].to_numpy()
            y_values = pq_data_set[pq]["data"].to_numpy()
            fig.add_trace(dict(name=doocs_properties[str(pq)]), hf_x=x_values, hf_y=process_data(y_values, remove_outliers))
            fig.update_layout(title=f"<b>History: {doocs_properties[str(pq)]}</b>", title_x=0.5)
        
    fig.update_layout(yaxis_tickformat = '.5f')
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
