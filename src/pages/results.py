from dash import Dash, html, dcc, Input, Output
from dash import dcc, html
from dash.dependencies import Input, Output, State
import json
import sys
import pandas as pd
from metrics import equality_opportunity_difference, predictive_equality_difference, metric_evaluation, get_metric_evaluation
from graphs import (
    create_df_ranges, 
    eval_metrics_graph, 
    fig_train_test, 
    create_df_metrics, 
    pareto_fig, 
    comparison_graph, 
    graph_eval_groups, 
    graph_eval_groups_metric,
    indicators,
    graph_opt_orig,
    create_df_groups_metric
)
import plotly.express as px

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

import dash
import dill

CONTENT_STYLE = {
    "margin-left": "3rem",
    "margin-right": "3rem",
    "padding": "1rem 1rem",
}

file_name = 'f1-parity-models-dashboard.pkl'
with open(file_name, 'rb') as in_strm:
    results_dict = dill.load(in_strm)

fair_metrics_dict = {}
fair_metrics_u_dict = {}
fair_metrics = [
    'demographic parity',
    'predictive parity',
    'equality opportunity',
    'predictive equality', 
    'average absolute odds',
    ]

fair_metrics_dict['train_fair'],_ = list(zip(*results_dict['res_sim'][0]))
for metric in fair_metrics:
    fair_metrics_dict[metric] = [get_metric_evaluation(metric_frame)[metric] for metric_frame in results_dict['metrics_sim'][0]]
    fair_metrics_u_dict[metric] = [get_metric_evaluation(metric_frame)[metric] for metric_frame in results_dict['metrics_sim_u'][0]]

model_metrics_dict = {}
model_metrics_u_dict = {}
metrics = [
    'recall', 
    'precision',
    'f1 score',
    'accuracy'
    ]

_,model_metrics_dict['train_model'] = list(zip(*results_dict['res_sim'][0]))
for metric in metrics:
    model_metrics_dict[metric] = [metric_frame.overall[metric] for metric_frame in results_dict['metrics_sim'][0]]
    model_metrics_u_dict[metric] = [metric_frame.overall[metric] for metric_frame in results_dict['metrics_sim_u'][0]]

df_metrics = create_df_metrics(fair_metrics_dict, model_metrics_dict)
df_metrics['model'] = results_dict['models_sim'][0]
df_metrics = df_metrics[df_metrics['train_model'] != 0]
df_metrics_sorted = df_metrics.sort_values(['train_fair'])
new_index = df_metrics_sorted.index
df_metrics_sorted = df_metrics_sorted.reset_index(drop = True)

colors = ['CornflowerBlue','LightCoral','MediumPurple','SandyBrown','lightseagreen']
model_title = 'Model metrics'
fair_title = 'Fairness metrics'

fig_eval_model = eval_metrics_graph(df_metrics_sorted, metrics, colors, model_title)
fig_eval_fairness = eval_metrics_graph(df_metrics_sorted, fair_metrics, colors, fair_title)

fair_metric_name = 'Demographic Parity Difference'
model_metric_name = 'F1 Score'
fair_col = 'demographic parity'
model_col = 'f1 score'
train_fair_col = 'train_fair'
train_model_col = 'train_model'

paretoFig = pareto_fig(df_metrics_sorted, fair_col, model_col, fair_metric_name, model_metric_name, colors)

fig_model = fig_train_test(
    df_metrics_sorted,
    metric_title = model_metric_name,
    train_col = train_model_col,
    test_col =  model_col)

fig_fair = fig_train_test(
    df_metrics_sorted,
    metric_title = fair_metric_name,
    train_col = train_fair_col,
    test_col = fair_col,
    title = 'Train vs Test')

df_fair_ranges = create_df_ranges(fair_metrics, df_metrics_sorted)
df_model_ranges = create_df_ranges(metrics, df_metrics_sorted)


model_mapping = {
    'LogisticRegression':'LR',
    'RandomForestClassifier':'RF',
    'GradientBoostingClassifier':'GBM',
    'LGBMClassifier' : 'LGBM'}
df_metrics_u = create_df_metrics(fair_metrics_u_dict, model_metrics_u_dict)
df_metrics_u['model'] = results_dict['models_sim_u'][0]
df_metrics_u['model_abrv'] = df_metrics_u['model'].map(model_mapping)

### Dash 
external_stylesheets = [dbc.themes.BOOTSTRAP] 
app = Dash(__name__,
           external_stylesheets=external_stylesheets,
           suppress_callback_exceptions=True) #,
           #use_pages=True)

navbar = dbc.NavbarSimple(
    brand=html.H4("Fairness App"),
    brand_href="#",
    color="#002654",
    dark=True,
    fluid= True,
    style={'padding-left': '15px', "height": "8vh", "margin-bottom":"0px"},
)

tab_results_layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Br(),
            dcc.Graph(
                id = "paretofig",
                figure = paretoFig,
                clickData={'points': [{'pointIndex': int(df_metrics_sorted.shape[0]/2)}]}
            )],
            width=4),
        dbc.Col([
            html.Br(),
            dcc.Graph(
                figure= fig_fair
                ),
        ], width=4),
        dbc.Col([
            html.Br(),
            dcc.Graph(
                figure= fig_model
            ),
        ],width=4),
            ]),
])

tab_evaluation_layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Br(),
            dcc.Dropdown(
                id='fair_comparison_choice',
                options=[{'label':'Scatter','value':'scatter'},{'label':'Bars','value':'bars'}],
                value= 'bars',
                style={'width': '70%'}
            ),
            dcc.Graph(
                id = 'fig_evaluation_fair',
                #figure = fig_fair_comparison,
            )],
            width=6),
        dbc.Col([
            html.Br(),
            dcc.Dropdown(
                id='model_comparison_choice',
                options=[{'label':'Scatter','value':'scatter'},{'label':'Bars','value':'bars'}],
                value= 'bars',
                style={'width': '70%'}
            ),
            #html.H5("Model Metrics", style={'color': '#455A64', 'text-align': 'center'}),
            dcc.Graph(
                id = 'fig_evaluation_model',
            )],
            width=6),
        ]),
    ])


tab_evaluation_groups_layout = html.Div(children = [
            dbc.Col(children = [
            html.Br(),   
            dcc.Dropdown(
                id='metric_eval_choice',
                options= [{'label': x.capitalize(), 'value': x} for x in results_dict['metrics_sim'][0][1].by_group.columns],
                value= 'f1 score',
                style={'width': '50%', '':''}
            ),
            html.Br(),
            dcc.Graph(id='fig_eval_groups', 
                        style={"width":600, "height":350, "margin": 0,  'display': 'inline-block'}),
            dcc.Graph(id='fig_eval_groups_metric', 
                        style={"width":350, "height":350, "margin": 0,  'display': 'inline-block'}
                    )]),
        ], className="row")
    
tab_comparison = html.Div([
        html.Br(),
        html.Div([
            dcc.Dropdown(
                id='metric_eval_opt_orig',
                options= [{'label': x.capitalize(), 'value': x} for x in results_dict['metrics_sim'][0][1].by_group.columns],
                value= 'f1 score',
                style={'width': '50%', '':''}
            ),
            #html.Pre(id='click-data'),
            dcc.Graph(
                id = "fig_groups_opt_orig",
            )

        ])
        ])

tab_indicators = html.Div([
        html.Br(),
        html.H5("Model Metrics", style={'color': '#455A64', 'text-align': 'center'}),
        html.Div([
            dcc.Graph(
                id = 'fig_indicators',
                style={"height": 140}
                )
        ], style={'background-color': '#78909C'}),
        html.Br(),
        html.H5("Fair Metrics", style={'color':'#455A64', 'text-align':'center'}),
        html.Div([
            dcc.Graph(
                id = 'fig_indicators_fair',
                style={"height": 140}
            )
        ],  style={'background-color': '#78909C'})
        ])

tab_height = '6vh'
content = html.Div(
    style=CONTENT_STYLE,
    children = [
    dcc.Tabs(id="tabs-example-graph", 
             value='tab-results',
             style={
                'font-size': '15px',
                'height': tab_height
    }, 
            children=[
                dcc.Tab(label='Results', value = 'tab-results', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = tab_results_layout ),
                dcc.Tab(label='Indicators', value = 'tab-indicators', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = tab_indicators),
                dcc.Tab(label = 'Evaluation', value = 'tab-groups', style={'padding': '0','line-height': tab_height}, children = [
                    dcc.Tabs(
                        id = 'tabs-comparison',
                        value = 'tab-evaluation',
                        style={
                            'font-size': '15px',
                            'height': tab_height
                        }, 
                        children = [
                        dcc.Tab(label='Overall', value = 'tab-evaluation', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = tab_evaluation_layout),
                        dcc.Tab(label='Groups', value = 'tab-groups', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = tab_evaluation_groups_layout),
                        dcc.Tab(label='Original vs Optimized', value = 'tab-comparison', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = tab_comparison)]
                        )])
        ]),
    html.Div(id='tabs-content-example-graph')
])

app.layout = dbc.Container(
        [
            navbar,
            dbc.Row(
                [
                    dbc.Col(content, width=12)
                ]
                ),
            ],
    fluid=True,
    class_name='px-0'
    
)


@app.callback(
    Output('fig_eval_groups', 'figure'),
    Output('fig_eval_groups_metric', 'figure'),
    [Input('paretofig', 'clickData'),
     Input('metric_eval_choice','value')])
def update_eval_figure(clickData, metric):
    n = clickData['points'][0]['pointIndex']
    metric_frame = results_dict['metrics_sim'][0][new_index[n]]
    fig_eval_groups = graph_eval_groups(metric_frame)
    fig_eval_groups_metric = graph_eval_groups_metric(metric_frame, metric = metric)
    return fig_eval_groups, fig_eval_groups_metric

@app.callback(
    Output('fig_indicators', 'figure'),
    Output('fig_indicators_fair', 'figure'),
    [Input('paretofig', 'clickData')])
def update_indicators_figure(clickData):
    n = clickData['points'][0]['pointIndex']
    fig_indicators = indicators(int(n), metrics, df_metrics_sorted, df_metrics_u)
    fig_indicators_fair = indicators(int(n), fair_metrics, df_metrics_sorted, df_metrics_u)
    return fig_indicators, fig_indicators_fair

@app.callback(
    Output('click-data', 'children'),
    Input('paretofig', 'clickData'))
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)

@app.callback(
    Output('fig_groups_opt_orig', 'figure'),
    [Input('paretofig', 'clickData'),
     Input('metric_eval_opt_orig', 'value')])
def update_groups_figure(clickData, metric_group):
    #metric_group = 'true positive rate'
    n = clickData['points'][0]['pointIndex']
    df_groups = create_df_groups_metric(int(new_index[n]), metric_group, results_dict, model_mapping)
    return graph_opt_orig(metric_group, df_groups)

@app.callback(  
    Output('fig_evaluation_fair', 'figure'),
    [Input('fair_comparison_choice', 'value'),
     Input('paretofig', 'clickData')])
def update_fair_eval_figure(selected_drop, clickData):
    n = clickData['points'][0]['pointIndex']
    if selected_drop == 'bars':
        fig = comparison_graph(df_ranges = df_fair_ranges, df_metrics = df_metrics_sorted, n = n, title = fair_title)
    elif selected_drop == 'scatter':
        fig = fig_eval_fairness
    return fig

@app.callback(
    Output('fig_evaluation_model', 'figure'),
    [Input('model_comparison_choice', 'value'),
     Input('paretofig', 'clickData')])
def update_model_eval_figure(selected_drop, clickData):
    n = clickData['points'][0]['pointIndex']
    if selected_drop == 'bars':
        fig = comparison_graph(df_ranges = df_model_ranges, df_metrics = df_metrics_sorted, n = n, title = model_title)
    elif selected_drop == 'scatter':
        fig = fig_eval_model
    return fig




if __name__ == '__main__':
    app.run_server(debug=True)