from dash import Dash, html, dcc, Input, Output
from dash import dcc, html
from dash.dependencies import Input, Output, State
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
    'predictive equality', 
    'equality opportunity',
    'average absolute odds',
    'disparity'
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
new_index = df_metrics[df_metrics['train_model'] != 0].index
df_metrics = df_metrics[df_metrics['train_model'] != 0].reset_index(drop = True)
df_metrics_sorted = df_metrics.sort_values(['train_fair']).reset_index(drop = True)

colors = ['SkyBlue','LightCoral','MediumPurple','SandyBrown']
model_title = 'Model metrics'
fair_title = 'Fairness metrics'

fig_eval_model = eval_metrics_graph(df_metrics_sorted, metrics, colors, model_title)
fig_eval_fairness = eval_metrics_graph(df_metrics_sorted, fair_metrics, colors, fair_title)

fair_metric_name = 'Demographic Parity Difference'
model_metric_name = 'F1 Score'
fair_col = 'disparity'
model_col = 'f1 score'
train_fair_col = 'train_fair'
train_model_col = 'train_model'

paretoFig = pareto_fig(df_metrics, train_fair_col, train_model_col, fair_metric_name, model_metric_name)

fig_model = fig_train_test(
    df_metrics_sorted,
    metric_title = model_metric_name,
    train_col = train_model_col,
    test_col =  model_col)

fig_fair = fig_train_test(
    df_metrics_sorted,
    metric_title = fair_metric_name,
    train_col = train_fair_col,
    test_col = fair_col)

df_fair_ranges = create_df_ranges(fair_metrics, df_metrics)
fig_fair_comparison = comparison_graph(df_ranges = df_fair_ranges, df_metrics = df_metrics, n = 25, title = fair_title)

df_model_ranges = create_df_ranges(metrics, df_metrics)
fig_model_comparison = comparison_graph(df_ranges = df_model_ranges, df_metrics = df_metrics, n = 25, title = model_title)

n=20
metric_frame = results_dict['metrics_sim'][0][new_index[20]]
fig_eval_groups = graph_eval_groups(metric_frame)
fig_eval_groups_metric = graph_eval_groups_metric(metric_frame, metric = 'accuracy')

model_mapping = {
    'LogisticRegression':'LR',
    'RandomForestClassifier':'RF',
    'GradientBoostingClassifier':'GBM',
    'LGBMClassifier' : 'LGBM'}
df_metrics_u = create_df_metrics(fair_metrics_u_dict, model_metrics_u_dict)
df_metrics_u['model'] = results_dict['models_sim_u'][0]
df_metrics_u['model_abrv'] = df_metrics_u['model'].map(model_mapping)
fig_indicators = indicators(n, metrics, df_metrics, df_metrics_u)
fig_indicators_fair = indicators(n, fair_metrics, df_metrics, df_metrics_u)

metric_group = 'true positive rate'
df_groups = create_df_groups_metric(new_index[20], metric_group, results_dict, model_mapping)
fig_groups_opt_orig = graph_opt_orig(metric_group, df_groups)
### Dash 
external_stylesheets = [dbc.themes.BOOTSTRAP] 
app = Dash(__name__,
           external_stylesheets=external_stylesheets) #,
           #use_pages=True)

navbar = dbc.NavbarSimple(
    brand=html.H4("Fairness App"),
    brand_href="#",
    color="#002654",
    dark=True,
    fluid= True,
    style={'padding-left': '15px', "height": "8vh", "margin-bottom":"15px"},
)

tab_height = '6vh'
content = html.Div(
    style=CONTENT_STYLE,
    children = [
    dcc.Tabs(id="tabs-example-graph", 
             value='tab-1-example-graph',
             style={
                #'width': '50%',
                'font-size': '15px',
                'height': tab_height
    }, 
            children=[
                dcc.Tab(label='Results', value='tab-results', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}),
                dcc.Tab(label='Summary', value='tab-summary',style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}),
                dcc.Tab(label='Evaluation', value='tab-evaluation',style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}),
                dcc.Tab(label='Evaluation Groups', value='tab-evaluation-groups',style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}),
                dcc.Tab(label='Indicators', value='tab-indicators',style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}),
                dcc.Tab(label='Comparison', value='tab-comparison',style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height})
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

tab_results_layout = html.Div([
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                figure = paretoFig
            ),
            width=4),
        dbc.Col([
            #html.H4("Train vs Test"),
            dcc.Graph(
                figure= fig_fair
                ),
        ], width=4),
        dbc.Col(
            dcc.Graph(
                figure= fig_model
            ),
            width=4),
            ]),
])

tab_summary_layout = html.Div([
    dbc.Row([
        dbc.Col([
            #html.H5("Fair Metrics", style={'color': '#455A64', 'text-align': 'center'}),
            dcc.Graph(
                figure = fig_fair_comparison,
            )],
            width=6),
        dbc.Col([
            #html.H5("Model Metrics", style={'color': '#455A64', 'text-align': 'center'}),
            dcc.Graph(
                figure = fig_model_comparison,
            )],
            width=6),
        ]),
    ])

tab_evaluation_layout = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                figure= fig_eval_model
                ),
        ], width=6),
        dbc.Col(
            dcc.Graph(
                figure= fig_eval_fairness
            ),
            width=6),
        ])
    ])


tab_evaluation_groups_layout = html.Div(children = [
            dbc.Col(children = [
            dcc.Graph(id='g1', 
                        figure= fig_eval_groups, 
                        style={"width":600, "margin": 0,  'display': 'inline-block'}),
            dcc.Graph(id='g2', 
                        figure=fig_eval_groups_metric, 
                        style={"width":350, "margin": 0,  'display': 'inline-block'}
                    )]),
        ], className="row")
    
tab_comparison = html.Div([
        html.Br(),
        #html.H5("Model Metrics", style={'color': '#455A64', 'text-align': 'center'}),
        html.Div([
            dcc.Graph(
                figure= fig_groups_opt_orig,
            )

        ])
        ])

tab_indicators = html.Div([
        html.Br(),
        html.H5("Model Metrics", style={'color': '#455A64', 'text-align': 'center'}),
        html.Div([
            dcc.Graph(
                figure= fig_indicators,
                )
        ], style={'background-color': '#78909C'}),
        html.Br(),
        html.H5("Fair Metrics", style={'color':'#455A64', 'text-align':'center'}),
        html.Div([
            dcc.Graph(
                figure= fig_indicators_fair,
            )
        ],  style={'background-color': '#78909C'})
        ])

@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'))
def render_content(tab):
    if tab == 'tab-results':
        return tab_results_layout 
    elif tab == 'tab-summary':
        return tab_summary_layout
    elif tab == 'tab-evaluation':
        return tab_evaluation_layout
    elif tab == 'tab-evaluation-groups':
        return tab_evaluation_groups_layout
    elif tab == 'tab-indicators':
        return tab_indicators
    elif tab == 'tab-comparison':
        return tab_comparison

if __name__ == '__main__':
    app.run_server(debug=True)