from dash import dcc, html, Dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import json
import pandas as pd
from metrics import (
    equality_opportunity_difference, 
    predictive_equality_difference, 
    metric_evaluation, 
    get_metric_evaluation,
    mse,
    mae,
    nmse,
    nmae
)
from graphs import (
    create_df_ranges, 
    eval_metrics_graph, 
    fig_train_test, 
    create_df_groups_metrics,
    create_df_metrics, 
    pareto_fig, 
    comparison_graph, 
    graph_eval_groups, 
    graph_eval_groups_metric,
    indicators,
    graph_opt_orig,
    graph_fair_opt_orig,
    create_df_groups_metrics
)
import plotly.express as px

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import dash
import dill
import json
import logging
import os

logger = logging.getLogger(__name__)

train_fair_col = 'fair_metric'
train_model_col = 'model_metric'
fair_metric_name = 'Predictive Equality Difference'
model_metric_name = 'F1 Score'
fair_col = 'predictive equality'
model_col = 'f1 score'
train_fair_col = 'fair_metric'
train_model_col = 'model_metric'

import glob

path = '../notebooks/results/sex/'
file_names = [file for file in glob.glob(os.path.join(path,'*'))]

# Reading files
file_name = '../notebooks/metrics.json'
with open(file_name, 'rb') as f:
    metrics_info = json.load(f)

model_metrics = [metrics_info[metric]['short_name'].lower()for metric in metrics_info if metrics_info[metric]['type'] == 'model']
fair_metrics = [metrics_info[metric]['short_name'].lower() for metric in metrics_info if metrics_info[metric]['type'] == 'fairness']
      
# Dictionaries
mapping = {
    "selection rate":"demographic parity",
    "precision":"predictive parity", 
    "recall":"equality opportunity", 
    "false positive rate":"predictive equality",
    #"recall" : "Recall TPR (Equality of Opportunity)",
    }

metrics_bygroup = {
    "accuracy" : "Accuracy",
    #"false positive rate": "False positive rate (Predictive Equality)",
    #"false negative rate" : "False negative rate",
    "f1 score" : "F1 score",
    "precision" : "Precision (Predictive Parity)",
    "recall" : "Recall TPR (Equality of Opportunity)",
    "selection rate": "Selection rate (Demographic Parity)",
    #"true negative rate" : "True negative rate"
}

colors = ['CornflowerBlue','LightCoral','MediumPurple','SandyBrown','lightseagreen']


### Dash 
CONTENT_STYLE = {
    "margin-left": "3rem",
    "margin-right": "3rem",
    "padding": "1rem 1rem",
}

external_stylesheets = [dbc.themes.BOOTSTRAP] 
app = Dash(__name__,
           external_stylesheets=external_stylesheets,
           suppress_callback_exceptions=True) 

nav_layout = dbc.NavbarSimple(
    brand=dbc.NavLink("Fairness App", href="home"),
    brand_href="#",
    color="#0d0d62",
    dark=True,
    fluid= True,
    style={'padding-left': '15px', "height": "8vh", "margin-bottom":"0px"},
    children = [
        dbc.NavItem(dbc.NavLink("Results", href="results")),
    ]
)



tab_results_layout = html.Div([
    html.H6(id='model_metric'), 
    html.H6(id='fair_metric'),
    html.Br(),
     dbc.Row([
        dbc.Col([
            html.P("File",
                className = 'text-center text-primary, mb-4 ',
                style={
                    'margin-left': '40px', 
                    'vertical-align' : 'middle'}

                    ),
                ], width = 2),
        dbc.Col([
                dcc.Dropdown(options =  [{'label': file, 'value': file} for file in file_names], 
                             placeholder="Choose a file.", 
                             id='dropdown_file',
                             value = "f1-ppv-models-motpe-succesivehalving-parallel-150trials-4sim-metrics.pkl",
                             style={'width': '100%',
                                    'vertical-align' : 'middle'}),
            ]),
        ]), 
    dbc.Row([
        dbc.Col([
            html.Br(),
            dcc.Graph(
                id = "paretoFig",
                #figure = paretoFig,
                clickData={'points': [{'customdata': 0}]}
            )
        ],
            width=4),
        dbc.Col([
            html.Br(),
            dcc.Graph(
                id = "fig_fair"
                #figure= fig_fair
                ),
        ], width=4),
        dbc.Col([
            html.Br(),
            dcc.Graph(
                id = "fig_model"
                #figure= fig_model
            ),
        ],width=4),
            ]),
])

tab_height = '6vh'

results_layout = html.Div(
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
                #dcc.Tab(label='Indicators', value = 'tab-indicators', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = tab_indicators),
                #dcc.Tab(label = 'Evaluation', value = 'tab-overall', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = [
                #    dcc.Tabs(
                #        id = 'tabs-comparison',
                #        value = 'tab-overall',
                #        style={
                #            'font-size': '15px',
                #            'height': tab_height
                #        }, 
                #        children = [
                #        dcc.Tab(label='Overall', value = 'tab-overall', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = tab_evaluation_layout),
                #        dcc.Tab(label='Groups', value = 'tab-groups', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = tab_evaluation_groups_layout),
                #        dcc.Tab(label='Original vs Optimized', value = 'tab-comparison', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = tab_comparison)
                #        ]
                #        )])
        ]),
    html.Div(id='tabs-content-example-graph')
])



app.layout = dbc.Container(
        [
            nav_layout,
            dbc.Row(
                [
                    dbc.Col(results_layout, width=12),
                    dcc.Store(id='df_overall'),
                    dcc.Store(id='df_bygroup'),
                    dcc.Store(id='df_default_bygroup')
                ]
                ),
            ],
    fluid=True,
    class_name='px-0'
    
)

    
@app.callback([  
    Output('df_overall', 'data'),
    Output('df_bygroup', 'data'),
    Output('df_default_bygroup', 'data'),
    Output('fair_metric', 'children'),
    Output('model_metric', 'children')],
    [Input('dropdown_file', 'value')])
def update_file(file_name):
    with open(file_name, 'rb') as f:
        metrics = dill.load(f)
    #metrics = results[0]
    fair_metric = metrics['overall'].iloc[1,1]
    model_metric = file_name
    #df_overall = metrics['overall'].to_json(date_format='iso', orient='split')
    df_bygroup = metrics['bygroup'].to_json(date_format='iso', orient='split')
    df_default_bygroup = metrics['default_bygroup'].to_json(date_format='iso', orient='split')
    df_default_overall = metrics['default_overall'].rename(columns = {'model':'model_name'})
    df_optimized_overall = metrics['overall']
    df_overall = pd.concat([df_optimized_overall,df_default_overall])
    df_overall = df_overall.sort_values([fair_col]).reset_index(drop = True).to_json(date_format='iso', orient='split')
    #model_metric = os.path.join(path,file_name)
    #fair_metric = file_name
    #model_metric = metrics['model_metric']
    #fair_metric = metrics['fair_metric']
    #return df_overall, df_bygroup, df_default_overall, df_default_bygroup, fair_metric, model_metric
    return df_overall, df_bygroup, df_default_bygroup, fair_metric, model_metric
    #return df_bygroup, fair_metric, model_metric

#@app.callback(  
#    [Output('fair_metric_name', 'value'),
#     Output('model_metric_name', 'value'),
#     Output('fair_col', 'value'),
#     Output('model_col', 'value'),
#    ],
#    [Input('fair_metric', 'value'),
#    Input('model_metric','value')])
#def update_metrics(fair_metric, model_metric):
#    fair_metric_name = metrics_info[fair_metric]['short_name']
#    model_metric_name = metrics_info[model_metric]['short_name']
#    fair_col = fair_metric_name.replace("_"," ").lower()
#    model_col = model_metric_name.replace("_"," ").lower()
#    return fair_metric_name, model_metric_name, fair_col, model_col

@app.callback(  
    [Output('paretoFig', 'figure'),
     Output('fig_model', 'figure'),
     Output('fig_fair', 'figure')
    ],
    [Input('df_overall', 'data'),
    #Input('fair_col', 'value'),
    #Input('model_col', 'value'),
    #Input('fair_metric_name', 'value'),
    #Input('model_metric_name', 'value')]
    ])
def update_results_figure(overall):
    df_overall = pd.read_json(overall, orient='split')
    df_overall_sim = df_overall[df_overall[train_fair_col].notna()]
    fair_mse = np.round(nmse(df_overall_sim, train_fair_col, fair_col),3)
    model_mse = np.round(nmse(df_overall_sim, train_model_col, model_col),3)
    paretoFig = pareto_fig(
        df_overall,
        fair_col, 
        model_col, 
        fair_metric_name, 
        model_metric_name, 
        colors
    )
    fig_model = fig_train_test(
        df_overall,
        metric_title = model_metric_name,
        train_col = train_model_col,
        test_col =  model_col,
        ranking_metric= fair_col,
        metric_result = model_mse)

    fig_fair = fig_train_test(
        df_overall,
        metric_title = fair_metric_name,
        train_col = train_fair_col,
        test_col = fair_col,
        ranking_metric = fair_col,
        title = 'Train vs Test',
        metric_result = fair_mse)
    return paretoFig, fig_model, fig_fair

if __name__ == '__main__':
    app.run_server(debug=True)
