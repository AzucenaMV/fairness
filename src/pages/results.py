from dash import Dash, html, dcc, Input, Output
from dash import dcc, html
from dash.dependencies import Input, Output, State
import json
import sys
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
    create_df_groups_metric,
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

fair_metric_name = 'Demographic Parity Difference'
model_metric_name = 'F1 Score'
fair_col = 'demographic parity'
model_col = 'f1 score'
train_fair_col = 'train_fair'
train_model_col = 'train_model'

fair_mse = np.round(nmse(df_metrics_sorted, train_fair_col, fair_col),3)
model_mse = np.round(nmse(df_metrics_sorted, train_model_col, model_col),3)

paretoFig = pareto_fig(df_metrics_sorted, fair_col, model_col, fair_metric_name, model_metric_name, colors)

fig_model = fig_train_test(
    df_metrics_sorted,
    metric_title = model_metric_name,
    train_col = train_model_col,
    test_col =  model_col,
    ranking_metric= fair_col,
    metric_result = model_mse)

fig_fair = fig_train_test(
    df_metrics_sorted,
    metric_title = fair_metric_name,
    train_col = train_fair_col,
    test_col = fair_col,
    ranking_metric = fair_col,
    title = 'Train vs Test',
    metric_result = fair_mse)



model_mapping = {
    'LogisticRegression':'LR',
    'RandomForestClassifier':'RF',
    'GradientBoostingClassifier':'GBM',
    'LGBMClassifier' : 'LGBM'}
df_metrics_u = create_df_metrics(fair_metrics_u_dict, model_metrics_u_dict)
df_metrics_u['model'] = results_dict['models_sim_u'][0]
df_metrics_u['model_abrv'] = df_metrics_u['model'].map(model_mapping)

### Dash 
CONTENT_STYLE = {
    "margin-left": "3rem",
    "margin-right": "3rem",
    "padding": "1rem 1rem",
}

external_stylesheets = [dbc.themes.BOOTSTRAP] 
app = Dash(__name__,
           external_stylesheets=external_stylesheets,
           suppress_callback_exceptions=True) #,
           #use_pages=True)

navbar = dbc.NavbarSimple(
    brand=html.H4("Fairness App"),
    brand_href="#",
    color="#0d0d62",
    dark=True,
    fluid= True,
    style={'padding-left': '15px', "height": "8vh", "margin-bottom":"0px"},
)

tab_results_layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Br(),
            #html.P('Select a model:'),
            dcc.Graph(
                id = "paretofig",
                figure = paretoFig,
                clickData={'points': [{'customdata': int(df_metrics_sorted.shape[0]/2)}]}
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
            html.Div([
                html.P("Select plot type:", style = {"vertical-align" : "top", "display":"inline-block", "padding-right": 10, "font-size":"1em","color":"#455A64",  "margin-bottom":0, "padding-bottom":0}),
                dcc.RadioItems(
                    id = 'comparison_choice',
                    options = [{'label': html.Span("Bars", style={'font-size': 15, 'padding-left': 10, 'padding-right':10, "color":"#455A64"}),'value':'bars'},
                            {'label': html.Span("Scatter", style={'font-size': 15, 'padding-left': 10, 'padding-right':10, "color":"#455A64"}),'value':'scatter'}],
                    value = 'bars',
                    inline = True,
                    style = {'padding-bottom':0, "margin-bottom":0, "display":"inline-block"}),
                    html.Abbr("\u2139", title="The bar plot shows the ranges in blue (minimum and maximum) from the optimized models for each of the metrics. The line with a value represents the selected model.", 
                        style = {'display':'inline-block',"font-size":"1.1em", "padding-left":20, 'text-decoration': 'none','border':'none'})
            ], style = {"display":"inline-block",  "margin-bottom":0, "padding-bottom":0}),
            dcc.Graph(
                id = 'fig_evaluation_fair',
                style = {"height":400, 'padding-top':0, "margin-top":0}
            )],
            width=6),
        dbc.Col([
            html.Br(),
            html.Div([
                html.P("Filter by model:", style = {"vertical-align" : "top", "display":"inline-block", "padding-right": 10, "font-size":"1em","color":"#455A64",  "margin-bottom":0, "padding-bottom":0}),
                dcc.Checklist(
                    id = 'model_selection',
                    options = [{'label': html.Span(model, style={'font-size': 15, 'padding-left': 10, 'padding-right':10, "color":"#455A64"}), 'value': model} for model in df_metrics.model.unique()],
                    value = df_metrics.model.unique(),
                    inline = True,
                    style = {'padding-bottom':0, "margin-bottom":0, "display":"inline-block"}),
            ], style = {"display":"inline-block",  "margin-bottom":0, "padding-bottom":0}),
            dcc.Graph(
                id = 'fig_evaluation_model',
                style={"height":400, 'padding-top':0, "margin-top":0}
            )],
            width=6),
        ]),
    ])


tab_evaluation_groups_layout = html.Div(children = [
            dbc.Col(children = [
            html.Br(),   
            html.P("Select a metric:", style = {"vertical-align" : "top","display":"inline-block", "padding-right": 15, "font-size":"1.1em","color":"#455A64"}),
            dcc.Dropdown(
                id='metric_eval_choice',
                options= [{'label': value, 'value': key} for key,value in metrics_bygroup.items()],
                value= 'f1 score',
                style={'width': '50%', "display":"inline-block"}
            ),
            html.Br(),
            html.Br(),
            dcc.Graph(id='fig_eval_groups_metric', 
                        style={"width":350, "height":350, "margin-right": 10,  'display': 'inline-block'}),
            dcc.Graph(id='fig_eval_groups', 
                        style={"width":600, "height":350, "margin": 0,  'display': 'inline-block'}),
                    ]),
        ], className="row")
    
tab_comparison = html.Div([
        html.Br(),
        html.Div([
            html.Abbr("\u2139", title="The bar plot shows the ranges in blue (minimum and maximum) from the optimized models for each of the metrics. The line with a value represents the selected model.", 
                style = {"font-size":"1.1em", "padding-left":20, 'text-decoration': 'none','border':'none','float': 'right'}),
            dcc.Graph(
                id = "fig_groups_opt_orig",
                style = {"padding":0,"margin":0} ,
            ),

            ])
        ])

tab_indicators = html.Div([
        html.Br(),
        html.Div([
            html.H5("Model Metrics", style={'color': '#455A64','display':'inline-block'}),
            #html.Button('\u2139', id='indicators-info', n_clicks=0, title= 'hola'),
            html.Abbr("\u2139", title="The main value corresponds to the optmized model that is selected in the results tab. The increase or decrease is relative to the same type of model as the selection with the default hyperparamets. A green arrow means that the optimized model is performing better for that specific metric.", 
                      style = {'display':'inline-block',"font-size":"1.1em", "padding-left":20, 'text-decoration': 'none','border':'none'})
    ], style = {'text-align': 'center'}),
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
                dcc.Tab(label = 'Evaluation', value = 'tab-overall', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = [
                    dcc.Tabs(
                        id = 'tabs-comparison',
                        value = 'tab-overall',
                        style={
                            'font-size': '15px',
                            'height': tab_height
                        }, 
                        children = [
                        dcc.Tab(label='Overall', value = 'tab-overall', style={'padding': '0','line-height': tab_height},selected_style={'padding': '0','line-height': tab_height}, children = tab_evaluation_layout),
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
    n = clickData['points'][0]['customdata']
    metric_frame = results_dict['metrics_sim'][0][new_index[n]]
    fig_eval_groups = graph_eval_groups(metric_frame)
    fig_eval_groups_metric = graph_eval_groups_metric(metric_frame, metric = metric)
    return fig_eval_groups, fig_eval_groups_metric

@app.callback(
    Output('fig_indicators', 'figure'),
    Output('fig_indicators_fair', 'figure'),
    [Input('paretofig', 'clickData')])
def update_indicators_figure(clickData):
    n = clickData['points'][0]['customdata']
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
    [Input('paretofig', 'clickData')])
    # Input('metric_eval_opt_orig', 'value')])
def update_groups_figure(clickData): #, metric_group):
    n = clickData['points'][0]['customdata']
    df_groups = create_df_groups_metrics(int(new_index[n]), results_dict, model_mapping)
    #df_groups = create_df_groups_metric(int(new_index[n]), metric_group, results_dict, model_mapping)
    return graph_fair_opt_orig(df_groups, mapping)

@app.callback(  
    Output('fig_evaluation_fair', 'figure'),
    [Input('comparison_choice', 'value'),
     Input('paretofig', 'clickData'),
     Input('model_selection', 'value')])
def update_fair_eval_figure(selected_drop, clickData, model_selection):
    fair_title = 'Fairness metrics'
    n = clickData['points'][0]['customdata']
    if selected_drop == 'bars':
        df_fair_ranges = create_df_ranges(fair_metrics, df_metrics_sorted, model_selection)
        fig = comparison_graph(df_ranges = df_fair_ranges, df_metrics = df_metrics_sorted, n = n, title = fair_title)
    elif selected_drop == 'scatter':
        fig = eval_metrics_graph(df_metrics_sorted, fair_metrics, colors, fair_title, model_selection= model_selection, n = n, ranking_metric = fair_col)
    return fig

@app.callback(
    Output('fig_evaluation_model', 'figure'),
    [Input('comparison_choice', 'value'),
     Input('paretofig', 'clickData'),
     Input('model_selection', 'value')])
def update_model_eval_figure(selected_drop, clickData, model_selection):
    model_title = 'Model metrics'
    n = clickData['points'][0]['customdata']
    if selected_drop == 'bars':
        df_model_ranges = create_df_ranges(metrics, df_metrics_sorted, model_selection)
        fig = comparison_graph(df_ranges = df_model_ranges, df_metrics = df_metrics_sorted, n = n, title = model_title)
    elif selected_drop == 'scatter':
        fig = eval_metrics_graph(df_metrics_sorted, metrics, colors, model_title, n = n, model_selection= model_selection, ranking_metric = fair_col)
    return fig




if __name__ == '__main__':
    app.run_server(debug=True)