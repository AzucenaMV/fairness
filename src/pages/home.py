from dash import Dash, html, dcc, Input, Output
from dash import dcc, html
from dash.dependencies import Input, Output, State
from subprocess import call
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import subprocess
import dash

#dash.register_page(__name__, path='/')
external_stylesheets = [dbc.themes.BOOTSTRAP] 
app = Dash(__name__,
           external_stylesheets=external_stylesheets) 

navbar = dbc.NavbarSimple(
    brand=html.H4("Fairness App"),
    brand_href="#",
    color="#002654",
    dark=True,
    fluid= True,
    style={'padding-left': '15px', "height": "8vh", "margin-bottom":"15px"},
)


content = html.Div(
    [
    dbc.Row([
        html.H6("Optimization Metrics",
        style={'margin-left': '30px','color':'#424949'}),
        ], 
        style={'margin-bottom' : '10px'}
     ),
    dbc.Row([
        dbc.Col([
            html.P("Performance Metric",
                className = 'text-center text-primary, mb-4 ',
                style={
                    'margin-left': '40px', 
                    'vertical-align' : 'middle'}
                    ),
                ], width = 4),
        dbc.Col([
            dcc.Dropdown(
                id='dropdown_opt_model_metrics',
                options=[
                    {'label': 'Recall', 'value': 'recall_score'},
                    {'label': 'Precision', 'value': 'precision_score'},
                    {'label': 'F1 score', 'value': 'f1_score'},
                    {'label': 'Accuracy', 'value': 'accuracy_score'},
                ],
                clearable = False,
                value='f1_score',
                style={'width': '70%',
                        'vertical-align' : 'middle'}
            ), 
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.P("Fairness Metric",
                className = 'text-center text-primary, mb-4 ',
                style={
                    'margin-left': '40px', 
                    'vertical-align' : 'middle'}
                    ),
                ], width = 4),
        dbc.Col([
            dcc.Dropdown(
                id='dropdown_opt_fair_metrics',
                options=[
                    {'label': 'Demographic Parity', 'value': 'demographic_parity_difference'},
                    {'label': 'Predictive Parity', 'value': 'predictive_parity_difference'},
                    {'label': 'Predictive Equality', 'value': 'predictive_equality_difference'},
                    {'label': 'Equal Opportunity', 'value': 'equal_opportunity_difference'},
                    {'label': 'Avg Absolute Odds', 'value': 'avg_abs_odds_difference'},
                ],
                clearable = False,
                value='ppd',
                style={'width': '70%', "margin-bottom":"10px"}
            ),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.P("Sensitive Attribute",
                className = 'text-center text-primary, mb-4 ',
                style={
                    'margin-left': '40px', 
                    'vertical-align' : 'middle',
                    }),
                ], width = 4),
        dbc.Col([
            dcc.Dropdown(
                id='dropdown_opt_sensitive_attribute',
                options=[
                    {'label': 'Gender', 'value': 'gender'},
                    {'label': 'Sex', 'value': 'sex'},
                    {'label': 'Capital Gain', 'value': 'capital_gain'},
                    {'label': 'Education Level', 'value': 'education_level'},
                ],
                clearable = False,
                value='gender',
                style={'width': '70%', "margin-bottom":"15px"}
            ),
        ])
    ]),
    # dbc.Row([
    #     html.H6("Evaluation Metrics",
    #     style={'margin-left': '30px','color':'#424949'}),
    #     ], 
    #     style={'margin-bottom' : '10px'}
    #  ),
    # dbc.Row([
    #     dbc.Col([
    #         html.P("Performance Metric",
    #             className = 'text-center text-primary, mb-4 ',
    #             style={
    #                 'margin-left': '40px', 
    #                 'vertical-align' : 'middle',
    #                 }),
    #             ], width = 4),
    #     dbc.Col([
    #         dcc.Dropdown(
    #             id='dropdown_eval_model_metrics',
    #             options=[
    #                 {'label': 'Recall', 'value': 'recall'},
    #                 {'label': 'Precision', 'value': 'precision'},
    #                 {'label': 'F1 score', 'value': 'f1_score'},
    #                 {'label': 'Accuracy', 'value': 'accuracy'},
    #             ],
    #             value=['recall','precision','f1_score','accuracy'],
    #             multi = True,
    #             style={'width': '100%', "margin-bottom":"15px"}
    #         ),
    #     ])
    # ]),
    # dbc.Row([
    #     dbc.Col([
    #         html.P("Fairness Metrics",
    #             className = 'text-center text-primary, mb-4 ',
    #             style={
    #                 'margin-left': '40px', 
    #                 'vertical-align' : 'middle',
    #                 }),
    #             ], width = 4),
    #     dbc.Col([
    #         dcc.Dropdown(
    #             id='dropdown_eval_fair_metrics',
    #             placeholder = "Fairness Metrics",
    #             options=[
    #                 {'label': 'Demographic Parity', 'value': 'dpd'},
    #                 {'label': 'Predictive Parity', 'value': 'ppd'},
    #                 {'label': 'Predictive Equality', 'value': 'ped'},
    #                 {'label': 'Equal Opportunity', 'value': 'eod'},
    #                 {'label': 'Avg Absolute Odds', 'value': 'aao'},
    #             ],
    #             value=['dpd','ppd','ped','eod','aao'],
    #             multi = True,
    #             style={'minWidth': '50%', "margin-bottom":"10px"}
    #         ),
    #     ])
    # ]),
    dbc.Row([
        html.H6("Advanced Options",
        style={'margin-left': '30px','color':'#424949'}),
        ], 
        style={'margin-bottom' : '10px'}
     ),
    dbc.Row([
        dbc.Col([
            html.P("Models",
                className = 'text-center text-primary, mb-4 ',
                style={
                    'margin-left': '40px', 
                    'vertical-align' : 'middle',
                    }),
                ], width = 4),
        dbc.Col([
            dcc.Dropdown(
                id='dropdown_model',
                options=[
                    {'label': 'Random Forest', 'value': 'RF'},
                    {'label': 'GBM', 'value': 'GBM'},
                    {'label': 'LGBM', 'value': 'LGBM'},
                    {'label': 'Logistic Regression', 'value': 'LR'},
                ],
                value=['RF','GBM','LGBM'],
                multi = True,
                style={'width': '100%', "margin-bottom":"15px"}
            ),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.P("Cross Validation Folds",
                className = 'text-center text-primary, mb-4 ',
                style={
                    'margin-left': '40px', 
                    'vertical-align' : 'middle',
                    }),
                ], width = 4),
        dbc.Col([
                dcc.Input(
                    id='input_nfolds',
                    value = 5,
                    type="number",
                    min = 3,
                    step = 1,
                    max = 10,
                    style={'minWidth': '50%'}
                ),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.P("Iterations",
                className = 'text-center text-primary, mb-4 ',
                style={
                    'margin-left': '40px', 
                    'vertical-align' : 'middle',
                    }),
                ], width = 4),
        dbc.Col([
                dcc.Input(
                    id = 'input_niter',
                    value = 100,
                    type="number",
                    min = 1,
                    step = 50,
                    max = 1000,
                    style={'minWidth': '50%'}
                ),
        ]),
    ]),
    html.Div([
        dbc.Button("Start", id="start-button", className="me-md-2")# , disabled = True),
        ],
        className="d-grid gap-2 d-md-flex justify-content-md-end",
        ),
    html.Div(id='output-container-button', 
             children='Hit the button to update.', 
             style={'margin-left': '30px','color':'#424949'})
    ]

)

app.layout = dbc.Container(
        [
            navbar,
            dbc.Row(
                [
                    dbc.Col(content, width=9)
                ]
                ),
            ],
    fluid=True,
    class_name='px-0'
    
)

@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('start-button', 'n_clicks'),
     dash.dependencies.Input('dropdown_opt_model_metrics', 'value'),
     dash.dependencies.Input('dropdown_opt_fair_metrics', 'value'),
     dash.dependencies.Input('dropdown_opt_sensitive_attribute','value'),
     dash.dependencies.Input('dropdown_model','value'),
     dash.dependencies.Input('input_nfolds','value'),
     dash.dependencies.Input('input_niter','value'),
     ])
def run_script_onClick(n_clicks, model_metric, fair_metric, sensitive_attribute, model, n_folds, n_trials):
    if not n_clicks:
        #raise dash.exceptions.PreventUpdate
        return dash.no_update 
    from subprocess import PIPE, Popen
    process = subprocess.Popen(['python', '/home/azucena/fairness/src/notebooks/fair_ho.py', 
                               '--fair_metric', fair_metric, '--model_metric', model_metric,
                               '--sensitive_attribute', sensitive_attribute, '--models', " ".join(model),
                               '--n_folds', str(n_folds), '--n_trials', str(n_trials)], stdout=PIPE)
    output = process.communicate()[0]
    return output

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)