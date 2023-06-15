from dash import Dash, html, dcc, Input, Output
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

#dash.register_page(__name__, path='/')
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
                    {'label': 'Recall', 'value': 'recall'},
                    {'label': 'Precision', 'value': 'precision'},
                    {'label': 'F1 score', 'value': 'f1_score'},
                    {'label': 'Accuracy', 'value': 'accuracy'},
                ],
                clearable = False,
                value='f1_score',
                style={'width': '70%',
                        #'margin-left': '15px', 
                        #"margin-top": "20px",
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
                    {'label': 'Statistical Parity Diff', 'value': 'spd'},
                    {'label': 'Equal Opportunity Diff', 'value': 'eod'},
                    {'label': 'Predictive Equality Diff', 'value': 'ped'},
                    {'label': 'Avg Odd Diff', 'value': 'avd'},
                ],
                clearable = False,
                value='spd',
                style={'width': '70%', "margin-bottom":"10px"}
            ),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.P("Sensitive Attributes",
                className = 'text-center text-primary, mb-4 ',
                style={
                    'margin-left': '40px', 
                    'vertical-align' : 'middle',
                    #'text-align':'right'
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
    #html.Br(),
    dbc.Row([
        html.H6("Evaluation Metrics",
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
                    'vertical-align' : 'middle',
                    #'text-align':'right'
                    }),
                ], width = 4),
        dbc.Col([
            dcc.Dropdown(
                id='dropdown_model',
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'GBM', 'value': 'gbm'},
                    {'label': 'LGBM', 'value': 'lgbm'},
                    {'label': 'Logistic Regression', 'value': 'lr'},
                ],
                value=['rf','gbm','lgbm'],
                multi = True,
                style={'width': '100%', "margin-bottom":"15px"}
            ),
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.P("Fairness Metrics",
                className = 'text-center text-primary, mb-4 ',
                style={
                    'margin-left': '40px', 
                    'vertical-align' : 'middle',
                    #'text-align':'right'
                    }),
                ], width = 4),
        dbc.Col([
            dcc.Dropdown(
                id='dropdown_eval_fair_metrics',
                placeholder = "Fairness Metrics",
                options=[
                    {'label': 'Statistical Parity Diff', 'value': 'spd'},
                    {'label': 'Equal Opportunity Diff', 'value': 'eod'},
                    {'label': 'Predictive Equality Diff', 'value': 'ped'},
                    {'label': 'Avg Odd Diff', 'value': 'avd'},
                ],
                value=['spd','eod','ped','avd'],
                multi = True,
                style={'minWidth': '50%', "margin-bottom":"10px"}
            ),
        ])
    ]),
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
                    #'text-align':'right'
                    }),
                ], width = 4),
        dbc.Col([
            dcc.Dropdown(
                id='dropdown_model',
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'GBM', 'value': 'gbm'},
                    {'label': 'LGBM', 'value': 'lgbm'},
                    {'label': 'Logistic Regression', 'value': 'lr'},
                ],
                value=['rf','gbm','lgbm'],
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
                    #'text-align':'right'
                    }),
                ], width = 4),
        dbc.Col([
                dcc.Input(
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
                    #'text-align':'right'
                    }),
                ], width = 4),
        dbc.Col([
                dcc.Input(
                    value = 100,
                    type="number",
                    min = 50,
                    step = 50,
                    max = 1000,
                    style={'minWidth': '50%'}
                ),
        ])
    ]),

    ]

)


app.layout = dbc.Container(
        [
            navbar,
            dbc.Row(
                [
                    #dbc.Col(sidebar, width=3, className='bg-light'),
                    dbc.Col(content, width=9)
                ]
                ),
            ],
    fluid=True,
    class_name='px-0'
    
)

if __name__ == '__main__':
    app.run_server(debug=True)