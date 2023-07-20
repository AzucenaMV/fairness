
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.style.use('seaborn-v0_8-darkgrid')

def create_df_groups_metrics(n, metrics):
    model = metrics['overall'].loc[metrics['overall'].best_trial == 5,'model_name']
    n_model = metrics['default_bygroup'].model.isin(model)
    dif = metrics['default_bygroup'][n_model].apply(pd.to_numeric, errors='coerce').diff().abs().iloc[-1,:]
    dif[0] = 'Difference'
    df_groups_u = metrics['default_bygroup'][n_model].T
    df_groups_u = pd.concat([df_groups_u,dif], axis = 1).T
    df_groups_u.columns = df_groups_u.columns + ' u'

    dif = metrics['bygroup'][metrics['bygroup'].best_trial == 5].apply(pd.to_numeric, errors='coerce').diff().abs().iloc[-1,:]
    dif[0] = 'Difference'
    df_groups_m = metrics['bygroup'][metrics['bygroup'].best_trial == 5].T
    df_groups_m = pd.concat([df_groups_m,dif], axis = 1).T
    df_groups = pd.concat([df_groups_u,df_groups_m],axis = 1).reset_index(drop=True)
    return df_groups

def graph_fair_opt_orig(df_groups, mapping):
    name = df_groups.columns[0]
    fig = make_subplots(rows=1, cols=len(mapping), horizontal_spacing=.01)
    for i,(col, metric) in enumerate(zip(mapping.keys(),mapping.values())):
        if i == 0:
            showlegend = True
        else:
            showlegend = False

        fig.add_trace(go.Bar(
                            name = 'Non-Optmized',
                            y=df_groups[name] + ' ', 
                            x=df_groups[col + ' u'],
                            orientation='h', 
                            width=0.4,
                            legendgroup="Non-Optmized",
                            showlegend= showlegend, 
                            marker_color='#d1d1e0'),
                            row = 1, col = i+1)
                            #row = int(np.ceil((i+1)/2)), col = (i % 2) + 1) 

        fig.add_trace(go.Bar(
                            #name = metric + ' optimized',
                            name = 'Optimized',
                            y=df_groups[name] + ' ', 
                            x=df_groups[col],
                            #legendgrouptitle_text="Optmized",
                            legendgroup="Optmized",
                            orientation='h', 
                            width=0.4, 
                            showlegend= showlegend, 
                            marker_color='#ADD8E6'),
                            row = 1, col = i+1)

        fig.update_xaxes(side = 'top', title_text="<span style='font-size:1em;color:#455A64'>"+metric.capitalize()+"</span><br><span style='font-size:.8em;color:#455A64'>"+ col.capitalize()+"</span>", row=1, col=i+1)
        if i != 0:
            fig.update_yaxes(showticklabels=False, row=1, col=i+1) 

    fig.update_layout(
        barmode='group',
        font_size = 12,
        yaxis=dict(type='category'),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=1
        ),
    )
    return fig

def graph_opt_orig(metric, df_groups):
    name = df_groups.columns[0]
    fig = go.Figure()
    fig.add_trace(go.Bar(
                        name = metric,
                        y=df_groups[name], 
                        x=df_groups[metric + ' u'],
                        orientation='h', 
                        width=0.4, 
                        showlegend= True, 
                        marker_color='#d1d1e0')) 

    fig.add_trace(go.Bar(
                        name = metric + ' optimized',
                        y=df_groups[name], 
                        x=df_groups[metric],
                        orientation='h', 
                        width=0.4, 
                        showlegend= True, 
                        marker_color='#ADD8E6')) 


    fig.update_layout(
        barmode='group',
        font_size = 14,
        yaxis=dict(type='category'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    fig.update_yaxes(categoryorder='array', categoryarray= df_groups[name])
    return fig

def indicators(n, metrics, df_metrics, df_metrics_u):
    df = df_metrics.loc[df_metrics.index == n]
    n_model = df_metrics_u['model'].isin(df['model_name'])
    fig = go.Figure()
    spacing = np.linspace(0, 1, num= len(metrics) + 1, endpoint=True)
    for i,metric in enumerate(metrics): 
        if metric in ['recall','precision','f1 score','accuracy']:
            decreasing_color = '#E74C3C'
            increasing_color = '#28B463'
        else:
            decreasing_color = '#28B463' #green
            increasing_color = '#E74C3C' # red

        fig.add_trace(go.Indicator(
            mode = "number+delta",
            value = np.round(df[metric].values[0],3),
            title = {"text": f"<span style='font-size:.8em;color:#455A64'>{metric.capitalize()}</span><br>"},
            delta = {
                'reference': np.round(df_metrics_u.loc[n_model,metric].values[0],3), 
                'relative': True, 
                'decreasing' : dict(color = decreasing_color),
                'increasing' : dict(color = increasing_color),
                'valueformat': ".2f"},
            domain = {'x': [spacing[i], spacing[i+1]], 'y': [0, 1]}))


    fig.update_layout(
        paper_bgcolor="#ECEFF1",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig

def graph_eval_groups(df):
    fig = make_subplots(rows=1, cols=2, shared_xaxes=False,
                        shared_yaxes=True, horizontal_spacing=0)
    
    metric1 = 'false negative rate'
    y =  df.iloc[:,0]  + "     "
    text1 = "<span style='font-size:.9em;color:#455A64;font-weight:bold'>FN </span> <span style='font-size:.9em;color:#455A64'>(P=0,T=1)</span>"
    x1 = df[metric1]
    fig.append_trace(go.Bar(
                        x=x1, 
                        y=y,
                        text=np.round(x1,3), #Display the numbers with thousands separators in hover-over tooltip 
                        textposition='outside',
                        orientation='h', 
                        width=0.4, 
                        showlegend=False, 
                        marker_color='#EEE9BF'), 
                        1, 1) # 1,1 represents row 1 column 1 in the plot grid

    metric2 = 'false positive rate'
    text2 = "<span style='font-size:1em;color:#455A64;font-weight:bold'>FP</span> <span style='font-size:.9em;color:#455A64'>(P=1,T=0)</span>"
    x2 = df[metric2]
    fig.append_trace(go.Bar(
                        x=x2, 
                        y=y,
                        text=np.round(x2,3), 
                        textposition='outside',
                        orientation='h', 
                        width=0.4, 
                        showlegend=False, 
                        marker_color='#ADD8E6'), 
                        1, 2) # 1,2 represents row 1 column 2 in the plot grid

    fig.update_xaxes(showticklabels=False,title_text= text1, row=1, col=1, range = [1,0], showgrid=False)
    fig.update_xaxes(showticklabels=False,title_text= text2, row=1, col=2, range = [0,1], showgrid=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        template = 'plotly_white',
    )
    return fig

def graph_eval_groups_metric(df, metric = 'accuracy'):
    fig = go.Figure()
    y = df.iloc[:,0]  + "     "
    x = df[metric]
    text = f"<span style='font-size:1em;color:#455A64;font-weight:bold'>{metric.capitalize()}</span>"
    fig.add_trace(go.Bar(
                        x=x, 
                        y=y,
                        text=np.round(x,3), 
                        textposition='inside',
                        orientation='h', 
                        width=0.4, 
                        showlegend=False, 
                        marker_color='#ADD8E6')) 
    
    fig.update_xaxes(showticklabels=False,title_text= text, range = [0,1], showgrid=False)
    fig.update_yaxes(tickfont=dict(size = 13), showgrid=False)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        template = 'plotly_white',
    )
    return fig

def fig_train_test(df_metrics, metric_title, train_col, test_col, ranking_metric, title = '', metric_result = ''):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_metrics.index, 
            y=df_metrics[train_col],
            name = 'Train',
            mode='markers', 
            marker_color='#4682B4',
            customdata = df_metrics['model_name'],
            hovertemplate="<br>".join([
                "Ranking: %{x}",
                metric_title +" (train): %{y:,.3f}",
                "Model: %{customdata}",
            ])
            ))
    fig.add_trace(
        go.Scatter(
            x= df_metrics.index, 
            y= df_metrics[test_col], 
            name = 'Test',
            customdata = df_metrics['model_name'],
            mode = 'markers',
            marker_color='#87CEEB',
            hovertemplate="<br>".join([
                "Ranking: %{x}",
                metric_title +" (test): %{y:,.3f}",
                "Model: %{customdata}",
                "<extra></extra>"
            ])
            ))
    fig.update_layout(
        title= "<span style='font-size:.9em;color:#455A64'>"+title+"</span><br><span style='font-size:.8em;color:#455A64'> NMSE: "+str(metric_result)+"</span>",
        xaxis_title="Ranked by "+ranking_metric,
        yaxis_title= metric_title,
        margin=dict(l=20, r=20, t=80, b=60),
        showlegend = True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size= 12),
            ),
    )
    return fig

def fig_train_test_bars(df_metrics, metric_title, train_col, test_col, ranking_metric, title = ''):
    fig = go.Figure()
    fig.add_trace(go.Bar(
            x=df_metrics.index, 
            y=df_metrics[train_col],
            name = 'Train',
            width=0.4, 
            customdata = df_metrics['model'],
            marker_color='#4682B4',
            hovertemplate="<br>".join([
                "Ranking: %{x}",
                metric_title +" (train): %{y:,.3f}",
                "Model: %{customdata}",
                "<extra></extra>"
            ])
            ))
    fig.add_trace(go.Bar(
            x=df_metrics.index, 
            y=df_metrics[test_col],
            name = 'Test',
            width=0.4, 
            customdata = df_metrics['model'],
            marker_color='#87CEEB',
            hovertemplate="<br>".join([
                "Ranking: %{x}",
                metric_title +" (test): %{y:,.3f}",
                "Model: %{customdata}",
                "<extra></extra>"
            ])
            ))

    fig.update_layout(
        title= title,
        xaxis_title="Ranking by "+ranking_metric,
        yaxis_title= metric_title,
        barmode='group',
        margin=dict(l=20, r=20, t=40, b=40),
        showlegend = True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size= 12),
            ),
    )
    return fig

def pareto_fig(df_metrics, df_metrics_u, train_fair_col, train_model_col, fair_metric_name, model_metric_name, colors):
    fig = go.Figure()
    for model, color in zip(df_metrics['model_name'].unique(),colors):
        condition = df_metrics.model_name == model
        fig.add_trace(
            go.Scatter(
                x=df_metrics.loc[condition, train_fair_col], 
                y=df_metrics.loc[condition, train_model_col],
                mode='markers', 
                name = model,
                marker = {'color': color,
                'size': 6 
                },
                customdata = df_metrics[condition].index,
                hovertemplate="<br>".join([
                    fair_metric_name+" (test): %{x:,.3f}",
                    model_metric_name+" (test): %{y:,.3f}",
                    "Ranking: %{customdata}",
                ])
                ))
        condition = df_metrics_u.model == model
        fig.add_trace(
            go.Scatter(
                    x=df_metrics_u.loc[condition, train_fair_col], 
                    y=df_metrics_u.loc[condition, train_model_col],
                    mode='markers', 
                    name = model,
                    marker = {
                        'color': color,
                        'size': 6,
                        'symbol' : 'x'
                    },
                    showlegend = False,
                    customdata = df_metrics_u[condition].index,
                    hovertemplate="<br>".join([
                        fair_metric_name+" (default): %{x:,.3f}",
                        model_metric_name+" (default): %{y:,.3f}",
                    ])
            )
        )

    fig.update_layout(
        title="<span style='font-size:.85em;color:#455A64'>Select a model from the Pareto Front:</span><br>",
        xaxis_title=fair_metric_name,
        yaxis_title=model_metric_name,
        margin=dict(l=20, r=20, t=80, b=60),
        clickmode='event+select',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size= 12),
            ),
        showlegend=True 
    )
    return fig

def create_df_metrics(fair_metrics, model_metrics):
    df_fair = pd.DataFrame.from_dict(fair_metrics)
    df_model = pd.DataFrame.from_dict(model_metrics)
    df_metrics = pd.concat([df_fair,df_model], axis = 1)
    return df_metrics

def create_df_ranges(metrics, df_metrics, model):
    df_metric_models = df_metrics[df_metrics.model_name.isin(model)]
    d = []
    for metric in metrics:
        metric_min = np.min(df_metric_models[metric])
        metric_max = np.max(df_metric_models[metric])
        d.append(
            {
                'metric': metric,
                'min': metric_min,
                'max':  metric_max,
                'diff': metric_max - metric_min
            }
        )

    return pd.DataFrame(d)

def eval_metrics_graph(df_metrics, labels, colors, title, ranking_metric, model_selection, n = 15):
    fig = go.Figure()
    df_metrics_model = df_metrics.loc[df_metrics.model_name.isin(model_selection)]
    df_metrics_model['ranking'] = df_metrics_model.index
    df_metrics_model = df_metrics_model.reset_index(drop = True)
    opacity = [.4] * df_metrics_model.shape[0]
    if n in df_metrics_model.ranking:
        n_opacity = df_metrics_model[df_metrics_model.ranking == n].index[0]
        opacity[n_opacity] = 1
    for i, (color, label) in enumerate(zip(colors,labels)):
        fig.add_trace(
            go.Scatter(
                name = label.capitalize(),
                x= df_metrics_model['ranking'], 
                y= df_metrics_model[label], 
                customdata = df_metrics_model.model_name,
                mode = 'markers',
                marker = dict(opacity = opacity),
                marker_color = color,
                hovertemplate = "<br>".join([
                    "Ranking: %{x}",
                    label.capitalize() +" (test): %{y:,.3f}",
                    "Model: %{customdata}",
                    "<extra></extra>",
                ])
        ))
    fig.update_layout(
        xaxis_title="Ranked by "+ranking_metric,
        margin=dict(l=20, r=20, t=85, b=40),
        title={
            'text': title,
            'y':.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1,
            font=dict(size= 9),
            ),
        #title = title
    )
    return fig

def comparison_graph(df_ranges, df_metrics, n, title):
    fig = go.Figure()
    df_ranges['metric'] = df_ranges['metric'].str.capitalize()
    for i in range(df_ranges.shape[0]):
        fig.add_trace(
            go.Bar(
                    x= [df_ranges['min'][i]],
                    y= [df_ranges.metric[i]],
                    width = .05,
                    marker = dict(
                        color = "#d1d1e0",
                    ),
                    orientation='h'))
        fig.add_trace(
            go.Bar(
                    x= [df_ranges['diff'][i]],
                    base = [df_ranges['min'][i]],
                    y= [df_ranges.metric[i]],
                    width = .15,
                    marker = dict(
                        color = '#ADD8E6',
                    ),
                    orientation='h'))
        fig.add_trace(
            go.Bar(
                    x= [1-df_ranges['diff'][i]-df_ranges['min'][i]],
                    base = [df_ranges['min'][i]+df_ranges['diff'][i]],
                    y= [df_ranges.metric[i]],
                    width = .05,
                    marker =dict(
                        color = "#d1d1e0"
                    ),
                    orientation='h'))

        fig.add_trace(
            go.Scatter(
                x = [1],
                y = [df_ranges.metric[i]], 
                mode = 'markers+text',
                text = ['1'],
                textposition="bottom center",
                marker=dict(size=10,
                            symbol = 'line-ns',
                            line=dict(width=2,
                            color="#d1d1e0")
            ))
        )

        fig.add_trace(
            go.Scatter(
                x = [0],
                y = [df_ranges.metric[i]], 
                mode = 'markers+text',
                text = ['0'],
                textfont_size=12,
                textposition="bottom center",
                marker=dict(size=10,
                            symbol = 'line-ns',
                            line=dict(width=2,
                            color="#d1d1e0")
            ))
        )

        fig.add_trace(
            go.Scatter(
                    x= [df_metrics.loc[n,df_ranges.metric[i].lower()]],
                    y= [df_ranges.metric[i]],
                    mode = 'markers+text',
                    textposition = "top center",
                    text = [np.round(df_metrics.loc[n,df_ranges.metric[i].lower()],2)],
                    marker =dict(
                        size = 5,
                        color = "mediumaquamarine",
                        symbol = 'line-ns',
                        line_width=2,
                    ),
                    orientation='h'))

    fig.update_layout(xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                margin=dict(l=20, r=20, t=60, b=60),
                showlegend=False,
                barmode='stack',
                title={
                    'text': title,
                    'y':.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
    )
    return fig