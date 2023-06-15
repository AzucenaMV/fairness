
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
plt.style.use('seaborn-v0_8-darkgrid')
from plotly.subplots import make_subplots



def graph_eval_groups(metric_frame):
    fig = make_subplots(rows=1, cols=2, shared_xaxes=False,
                        shared_yaxes=True, horizontal_spacing=0)
    
    metric1 = 'false negative rate'
    y = metric_frame.by_group[metric1].index  + "  "
    text1 = 'FN (P=0,T=1)'
    x1 = metric_frame.by_group[metric1]
    fig.append_trace(go.Bar(
                        x=x1, 
                        y=y,
                        text=np.round(x1,3), #Display the numbers with thousands separators in hover-over tooltip 
                        textposition='inside',
                        orientation='h', 
                        width=0.3, 
                        showlegend=False, 
                        marker_color='#EEE9BF'), 
                        1, 1) # 1,1 represents row 1 column 1 in the plot grid

    metric2 = 'false positive rate'
    text2 = 'FP (P=1,T=0)'
    x2 = metric_frame.by_group[metric2]
    fig.append_trace(go.Bar(
                        x=x2, 
                        y=y,
                        text=np.round(x2,3), 
                        textposition='inside',
                        orientation='h', 
                        width=0.3, 
                        showlegend=False, 
                        marker_color='#ADD8E6'), 
                        1, 2) # 1,2 represents row 1 column 2 in the plot grid

    fig.update_xaxes(showticklabels=False,title_text= text1, row=1, col=1, range = [1,0], title_font = {"size": 16})
    fig.update_xaxes(showticklabels=False,title_text= text2, row=1, col=2, title_font = {"size": 16}, range = [0,1])
    #fig.update_xaxes(showticklabels=False,title_text= text3, row=1, col=3, title_font = {"size": 16})
    fig.update_yaxes(tickfont=dict(size = 16))
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        template = 'plotly_white'
        #yaxis = {'side': 'right'}
    )
    return fig

def graph_eval_groups_metric(metric_frame, metric = 'accuracy'):
    fig = go.Figure()
    y = metric_frame.by_group[metric].index  + "      "
    x = metric_frame.by_group[metric]
    text = metric.capitalize()
    fig.add_trace(go.Bar(
                        x=x, 
                        y=y,
                        text=np.round(x,3), 
                        textposition='inside',
                        orientation='h', 
                        width=0.3, 
                        showlegend=False, 
                        marker_color='#ADD8E6')) 
    
    fig.update_xaxes(showticklabels=False,title_text= text, title_font = {"size": 16}, range = [0,1])
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        template = 'plotly_white',
    )
    #fig.update_layout(yaxis_title=None)
    #fig.update_xaxes(showticklabels=False,title_text= text1, row=1, col=1, range = [1,0], title_font = {"size": 16})
    return fig

def metrics_scatter(metrics, labels, colors):
    fig, ax = plt.subplots(figsize=(10,6))
    for i, (color, label) in enumerate(zip(colors,labels)):
        if i != 0:
            y = list(zip(*metrics))[i]
            ax.scatter(range(len(y)), y, c=color, label=label)

    ax.legend()
    ax.grid(True)
    plt.show()

def pareto_graph(fair_metrics_dict, model_metrics_dict, train_col = 'train'):
    pareto = list(zip(fair_metrics_dict[train_col], model_metrics_dict[train_col]))
    pareto_sorted = sorted(pareto, key = lambda x: x[0])
    return plt.scatter(list(zip(*pareto_sorted))[0], list(zip(*pareto_sorted))[1], color = '#4682B4')

def fig_train_test(df_metrics, metric_title, train_col, test_col):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_metrics.index, 
            y=df_metrics[train_col],
            mode='markers', 
            marker_color='#4682B4',
            customdata = df_metrics['model'],
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
            customdata = df_metrics['model'],
            mode = 'markers',
            marker_color='#87CEEB',
            hovertemplate="<br>".join([
                "Ranking: %{x}",
                metric_title +" (test): %{y:,.3f}",
                "Model: %{customdata}",
            ])
            ))
    fig.update_layout(
        title="Train vs Test",
        xaxis_title="Ranking",
        yaxis_title= metric_title,
        #legend_title="Legend Title",
        showlegend = False,
    )
    return fig

def pareto_fig(df_metrics, train_fair_col, train_model_col, fair_metric_name, model_metric_name):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_metrics[train_fair_col], 
            y=df_metrics[train_model_col],
            mode='markers', 
            marker_color='#4682B4',
            marker_symbol = df_metrics['model'].astype('category').cat.codes,
            customdata = df_metrics['model'],
            hovertemplate="<br>".join([
                fair_metric_name+": %{x:,.3f}",
                model_metric_name+": %{y:,.3f}",
                "Model: %{customdata}",
            ])
            ))
    fig.update_layout(
        title="Pareto Front",
        xaxis_title=fair_metric_name,
        yaxis_title=model_metric_name,
        #legend_title="Legend Title",
        #showlegend = False,
    )
    return fig

def create_df_metrics(fair_metrics, model_metrics):
    df_fair = pd.DataFrame.from_dict(fair_metrics)
    df_model = pd.DataFrame.from_dict(model_metrics)
    df_metrics = pd.concat([df_fair,df_model], axis = 1)
    return df_metrics

def create_df_ranges(metrics, df_metrics):
    d = []
    for metric in metrics:
        metric_min = np.min(df_metrics[metric])
        metric_max = np.max(df_metrics[metric])
        d.append(
            {
                'metric': metric,
                'min': metric_min,
                'max':  metric_max,
                'diff': metric_max - metric_min
            }
        )

    return pd.DataFrame(d)

def eval_metrics_graph(df_metrics, labels, colors, title):
    fig = go.Figure()
    for i, (color, label) in enumerate(zip(colors,labels)):
        fig.add_trace(
            go.Scatter(
                name = label.capitalize(),
                x= df_metrics.index, 
                y= df_metrics[label], 
                customdata = df_metrics.model,
                mode = 'markers',
                marker_color = color,
                hovertemplate = "<br>".join([
                    "Ranking: %{x}",
                    label.capitalize() +" (test): %{y:,.3f}",
                    "Model: %{customdata}",
                ])
        ))
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
            ),
        title = title
    )
    return fig

def comparison_graph(df_ranges, df_metrics, n):
    fig = go.Figure()
    df_ranges['metric'] = df_ranges['metric'].str.capitalize()
    for i in range(df_ranges.shape[0]):
        fig.add_trace(
            go.Bar(
                    x= [df_ranges['min'][i]],
                    y= [df_ranges.metric[i]],
                    width = .05,
                    marker = dict(
                        color = 'lightslategray',
                    ),
                    orientation='h'))
        fig.add_trace(
            go.Bar(
                    x= [df_ranges['diff'][i]],
                    base = [df_ranges['min'][i]],
                    y= [df_ranges.metric[i]],
                    width = .15,
                    marker = dict(
                        color = 'LightSkyBlue',
                    ),
                    orientation='h'))
        fig.add_trace(
            go.Bar(
                    x= [1-df_ranges['diff'][i]-df_ranges['min'][i]],
                    base = [df_ranges['min'][i]+df_ranges['diff'][i]],
                    y= [df_ranges.metric[i]],
                    width = .05,
                    marker =dict(
                        color = "lightslategray"
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
                            color='lightslategray')
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
                            color='lightslategray')
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
                showlegend=False,
                barmode='stack',
    )
    return fig