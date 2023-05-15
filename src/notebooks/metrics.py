from sklearn.metrics import f1_score, confusion_matrix, make_scorer
from fairlearn.metrics import (
    MetricFrame,
    count,
    selection_rate,
    equalized_odds_difference,
    false_positive_rate,
    false_negative_rate,
    demographic_parity_difference,
    true_positive_rate,
    true_negative_rate
)
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score, 
    confusion_matrix, 
    make_scorer
)
import numpy as np


metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall":recall_score,
    "f1 score": f1_score,
    "selection rate": selection_rate,
    "false positive rate": false_positive_rate,
    "true positive rate": true_positive_rate,
    "false negative rate": false_negative_rate,
    "true negative rate": true_negative_rate,
    "count": count,
}

def metric_evaluation(y_true, y_pred, sensitive_features, metrics = metrics):
    return MetricFrame(
        metrics=metrics, 
        y_true=y_true, 
        y_pred=y_pred, 
        sensitive_features=sensitive_features
    )

def metrics(model_metric, fair_metric,sensitive_col):
    def metric_scorer(clf, X, y):
        y_pred = clf.predict(X)
        performance_metric = model_metric(y,y_pred)
        fairness_metric = np.abs(fair_metric(y, y_pred, sensitive_features = X[sensitive_col]))
        return {'model': performance_metric, 'fairness': fairness_metric}
    return metric_scorer

def equality_opportunity_difference(y_true, y_pred, sensitive_features):
    return MetricFrame(metrics=true_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features).difference()

def predictive_equality_difference(y_true, y_pred, sensitive_features):
    return MetricFrame(metrics=false_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features).difference()

def average_absolute_odds_difference(y_true, y_pred, sensitive_features):
    fpr = MetricFrame(metrics=false_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features).difference()
    tpr = MetricFrame(metrics=true_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features).difference()
    return (np.abs(fpr)+np.abs(tpr))*.5

def disparate_impact(y_true, y_pred, sensitive_features):
    group_min = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features).group_min()
    group_max = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features).group_max()
    return group_min/ group_max