from sklearn.metrics import (
    f1_score, 
    confusion_matrix, 
    make_scorer, 
    accuracy_score, 
    recall_score, 
    matthews_corrcoef,
    accuracy_score,
    precision_score,
)

from metrics import (
    equality_opportunity_difference,
    predictive_equality_difference,
    predictive_parity_difference,
    metrics,
    average_absolute_odds_difference,
    get_functions
    
)
from fairlearn.metrics import demographic_parity_difference
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from fairlearn.datasets import fetch_adult
from sklearn.pipeline import Pipeline

from sklearn.utils import resample
import optuna
import json 
import dill
import argparse
import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer

parser = argparse.ArgumentParser() # Parser for command-line options
parser.add_argument("--fair_metric", help = "Metric to optimize fairness", type = str)
parser.add_argument("--model_metric", help = "Metric to optimize model performance", type = str)
parser.add_argument("--sensitive_attribute", help = "Column of the sensitive attribute", type = str)
parser.add_argument("--n_trials", help = "Number of trials", type = str)
parser.add_argument('--models', nargs='+', help='Models to optimize')
parser.add_argument("--n_folds", help = "Number of folds for Cross Validation", type = str)

FOLDER_ID = 'adult'
MODEL_PATH_IN_FOLDER = 'results'

args = parser.parse_args()
fair_metric_selection = args.fair_metric
model_metric_selection = args.model_metric
sensitive_attribute = args.sensitive_attribute
print(args.n_trials)
n_trials = int(args.n_trials)
models =  args.models[0].split(" ")
n_folds = int(args.n_folds)



with open('/home/azucena/fairness/src/notebooks/metrics.json', 'r') as f:
  metrics_dict = json.load(f)

def objective_decorator(metric_scorer, X_train, y_train, models, preprocessor, n_folds = 5):
    def objective_fn(trial):

        classifier_name = trial.suggest_categorical("classifier",models)

        if classifier_name == "logit":        
            params = {
                "penalty" : trial.suggest_categorical('logit_penalty', ['l1','l2']),
                "C" : trial.suggest_float('logit_c', 0.001, 10),
                "max_iter": 2000,
                "solver" : 'saga'
                }
            classifier = LogisticRegression(**params)

        elif classifier_name =="RF":
            params = {
                'n_estimators': trial.suggest_int("rf_n_estimators", 100, 1000),
                'criterion': trial.suggest_categorical("rf_criterion", ['gini', 'entropy']),
                'max_depth': trial.suggest_int("rf_max_depth", 1, 4),
                'min_samples_split': trial.suggest_float("rf_min_samples_split", 0.01, 1),
                }
            classifier = RandomForestClassifier(**params)

        elif classifier_name =="LGBM":
            params = {
                'n_estimators': trial.suggest_int("lgbm_n_estimators", 20, 10000),
                'num_leaves': trial.suggest_int("lgbm_num_leaves", 10, 1000),
                'max_depth': trial.suggest_int("lgbm_max_depth", 2, 20),
                'min_child_samples': trial.suggest_int("lgbm_min_child_samples", 5, 300),
                'learning_rate': trial.suggest_float('lgbm_learning_rate', 1e-5, 1e-2),
                'boosting_type': trial.suggest_categorical("lgbm_boosting_type", ['goss', 'gbdt'])
                }
            classifier = LGBMClassifier(**params)  

        elif classifier_name =="GBM":
            params = {
                'n_estimators': trial.suggest_int("gbm_n_estimators", 100, 1000), 
                'criterion': trial.suggest_categorical("gbm_criterion", ['squared_error', 'friedman_mse']),
                'max_depth': trial.suggest_int("gbm_max_depth", 1, 4),
                'min_samples_split': trial.suggest_int("gbm_min_samples_split", 5, 300),
                }
            classifier = GradientBoostingClassifier(**params)            

        else:
            raise Exception(f"{models} does not exist")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        scores = cross_validate(
                pipeline, 
                X_train,
                y_train, 
                cv=n_folds,
                scoring = metric_scorer,
                return_train_score=True)

        fair_metric = scores['test_fairness'].mean()
        model_metric = scores['test_model'].mean()
        return fair_metric, model_metric
    return objective_fn

def split_data(X, y, sensitive_attribute, test_size = 0.2, perc_sample = .5, sample = True, random_state = None):    
    if sensitive_attribute == 'race':
        mapping = {'White':'white','Black':'black','Asian-Pac-Islander':'others','Amer-Indian-Eskimo':'others','Other':'others'}
        X.loc[:,'race'] = X['race'].map(mapping).astype("category")
    if sample:
        X, y= resample(X, y, n_samples=int(perc_sample*X.shape[0]), random_state = random_state)

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size= test_size, random_state=random_state, stratify=y
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test

def get_adult_dataset():
    from fairlearn.datasets import fetch_adult
    data = fetch_adult(as_frame=True)
    X_raw = data.data
    y = (data.target == ">50K") * 1
    return X_raw, y


def save_study(study_name, study, in_dataiku = False):
    #import dataiku
    if in_dataiku:
        None
        #with dataiku.Folder(FOLDER_ID).get_writer(MODEL_PATH_IN_FOLDER) as writer:
        #    writeable = dill.dumps(study)
        #writer.write(writeable)
    else:
        file_name = '/home/azucena/fairness/src/notebooks/' + study_name +'.pkl'
        with open(file_name, 'wb') as file:
            dill.dump(study, file)
            print(f'Object successfully saved to "{file_name}"')

#def fair_ho_optimize(X_train, y_train, models, fair_metric, model_metric, sensitive_attribute, directions, n_trials):
if __name__ == "__main__":
    functions = get_functions()
    fair_metric = functions[fair_metric_selection]
    model_metric = functions[model_metric_selection]
    fair_direction = metrics_dict[fair_metric_selection]['optimization']
    model_direction = metrics_dict[model_metric_selection]['optimization']
    directions = [fair_direction, model_direction]
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.SuccessiveHalvingPruner()

    numeric_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, selector(dtype_exclude="category")),
            ("cat", categorical_transformer, selector(dtype_include="category")),
        ]
    )

    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f'{model_metric_selection}_{fair_metric_selection}_{n_trials}_{time}.pkl'
    X_raw, y = get_adult_dataset()
    X_train, X_test, y_train, y_test = split_data(X_raw, y, sensitive_attribute = sensitive_attribute)
    metric_scorer =  metrics(model_metric, fair_metric, sensitive_attribute)
    objective = objective_decorator(metric_scorer, X_train, y_train, models, preprocessor)
    study = optuna.create_study(
        directions = directions, 
        pruner = pruner, 
        sampler = sampler,
    )
    study.set_user_attr("fair_metric", fair_metric_selection)
    study.set_user_attr("model_metric", model_metric_selection)
    study.set_user_attr("n_folds", n_folds)
    study.set_user_attr("dataset", "adult")

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    save_study(file_name, study)
    