from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

import copy
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from sklearn.decomposition import PCA

from sklearn.exceptions import FitFailedWarning
import numpy as np


def get_est_params_dict(keys):
    clf_params_dict = {
        'LogisticRegression': {
            'class': LogisticRegression,
            'params': {
                'solver': ['lbfgs'],
                'max_iter': [1000],
                'penalty': ['l2'],
                'C': [0.1, 1.0, 10],
                #'multi_class': ['auto']  # o 'ovr', 'multinomial'

            }
        },
        'DecisionTreeClassifier': {
            'class': DecisionTreeClassifier,
            'params': {
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini', 'entropy']
            }
        },
        'RandomForestClassifier': {
            'class': RandomForestClassifier,
            'params': {
                'bootstrap': [True],
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'random_state': [11]
            }
        },
        'GradientBoostingClassifier': {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'objective': ['multi:softprob'],
                'num_class': [3]  # Reemplaza n_classes por el número de clases en tu problema

            }
        },
        'XGBClassifier': {
            'class': XGBClassifier,
            'params': {
                'n_estimators': [5, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'eval_metric': ['logloss']
            }
        },
        'MLPClassifier': {
            'class': MLPClassifier,
            'params': {
                'hidden_layer_sizes': [5, 10, (10, 10)],
                'random_state': [11],
                'max_iter': [2000],
                'solver': ['lbfgs'],
                'activation': ['relu'],
                'early_stopping': [True],
                'validation_fraction': [0.2],
                'n_iter_no_change': [50],
                'verbose': [True]
            }
        },

        'SVC': {
            'class': SVC,
            'params': {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1.0, 10],
                'gamma': ['scale', 'auto'],
                'probability': [True],  # necesario si quieres usar y_proba para AUROC, etc.
                'random_state': [11]
            }
        },

        'SGDClassifier': {
            'class': SGDClassifier,
            'params': {
                'loss': ['hinge', 'log_loss'],  # hinge = SVM, log_loss = regresión logística
                'penalty': ['l2', 'l1'],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [1000],
                'tol': [1e-3],
                'random_state': [11],
                'early_stopping': [True],

            }
        }



    }

    # Filtrar y retornar solo los modelos solicitados
    return {key: clf_params_dict[key] for key in keys if key in clf_params_dict}

def setup_model(dicc):
    model_ = dicc['class']()
    scaler = StandardScaler()
    model = Pipeline([('scaler', scaler), ('model', model_)])
    param_grid = {'model__' + param_name: param_value for param_name, param_value in dicc['params'].items()}
    return model, param_grid

def run_grid_search(model, param_grid, X_train, y_train, scoring, sample_weight=None):


    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=None,
        scoring=scoring,
        n_jobs=4,
        error_score='raise'  # Para forzar el error y capturarlo
    )

    try:
        fit_params = {'model__sample_weight': sample_weight} if sample_weight is not None else {}
        grid_search.fit(X_train, y_train, **fit_params)
    except TypeError as e:
        print(f"⚠️ Modelo {model.named_steps['model'].__class__.__name__} no acepta sample_weight. Reintentando sin él.")
        grid_search.fit(X_train, y_train)

    return grid_search

def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    if y_proba is not None:
        try:
            # AUROC y AUPRC en multiclase deben especificar cómo agregarlas
            metrics["auroc_macro"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics["auprc_macro"] = average_precision_score(y_true, y_proba, average='macro')
        except:
            metrics["auroc_macro"] = np.nan
            metrics["auprc_macro"] = np.nan

    return metrics

def get_sample_weight(y_train):
    """
    Devuelve un array de sample weights donde las muestras con y=1
    tienen 10 veces más peso que las muestras con y=0.
    """
    y_train = np.asarray(y_train)
    weights = np.ones_like(y_train, dtype=np.float64)
    weights[y_train == 1] = 10
    return weights

def custom_rus_keep_all_minority(X, y, ratio=0.1, random_state=42):
    print("📊 Antes del undersampling:", Counter(y))
    minority_class = min(Counter(y), key=Counter(y).get)
    majority_class = max(Counter(y), key=Counter(y).get)
    n_minority = Counter(y)[minority_class]
    n_majority_target = int(Counter(y)[majority_class]*ratio)

    strategy = {majority_class: n_majority_target, minority_class: n_minority}

    rus = RandomUnderSampler(sampling_strategy=strategy, random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    print("📊 Después del undersampling:", Counter(y_res))

    return X_res, y_res


def run_short_version(est_params_dict, X_train, X_test, y_train, y_test, sample_weight_On = None, SMOTE_on = None, RandomUnderSampler_on = None):

    # creacion de diccionarios para almacenamiento
    results_test = {}
    cm_test = {}
    results_val = {}
    models_dicc = {}

    #Obtener dataset para el target
    #X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42, stratify=labels)

    #Reorder data
    #X_train, X_test, y_train, y_test = generate_datasets(X_train, X_test, y_train, y_test)

    for model_name, dicc in est_params_dict.items():

        print(model_name)
        model, param_grid = setup_model(dicc)

        #Random undersampling
        if RandomUnderSampler_on is not None:
          X_train, y_train = custom_rus_keep_all_minority(X_train, y_train, ratio=0.1, random_state=42)


        #SMOTE strategy
        if SMOTE_on is not None:
          print('computing SMOTE')
          print("📊 Antes del oversampling:", Counter(y_train))
          smote = SMOTE(random_state=42, sampling_strategy='minority')
          X_train, y_train = smote.fit_resample(X_train, y_train)
          print("📊 Despues del oversampling:", Counter(y_train))

        #Sample weights strategy
        if sample_weight_On is not None:
          print('Compute sw')
          sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        else:
          sample_weight = None

        #Run gridsearch
        grid_search = run_grid_search(model, param_grid, X_train, y_train, scoring = None, sample_weight = sample_weight)

        # Guardar best model
        best_model = grid_search.best_estimator_
        models_dicc[model_name] = copy.deepcopy(best_model)

        mean_val_score = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
        std_val_score = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
        mean_val_score = np.round(mean_val_score,2)
        std_val_score = np.round(std_val_score,2)

        metrics_val = {
          'mean_test_score': float(np.round(mean_val_score, 2)),
          'std_test_score': float(np.round(std_val_score, 2)
        )}

        results_val[model_name]  = metrics_val

        # predecir sobre test
        y_test_pred = best_model.predict(X_test)

        # métricas de test
        metrics_test = calculate_metrics(y_test, y_test_pred)
        metrics_test = {k: float(np.round(v, 2)) for k, v in metrics_test.items()}
        cm = confusion_matrix(y_test, y_test_pred)
        results_test[model_name]  = metrics_test
        cm_test[model_name] = cm

        #print(metrics_test, cm)

    return results_val, results_test, cm_test, models_dicc
