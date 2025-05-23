import os
import joblib
import logging
import numpy as np
from datetime import datetime

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import optuna
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from config import Config


logger = logging.getLogger(__name__)


def treniruoti_modeli(modelis, X_train, y_train, modelio_tipas=None):
    
    modelio_pavadinimas = modelis.__class__.__name__
    
    try:
        logger.info(f"Pradedama treniruoti modelį: {modelio_pavadinimas}")
        modelis.fit(X_train, y_train)
        logger.info(f"Modelis {modelio_pavadinimas} sėkmingai apmokytas")
        return modelis
    except Exception as e:
        logger.error(f"Klaida treniruojant modelį {modelio_pavadinimas}: {e}")
        return None


def ivertinti_modeli(modelis, X_val, y_val, modelio_tipas):
    y_pred = modelis.predict(X_val)
    if modelio_tipas == 'regresija':
        return {
            'MSE': mean_squared_error(y_val, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
            'MAE': mean_absolute_error(y_val, y_pred),
            'R2': r2_score(y_val, y_pred)
        }
    else:
        return {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1': f1_score(y_val, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
            'report': classification_report(y_val, y_pred, output_dict=True)
        }


def optuna_optimizuoti(modelio_tipas, X_train, y_train, X_val, y_val, uzdavinio_tipas='klasifikavimas'):
    def objective(trial):
        if modelio_tipas == 'xgboost':
            if uzdavinio_tipas == 'regresija':
                model = XGBRegressor(
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    early_stopping_rounds=10,
                    verbosity=0
                )
            else:
                model = XGBClassifier(
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    verbosity=0
                )
        elif modelio_tipas == 'svm':
            if uzdavinio_tipas == 'regresija':
                model = SVR(C=trial.suggest_float("C", 0.1, 10.0))
            else:
                model = SVC(C=trial.suggest_float("C", 0.1, 10.0), probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if uzdavinio_tipas == 'regresija':
            return mean_squared_error(y_val, y_pred)
        else:
            return 1.0 - f1_score(y_val, y_pred, average='weighted')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    best_trial = study.best_trial

    logger.info(f"Optimizuotas modelio {modelio_tipas} rezultatas: {best_trial.value:.4f}")
    logger.info(f"Geriausi parametrai: {best_trial.params}")

    return best_trial


def issaugoti_modeli(modelis, pavadinimas, preprocessorius=None):
    kelias = os.path.join(Config.MODELIU_DIREKTORIJA, f"{pavadinimas}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
    joblib.dump({
        'modelis': modelis,
        'preprocessorius': preprocessorius
    }, kelias)
    logger.info(f"Modelis išsaugotas: {kelias}")
    return kelias


def palyginti_modelius(modeliai, X_train, y_train, X_val, y_val, modelio_tipas='regresija'):
    """
    Palygina kelis modelius ir grąžina geriausią modelį bei visų rezultatų metrikas.

    Args:
        modeliai (list): Modelių sąrašas
        X_train (ndarray): Treniravimo požymiai
        y_train (ndarray): Treniravimo tikslai
        X_val (ndarray): Validavimo požymiai
        y_val (ndarray): Validavimo tikslai
        modelio_tipas (str): 'regresija' arba 'klasifikavimas'

    Returns:
        tuple: (geriausias_modelis, rezultatai)
    """
    rezultatai = {}
    treniruoti_modeliai = {}

    logger.info(f"Pradedamas {len(modeliai)} modelių palyginimas ({modelio_tipas})")

    for i, modelis in enumerate(modeliai):
        modelio_pavadinimas = modelis.__class__.__name__
        logger.info(f"Modelis {i+1}/{len(modeliai)}: {modelio_pavadinimas}")

        try:
            modelis_apmokytas = treniruoti_modeli(modelis, X_train, y_train)
            if modelis_apmokytas is not None:
                metrikos = ivertinti_modeli(modelis_apmokytas, X_val, y_val, modelio_tipas)
                rezultatai[modelio_pavadinimas] = metrikos
                treniruoti_modeliai[modelio_pavadinimas] = modelis_apmokytas
        except Exception as e:
            logger.error(f"Klaida modelyje {modelio_pavadinimas}: {e}")

    if not rezultatai:
        logger.warning("Nerasta tinkamų modelių")
        return None, {}

    if modelio_tipas == 'regresija':
        geriausias = min(rezultatai, key=lambda k: rezultatai[k]['RMSE'])
    else:
        geriausias = max(rezultatai, key=lambda k: rezultatai[k]['f1'])

    logger.info(f"Geriausias modelis: {geriausias}")
    return treniruoti_modeliai[geriausias], rezultatai


