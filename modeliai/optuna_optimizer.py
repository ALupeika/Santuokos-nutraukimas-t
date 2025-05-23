"""
Optuna optimizavimo modulis

Šis modulis naudoja Optuna biblioteką hiperparametrų optimizavimui.
"""

import logging
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from modeliai.model_train import treniruoti_modeli
from config import Config

logger = logging.getLogger(__name__)

def optimizuoti_regresijos_modeli(modelio_klase, X_train, y_train, X_val, y_val, n_trials=None):
    """
    Optimizuoja regresijos modelio hiperparametrus naudojant Optuna
    
    Args:
        modelio_klase: Modelio klasė (pvz., RandomForestRegressor)
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        n_trials (int): Bandymų skaičius
    
    Returns:
        tuple: (geriausias_modelis, geriausi_parametrai, geriausia_verte)
    """
    if n_trials is None:
        n_trials = Config.OPTIMIZAVIMO_BANDYMU_SKAICIUS
    
    modelio_pavadinimas = modelio_klase.__name__
    logger.info(f"Pradedama {modelio_pavadinimas} hiperparametrų optimizacija su {n_trials} bandymais")
    
    # Apibrėžiame objektyvinę funkciją
    def objektyvine_funkcija(trial):
        # Parametrų erdvė priklauso nuo modelio tipo
        parametrai = {}
        
        if modelio_klase == RandomForestRegressor:
            parametrai = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': Config.ATSITIKTINIS_SEED
            }
        
        elif modelio_klase == GradientBoostingRegressor:
            parametrai = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': Config.ATSITIKTINIS_SEED
            }
        
        elif modelio_klase == Ridge:
            parametrai = {
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
                'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
                'random_state': Config.ATSITIKTINIS_SEED
            }
        
        elif modelio_klase == Lasso:
            parametrai = {
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
                'random_state': Config.ATSITIKTINIS_SEED
            }
        
        elif modelio_klase == SVR:
            parametrai = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0)
            }
        
        elif modelio_klase == KNeighborsRegressor:
            parametrai = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': trial.suggest_int('leaf_size', 20, 100)
            }
        
        else:
            logger.warning(f"Nežinoma modelio klasė: {modelio_pavadinimas}. Naudojame numatytuosius parametrus.")
        
        # Treniruojame modelį su šiais parametrais
        try:
            model = modelio_klase(**parametrai)
            model.fit(X_train, y_train)
            
            # Gauname prognozes validavimo duomenims
            y_pred = model.predict(X_val)
            
            # Skaičiuojame metriką (MSE - mažesnė reikšmė geriau)
            mse = mean_squared_error(y_val, y_pred)
            
            return mse
        
        except Exception as e:
            logger.error(f"Klaida treniruojant modelį su parametrais {parametrai}: {e}")
            # Grąžiname didelę MSE reikšmę, kad šie parametrai būtų atmesti
            return float('inf')
    
    # Sukuriame ir vykdome optimizavimą
    study = optuna.create_study(direction='minimize')  # minimizuojame MSE
    study.optimize(objektyvine_funkcija, n_trials=n_trials)
    
    # Gauname geriausius parametrus
    geriausi_parametrai = study.best_params
    geriausia_verte = study.best_value
    
    logger.info(f"Geriausi {modelio_pavadinimas} parametrai: {geriausi_parametrai}")
    logger.info(f"Geriausias MSE: {geriausia_verte:.6f}")
    
    # Pridedame random_state, jei modelis jį palaiko ir jis nebuvo įtrauktas
    if hasattr(modelio_klase, 'random_state') and 'random_state' not in geriausi_parametrai:
        geriausi_parametrai['random_state'] = Config.ATSITIKTINIS_SEED
    
    # Treniruojame galutinį modelį su geriausiais parametrais
    geriausias_modelis = modelio_klase(**geriausi_parametrai)
    geriausias_modelis.fit(X_train, y_train)
    
    return geriausias_modelis, geriausi_parametrai, geriausia_verte

def optimizuoti_klasifikavimo_modeli(modelio_klase, X_train, y_train, X_val, y_val, n_trials=None):
    """
    Optimizuoja klasifikavimo modelio hiperparametrus naudojant Optuna
    
    Args:
        modelio_klase: Modelio klasė (pvz., RandomForestClassifier)
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        n_trials (int): Bandymų skaičius
    
    Returns:
        tuple: (geriausias_modelis, geriausi_parametrai, geriausia_verte)
    """
    if n_trials is None:
        n_trials = Config.OPTIMIZAVIMO_BANDYMU_SKAICIUS
    
    modelio_pavadinimas = modelio_klase.__name__
    logger.info(f"Pradedama {modelio_pavadinimas} hiperparametrų optimizacija su {n_trials} bandymais")
    
    # Apibrėžiame objektyvinę funkciją
    def objektyvine_funkcija(trial):
        # Parametrų erdvė priklauso nuo modelio tipo
        parametrai = {}
        
        if modelio_klase == RandomForestClassifier:
            parametrai = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': Config.ATSITIKTINIS_SEED
            }
        
        elif modelio_klase == GradientBoostingClassifier:
            parametrai = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': Config.ATSITIKTINIS_SEED
            }
        
        elif modelio_klase == LogisticRegression:
            parametrai = {
                'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none']),
                'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'random_state': Config.ATSITIKTINIS_SEED
            }
            
            # Korektiški solver+penalty kombinacijos
            if parametrai['penalty'] == 'l1' and parametrai['solver'] in ['newton-cg', 'lbfgs', 'sag']:
                parametrai['solver'] = 'saga'
            elif parametrai['penalty'] == 'elasticnet' and parametrai['solver'] != 'saga':
                parametrai['solver'] = 'saga'
        
        elif modelio_klase == SVC:
            parametrai = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                'probability': True,  # Būtina norint gauti tikimybes
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'random_state': Config.ATSITIKTINIS_SEED
            }
        
        elif modelio_klase == KNeighborsClassifier:
            parametrai = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': trial.suggest_int('leaf_size', 20, 100)
            }
        
        else:
            logger.warning(f"Nežinoma modelio klasė: {modelio_pavadinimas}. Naudojame numatytuosius parametrus.")
        
        # Treniruojame modelį su šiais parametrais
        try:
            model = modelio_klase(**parametrai)
            model.fit(X_train, y_train)
            
            # Gauname prognozes validavimo duomenims
            y_pred = model.predict(X_val)
            
            # Skaičiuojame metriką (f1_score - didesnė reikšmė geriau)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            # Optuna minimizuoja funkcijos reikšmę, todėl invertuojame f1 (1 - f1)
            return 1.0 - f1
        
        except Exception as e:
            logger.error(f"Klaida treniruojant modelį su parametrais {parametrai}: {e}")
            # Grąžiname didelę reikšmę (blogas f1), kad šie parametrai būtų atmesti
            return 1.0
    
    # Sukuriame ir vykdome optimizavimą
    study = optuna.create_study(direction='minimize')  # minimizuojame (1 - f1_score)
    study.optimize(objektyvine_funkcija, n_trials=n_trials)
    
    # Gauname geriausius parametrus
    geriausi_parametrai = study.best_params
    geriausia_verte = 1.0 - study.best_value  # Konvertuojame atgal į f1_score
    
    logger.info(f"Geriausi {modelio_pavadinimas} parametrai: {geriausi_parametrai}")
    logger.info(f"Geriausias F1 score: {geriausia_verte:.6f}")
    
    # Pridedame random_state, jei modelis jį palaiko ir jis nebuvo įtrauktas
    if hasattr(modelio_klase, 'random_state') and 'random_state' not in geriausi_parametrai:
        geriausi_parametrai['random_state'] = Config.ATSITIKTINIS_SEED
    
    # Treniruojame galutinį modelį su geriausiais parametrais
    geriausias_modelis = modelio_klase(**geriausi_parametrai)
    geriausias_modelis.fit(X_train, y_train)
    
    return geriausias_modelis, geriausi_parametrai, geriausia_verte

def optimizuoti_neuronini_tinkla(X_train, y_train, X_val, y_val, modelio_tipas='regresija', n_trials=None):
    """
    Optimizuoja neuroninio tinklo hiperparametrus naudojant Optuna
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        modelio_tipas (str): 'regresija' arba 'klasifikavimas'
        n_trials (int): Bandymų skaičius
    
    Returns:
        tuple: (geriausias_modelis, geriausi_parametrai, geriausia_verte, istorija)
    """
    if n_trials is None:
        n_trials = Config.OPTIMIZAVIMO_BANDYMU_SKAICIUS
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    logger.info(f"Pradedama neuroninio tinklo hiperparametrų optimizacija su {n_trials} bandymais")
    
    # Apibrėžiame objektyvinę funkciją
    def objektyvine_funkcija(trial):
        # Hiperparametrų erdvė
        parametrai = {
            'hidden_layers': trial.suggest_int('hidden_layers', 1, 5),
            'pirmo_sluoksnio_neuronai': trial.suggest_int('pirmo_sluoksnio_neuronai', 16, 256),
            'neutronu_mazejimo_faktorius': trial.suggest_float('neutronu_mazejimo_faktorius', 1.2, 3.0),
            'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'selu']),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
            'l2_regularizacija': trial.suggest_float('l2_regularizacija', 1e-5, 1e-1, log=True),
            'batch_normalization': trial.suggest_categorical('batch_normalization', [True, False]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        }
        
        # Sukuriame modelį su šiais parametrais
        model = keras.Sequential()
        
        # Įvesties sluoksnis
        model.add(layers.Input(shape=(X_train.shape[1],)))
        
        # Paslėpti sluoksniai
        neurons = parametrai['pirmo_sluoksnio_neuronai']
        for i in range(parametrai['hidden_layers']):
            model.add(layers.Dense(
                units=int(neurons),
                activation=parametrai['activation'],
                kernel_regularizer=regularizers.l2(parametrai['l2_regularizacija'])
            ))
            
            if parametrai['batch_normalization']:
                model.add(layers.BatchNormalization())
            
            if parametrai['dropout_rate'] > 0:
                model.add(layers.Dropout(parametrai['dropout_rate']))
            
            # Mažiname neuronų skaičių kiekviename paslėptame sluoksnyje
            neurons = neurons / parametrai['neutronu_mazejimo_faktorius']
        
        # Išvesties sluoksnis
        if modelio_tipas == 'regresija':
            model.add(layers.Dense(1))  # Regresijos atveju - vienas neuronas be aktyvacijos
            loss = 'mse'
            metrikos = ['mae']
        else:  # klasifikavimas
            # Nustatome unikalių klasių skaičių
            klasiu_skaicius = len(np.unique(y_train))
            
            if klasiu_skaicius == 2:  # binarinė klasifikacija
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrikos = ['accuracy']
            else:  # daugiaklasė klasifikacija
                model.add(layers.Dense(klasiu_skaicius, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrikos = ['accuracy']
        
        # Nustatome optimizatorių
        if parametrai['optimizer'] == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=parametrai['learning_rate'])
        elif parametrai['optimizer'] == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=parametrai['learning_rate'])
        else:  # sgd
            optimizer = keras.optimizers.SGD(learning_rate=parametrai['learning_rate'])
        
        # Kompiliuojame modelį
        model.compile(optimizer=optimizer, loss=loss, metrics=metrikos)
        
        # Sukuriame ankstyvojo sustabdymo ir mokymosi tempo mažinimo callback'us
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Treniruojame modelį
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,  # Didelis epochų skaičius, bet su ankstyvuoju sustabdymu
                batch_size=parametrai['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
            
            # Gauname metriką pagal modelio tipą
            if modelio_tipas == 'regresija':
                # Regresijos atveju minimizuojame MSE
                y_pred = model.predict(X_val, verbose=0)
                mse = mean_squared_error(y_val, y_pred)
                return mse
            else:
                # Klasifikavimo atveju maksimizuojame tikslumą (minimize 1-accuracy)
                val_loss = history.history['val_loss'][-1]
                val_accuracy = history.history.get('val_accuracy', [0])[-1]
                return 1.0 - val_accuracy
        
        except Exception as e:
            logger.error(f"Klaida treniruojant modelį su parametrais {parametrai}: {e}")
            return float('inf') if modelio_tipas == 'regresija' else 1.0
    
    # Sukuriame ir vykdome optimizavimą
    study = optuna.create_study(direction='minimize')
    study.optimize(objektyvine_funkcija, n_trials=n_trials)
    
    # Gauname geriausius parametrus
    geriausi_parametrai = study.best_params
    geriausia_verte = study.best_value
    
    if modelio_tipas == 'regresija':
        logger.info(f"Geriausi neuroninio tinklo parametrai: {geriausi_parametrai}")
        logger.info(f"Geriausias MSE: {geriausia_verte:.6f}")
    else:
        logger.info(f"Geriausi neuroninio tinklo parametrai: {geriausi_parametrai}")
        logger.info(f"Geriausias tikslumas: {1.0 - geriausia_verte:.6f}")
    
    # Sukuriame galutinį modelį su geriausiais parametrais
    from modeliai.neuroninis_tinklas import (
        sukurti_regresijos_tinkla,
        sukurti_klasifikavimo_tinkla,
        treniruoti_tinkla
    )
    
    if modelio_tipas == 'regresija':
        # Sukuriame regresijos tinklą
        hidden_layers = []
        neurons = geriausi_parametrai['pirmo_sluoksnio_neuronai']
        for i in range(geriausi_parametrai['hidden_layers']):
            hidden_layers.append(int(neurons))
            neurons = neurons / geriausi_parametrai['neutronu_mazejimo_faktorius']
        
        geriausias_modelis = sukurti_regresijos_tinkla(
            ivesties_dydis=X_train.shape[1],
            hidden_layers=hidden_layers,
            activation=geriausi_parametrai['activation'],
            learning_rate=geriausi_parametrai['learning_rate'],
            dropout_rate=geriausi_parametrai['dropout_rate']
        )
    else:
        # Sukuriame klasifikavimo tinklą
        klasiu_skaicius = len(np.unique(y_train))
        
        hidden_layers = []
        neurons = geriausi_parametrai['pirmo_sluoksnio_neuronai']
        for i in range(geriausi_parametrai['hidden_layers']):
            hidden_layers.append(int(neurons))
            neurons = neurons / geriausi_parametrai['neutronu_mazejimo_faktorius']
        
        geriausias_modelis = sukurti_klasifikavimo_tinkla(
            ivesties_dydis=X_train.shape[1],
            klasiu_skaicius=klasiu_skaicius,
            hidden_layers=hidden_layers,
            activation=geriausi_parametrai['activation'],
            learning_rate=geriausi_parametrai['learning_rate'],
            dropout_rate=geriausi_parametrai['dropout_rate']
        )
    
    # Treniruojame galutinį modelį
    istorija, geriausias_modelis = treniruoti_tinkla(
        model=geriausias_modelis,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochos=100,
        batch_dydis=geriausi_parametrai['batch_size'],
        ankstyvasis_sustabdymas=True
    )
    
    return geriausias_modelis, geriausi_parametrai, geriausia_verte, istorija