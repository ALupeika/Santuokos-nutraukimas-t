"""
Regresijos modelių modulis

Šis modulis apima visas regresijos modelių implementacijas.
"""

import logging
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from modeliai.model_train import (
    treniruoti_modeli, 
    ivertinti_modeli, 
    palyginti_modelius
)

from config import Config

logger = logging.getLogger(__name__)

def gauti_regresijos_modelius():
    """
    Grąžina regresijos modelių sąrašą
    
    Returns:
        list: Regresijos modelių sąrašas
    """
    # Sukuriame modelių sąrašą
    modeliai = [
        {
            'pavadinimas': 'Tiesinė regresija',
            'modelis': LinearRegression()
        },
        {
            'pavadinimas': 'Ridge regresija',
            'modelis': Ridge(alpha=1.0, random_state=Config.ATSITIKTINIS_SEED)
        },
        {
            'pavadinimas': 'Lasso regresija',
            'modelis': Lasso(alpha=0.1, random_state=Config.ATSITIKTINIS_SEED)
        },
        {
            'pavadinimas': 'ElasticNet regresija',
            'modelis': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=Config.ATSITIKTINIS_SEED)
        },
        {
            'pavadinimas': 'Sprendimų medis',
            'modelis': DecisionTreeRegressor(random_state=Config.ATSITIKTINIS_SEED)
        },
        {
            'pavadinimas': 'Atsitiktinis miškas',
            'modelis': RandomForestRegressor(
                n_estimators=100, 
                max_depth=None, 
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'Gradientinis stiprinimas',
            'modelis': GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'AdaBoost',
            'modelis': AdaBoostRegressor(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'SVR',
            'modelis': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        },
        {
            'pavadinimas': 'K kaimynai',
            'modelis': KNeighborsRegressor(n_neighbors=5)
        }
    ]
    
    return modeliai

def treniruoti_regresijos_modelius(X_train, y_train, X_val, y_val):
    """
    Treniruoja visus regresijos modelius ir grąžina geriausią
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
    
    Returns:
        tuple: (geriausias_modelis, visi_rezultatai)
    """
    logger.info("Pradedamas regresijos modelių treniravimas")
    
    # Gauname modelių sąrašą
    modeliai = gauti_regresijos_modelius()
    
    # Lyginame modelius
    geriausias_modelis, rezultatai = palyginti_modelius(
        modeliai, X_train, y_train, X_val, y_val, modelio_tipas='regresija'
    )
    
    return geriausias_modelis, rezultatai

def sukurti_islaikymo_regresijos_modeli(X_train, y_train, X_val, y_val):
    """
    Sukuria regresijos modelį vaiko išlaikymo sumos prognozavimui
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis (išlaikymo suma)
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        
    Returns:
        tuple: (geriausias_modelis, rezultatai)
    """
    # Galime pritaikyti specialų algoritmą arba parametrus pagal uždavinį
    # Pavyzdžiui, pritaikyti modelį, kad geriau dirbtų su piniginėmis sumomis
    
    logger.info("Treniruojamas išlaikymo sumos regresijos modelis")
    
    # Naudosime tuos pačius regresijos modelius kaip ir baziniame metode
    modeliai = gauti_regresijos_modelius()
    
    # Gradientinio stiprinimo modelis dažnai gerai veikia su piniginiais duomenimis,
    # tad galime jį naudoti su kitokiais hiperparametrais
    modeliai.append({
        'pavadinimas': 'Gradientinis stiprinimas (specialus)',
        'modelis': GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5,
            min_samples_split=5,
            random_state=Config.ATSITIKTINIS_SEED
        )
    })
    
    # Treniruojame ir lyginame modelius
    geriausias_modelis, rezultatai = palyginti_modelius(
        modeliai, X_train, y_train, X_val, y_val, modelio_tipas='regresija'
    )
    
    return geriausias_modelis, rezultatai

def sukurti_turto_padalijimo_modeli(X_train, y_train, X_val, y_val):
    """
    Sukuria regresijos modelį turto padalijimo procentų prognozavimui
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis (padalijimo procentas)
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        
    Returns:
        tuple: (geriausias_modelis, rezultatai)
    """
    logger.info("Treniruojamas turto padalijimo regresijos modelis")
    
    # Naudosime bazinę regresijos modelių kolekciją
    modeliai = gauti_regresijos_modelius()
    
    # Treniruojame ir lyginame modelius
    geriausias_modelis, rezultatai = palyginti_modelius(
        modeliai, X_train, y_train, X_val, y_val, modelio_tipas='regresija'
    )
    
    return geriausias_modelis, rezultatai

def sukurti_bylinejimosi_islaidu_modeli(X_train, y_train, X_val, y_val):
    """
    Sukuria regresijos modelį bylinėjimosi išlaidų prognozavimui
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis (bylinėjimosi išlaidos)
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        
    Returns:
        tuple: (geriausias_modelis, rezultatai)
    """
    logger.info("Treniruojamas bylinėjimosi išlaidų regresijos modelis")
    
    # Naudosime tuos pačius regresijos modelius kaip ir baziniame metode
    modeliai = gauti_regresijos_modelius()
    
    # Galime pridėti specializuotą modelį
    modeliai.append({
        'pavadinimas': 'Atsitiktinis miškas (specialus)',
        'modelis': RandomForestRegressor(
            n_estimators=200, 
            max_depth=10,
            min_samples_split=5,
            random_state=Config.ATSITIKTINIS_SEED
        )
    })
    
    # Treniruojame ir lyginame modelius
    geriausias_modelis, rezultatai = palyginti_modelius(
        modeliai, X_train, y_train, X_val, y_val, modelio_tipas='regresija'
    )
    
    return geriausias_modelis, rezultatai

def sukurti_islaikymo_skolos_modeli(X_train, y_train, X_val, y_val):
    """
    Sukuria regresijos modelį išlaikymo skolos prognozavimui
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis (išlaikymo skola)
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        
    Returns:
        tuple: (geriausias_modelis, rezultatai)
    """
    logger.info("Treniruojamas išlaikymo skolos regresijos modelis")
    
    # Dažnai išlaikymo skola yra proporcinga išlaikymo sumai, tad galime
    # naudoti paprastesnį modelį, pvz., tiesinę regresiją
    modeliai = [
        {
            'pavadinimas': 'Tiesinė regresija',
            'modelis': LinearRegression()
        },
        {
            'pavadinimas': 'Ridge regresija',
            'modelis': Ridge(alpha=1.0, random_state=Config.ATSITIKTINIS_SEED)
        },
        {
            'pavadinimas': 'Atsitiktinis miškas',
            'modelis': RandomForestRegressor(
                n_estimators=100, 
                max_depth=None, 
                random_state=Config.ATSITIKTINIS_SEED
            )
        }
    ]
    
    # Treniruojame ir lyginame modelius
    geriausias_modelis, rezultatai = palyginti_modelius(
        modeliai, X_train, y_train, X_val, y_val, modelio_tipas='regresija'
    )
    
    return geriausias_modelis, rezultatai