"""
Klasifikavimo modelių modulis

Šis modulis apima visas klasifikavimo modelių implementacijas.
"""

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from modeliai.model_train import (
    treniruoti_modeli, 
    ivertinti_modeli, 
    palyginti_modelius
)

from config import Config

logger = logging.getLogger(__name__)

def gauti_klasifikavimo_modelius():
    """
    Grąžina klasifikavimo modelių sąrašą
    
    Returns:
        list: Klasifikavimo modelių sąrašas
    """
    # Sukuriame modelių sąrašą
    modeliai = [
        {
            'pavadinimas': 'Logistinė regresija',
            'modelis': LogisticRegression(
                max_iter=1000,
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'Sprendimų medis',
            'modelis': DecisionTreeClassifier(
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'Atsitiktinis miškas',
            'modelis': RandomForestClassifier(
                n_estimators=100, 
                max_depth=None, 
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'Gradientinis stiprinimas',
            'modelis': GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'AdaBoost',
            'modelis': AdaBoostClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'SVC',
            'modelis': SVC(
                kernel='rbf', 
                C=1.0, 
                probability=True,
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'K kaimynai',
            'modelis': KNeighborsClassifier(
                n_neighbors=5
            )
        },
        {
            'pavadinimas': 'Naive Bayes',
            'modelis': GaussianNB()
        }
    ]
    
    return modeliai

def treniruoti_klasifikavimo_modelius(X_train, y_train, X_val, y_val):
    """
    Treniruoja visus klasifikavimo modelius ir grąžina geriausią
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
    
    Returns:
        tuple: (geriausias_modelis, visi_rezultatai)
    """
    logger.info("Pradedamas klasifikavimo modelių treniravimas")
    
    # Gauname modelių sąrašą
    modeliai = gauti_klasifikavimo_modelius()
    
    # Lyginame modelius
    geriausias_modelis, rezultatai = palyginti_modelius(
        modeliai, X_train, y_train, X_val, y_val, modelio_tipas='klasifikavimas'
    )
    
    return geriausias_modelis, rezultatai

def sukurti_gyvenamosios_vietos_modeli(X_train, y_train, X_val, y_val):
    """
    Sukuria klasifikavimo modelį vaiko gyvenamosios vietos prognozavimui
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis (gyvenamoji vieta)
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        
    Returns:
        tuple: (geriausias_modelis, rezultatai)
    """
    logger.info("Treniruojamas gyvenamosios vietos klasifikavimo modelis")
    
    # Šiam uždaviniui naudosime tik keletą modelių
    modeliai = [
        {
            'pavadinimas': 'Logistinė regresija',
            'modelis': LogisticRegression(
                max_iter=1000,
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'Atsitiktinis miškas',
            'modelis': RandomForestClassifier(
                n_estimators=100, 
                max_depth=None, 
                random_state=Config.ATSITIKTINIS_SEED
            )
        },
        {
            'pavadinimas': 'Gradientinis stiprinimas',
            'modelis': GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                random_state=Config.ATSITIKTINIS_SEED
            )
        }
    ]
    
    # Treniruojame ir lyginame modelius
    geriausias_modelis, rezultatai = palyginti_modelius(
        modeliai, X_train, y_train, X_val, y_val, modelio_tipas='klasifikavimas'
    )
    
    return geriausias_modelis, rezultatai

def sukurti_bendravimo_tvarkos_modeli(X_train, y_train, X_val, y_val):
    """
    Sukuria klasifikavimo modelį bendravimo tvarkos prognozavimui
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis (bendravimo tvarka)
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        
    Returns:
        tuple: (geriausias_modelis, rezultatai)
    """
    logger.info("Treniruojamas bendravimo tvarkos klasifikavimo modelis")
    
    # Naudosime visus klasifikavimo modelius
    modeliai = gauti_klasifikavimo_modelius()
    
    # Kadangi bendravimo tvarkos klasifikacija gali būti sudėtinga,
    # galime pridėti specializuotą modelį su kitokiais parametrais
    modeliai.append({
        'pavadinimas': 'Atsitiktinis miškas (specialus)',
        'modelis': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=Config.ATSITIKTINIS_SEED
        )
    })
    
    # Treniruojame ir lyginame modelius
    geriausias_modelis, rezultatai = palyginti_modelius(
        modeliai, X_train, y_train, X_val, y_val, modelio_tipas='klasifikavimas'
    )
    
    return geriausias_modelis, rezultatai