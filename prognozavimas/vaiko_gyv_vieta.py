"""
Vaiko gyvenamosios vietos prognozavimo modulis

Šis modulis apima funkcijas, skirtas prognozuoti vaiko gyvenamąją vietą santuokos nutraukimo bylose.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

from config import Config
from modeliai.klasifikatoriai import sukurti_gyvenamosios_vietos_modeli

logger = logging.getLogger(__name__)

# Modelio failo pavadinimas
MODELIO_FAILAS = 'gyvenamoji_vieta_model.joblib'
PREPROCESSOR_FAILAS = 'gyvenamoji_vieta_preprocessor.joblib'

def ikrauti_gyvenamosios_vietos_modeli():
    """
    Įkrauna treniruotą vaiko gyvenamosios vietos modelį
    
    Returns:
        tuple: (modelis, preprocessorius) - treniruotas modelis ir duomenų transformatorius
    """
    modelio_kelias = os.path.join(Config.MODELIU_DIREKTORIJA, MODELIO_FAILAS)
    preprocessor_kelias = os.path.join(Config.MODELIU_DIREKTORIJA, PREPROCESSOR_FAILAS)
    
    try:
        # Bandome įkrauti modelį
        modelio_objektas = joblib.load(modelio_kelias)
        
        modelis = modelio_objektas.get('modelis')
        meta_info = modelio_objektas.get('meta_info', {})
        
        # Bandome įkrauti preprocessorių
        preprocessorius = joblib.load(preprocessor_kelias)
        
        logger.info(f"Sėkmingai įkrautas gyvenamosios vietos modelis: {meta_info.get('modelio_klase', 'Nežinomas')}")
        
        return modelis, preprocessorius
    
    except FileNotFoundError:
        logger.warning("Gyvenamosios vietos modelis nerastas. Bus naudojamas fiktyvus modelis.")
        return None, None
    except Exception as e:
        logger.error(f"Klaida įkraunant gyvenamosios vietos modelį: {e}")
        return None, None

def prognozuoti_gyvenamaja_vieta(duomenys):
    """
    Prognozuoja vaiko gyvenamąją vietą pagal pateiktus duomenis
    
    Args:
        duomenys (dict): Duomenys, naudojami prognozavimui
    
    Returns:
        tuple: (prognozė, tikimybė) - prognozuota gyvenamoji vieta ir prognozės tikimybė
    """
    # Įkrauname modelį
    modelis, preprocessorius = ikrauti_gyvenamosios_vietos_modeli()
    
    # Jei modelis nerastas, naudojame fiktyvų modelį
    if modelis is None or preprocessorius is None:
        # Fiktyvus prognozavimas
        return fiktyvus_gyvenamosios_vietos_prognozavimas(duomenys)
    
    try:
        # Konvertuojame duomenis į DataFrame
        df = pd.DataFrame([duomenys])
        
        # Transformuojame duomenis
        X_transformed = preprocessorius.transform(df)
        
        # Atliekame prognozę
        prognoze_probs = modelis.predict_proba(X_transformed)[0]
        klases = modelis.classes_
        
        # Randame labiausiai tikėtiną klasę ir jos tikimybę
        max_index = np.argmax(prognoze_probs)
        prognoze = klases[max_index]
        tikimybe = prognoze_probs[max_index]
        
        logger.info(f"Prognozuota gyvenamoji vieta: {prognoze} (tikimybė: {tikimybe:.4f})")
        
        return prognoze, tikimybe
    
    except Exception as e:
        logger.error(f"Klaida prognozuojant gyvenamąją vietą: {e}")
        return fiktyvus_gyvenamosios_vietos_prognozavimas(duomenys)

def fiktyvus_gyvenamosios_vietos_prognozavimas(duomenys):
    """
    Fiktyvus gyvenamosios vietos prognozavimas, kai nėra modelio
    
    Args:
        duomenys (dict): Duomenys, naudojami prognozavimui
    
    Returns:
        tuple: (prognozė, tikimybė) - prognozuota gyvenamoji vieta ir prognozės tikimybė
    """
    logger.warning("Naudojamas fiktyvus gyvenamosios vietos prognozavimas")
    
    # Paprastas taisyklėmis pagrįstas sprendimas
    emocinis_rysys_mama = duomenys.get('emocinis_rysys_mama', 'vidutinis')
    emocinis_rysys_tevas = duomenys.get('emocinis_rysys_tevas', 'vidutinis')
    vaiko_amzius = duomenys.get('vaiko_amzius', 10)
    
    # Jaunesni vaikai dažniau lieka su mama
    if vaiko_amzius < 10:
        bazine_tikimybe_mama = 0.7
    else:
        bazine_tikimybe_mama = 0.6
    
    # Koreguojame tikimybę pagal emocinį ryšį
    if emocinis_rysys_mama == 'stiprus':
        tikimybe_mama = bazine_tikimybe_mama + 0.2
    elif emocinis_rysys_mama == 'silpnas':
        tikimybe_mama = bazine_tikimybe_mama - 0.3
    else:  # vidutinis
        tikimybe_mama = bazine_tikimybe_mama
    
    # Ribojame tikimybę intervale [0.1, 0.9]
    tikimybe_mama = max(0.1, min(0.9, tikimybe_mama))
    
    # Jei tėvo emocinis ryšys stiprus, o mamos ne
    if emocinis_rysys_tevas == 'stiprus' and emocinis_rysys_mama != 'stiprus':
        tikimybe_mama -= 0.15
    
    # Galutinė prognozė
    if tikimybe_mama > 0.5:
        return 'mama', tikimybe_mama
    else:
        return 'tevas', 1 - tikimybe_mama

def treniruoti_gyvenamosios_vietos_modeli(X_train, y_train, X_val, y_val):
    """
    Treniruoja vaiko gyvenamosios vietos modelį
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
    
    Returns:
        tuple: (geriausias_modelis, rezultatai) - geriausias modelis ir jo metrikos
    """
    from modeliai.model_train import issaugoti_modeli
    
    logger.info("Pradedamas gyvenamosios vietos modelio treniravimas")
    
    try:
        # Importuojame modelio sukūrimo funkciją
        geriausias_modelis, rezultatai = sukurti_gyvenamosios_vietos_modeli(X_train, y_train, X_val, y_val)
        
        # Išsaugome modelį
        modelio_kelias = os.path.join(Config.MODELIU_DIREKTORIJA, MODELIO_FAILAS)
        issaugoti_modeli(geriausias_modelis, "Gyvenamoji vieta", meta_info=rezultatai)
        
        logger.info(f"Gyvenamosios vietos modelis sėkmingai ištreniruotas ir išsaugotas: {modelio_kelias}")
        
        return geriausias_modelis, rezultatai
    
    except Exception as e:
        logger.error(f"Klaida treniruojant gyvenamosios vietos modelį: {e}")
        return None, None

def apdoroti_naujas_prognozes(bylų_duomenys):
    """
    Atlieka gyvenamosios vietos prognozes kelioms byloms vienu metu
    
    Args:
        bylų_duomenys (list): Bylų duomenų sąrašas
    
    Returns:
        list: Prognozių sąrašas
    """
    prognozes = []
    
    for bylos_duomenys in bylų_duomenys:
        prognoze, tikimybe = prognozuoti_gyvenamaja_vieta(bylos_duomenys)
        
        prognozes.append({
            'byla_id': bylos_duomenys.get('byla_id', 'Nežinoma'),
            'vaiko_amzius': bylos_duomenys.get('vaiko_amzius', 'Nežinoma'),
            'emocinis_rysys_mama': bylos_duomenys.get('emocinis_rysys_mama', 'Nežinoma'),
            'emocinis_rysys_tevas': bylos_duomenys.get('emocinis_rysys_tevas', 'Nežinoma'),
            'prognoze': prognoze,
            'tikimybe': tikimybe
        })
    
    return prognozes

def optimizuoti_gyvenamosios_vietos_modeli(X_train, y_train, X_val, y_val, n_trials=None):
    """
    Optimizuoja gyvenamosios vietos modelio hiperparametrus
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        n_trials (int): Bandymų skaičius
    
    Returns:
        tuple: (geriausias_modelis, geriausi_parametrai, tikslumas)
    """
    from modeliai.optuna_optimizer import optimizuoti_klasifikavimo_modeli
    from sklearn.ensemble import RandomForestClassifier
    
    logger.info("Pradedama gyvenamosios vietos modelio hiperparametrų optimizacija")
    
    try:
        # Optimizuojame RandomForest modelį
        geriausias_modelis, geriausi_parametrai, tikslumas = optimizuoti_klasifikavimo_modeli(
            RandomForestClassifier,
            X_train, y_train,
            X_val, y_val,
            n_trials=n_trials
        )
        
        # Išsaugome modelį
        from modeliai.model_train import issaugoti_modeli
        
        meta_info = {
            'hiperparametrai': geriausi_parametrai,
            'tikslumas': tikslumas
        }
        
        issaugoti_modeli(geriausias_modelis, "Gyvenamoji vieta (optimizuotas)", meta_info=meta_info)
        
        logger.info(f"Gyvenamosios vietos modelis sėkmingai optimizuotas: tikslumas={tikslumas:.4f}")
        
        return geriausias_modelis, geriausi_parametrai, tikslumas
    
    except Exception as e:
        logger.error(f"Klaida optimizuojant gyvenamosios vietos modelį: {e}")
        return None, None, None