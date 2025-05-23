"""
Vaiko išlaikymo prognozavimo modulis

Šis modulis apima funkcijas, skirtas prognozuoti vaiko išlaikymo sumą santuokos nutraukimo bylose.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

from config import Config
from modeliai.regresoriai import sukurti_islaikymo_regresijos_modeli

logger = logging.getLogger(__name__)

# Modelio failo pavadinimas
MODELIO_FAILAS = 'islaikymo_model.joblib'
PREPROCESSOR_FAILAS = 'islaikymo_preprocessor.joblib'

def ikrauti_islaikymo_modeli():
    """
    Įkrauna treniruotą vaiko išlaikymo modelį
    
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
        
        logger.info(f"Sėkmingai įkrautas išlaikymo modelis: {meta_info.get('modelio_klase', 'Nežinomas')}")
        
        return modelis, preprocessorius
    
    except FileNotFoundError:
        logger.warning("Išlaikymo modelis nerastas. Bus naudojamas fiktyvus modelis.")
        return None, None
    except Exception as e:
        logger.error(f"Klaida įkraunant išlaikymo modelį: {e}")
        return None, None

def prognozuoti_islaikyma(duomenys):
    """
    Prognozuoja vaiko išlaikymo sumą pagal pateiktus duomenis
    
    Args:
        duomenys (dict): Duomenys, naudojami prognozavimui
    
    Returns:
        tuple: (prognozė, tikimybė) - prognozuota išlaikymo suma ir prognozės tikimybė
    """
    # Įkrauname modelį
    modelis, preprocessorius = ikrauti_islaikymo_modeli()
    
    # Jei modelis nerastas, naudojame fiktyvų modelį
    if modelis is None or preprocessorius is None:
        # Fiktyvus prognozavimas
        return fiktyvus_islaikymo_prognozavimas(duomenys)
    
    try:
        # Konvertuojame duomenis į DataFrame
        df = pd.DataFrame([duomenys])
        
        # Transformuojame duomenis
        X_transformed = preprocessorius.transform(df)
        
        # Atliekame prognozę
        prognoze = modelis.predict(X_transformed)[0]
        
        # Kadangi tai regresijos modelis, tikimybę įvertintiname fiktyviai
        tikimybe = 0.8  # Tiesiog fiksuota reikšmė, nes regresijos modeliai negrąžina tikimybių
        
        logger.info(f"Prognozuota išlaikymo suma: {prognoze:.2f} EUR (tikimybė: {tikimybe:.4f})")
        
        return prognoze, tikimybe
    
    except Exception as e:
        logger.error(f"Klaida prognozuojant išlaikymo sumą: {e}")
        return fiktyvus_islaikymo_prognozavimas(duomenys)

def fiktyvus_islaikymo_prognozavimas(duomenys):
    """
    Fiktyvus išlaikymo sumos prognozavimas, kai nėra modelio
    
    Args:
        duomenys (dict): Duomenys, naudojami prognozavimui
    
    Returns:
        tuple: (prognozė, tikimybė) - prognozuota išlaikymo suma ir prognozės tikimybė
    """
    logger.warning("Naudojamas fiktyvus išlaikymo prognozavimas")
    
    # Paprastas taisyklėmis pagrįstas sprendimas
    vaiko_amzius = duomenys.get('vaiko_amzius', 10)
    poreikiai = duomenys.get('poreikiai', 500)
    pajamos_tevas = duomenys.get('pajamos_vyras', 1000)
    pajamos_mama = duomenys.get('pajamos_moteris', 800)
    gyvenamoji_vieta = duomenys.get('gyvenamoji_vieta', 'mama')
    
    # Bazinė išlaikymo suma priklauso nuo vaiko amžiaus ir poreikių
    if vaiko_amzius < 7:
        bazine_suma = poreikiai * 0.3
    elif vaiko_amzius < 14:
        bazine_suma = poreikiai * 0.4
    else:
        bazine_suma = poreikiai * 0.5
    
    # Koreguojame pagal tėvų pajamas
    if gyvenamoji_vieta == 'mama':
        # Tėvas moka
        pajamu_santykis = pajamos_tevas / (pajamos_tevas + pajamos_mama)
        galutine_suma = bazine_suma * (pajamu_santykis + 0.1)  # Pridedame mažą korekciją
    else:
        # Mama moka
        pajamu_santykis = pajamos_mama / (pajamos_tevas + pajamos_mama)
        galutine_suma = bazine_suma * (pajamu_santykis + 0.1)  # Pridedame mažą korekciją
    
    # Ribojame sumą, kad būtų realistiška
    galutine_suma = max(100, min(1000, galutine_suma))
    
    # Fiktyviai priskiriama tikimybė
    tikimybe = 0.7
    
    return galutine_suma, tikimybe

def treniruoti_islaikymo_modeli(X_train, y_train, X_val, y_val):
    """
    Treniruoja vaiko išlaikymo modelį
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
    
    Returns:
        tuple: (geriausias_modelis, rezultatai) - geriausias modelis ir jo metrikos
    """
    from modeliai.model_train import issaugoti_modeli
    
    logger.info("Pradedamas išlaikymo modelio treniravimas")
    
    try:
        # Importuojame modelio sukūrimo funkciją
        geriausias_modelis, rezultatai = sukurti_islaikymo_regresijos_modeli(X_train, y_train, X_val, y_val)
        
        # Išsaugome modelį
        modelio_kelias = os.path.join(Config.MODELIU_DIREKTORIJA, MODELIO_FAILAS)
        issaugoti_modeli(geriausias_modelis, "Išlaikymas", meta_info=rezultatai)
        
        logger.info(f"Išlaikymo modelis sėkmingai ištreniruotas ir išsaugotas: {modelio_kelias}")
        
        return geriausias_modelis, rezultatai
    
    except Exception as e:
        logger.error(f"Klaida treniruojant išlaikymo modelį: {e}")
        return None, None

def optimizuoti_islaikymo_modeli(X_train, y_train, X_val, y_val, n_trials=None):
    """
    Optimizuoja išlaikymo modelio hiperparametrus
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        n_trials (int): Bandymų skaičius
    
    Returns:
        tuple: (geriausias_modelis, geriausi_parametrai, geriausia_verte)
    """
    from modeliai.optuna_optimizer import optimizuoti_regresijos_modeli
    from sklearn.ensemble import GradientBoostingRegressor
    
    logger.info("Pradedama išlaikymo modelio hiperparametrų optimizacija")
    
    try:
        # Optimizuojame GradientBoosting modelį
        geriausias_modelis, geriausi_parametrai, geriausia_verte = optimizuoti_regresijos_modeli(
            GradientBoostingRegressor,
            X_train, y_train,
            X_val, y_val,
            n_trials=n_trials
        )
        
        # Išsaugome modelį
        from modeliai.model_train import issaugoti_modeli
        
        meta_info = {
            'hiperparametrai': geriausi_parametrai,
            'mse': geriausia_verte
        }
        
        issaugoti_modeli(geriausias_modelis, "Išlaikymas (optimizuotas)", meta_info=meta_info)
        
        logger.info(f"Išlaikymo modelis sėkmingai optimizuotas: MSE={geriausia_verte:.4f}")
        
        return geriausias_modelis, geriausi_parametrai, geriausia_verte
    
    except Exception as e:
        logger.error(f"Klaida optimizuojant išlaikymo modelį: {e}")
        return None, None, None

def sukurti_neuronini_islaikymo_modeli(X_train, y_train, X_val, y_val):
    """
    Sukuria ir treniruoja neuroninį tinklą išlaikymo sumos prognozavimui
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        
    Returns:
        tuple: (modelis, istorija) - treniruotas neuroninis tinklas ir mokymosi istorija
    """
    from modeliai.neuroninis_tinklas import (
        sukurti_regresijos_tinkla,
        treniruoti_tinkla,
        ivertinti_regresijos_tinkla,
        issaugoti_tinkla
    )
    
    logger.info("Kuriamas neuroninis tinklas išlaikymo prognozavimui")
    
    try:
        # Sukuriame neuroninį tinklą
        ivesties_dydis = X_train.shape[1]
        neuroninis_tinklas = sukurti_regresijos_tinkla(
            ivesties_dydis=ivesties_dydis,
            hidden_layers=[128, 64, 32],  # Gilesnis tinklas
            activation='relu',
            learning_rate=0.001,
            dropout_rate=0.3
        )
        
        # Treniruojame tinklą
        istorija, modelis = treniruoti_tinkla(
            model=neuroninis_tinklas,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochos=100,
            batch_dydis=32,
            ankstyvasis_sustabdymas=True
        )
        
        # Įvertiname modelį
        metrikos = ivertinti_regresijos_tinkla(modelis, X_val, y_val)
        
        logger.info(f"Neuroninis išlaikymo modelis treniruotas: MSE={metrikos['MSE']:.4f}, R2={metrikos['R2']:.4f}")
        
        # Išsaugome modelį
        modelio_kelias = issaugoti_tinkla(modelis, "Neuroninis_islaikymo_modelis")
        
        return modelis, istorija
    
    except Exception as e:
        logger.error(f"Klaida kuriant neuroninį išlaikymo modelį: {e}")
        return None, None

def apskaiciuoti_rekomendacijas(islaikymo_suma, duomenys):
    """
    Apskaičiuoja rekomenduojamą išlaikymo sumą ir jos paskirstymą tėvams
    
    Args:
        islaikymo_suma (float): Prognozuota išlaikymo suma
        duomenys (dict): Bylos duomenys
        
    Returns:
        dict: Rekomendacijos
    """
    # Gauname reikalingus duomenis
    pajamos_tevas = duomenys.get('pajamos_vyras', 0)
    pajamos_mama = duomenys.get('pajamos_moteris', 0)
    bendros_pajamos = pajamos_tevas + pajamos_mama
    
    # Apskaičiuojame pajamų santykį
    if bendros_pajamos > 0:
        tevas_dalis = pajamos_tevas / bendros_pajamos
        mama_dalis = pajamos_mama / bendros_pajamos
    else:
        tevas_dalis = 0.5
        mama_dalis = 0.5
    
    # Apskaičiuojame rekomenduojamas sumas
    if duomenys.get('gyvenamoji_vieta') == 'mama':
        # Tėvas moka išlaikymą
        suma_tevas = islaikymo_suma
        suma_mama = 0
        
        return {
            'islaikymo_suma': round(islaikymo_suma, 2),
            'tevas_moka': round(suma_tevas, 2),
            'tevas_procentais': round(tevas_dalis * 100, 1),
            'mama_procentais': round(mama_dalis * 100, 1),
            'komentaras': 'Vaikas gyvena su mama, išlaikymą moka tėvas'
        }
    elif duomenys.get('gyvenamoji_vieta') == 'tevas':
        # Mama moka išlaikymą
        suma_tevas = 0
        suma_mama = islaikymo_suma
        
        return {
            'islaikymo_suma': round(islaikymo_suma, 2),
            'mama_moka': round(suma_mama, 2),
            'tevas_procentais': round(tevas_dalis * 100, 1),
            'mama_procentais': round(mama_dalis * 100, 1),
            'komentaras': 'Vaikas gyvena su tėvu, išlaikymą moka mama'
        }
    else:
        # Kita situacija (pvz., gyvenama lygiomis dalimis)
        # Paskirstome išlaikymą pagal pajamų santykį
        suma_tevas = islaikymo_suma * mama_dalis
        suma_mama = islaikymo_suma * tevas_dalis
        
        return {
            'islaikymo_suma': round(islaikymo_suma, 2),
            'tevas_moka': round(suma_tevas, 2),
            'mama_moka': round(suma_mama, 2),
            'tevas_procentais': round(tevas_dalis * 100, 1),
            'mama_procentais': round(mama_dalis * 100, 1),
            'komentaras': 'Vaikas gyvena su abiem tėvais, išlaikymas paskirstytas pagal pajamų santykį'
        }