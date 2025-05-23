"""
Bendravimo tvarkos prognozavimo modulis

Šis modulis apima funkcijas, skirtas prognozuoti bendravimo tvarką santuokos nutraukimo bylose.
Šiam modeliui naudojamas neuroninis tinklas.
"""

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from config import Config
from modeliai.neuroninis_tinklas import sukurti_klasifikavimo_tinkla, ikrauti_tinkla

logger = logging.getLogger(__name__)

# Modelio failo pavadinimas
MODELIO_DIREKTORIJA = 'bendravimo_tvarka_model'

# Galimos bendravimo tvarkos kategorijos
BENDRAVIMO_TVARKOS = [
    'konkreti tvarka, nustatant bendravimą atostogų metu, šventėmis',
    'kas antrą savaitgalį',
    'neribota bendravimo tvarka',
    'kita'
]

def ikrauti_bendravimo_tvarkos_modeli():
    """
    Įkrauna treniruotą bendravimo tvarkos modelį
    
    Returns:
        tuple: (modelis, preprocessorius) - treniruotas modelis ir duomenų transformatorius
    """
    modelio_kelias = os.path.join(Config.MODELIU_DIREKTORIJA, MODELIO_DIREKTORIJA)
    
    try:
        # Bandome įkrauti modelį
        modelis = ikrauti_tinkla(modelio_kelias)
        
        # Bandome įkrauti preprocessorių
        preprocessor_kelias = os.path.join(modelio_kelias, 'preprocessor.joblib')
        preprocessorius = None
        
        if os.path.exists(preprocessor_kelias):
            import joblib
            preprocessorius = joblib.load(preprocessor_kelias)
        
        if modelis is not None:
            logger.info(f"Sėkmingai įkrautas bendravimo tvarkos modelis")
            return modelis, preprocessorius
        else:
            logger.warning("Bendravimo tvarkos modelis nerastas. Bus naudojamas fiktyvus modelis.")
            return None, None
    
    except Exception as e:
        logger.error(f"Klaida įkraunant bendravimo tvarkos modelį: {e}")
        return None, None

def prognozuoti_bendravimo_tvarka(duomenys):
    """
    Prognozuoja bendravimo tvarką pagal pateiktus duomenis
    
    Args:
        duomenys (dict): Duomenys, naudojami prognozavimui
    
    Returns:
        tuple: (prognozė, tikimybė) - prognozuota bendravimo tvarka ir prognozės tikimybė
    """
    # Įkrauname modelį
    modelis, preprocessorius = ikrauti_bendravimo_tvarkos_modeli()
    
    # Jei modelis nerastas, naudojame fiktyvų modelį
    if modelis is None:
        # Fiktyvus prognozavimas
        return fiktyvus_bendravimo_tvarkos_prognozavimas(duomenys)
    
    try:
        # Konvertuojame duomenis į DataFrame
        df = pd.DataFrame([duomenys])
        
        # Transformuojame duomenis, jei yra preprocessorius
        if preprocessorius is not None:
            X_transformed = preprocessorius.transform(df)
        else:
            # Jei nėra preprocessoriaus, naudojame originalius duomenis
            # Išrenkame tik skaitinius stulpelius
            skaitiniai_stulpeliai = df.select_dtypes(include=['number']).columns.tolist()
            X_transformed = df[skaitiniai_stulpeliai].values
        
        # Atliekame prognozę
        prognoze_probs = modelis.predict(X_transformed)[0]
        
        # Randame labiausiai tikėtiną kategoriją ir jos tikimybę
        max_index = np.argmax(prognoze_probs)
        prognoze = BENDRAVIMO_TVARKOS[max_index]
        tikimybe = prognoze_probs[max_index]
        
        logger.info(f"Prognozuota bendravimo tvarka: {prognoze} (tikimybė: {tikimybe:.4f})")
        
        return prognoze, float(tikimybe)
    
    except Exception as e:
        logger.error(f"Klaida prognozuojant bendravimo tvarką: {e}")
        return fiktyvus_bendravimo_tvarkos_prognozavimas(duomenys)

def fiktyvus_bendravimo_tvarkos_prognozavimas(duomenys):
    """
    Fiktyvus bendravimo tvarkos prognozavimas, kai nėra modelio
    
    Args:
        duomenys (dict): Duomenys, naudojami prognozavimui
    
    Returns:
        tuple: (prognozė, tikimybė) - prognozuota bendravimo tvarka ir prognozės tikimybė
    """
    logger.warning("Naudojamas fiktyvus bendravimo tvarkos prognozavimas")
    
    # Paprastas taisyklėmis pagrįstas sprendimas
    vaiko_amzius = duomenys.get('vaiko_amzius', 10)
    gyvenamoji_vieta = duomenys.get('gyvenamoji_vieta', 'mama')
    emocinis_rysys_neresidencinistevo = 'silpnas'
    
    if gyvenamoji_vieta == 'mama':
        emocinis_rysys_neresidencinistevo = duomenys.get('emocinis_rysys_tevas', 'vidutinis')
    else:
        emocinis_rysys_neresidencinistevo = duomenys.get('emocinis_rysys_mama', 'vidutinis')
    
    # Sprendžiame pagal vaiko amžių ir emocinį ryšį
    if vaiko_amzius <= 5:
        # Mažesni vaikai - dažniau konkreti tvarka
        if emocinis_rysys_neresidencinistevo == 'silpnas':
            prognoze = 'konkreti tvarka, nustatant bendravimą atostogų metu, šventėmis'
            tikimybe = 0.7
        else:
            # Jei ryšys su nerezidenciniu tėvu stiprus ar vidutinis
            prognoze = 'kas antrą savaitgalį'
            tikimybe = 0.6
    elif vaiko_amzius <= 12:
        # Vidutinio amžiaus vaikai
        if emocinis_rysys_neresidencinistevo == 'stiprus':
            prognoze = 'kas antrą savaitgalį'
            tikimybe = 0.8
        elif emocinis_rysys_neresidencinistevo == 'vidutinis':
            prognoze = 'kas antrą savaitgalį'
            tikimybe = 0.6
        else:
            prognoze = 'konkreti tvarka, nustatant bendravimą atostogų metu, šventėmis'
            tikimybe = 0.7
    else:
        # Vyresni vaikai
        if emocinis_rysys_neresidencinistevo == 'stiprus':
            prognoze = 'neribota bendravimo tvarka'
            tikimybe = 0.7
        elif emocinis_rysys_neresidencinistevo == 'vidutinis':
            prognoze = 'kas antrą savaitgalį'
            tikimybe = 0.6
        else:
            prognoze = 'konkreti tvarka, nustatant bendravimą atostogų metu, šventėmis'
            tikimybe = 0.65
    
    return prognoze, tikimybe

def sukurti_bendravimo_tvarkos_neuronini_tinkla(X_train, y_train, X_val, y_val):
    """
    Sukuria ir treniruoja neuroninį tinklą bendravimo tvarkos prognozavimui
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
    
    Returns:
        tuple: (modelis, istorija) - treniruotas neuroninis tinklas ir mokymosi istorija
    """
    from modeliai.neuroninis_tinklas import (
        sukurti_klasifikavimo_tinkla, 
        treniruoti_tinkla, 
        ivertinti_klasifikavimo_tinkla,
        issaugoti_tinkla
    )
    
    logger.info("Kuriamas neuroninis tinklas bendravimo tvarkos prognozavimui")
    
    try:
        # Apskaičiuojame klasių skaičių
        # Jei y yra kategoriškos reikšmės, turime jas konvertuoti į skaičius
        if isinstance(y_train[0], str):
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            y_train_encoded = encoder.fit_transform(y_train)
            y_val_encoded = encoder.transform(y_val)
            klasiu_skaicius = len(encoder.classes_)
        else:
            y_train_encoded = y_train
            y_val_encoded = y_val
            klasiu_skaicius = len(np.unique(y_train))
        
        logger.info(f"Bendravimo tvarkos klasių skaičius: {klasiu_skaicius}")
        
        # Sukuriame neuroninį tinklą
        ivesties_dydis = X_train.shape[1]
        
        neuroninis_tinklas = sukurti_klasifikavimo_tinkla(
            ivesties_dydis=ivesties_dydis,
            klasiu_skaicius=klasiu_skaicius,
            hidden_layers=[256, 128, 64],  # Gilesnis ir platesnis tinklas
            activation='relu',
            learning_rate=0.001,
            dropout_rate=0.4  # Didesnis dropout, kad išvengtume permokymo
        )
        
        # Treniruojame tinklą
        istorija, modelis = treniruoti_tinkla(
            model=neuroninis_tinklas,
            X_train=X_train,
            y_train=y_train_encoded,
            X_val=X_val,
            y_val=y_val_encoded,
            epochos=200,  # Daugiau epochų, bet su early stopping
            batch_dydis=32,
            ankstyvasis_sustabdymas=True
        )
        
        # Įvertiname modelį
        metrikos = ivertinti_klasifikavimo_tinkla(modelis, X_val, y_val_encoded, klasiu_skaicius)
        
        logger.info(f"Neuroninis bendravimo tvarkos modelis treniruotas: "
                  f"accuracy={metrikos['accuracy']:.4f}, f1={metrikos['f1']:.4f}")
        
        # Išsaugome modelį
        # Sukuriame direktoriją modeliui
        modelio_direktorija = os.path.join(Config.MODELIU_DIREKTORIJA, MODELIO_DIREKTORIJA)
        os.makedirs(modelio_direktorija, exist_ok=True)
        
        # Išsaugome encoderį
        if isinstance(y_train[0], str):
            import joblib
            joblib.dump(encoder, os.path.join(modelio_direktorija, 'label_encoder.joblib'))
        
        # Išsaugome modelį
        modelio_kelias = issaugoti_tinkla(modelis, "Bendravimo_tvarka", direktorija=modelio_direktorija)
        
        return modelis, istorija
    
    except Exception as e:
        logger.error(f"Klaida kuriant neuroninį bendravimo tvarkos modelį: {e}")
        return None, None

def optimizuoti_bendravimo_tvarkos_neuronini_tinkla(X_train, y_train, X_val, y_val):
    """
    Optimizuoja bendravimo tvarkos neuroninio tinklo hiperparametrus
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
    
    Returns:
        tuple: (geriausias_modelis, geriausi_parametrai, geriausia_verte, istorija)
    """
    from modeliai.optuna_optimizer import optimizuoti_neuronini_tinkla
    
    logger.info("Pradedama bendravimo tvarkos neuroninio tinklo hiperparametrų optimizacija")
    
    try:
        # Apskaičiuojame klasių skaičių
        # Jei y yra kategoriškos reikšmės, turime jas konvertuoti į skaičius
        if isinstance(y_train[0], str):
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            y_train_encoded = encoder.fit_transform(y_train)
            y_val_encoded = encoder.transform(y_val)
            klasiu_skaicius = len(encoder.classes_)
        else:
            y_train_encoded = y_train
            y_val_encoded = y_val
            klasiu_skaicius = len(np.unique(y_train))
        
        # Optimizuojame neuroninį tinklą
        geriausias_modelis, geriausi_parametrai, geriausia_verte, istorija = optimizuoti_neuronini_tinkla(
            X_train, y_train_encoded, X_val, y_val_encoded, 
            modelio_tipas='klasifikavimas',
            n_trials=50  # Bandymų skaičius
        )
        
        # Išsaugome modelį
        modelio_direktorija = os.path.join(Config.MODELIU_DIREKTORIJA, MODELIO_DIREKTORIJA)
        os.makedirs(modelio_direktorija, exist_ok=True)
        
        # Išsaugome encoderį
        if isinstance(y_train[0], str):
            import joblib
            joblib.dump(encoder, os.path.join(modelio_direktorija, 'label_encoder.joblib'))
        
        # Išsaugome modelį
        from modeliai.neuroninis_tinklas import issaugoti_tinkla
        modelio_kelias = issaugoti_tinkla(
            geriausias_modelis, 
            "Bendravimo_tvarka_optimizuotas", 
            direktorija=modelio_direktorija
        )
        
        logger.info(f"Bendravimo tvarkos neuroninis tinklas sėkmingai optimizuotas: tikslumas={1.0 - geriausia_verte:.4f}")
        
        return geriausias_modelis, geriausi_parametrai, geriausia_verte, istorija
    
    except Exception as e:
        logger.error(f"Klaida optimizuojant bendravimo tvarkos neuroninį tinklą: {e}")
        return None, None, None, None