import os
import logging
import json
import numpy as np
import pandas as pd
from config import Config

from duomenu_apdorojimas.duomenu_analizatorius import DuomenuAnalizatorius
from duomenu_apdorojimas.duomenu_paruosimas import DuomenuParuosejas
from modeliai.model_train import (
    palyginti_modelius,
    issaugoti_modeli
)
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


MODELIO_PARINKTYS = {
    'gyvenamoji_vieta': {
        'tipas': 'klasifikavimas',
        'modeliai': [
            RandomForestClassifier(n_estimators=100),
            SVC(probability=True),
            XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        ]
    },
    'islaikymas': {
        'tipas': 'regresija',
        'modeliai': [
            RandomForestRegressor(),
            SVR(),
            XGBRegressor()
        ]
    },
    'bendravimo_tvarka': {
        'tipas': 'klasifikavimas',
        'modeliai': [
            RandomForestClassifier(n_estimators=100),
            SVC(probability=True),
            XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        ]
    },
    'turto_padalijimas': {
        'tipas': 'regresija',
        'modeliai': [
            RandomForestRegressor(),
            SVR(),
            XGBRegressor()
        ]
    },
    'prievoles': {
        'tipas': 'regresija',
        'modeliai': [
            RandomForestRegressor(),
            SVR(),
            XGBRegressor()
        ]
    },
    'bylinejimosi_islaidos': {
        'tipas': 'regresija',
        'modeliai': [
            RandomForestRegressor(),
            SVR(),
            XGBRegressor()
        ]
    },
}

def sukurti_direktorijas():
    os.makedirs(Config.MODELIU_DIREKTORIJA, exist_ok=True)
    os.makedirs(Config.DUOMENU_DIREKTORIJA, exist_ok=True)
    os.makedirs(os.path.join(Config.MODELIU_DIREKTORIJA, 'metrikos'), exist_ok=True)


def main():
    sukurti_direktorijas()

    logger.info("Duomenų nuskaitymas ir paruošimas")
    analizatorius = DuomenuAnalizatorius()
    duomenys = analizatorius.nuskaityti_duomenis()
    paruosejas = DuomenuParuosejas()
    paruosejas.nuskaityti_duomenis(duomenys)
    paruosejas.valyti_duomenis()
    paruosejas.paruosti_duomenis()

    for uzdavinys, info in MODELIO_PARINKTYS.items():
        tipas = info['tipas']
        modeliai = info['modeliai']
        
        logger.info(f"\n------ Mokymas: {uzdavinys} ({tipas}) ------")
        X_train, y_train, X_val, y_val = paruosejas.gauti_mokymo_duomenis(uzdavinys)

        geriausias_modelis, rezultatai = palyginti_modelius(
            modeliai, X_train, y_train, X_val, y_val, modelio_tipas=tipas
        )

        if geriausias_modelis:
            logger.info(f"Saugomas geriausias modelis uždavinio '{uzdavinys}'")
            issaugoti_modeli(geriausias_modelis, pavadinimas=uzdavinys)

            metrikos_failas = os.path.join(Config.MODELIU_DIREKTORIJA, "metrikos", f"{uzdavinys}_metrikos.json")
            with open(metrikos_failas, "w", encoding="utf-8") as f:
                json.dump(rezultatai, f, ensure_ascii=False, indent=2)
        else:
            logger.warning(f"Nepavyko apmokyti modelio uždaviniui: {uzdavinys}")

    logger.info("Modelių treniravimas baigtas.")

if __name__ == '__main__':
    main()
