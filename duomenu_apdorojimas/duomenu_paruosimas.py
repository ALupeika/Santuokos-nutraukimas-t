"""
Duomenų paruošimo modulis

Šis modulis apima duomenų paruošimą ir transformavimą prieš modelių treniravimą
"""

import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

from config import Config
from duomenu_apdorojimas.duomenu_valymas import (
    uzpildyti_trukstamas_reiksmes,
    pašalinti_isskirties,
    konvertuoti_i_skaicius
)

logger = logging.getLogger(__name__)

class DuomenuParuosejas:
    """
    Klasė, atsakinga už duomenų paruošimą mašininio mokymosi modeliams
    """
    
    def __init__(self):
        """
        Inicializuoja duomenų paruošėjo objektą
        """
        self.duomenys = {}  # Pradinis duomenų rinkinys
        self.paruosti_duomenys = {}  # Paruošti duomenys modeliavimui
        self.preprocessoriai = {}  # Duomenų transformavimo objektai kiekvienam uždaviniui
        
        # Treniravimo, validavimo ir testavimo duomenys
        self.training_data = {}
        self.validation_data = {}
        self.testing_data = {}
    
    def nuskaityti_duomenis(self, duomenys=None):
        """
        Nuskaito duomenis iš CSV failų arba naudoja pateiktą duomenų žodyną
        
        Args:
            duomenys (dict, optional): Žodynas su DataFrame objektais
        """
        if duomenys is not None:
            self.duomenys = duomenys
            logger.info(f"Naudojami pateikti duomenys: {len(duomenys)} failai")
        else:
            # Jei duomenys nepateikti, nuskaitome iš CSV
            from duomenu_apdorojimas.duomenu_analizatorius import DuomenuAnalizatorius
            analizatorius = DuomenuAnalizatorius()
            self.duomenys = analizatorius.nuskaityti_duomenis()
            logger.info(f"Nuskaityti duomenys iš CSV: {len(self.duomenys)} failai")
    
    def valyti_duomenis(self):
        """
        Valo duomenis: užpildo trūkstamas reikšmes, pašalina išskirtis
        """
        if not self.duomenys:
            logger.error("Nėra duomenų valymui. Pirmiau nuskaitykite duomenis.")
            return
            
        # Kopijuojame duomenis, kad išsaugotume originalius
        self.paruosti_duomenys = {k: df.copy() for k, df in self.duomenys.items()}
        
        # Atliekame duomenų valymą kiekvienam DataFrame
        for pavadinimas, df in self.paruosti_duomenys.items():
            logger.info(f"Valomi duomenys: {pavadinimas}")
            
            # Pirmiausia konvertuojame skaitinius stulpelius
            for stulpelis in df.columns:
                # Tikriname, ar stulpelis gali būti skaitinis
                if (df[stulpelis].dtype == 'object' and
                    not pd.api.types.is_datetime64_any_dtype(df[stulpelis])):
                    try:
                        # Bandome konvertuoti į skaičius
                        df = konvertuoti_i_skaicius(df, stulpelis)
                    except Exception as e:
                        logger.debug(f"Nepavyko konvertuoti stulpelio {stulpelis} į skaičius: {e}")
            
            # Užpildome trūkstamas reikšmes skaitiniuose stulpeliuose
            for stulpelis in df.select_dtypes(include=['number']).columns:
                df = uzpildyti_trukstamas_reiksmes(df, stulpelis, 'mean')
            
            # Užpildome trūkstamas reikšmes kategoriniuose stulpeliuose
            for stulpelis in df.select_dtypes(exclude=['number']).columns:
                df = uzpildyti_trukstamas_reiksmes(df, stulpelis, 'mode')
            
            # Pašaliname išskirtis skaitiniuose stulpeliuose
            for stulpelis in df.select_dtypes(include=['number']).columns:
                if stulpelis != 'byla_id' and 'nr' not in stulpelis.lower():
                    df = pašalinti_isskirties(df, stulpelis, metodas='iqr')
            
            # Atnaujineme išvalytą DataFrame
            self.paruosti_duomenys[pavadinimas] = df
            
            # Pateikiame statistiką apie išvalytus duomenis
            logger.info(f"Išvalyti duomenys {pavadinimas}: {df.shape[0]} eilutės, {df.shape[1]} stulpeliai")
            logger.info(f"Trūkstamos reikšmės po valymo: {df.isnull().sum().sum()}")
    
    def paruosti_duomenis(self):
        """
        Paruošia duomenis mašininio mokymosi modeliams:
        1. Sujungia duomenis
        2. Kodavimas (one-hot encoding)
        3. Normalizavimas
        4. Skaido į treniravimo, validavimo ir testavimo imtis
        """
        if not self.paruosti_duomenys:
            logger.error("Nėra paruoštų duomenų. Pirmiau išvalykite duomenis.")
            return
        
        # Paruošimas gyvenamosios vietos modeliui
        self._paruosti_gyvenamoji_vieta()
        
        # Paruošimas išlaikymo modeliui
        self._paruosti_islaikymas()
        
        # Paruošimas bendravimo tvarkos modeliui
        self._paruosti_bendravimo_tvarka()
        
        # Paruošimas turto padalijimo modeliui
        self._paruosti_turto_padalijimas()
        
        # Paruošimas prievolių modeliui
        self._paruosti_prievoles()
        
        # Paruošimas bylinėjimosi išlaidų modeliui
        self._paruosti_bylinejimosi_islaidos()
    
    def _paruosti_duomenis_jungimui(self, df1_pavadinimas, df2_pavadinimas, imties_dydis=5000, su_vaikais=False):
        """
        Paruošia du duomenų rinkinius sujungimui, išsaugant byla_id reikšmes ir tvarkant NaN reikšmes.
        
        Args:
            df1_pavadinimas (str): Pirmojo DataFrame pavadinimas
            df2_pavadinimas (str): Antrojo DataFrame pavadinimas
            imties_dydis (int): Maksimalus duomenų imties dydis
            su_vaikais (bool): Ar modelis yra susijęs su vaikų klausimais
            
        Returns:
            tuple: (df1, df2) paruošti sujungimui DataFrame objektai arba (None, None) jei neįmanoma paruošti
        """
        # Gauname pradinius duomenis (PRIEŠ valymą)
        df1_pradinis = self.duomenys.get(df1_pavadinimas)
        df2_pradinis = self.duomenys.get(df2_pavadinimas)
        
        if df1_pradinis is None or df2_pradinis is None:
            logger.error(f"Nerasti pradiniai duomenys: {df1_pavadinimas} arba {df2_pavadinimas}")
            return None, None
        
        # Kopijuojame tik byla_id stulpelį iš pradinių duomenų
        df1_byla_id = df1_pradinis['byla_id'].copy()
        df2_byla_id = df2_pradinis['byla_id'].copy()
        
        # Konvertuojame byla_id į string tipą
        df1_byla_id = df1_byla_id.astype(str)
        df2_byla_id = df2_byla_id.astype(str)
        
        # Gauname išvalytus duomenis (BET be byla_id)
        df1 = self.paruosti_duomenys.get(df1_pavadinimas).copy()
        df2 = self.paruosti_duomenys.get(df2_pavadinimas).copy()
        
        if df1 is None or df2 is None:
            logger.error(f"Nerasti išvalyti duomenys: {df1_pavadinimas} arba {df2_pavadinimas}")
            return None, None
        
        # Pakeičiame išvalytų duomenų byla_id su originaliais byla_id
        df1['byla_id'] = df1_byla_id.values
        df2['byla_id'] = df2_byla_id.values
        
        # Pašaliname eilutes, kur byla_id yra tuščias arba 'nan'
        df1 = df1[~df1['byla_id'].isin(['nan', ''])]
        df2 = df2[~df2['byla_id'].isin(['nan', ''])]
        
        # Tikriname, ar yra likusių eilučių
        if len(df1) == 0 or len(df2) == 0:
            logger.error(f"Po byla_id filtravimo nebelieka duomenų: {df1_pavadinimas}({len(df1)}) arba {df2_pavadinimas}({len(df2)})")
            return None, None
        
        # Jei modelis susijęs su vaikų klausimais ir vienas iš dataframe yra 'seima_sugeneruota'
        if su_vaikais and (df1_pavadinimas == 'seima_sugeneruota' or df2_pavadinimas == 'seima_sugeneruota'):
            # Pašaliname šeimas, kurios neturi vaikų
            seima_df = df1 if df1_pavadinimas == 'seima_sugeneruota' else df2
            seima_df = seima_df[seima_df['vaiku_skaicius'] > 0]
            
            if df1_pavadinimas == 'seima_sugeneruota':
                df1 = seima_df
            else:
                df2 = seima_df
            
            logger.info(f"Pašalintos šeimos be vaikų, liko {len(seima_df)} šeimų su vaikais")
        
        # Jei reikia, sumažiname duomenų kiekį
        if len(df1) > imties_dydis:
            df1 = df1.sample(n=imties_dydis, random_state=Config.ATSITIKTINIS_SEED)
            logger.info(f"Imama {df1_pavadinimas} atsitiktinė imtis: {len(df1)} eilutės")
        
        if len(df2) > imties_dydis:
            df2 = df2.sample(n=imties_dydis, random_state=Config.ATSITIKTINIS_SEED)
            logger.info(f"Imama {df2_pavadinimas} atsitiktinė imtis: {len(df2)} eilutės")
        
        # Skaičiuojame, kiek yra bendrų byla_id reikšmių
        bendros_ids = set(df1['byla_id']).intersection(set(df2['byla_id']))
        logger.info(f"Bendros byla_id reikšmės tarp {df1_pavadinimas} ir {df2_pavadinimas}: {len(bendros_ids)}")
        
        if len(bendros_ids) == 0:
            logger.error(f"Nėra bendrų byla_id reikšmių tarp {df1_pavadinimas} ir {df2_pavadinimas}")
            return None, None
        
        # Filtruojame tik tas eilutes, kurios turi bendras byla_id reikšmes
        df1 = df1[df1['byla_id'].isin(bendros_ids)]
        df2 = df2[df2['byla_id'].isin(bendros_ids)]
        
        logger.info(f"Paruošti duomenys jungimui: {df1_pavadinimas}({len(df1)}) ir {df2_pavadinimas}({len(df2)})")
        
        # Užpildome trūkstamas reikšmes (NaN)
        for stulpelis in df1.columns:
            if stulpelis != 'byla_id':  # Neliečiame byla_id stulpelio
                if df1[stulpelis].dtype.kind in 'ifc':  # Skaitiniai stulpeliai
                    df1[stulpelis] = df1[stulpelis].fillna(df1[stulpelis].median())
                else:  # Kategoriniai stulpeliai
                    df1[stulpelis] = df1[stulpelis].fillna(df1[stulpelis].mode()[0] if not df1[stulpelis].mode().empty else "nežinoma")
        
        for stulpelis in df2.columns:
            if stulpelis != 'byla_id':  # Neliečiame byla_id stulpelio
                if df2[stulpelis].dtype.kind in 'ifc':  # Skaitiniai stulpeliai
                    df2[stulpelis] = df2[stulpelis].fillna(df2[stulpelis].median())
                else:  # Kategoriniai stulpeliai
                    df2[stulpelis] = df2[stulpelis].fillna(df2[stulpelis].mode()[0] if not df2[stulpelis].mode().empty else "nežinoma")
        
        # Pašaliname visas likusias NaN reikšmes, jei tokių būtų
        df1 = df1.fillna(0)
        df2 = df2.fillna(0)
        
        return df1, df2
    
    def _paruosti_gyvenamoji_vieta(self):
        """
        Paruošia duomenis vaiko gyvenamosios vietos prognozavimui
        """
        logger.info("Paruošiami duomenys gyvenamosios vietos modeliui")
        
        # Naudojame pagalbinę funkciją duomenų paruošimui su parametru su_vaikais=True
        seima_df, vaikai_df = self._paruosti_duomenis_jungimui('seima_sugeneruota', 'vaikai_sugeneruota', su_vaikais=True)
        
        # Jei negalima paruošti duomenų, grįžtame
        if seima_df is None or vaikai_df is None:
            return
        
        # Pasirenkame tik reikalingus stulpelius iš seima_df, kad sumažintume atminties naudojimą
        seima_df_mini = seima_df[['byla_id', 'santuokos_trukme', 'amzius_vyras', 'amzius_moteris', 
                                'pajamos_vyras', 'pajamos_moteris', 'vaiku_skaicius']]
        
        # Atliekame sujungimą
        sujungti_df = pd.merge(
            vaikai_df, 
            seima_df_mini, 
            on='byla_id',
            how='inner'
        )
        
        # Tikriname, ar yra duomenų po sujungimo
        if len(sujungti_df) == 0:
            logger.error("Po duomenų sujungimo nebeliko eilučių analizei.")
            return
        
        # Tikslo kintamasis
        y = sujungti_df['gyvenamoji_vieta']
        
        # Užpildome NaN reikšmes tikslo kintamajame (jei tokių yra)
        if y.dtype.kind in 'ifc':  # Jei skaitinis
            y = y.fillna(y.median())
        else:  # Jei kategorinis
            y = y.fillna(y.mode()[0] if not y.mode().empty else "nežinoma")
        
        # Požymiai
        X = sujungti_df.drop([
            'gyvenamoji_vieta', 'bendravimo_tvarka', 'islaikymas', 
            'islaikymo_iskolinimas', 'byla_id', 'vaiko_nr'
        ], axis=1)
        
        # Užpildome visas likusias NaN reikšmes
        for stulpelis in X.columns:
            if X[stulpelis].dtype.kind in 'ifc':  # Skaitiniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].median())
            else:  # Kategoriniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].mode()[0] if not X[stulpelis].mode().empty else "nežinoma")
        
        # Pašaliname visas likusias NaN reikšmes, jei tokių būtų
        X = X.fillna(0)
        
        # Nustatome skaitinius ir kategorinius stulpelius
        skaitiniai_stulpeliai = X.select_dtypes(include=['number']).columns.tolist()
        kategoriniai_stulpeliai = X.select_dtypes(exclude=['number']).columns.tolist()
        
        # Sukuriame transformavimo pipelineą
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), skaitiniai_stulpeliai),
                ('cat', OneHotEncoder(handle_unknown='ignore'), kategoriniai_stulpeliai)
            ],
            remainder='passthrough'
        )
        
        # Pritaikome transformaciją duomenims
        X_transformed = preprocessor.fit_transform(X)
        
        # Skaidome duomenis
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_transformed, y, 
            test_size=(Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS),
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Skaidome likusią dalį į validavimo ir testavimo
        validation_ratio = Config.VALIDAVIMO_DALIS / (Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - validation_ratio,
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Išsaugome duomenis ir transformatorių
        self.training_data['gyvenamoji_vieta'] = (X_train, y_train)
        self.validation_data['gyvenamoji_vieta'] = (X_val, y_val)
        self.testing_data['gyvenamoji_vieta'] = (X_test, y_test)
        self.preprocessoriai['gyvenamoji_vieta'] = preprocessor
        
        # Išsaugome preprocessorių į diską
        preprocessor_path = os.path.join(Config.MODELIU_DIREKTORIJA, 'gyvenamoji_vieta_preprocessor.joblib')
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Gyvenamosios vietos preprocessorius išsaugotas: {preprocessor_path}")
        
        logger.info(f"Paruošti gyvenamosios vietos duomenys: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    def _paruosti_islaikymas(self):
        """
        Paruošia duomenis vaiko išlaikymo prognozavimui
        """
        logger.info("Paruošiami duomenys išlaikymo modeliui")
        
        # Naudojame pagalbinę funkciją duomenų paruošimui su parametru su_vaikais=True
        seima_df, vaikai_df = self._paruosti_duomenis_jungimui('seima_sugeneruota', 'vaikai_sugeneruota', su_vaikais=True)
        
        # Jei negalima paruošti duomenų, grįžtame
        if seima_df is None or vaikai_df is None:
            return
        
        # Pasirenkame tik reikalingus stulpelius iš seima_df, kad sumažintume atminties naudojimą
        seima_df_mini = seima_df[['byla_id', 'santuokos_trukme', 'amzius_vyras', 'amzius_moteris', 
                                'pajamos_vyras', 'pajamos_moteris', 'vaiku_skaicius']]
        
        # Atliekame sujungimą
        sujungti_df = pd.merge(
            vaikai_df, 
            seima_df_mini, 
            on='byla_id',
            how='inner'
        )
        
        # Tikriname, ar yra duomenų po sujungimo
        if len(sujungti_df) == 0:
            logger.error("Po duomenų sujungimo nebeliko eilučių analizei.")
            return
        
        # Tikslo kintamasis - išlaikymo suma
        y = sujungti_df['islaikymas']
        
        # Užpildome NaN reikšmes tikslo kintamajame (jei tokių yra)
        if y.dtype.kind in 'ifc':  # Jei skaitinis
            y = y.fillna(y.median())
        else:  # Jei kategorinis
            y = y.fillna(y.mode()[0] if not y.mode().empty else "nežinoma")
        
        # Požymiai
        X = sujungti_df.drop([
            'gyvenamoji_vieta', 'bendravimo_tvarka', 'islaikymas', 
            'islaikymo_iskolinimas', 'byla_id', 'vaiko_nr'
        ], axis=1)
        
        # Užpildome visas likusias NaN reikšmes
        for stulpelis in X.columns:
            if X[stulpelis].dtype.kind in 'ifc':  # Skaitiniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].median())
            else:  # Kategoriniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].mode()[0] if not X[stulpelis].mode().empty else "nežinoma")
        
        # Pašaliname visas likusias NaN reikšmes, jei tokių būtų
        X = X.fillna(0)
        
        # Nustatome skaitinius ir kategorinius stulpelius
        skaitiniai_stulpeliai = X.select_dtypes(include=['number']).columns.tolist()
        kategoriniai_stulpeliai = X.select_dtypes(exclude=['number']).columns.tolist()
        
        # Sukuriame transformavimo pipelineą
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), skaitiniai_stulpeliai),
                ('cat', OneHotEncoder(handle_unknown='ignore'), kategoriniai_stulpeliai)
            ],
            remainder='passthrough'
        )
        
        # Pritaikome transformaciją duomenims
        X_transformed = preprocessor.fit_transform(X)
        
        # Skaidome duomenis
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_transformed, y, 
            test_size=(Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS),
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Skaidome likusią dalį į validavimo ir testavimo
        validation_ratio = Config.VALIDAVIMO_DALIS / (Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - validation_ratio,
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Išsaugome duomenis ir transformatorių
        self.training_data['islaikymas'] = (X_train, y_train)
        self.validation_data['islaikymas'] = (X_val, y_val)
        self.testing_data['islaikymas'] = (X_test, y_test)
        self.preprocessoriai['islaikymas'] = preprocessor
        
        # Išsaugome preprocessorių į diską
        preprocessor_path = os.path.join(Config.MODELIU_DIREKTORIJA, 'islaikymas_preprocessor.joblib')
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Išlaikymo preprocessorius išsaugotas: {preprocessor_path}")
        
        logger.info(f"Paruošti išlaikymo duomenys: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    def _paruosti_bendravimo_tvarka(self):
        """
        Paruošia duomenis bendravimo tvarkos prognozavimui
        """
        logger.info("Paruošiami duomenys bendravimo tvarkos modeliui")
        
        # Naudojame pagalbinę funkciją duomenų paruošimui su parametru su_vaikais=True
        seima_df, vaikai_df = self._paruosti_duomenis_jungimui('seima_sugeneruota', 'vaikai_sugeneruota', su_vaikais=True)
        
        # Jei negalima paruošti duomenų, grįžtame
        if seima_df is None or vaikai_df is None:
            return
        
        # Pasirenkame tik reikalingus stulpelius iš seima_df, kad sumažintume atminties naudojimą
        seima_df_mini = seima_df[['byla_id', 'santuokos_trukme', 'amzius_vyras', 'amzius_moteris', 
                                'pajamos_vyras', 'pajamos_moteris', 'vaiku_skaicius']]
        
        # Atliekame sujungimą
        sujungti_df = pd.merge(
            vaikai_df, 
            seima_df_mini, 
            on='byla_id',
            how='inner'
        )
        
        # Tikriname, ar yra duomenų po sujungimo
        if len(sujungti_df) == 0:
            logger.error("Po duomenų sujungimo nebeliko eilučių analizei.")
            return
        
        # Tikslo kintamasis
        y = sujungti_df['bendravimo_tvarka']
        
        # Užpildome NaN reikšmes tikslo kintamajame (jei tokių yra)
        if y.dtype.kind in 'ifc':  # Jei skaitinis
            y = y.fillna(y.median())
        else:  # Jei kategorinis
            y = y.fillna(y.mode()[0] if not y.mode().empty else "nežinoma")
        
        # Požymiai - pašaliname kitus tikslo kintamuosius
        X = sujungti_df.drop([
            'gyvenamoji_vieta', 'bendravimo_tvarka', 'islaikymas', 
            'islaikymo_iskolinimas', 'byla_id', 'vaiko_nr'
        ], axis=1)
        
        # Užpildome visas likusias NaN reikšmes
        for stulpelis in X.columns:
            if X[stulpelis].dtype.kind in 'ifc':  # Skaitiniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].median())
            else:  # Kategoriniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].mode()[0] if not X[stulpelis].mode().empty else "nežinoma")
        
        # Pašaliname visas likusias NaN reikšmes, jei tokių būtų
        X = X.fillna(0)
        
        # Nustatome skaitinius ir kategorinius stulpelius
        skaitiniai_stulpeliai = X.select_dtypes(include=['number']).columns.tolist()
        kategoriniai_stulpeliai = X.select_dtypes(exclude=['number']).columns.tolist()
        
        # Sukuriame transformavimo pipelineą
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), skaitiniai_stulpeliai),
                ('cat', OneHotEncoder(handle_unknown='ignore'), kategoriniai_stulpeliai)
            ],
            remainder='passthrough'
        )
        
        # Pritaikome transformaciją
        X_transformed = preprocessor.fit_transform(X)
        
        # Skaidome duomenis
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_transformed, y, 
            test_size=(Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS),
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Skaidome likusią dalį į validavimo ir testavimo
        validation_ratio = Config.VALIDAVIMO_DALIS / (Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - validation_ratio,
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Išsaugome duomenis ir transformatorių
        self.training_data['bendravimo_tvarka'] = (X_train, y_train)
        self.validation_data['bendravimo_tvarka'] = (X_val, y_val)
        self.testing_data['bendravimo_tvarka'] = (X_test, y_test)
        self.preprocessoriai['bendravimo_tvarka'] = preprocessor
        
        # Išsaugome preprocessorių į diską
        preprocessor_path = os.path.join(Config.MODELIU_DIREKTORIJA, 'bendravimo_tvarka_preprocessor.joblib')
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Bendravimo tvarkos preprocessorius išsaugotas: {preprocessor_path}")
        
        logger.info(f"Paruošti bendravimo tvarkos duomenys: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    def _paruosti_turto_padalijimas(self):
        """
        Paruošia duomenis turto padalijimo prognozavimui
        """
        logger.info("Paruošiami duomenys turto padalijimo modeliui")
        
        # Naudojame pagalbinę funkciją duomenų paruošimui (su_vaikais=False yra numatytoji reikšmė)
        seima_df, turtas_df = self._paruosti_duomenis_jungimui('seima_sugeneruota', 'turtas_sugeneruota')
        
        # Jei negalima paruošti duomenų, grįžtame
        if seima_df is None or turtas_df is None:
            return
        
        # Pasirenkame tik reikalingus stulpelius iš seima_df, kad sumažintume atminties naudojimą
        seima_df_mini = seima_df[['byla_id', 'santuokos_trukme', 'amzius_vyras', 'amzius_moteris', 
                                'pajamos_vyras', 'pajamos_moteris', 'vaiku_skaicius', 'nutraukimo_budas']]
        
        # Atliekame sujungimą
        sujungti_df = pd.merge(
            turtas_df, 
            seima_df_mini, 
            on='byla_id',
            how='inner'
        )
        
        # Patikriname, ar yra duomenų po sujungimo
        if len(sujungti_df) == 0:
            logger.error("Po duomenų sujungimo nebeliko eilučių analizei.")
            return
        
        # Tikslo kintamasis - vyro turto procentas
        y = sujungti_df['padalijimas_proc_vyras']
        
        # Užpildome NaN reikšmes tikslo kintamajame (jei tokių yra)
        if y.dtype.kind in 'ifc':  # Jei skaitinis
            y = y.fillna(y.median())
        else:  # Jei kategorinis
            y = y.fillna(y.mode()[0] if not y.mode().empty else "nežinoma")
        
        # Požymiai - pašaliname kitus tikslo kintamuosius
        X = sujungti_df.drop([
            'padalijimas_proc_vyras', 'padalijimas_proc_moteris', 
            'turto_vyras_eur', 'turto_moteris_eur',
            'turto_vyras_po_asmeniniu_lesu', 'turto_moteris_po_asmeniniu_lesu',
            'byla_id'
        ], axis=1)
        
        # Užpildome visas likusias NaN reikšmes
        for stulpelis in X.columns:
            if X[stulpelis].dtype.kind in 'ifc':  # Skaitiniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].median())
            else:  # Kategoriniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].mode()[0] if not X[stulpelis].mode().empty else "nežinoma")
        
        # Pašaliname visas likusias NaN reikšmes, jei tokių būtų
        X = X.fillna(0)
        
        # Nustatome skaitinius ir kategorinius stulpelius
        skaitiniai_stulpeliai = X.select_dtypes(include=['number']).columns.tolist()
        kategoriniai_stulpeliai = X.select_dtypes(exclude=['number']).columns.tolist()
        
        # Sukuriame transformavimo pipelineą
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), skaitiniai_stulpeliai),
                ('cat', OneHotEncoder(handle_unknown='ignore'), kategoriniai_stulpeliai)
            ],
            remainder='passthrough'
        )
        
        # Pritaikome transformaciją
        X_transformed = preprocessor.fit_transform(X)
        
        # Skaidome duomenis
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_transformed, y, 
            test_size=(Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS),
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Skaidome likusią dalį į validavimo ir testavimo
        validation_ratio = Config.VALIDAVIMO_DALIS / (Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - validation_ratio,
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Išsaugome duomenis ir transformatorių
        self.training_data['turto_padalijimas'] = (X_train, y_train)
        self.validation_data['turto_padalijimas'] = (X_val, y_val)
        self.testing_data['turto_padalijimas'] = (X_test, y_test)
        self.preprocessoriai['turto_padalijimas'] = preprocessor
        
        # Išsaugome preprocessorių į diską
        preprocessor_path = os.path.join(Config.MODELIU_DIREKTORIJA, 'turto_padalijimas_preprocessor.joblib')
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Turto padalijimo preprocessorius išsaugotas: {preprocessor_path}")
        
        logger.info(f"Paruošti turto padalijimo duomenys: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    def _paruosti_prievoles(self):
        """
        Paruošia duomenis prievolių paskirstymo prognozavimui
        """
        logger.info("Paruošiami duomenys prievolių modeliui")
        
        # Naudojame pagalbinę funkciją duomenų paruošimui (su_vaikais=False yra numatytoji reikšmė)
        seima_df, prievoles_df = self._paruosti_duomenis_jungimui('seima_sugeneruota', 'prievoles_sugeneruota')
        
        # Jei negalima paruošti duomenų, grįžtame
        if seima_df is None or prievoles_df is None:
            return
        
        # Pasirenkame tik reikalingus stulpelius iš seima_df, kad sumažintume atminties naudojimą
        seima_df_mini = seima_df[['byla_id', 'santuokos_trukme', 'amzius_vyras', 'amzius_moteris', 
                                'pajamos_vyras', 'pajamos_moteris', 'vaiku_skaicius', 'nutraukimo_budas']]
        
        # Atliekame sujungimą
        sujungti_df = pd.merge(
            prievoles_df, 
            seima_df_mini, 
            on='byla_id',
            how='inner'
        )
        
        # Patikriname, ar yra duomenų po sujungimo
        if len(sujungti_df) == 0:
            logger.error("Po duomenų sujungimo nebeliko eilučių analizei.")
            return
        
        # Tikslo kintamasis - vyro prievolių dalis
        y = sujungti_df['prievoles_vyras']
        
        # Užpildome NaN reikšmes tikslo kintamajame (jei tokių yra)
        if y.dtype.kind in 'ifc':  # Jei skaitinis
            y = y.fillna(y.median())
        else:  # Jei kategorinis
            y = y.fillna(y.mode()[0] if not y.mode().empty else "nežinoma")
        
        # Požymiai - pašaliname kitus tikslo kintamuosius
        X = sujungti_df.drop([
            'prievoles_vyras', 'prievoles_moteris',
            'byla_id'
        ], axis=1)
        
        # Užpildome visas likusias NaN reikšmes
        for stulpelis in X.columns:
            if X[stulpelis].dtype.kind in 'ifc':  # Skaitiniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].median())
            else:  # Kategoriniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].mode()[0] if not X[stulpelis].mode().empty else "nežinoma")
        
        # Pašaliname visas likusias NaN reikšmes, jei tokių būtų
        X = X.fillna(0)
        
        # Nustatome skaitinius ir kategorinius stulpelius
        skaitiniai_stulpeliai = X.select_dtypes(include=['number']).columns.tolist()
        kategoriniai_stulpeliai = X.select_dtypes(exclude=['number']).columns.tolist()
        
        # Sukuriame transformavimo pipelineą
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), skaitiniai_stulpeliai),
                ('cat', OneHotEncoder(handle_unknown='ignore'), kategoriniai_stulpeliai)
            ],
            remainder='passthrough'
        )
        
        # Pritaikome transformaciją
        X_transformed = preprocessor.fit_transform(X)
        
        # Skaidome duomenis
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_transformed, y, 
            test_size=(Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS),
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Skaidome likusią dalį į validavimo ir testavimo
        validation_ratio = Config.VALIDAVIMO_DALIS / (Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - validation_ratio,
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Išsaugome duomenis ir transformatorių
        self.training_data['prievoles'] = (X_train, y_train)
        self.validation_data['prievoles'] = (X_val, y_val)
        self.testing_data['prievoles'] = (X_test, y_test)
        self.preprocessoriai['prievoles'] = preprocessor
        
        # Išsaugome preprocessorių į diską
        preprocessor_path = os.path.join(Config.MODELIU_DIREKTORIJA, 'prievoles_preprocessor.joblib')
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Prievolių preprocessorius išsaugotas: {preprocessor_path}")
        
        logger.info(f"Paruošti prievolių duomenys: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    def _paruosti_bylinejimosi_islaidos(self):
        """
        Paruošia duomenis bylinėjimosi išlaidų prognozavimui
        """
        logger.info("Paruošiami duomenys bylinėjimosi išlaidų modeliui")
        
        # Naudojame pagalbinę funkciją duomenų paruošimui (su_vaikais=False yra numatytoji reikšmė)
        seima_df, procesas_df = self._paruosti_duomenis_jungimui('seima_sugeneruota', 'procesas_sugeneruota')
        
        # Jei negalima paruošti duomenų, grįžtame
        if seima_df is None or procesas_df is None:
            return
        
        # Pasirenkame tik reikalingus stulpelius
        seima_df_mini = seima_df[['byla_id', 'santuokos_trukme', 'amzius_vyras', 'amzius_moteris', 
                                'pajamos_vyras', 'pajamos_moteris', 'vaiku_skaicius', 'nutraukimo_budas']]
        
        # Atliekame sujungimą
        sujungti_df = pd.merge(
            procesas_df, 
            seima_df_mini, 
            on='byla_id',
            how='inner'
        )
        
        # Patikriname, ar yra duomenų po sujungimo
        if len(sujungti_df) == 0:
            logger.error("Po duomenų sujungimo nebeliko eilučių analizei.")
            return
        
        # Tikslo kintamasis
        y = sujungti_df['bylinejimosi_islaidos']
        
        # Užpildome NaN reikšmes tikslo kintamajame (jei tokių yra)
        if y.dtype.kind in 'ifc':  # Jei skaitinis
            y = y.fillna(y.median())
        else:  # Jei kategorinis
            y = y.fillna(y.mode()[0] if not y.mode().empty else "nežinoma")
        
        # Požymiai
        X = sujungti_df.drop(['bylinejimosi_islaidos', 'byla_id'], axis=1)
        
        # Užpildome visas likusias NaN reikšmes
        for stulpelis in X.columns:
            if X[stulpelis].dtype.kind in 'ifc':  # Skaitiniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].median())
            else:  # Kategoriniai stulpeliai
                X[stulpelis] = X[stulpelis].fillna(X[stulpelis].mode()[0] if not X[stulpelis].mode().empty else "nežinoma")
        
        # Pašaliname visas likusias NaN reikšmes, jei tokių būtų
        X = X.fillna(0)
        
        # Nustatome skaitinius ir kategorinius stulpelius
        skaitiniai_stulpeliai = X.select_dtypes(include=['number']).columns.tolist()
        kategoriniai_stulpeliai = X.select_dtypes(exclude=['number']).columns.tolist()
        
        # Sukuriame transformavimo pipelineą
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), skaitiniai_stulpeliai),
                ('cat', OneHotEncoder(handle_unknown='ignore'), kategoriniai_stulpeliai)
            ],
            remainder='passthrough'
        )
        
        # Pritaikome transformaciją
        X_transformed = preprocessor.fit_transform(X)
        
        # Skaidome duomenis
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_transformed, y, 
            test_size=(Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS),
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Skaidome likusią dalį į validavimo ir testavimo
        validation_ratio = Config.VALIDAVIMO_DALIS / (Config.VALIDAVIMO_DALIS + Config.TESTAVIMO_DALIS)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - validation_ratio,
            random_state=Config.ATSITIKTINIS_SEED
        )
        
        # Išsaugome duomenis ir transformatorių
        self.training_data['bylinejimosi_islaidos'] = (X_train, y_train)
        self.validation_data['bylinejimosi_islaidos'] = (X_val, y_val)
        self.testing_data['bylinejimosi_islaidos'] = (X_test, y_test)
        self.preprocessoriai['bylinejimosi_islaidos'] = preprocessor
        
        # Išsaugome preprocessorių į diską
        preprocessor_path = os.path.join(Config.MODELIU_DIREKTORIJA, 'bylinejimosi_islaidos_preprocessor.joblib')
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Bylinėjimosi išlaidų preprocessorius išsaugotas: {preprocessor_path}")
        
        logger.info(f"Paruošti bylinėjimosi išlaidų duomenys: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    def gauti_mokymo_duomenis(self, uzdavinys):
        """
        Grąžina paruoštus treniravimo duomenis nurodytam uždaviniui
        
        Args:
            uzdavinys (str): Uždavinio pavadinimas (pvz., 'gyvenamoji_vieta')
        
        Returns:
            tuple: (X_train, y_train, X_val, y_val) - treniravimo ir validavimo duomenys
        """
        if uzdavinys not in self.training_data or uzdavinys not in self.validation_data:
            logger.error(f"Nėra paruoštų duomenų uždaviniui '{uzdavinys}'. Pirmiau paruoškite duomenis.")
            return None, None, None, None
        
        X_train, y_train = self.training_data[uzdavinys]
        X_val, y_val = self.validation_data[uzdavinys]
        
        logger.info(f"Gaunami mokymo duomenys uždaviniui '{uzdavinys}': train={X_train.shape}, val={X_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    def gauti_testavimo_duomenis(self, uzdavinys):
        """
        Grąžina paruoštus testavimo duomenis nurodytam uždaviniui
        
        Args:
            uzdavinys (str): Uždavinio pavadinimas (pvz., 'gyvenamoji_vieta')
        
        Returns:
            tuple: (X_test, y_test) - testavimo duomenys
        """
        if uzdavinys not in self.testing_data:
            logger.error(f"Nėra paruoštų testavimo duomenų uždaviniui '{uzdavinys}'. Pirmiau paruoškite duomenis.")
            return None, None
        
        X_test, y_test = self.testing_data[uzdavinys]
        
        logger.info(f"Gaunami testavimo duomenys uždaviniui '{uzdavinys}': test={X_test.shape}")
        
        return X_test, y_test
    
    def gauti_preprocessoriu(self, uzdavinys):
        """
        Grąžina duomenų transformavimo objektą nurodytam uždaviniui
        
        Args:
            uzdavinys (str): Uždavinio pavadinimas (pvz., 'gyvenamoji_vieta')
        
        Returns:
            object: Duomenų transformavimo objektas
        """
        if uzdavinys not in self.preprocessoriai:
            logger.error(f"Nėra preprocessoriaus uždaviniui '{uzdavinys}'. Pirmiau paruoškite duomenis.")
            return None
        
        return self.preprocessoriai[uzdavinys]