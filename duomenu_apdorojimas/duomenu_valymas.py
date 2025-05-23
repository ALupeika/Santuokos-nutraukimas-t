"""
Duomenų valymo modulis

Šiame modulyje yra funkcijos, skirtos duomenų valymui ir paruošimui
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def uzpildyti_trukstamas_reiksmes(df, stulpelis, metodas='median'):
    """
    Užpildo trūkstamas reikšmes nurodytame stulpelyje
    
    Args:
        df (pd.DataFrame): DataFrame objektas
        stulpelis (str): Stulpelio pavadinimas
        metodas (str): Užpildymo metodas: 'median', 'mean', 'mode', 'zero'
    
    Returns:
        pd.DataFrame: DataFrame su užpildytomis reikšmėmis
    """
    # Sukuriame kopiją, kad nekeistume originalaus DataFrame
    df_kopija = df.copy()
    
    # Patikriname, ar stulpelis egzistuoja
    if stulpelis not in df_kopija.columns:
        logger.warning(f"Stulpelis {stulpelis} nerastas DataFrame")
        return df_kopija
    
    # Skaičiuojame trūkstamų reikšmių skaičių
    trukstamos = df_kopija[stulpelis].isnull().sum()
    
    # Jei nėra trūkstamų reikšmių, nėra ką daryti
    if trukstamos == 0:
        return df_kopija
    
    # Užpildome trūkstamas reikšmes pagal nurodytą metodą
    if metodas == 'median':
        reiksme = df_kopija[stulpelis].median()
        df_kopija[stulpelis].fillna(reiksme, inplace=True)
    elif metodas == 'mean':
        reiksme = df_kopija[stulpelis].mean()
        df_kopija[stulpelis].fillna(reiksme, inplace=True)
    elif metodas == 'mode':
        reiksme = df_kopija[stulpelis].mode()[0]  # Imame pirmą modinę reikšmę
        df_kopija[stulpelis].fillna(reiksme, inplace=True)
    elif metodas == 'zero':
        df_kopija[stulpelis].fillna(0, inplace=True)
    else:
        logger.warning(f"Nežinomas užpildymo metodas: {metodas}. Naudojame 'median'")
        reiksme = df_kopija[stulpelis].median()
        df_kopija[stulpelis].fillna(reiksme, inplace=True)
    
    logger.debug(f"Stulpelyje {stulpelis} užpildyta {trukstamos} trūkstamų reikšmių naudojant metodą '{metodas}'")
    
    return df_kopija

def konvertuoti_i_skaicius(df, stulpelis, pašalinti_neskaiciuojamus=True):
    """
    Konvertuoja stulpelio reikšmes į skaičius
    
    Args:
        df (pd.DataFrame): DataFrame objektas
        stulpelis (str): Stulpelio pavadinimas
        pašalinti_neskaiciuojamus (bool): Ar pašalinti reikšmes, kurios negali būti konvertuotos į skaičius
    
    Returns:
        pd.DataFrame: DataFrame su konvertuotomis reikšmėmis
    """
    # Sukuriame kopiją, kad nekeistume originalaus DataFrame
    df_kopija = df.copy()
    
    # Patikriname, ar stulpelis egzistuoja
    if stulpelis not in df_kopija.columns:
        logger.warning(f"Stulpelis {stulpelis} nerastas DataFrame")
        return df_kopija
    
    # Bandome konvertuoti į skaičius
    try:
        df_kopija[stulpelis] = pd.to_numeric(df_kopija[stulpelis], errors='coerce')
        
        # Jei reikia pašalinti neskaičiuojamas reikšmes (NaN), pakeičiame jas trūkstamomis
        if pašalinti_neskaiciuojamus:
            trukstamos_po = df_kopija[stulpelis].isnull().sum()
            logger.debug(f"Stulpelyje {stulpelis} {trukstamos_po} reikšmių negalėjo būti konvertuotos į skaičius")
    except Exception as e:
        logger.error(f"Klaida konvertuojant stulpelį {stulpelis} į skaičius: {e}")
    
    return df_kopija

def pašalinti_isskirties(df, stulpelis, metodas='iqr', k=1.5):
    """
    Pašalina išskirties (outliers) iš nurodyto stulpelio
    
    Args:
        df (pd.DataFrame): DataFrame objektas
        stulpelis (str): Stulpelio pavadinimas
        metodas (str): Metodas išskirtims nustatyti: 'iqr' (Tarpkvartilinis plotis) arba 'zscore'
        k (float): Koeficientas, naudojamas su IQR metodu (paprastai 1.5)
    
    Returns:
        pd.DataFrame: DataFrame be išskirčių
    """
    # Sukuriame kopiją, kad nekeistume originalaus DataFrame
    df_kopija = df.copy()
    
    # Patikriname, ar stulpelis egzistuoja
    if stulpelis not in df_kopija.columns:
        logger.warning(f"Stulpelis {stulpelis} nerastas DataFrame")
        return df_kopija
    
    # Patikriname, ar stulpelis yra skaitinis
    if df_kopija[stulpelis].dtype.kind not in 'ifc':
        logger.warning(f"Stulpelis {stulpelis} nėra skaitinis, išskirčių šalinimas negalimas")
        return df_kopija
    
    # Pašaliname išskirtis pagal nurodytą metodą
    if metodas == 'iqr':
        Q1 = df_kopija[stulpelis].quantile(0.25)
        Q3 = df_kopija[stulpelis].quantile(0.75)
        IQR = Q3 - Q1
        
        apatine_riba = Q1 - k * IQR
        viršutinė_riba = Q3 + k * IQR
        
        # Pažymime išskirtis kaip NaN
        df_kopija.loc[(df_kopija[stulpelis] < apatine_riba) | (df_kopija[stulpelis] > viršutinė_riba), stulpelis] = np.nan
        
        # Skaičiuojame, kiek išskirčių pašalinta
        trukstamos = df_kopija[stulpelis].isnull().sum()
        logger.debug(f"Stulpelyje {stulpelis} pašalinta {trukstamos} išskirčių naudojant IQR metodą")
    
    elif metodas == 'zscore':
        from scipy import stats
        
        # Apskaičiuojame z-scores
        z_scores = np.abs(stats.zscore(df_kopija[stulpelis].dropna()))
        
        # Pažymime išskirtis (z-score > 3) kaip NaN
        df_kopija.loc[np.abs(stats.zscore(df_kopija[stulpelis].dropna())) > 3, stulpelis] = np.nan
        
        # Skaičiuojame, kiek išskirčių pašalinta
        trukstamos = df_kopija[stulpelis].isnull().sum()
        logger.debug(f"Stulpelyje {stulpelis} pašalinta {trukstamos} išskirčių naudojant Z-score metodą")
    
    else:
        logger.warning(f"Nežinomas išskirčių šalinimo metodas: {metodas}")
    
    return df_kopija

def pašalinti_duplikatus(df, stulpeliai=None):
    """
    Pašalina duplikatus iš DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame objektas
        stulpeliai (list): Stulpelių sąrašas, pagal kuriuos ieškoti duplikatų.
                          Jei nenurodyta, naudojami visi stulpeliai.
    
    Returns:
        pd.DataFrame: DataFrame be duplikatų
    """
    # Sukuriame kopiją, kad nekeistume originalaus DataFrame
    df_kopija = df.copy()
    
    # Skaičiuojame pradinį eilučių skaičių
    pradinis_kiekis = df_kopija.shape[0]
    
    # Pašaliname duplikatus
    if stulpeliai is not None:
        # Patikriname, ar visi nurodyti stulpeliai egzistuoja
        neegzistuoja = [stulpelis for stulpelis in stulpeliai if stulpelis not in df_kopija.columns]
        if neegzistuoja:
            logger.warning(f"Stulpeliai {neegzistuoja} nerasti DataFrame")
            # Filtruojame tik egzistuojančius stulpelius
            stulpeliai = [stulpelis for stulpelis in stulpeliai if stulpelis in df_kopija.columns]
        
        if not stulpeliai:
            logger.warning("Nėra tinkamų stulpelių duplikatų pašalinimui")
            return df_kopija
        
        # Pašaliname duplikatus pagal nurodytus stulpelius
        df_kopija.drop_duplicates(subset=stulpeliai, keep='first', inplace=True)
    else:
        # Pašaliname duplikatus pagal visus stulpelius
        df_kopija.drop_duplicates(keep='first', inplace=True)
    
    # Skaičiuojame, kiek duplikatų pašalinta
    pašalinta = pradinis_kiekis - df_kopija.shape[0]
    logger.debug(f"Pašalinta {pašalinta} duplikatų")
    
    return df_kopija

def diskretizuoti_stulpeli(df, stulpelis, bins=5, metodas='equal_width', labels=None):
    """
    Diskretizuoja skaitinį stulpelį į kategorijas
    
    Args:
        df (pd.DataFrame): DataFrame objektas
        stulpelis (str): Stulpelio pavadinimas
        bins (int): Kategorijų skaičius
        metodas (str): Diskretizavimo metodas: 'equal_width' arba 'equal_freq'
        labels (list): Kategorijų pavadinimai
    
    Returns:
        pd.DataFrame: DataFrame su diskretizuotu stulpeliu
    """
    # Sukuriame kopiją, kad nekeistume originalaus DataFrame
    df_kopija = df.copy()
    
    # Patikriname, ar stulpelis egzistuoja
    if stulpelis not in df_kopija.columns:
        logger.warning(f"Stulpelis {stulpelis} nerastas DataFrame")
        return df_kopija
    
    # Patikriname, ar stulpelis yra skaitinis
    if df_kopija[stulpelis].dtype.kind not in 'ifc':
        logger.warning(f"Stulpelis {stulpelis} nėra skaitinis, diskretizavimas negalimas")
        return df_kopija
    
    # Sukuriame naują stulpelį su diskretizuotomis reikšmėmis
    naujas_stulpelis = f"{stulpelis}_kategorizuotas"
    
    if metodas == 'equal_width':
        # Diskretizuojame pagal vienodą plotį
        df_kopija[naujas_stulpelis] = pd.cut(df_kopija[stulpelis], bins=bins, labels=labels)
    elif metodas == 'equal_freq':
        # Diskretizuojame pagal vienodą dažnį
        df_kopija[naujas_stulpelis] = pd.qcut(df_kopija[stulpelis], q=bins, labels=labels)
    else:
        logger.warning(f"Nežinomas diskretizavimo metodas: {metodas}")
        return df_kopija
    
    logger.debug(f"Stulpelis {stulpelis} sėkmingai diskretizuotas į {bins} kategorijas")
    
    return df_kopija

def apjungti_kategorijas(df, stulpelis, kategoriju_map):
    """
    Apjungia (grupuoja) kategorines reikšmes
    
    Args:
        df (pd.DataFrame): DataFrame objektas
        stulpelis (str): Stulpelio pavadinimas
        kategoriju_map (dict): Žodynas, nurodantis, kaip grupuoti kategorijas
                              {senoji_kategorija: nauja_kategorija}
    
    Returns:
        pd.DataFrame: DataFrame su apjungtomis kategorijomis
    """
    # Sukuriame kopiją, kad nekeistume originalaus DataFrame
    df_kopija = df.copy()
    
    # Patikriname, ar stulpelis egzistuoja
    if stulpelis not in df_kopija.columns:
        logger.warning(f"Stulpelis {stulpelis} nerastas DataFrame")
        return df_kopija
    
    # Apjungiame kategorijas pagal nurodytą žodyną
    df_kopija[stulpelis] = df_kopija[stulpelis].map(kategoriju_map).fillna(df_kopija[stulpelis])
    
    logger.debug(f"Stulpelio {stulpelis} kategorijos sėkmingai apjungtos")
    
    return df_kopija

def koduoti_kategorinius(df, stulpelis, metodas='onehot', nauji_stulpeliai=True):
    """
    Koduoja kategorinius kintamuosius
    
    Args:
        df (pd.DataFrame): DataFrame objektas
        stulpelis (str): Stulpelio pavadinimas
        metodas (str): Kodavimo metodas: 'onehot', 'label', arba 'ordinal'
        nauji_stulpeliai (bool): Ar kurti naujus stulpelius (tik 'onehot' metodui)
    
    Returns:
        pd.DataFrame: DataFrame su užkoduotais kategoriniais kintamaisiais
    """
    # Sukuriame kopiją, kad nekeistume originalaus DataFrame
    df_kopija = df.copy()
    
    # Patikriname, ar stulpelis egzistuoja
    if stulpelis not in df_kopija.columns:
        logger.warning(f"Stulpelis {stulpelis} nerastas DataFrame")
        return df_kopija
    
    if metodas == 'onehot':
        # Naudojame one-hot kodavimą
        dummies = pd.get_dummies(df_kopija[stulpelis], prefix=stulpelis)
        
        if nauji_stulpeliai:
            # Pridedame naujus stulpelius į DataFrame
            df_kopija = pd.concat([df_kopija, dummies], axis=1)
        else:
            # Pašaliname originalų stulpelį ir pridedame naujus
            df_kopija = df_kopija.drop(columns=[stulpelis])
            df_kopija = pd.concat([df_kopija, dummies], axis=1)
        
        logger.debug(f"Stulpelis {stulpelis} užkoduotas naudojant one-hot kodavimą")
    
    elif metodas == 'label':
        # Naudojame label kodavimą
        label_encoder = LabelEncoder()
        df_kopija[f"{stulpelis}_encoded"] = label_encoder.fit_transform(df_kopija[stulpelis])
        logger.debug(f"Stulpelis {stulpelis} užkoduotas naudojant label kodavimą")
    
    elif metodas == 'ordinal':
        # Naudojame ordinalinį kodavimą (reikia nurodyti tvarką)
        # Čia tiesiog sukuriame žodyną, kuriame kiekviena unikali reikšmė gauna unikalų skaičių
        unikalios_reiksmes = df_kopija[stulpelis].unique()
        ordinal_map = {val: i for i, val in enumerate(unikalios_reiksmes)}
        
        df_kopija[f"{stulpelis}_ordinal"] = df_kopija[stulpelis].map(ordinal_map)
        logger.debug(f"Stulpelis {stulpelis} užkoduotas naudojant ordinalinį kodavimą")
    
    else:
        logger.warning(f"Nežinomas kodavimo metodas: {metodas}")
    
    return df_kopija