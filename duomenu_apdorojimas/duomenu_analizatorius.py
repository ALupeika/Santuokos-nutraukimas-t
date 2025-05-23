
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)

class DuomenuAnalizatorius:
   
    def __init__(self):
     
        self.duomenys = {}  
        self.sujungti_duomenys = None 
        
    def nuskaityti_duomenis(self, failu_sarasas=None, direktorija=None):
       
        if failu_sarasas is None:
            failu_sarasas = [
                'seima_sugeneruota.csv',
                'turtas_sugeneruota.csv',
                'vaikai_sugeneruota.csv',
                'prievoles_sugeneruota.csv',
                'procesas_sugeneruota.csv'
            ]
        
        if direktorija is None:
            direktorija = Config.DUOMENU_DIREKTORIJA
        
        for failas in failu_sarasas:
           
            failo_pavadinimas = failas.split('.')[0]
            try:
                
                kelias = os.path.join(direktorija, failas)
                df = pd.read_csv(kelias)
                logger.info(f"Sėkmingai nuskaitytas failas {failas}: {df.shape[0]} eilutės, {df.shape[1]} stulpeliai")
                
                
                self.duomenys[failo_pavadinimas] = df
            except Exception as e:
                logger.error(f"Klaida nuskaitant failą {failas}: {e}")
        
        return self.duomenys
    
    def analizuoti_duomenis(self):
        
        rezultatai = {}
        
        for pavadinimas, df in self.duomenys.items():
           
            failo_analize = {
                'eilučių_skaičius': df.shape[0],
                'stulpelių_skaičius': df.shape[1],
                'stulpelių_pavadinimai': df.columns.tolist(),
                'duomenų_tipai': df.dtypes.astype(str).to_dict(),
                'trūkstamos_reikšmės': df.isnull().sum().to_dict(),
                'aprašomoji_statistika': df.describe().to_dict() if df.select_dtypes(include=['number']).shape[1] > 0 else None,
                'kategorinių_stulpelių_unikalios_reikšmės': {
                    stulpelis: df[stulpelis].value_counts().to_dict() 
                    for stulpelis in df.select_dtypes(include=['object']).columns
                }
            }
            
            rezultatai[pavadinimas] = failo_analize
            
            
            logger.info(f"\n=== {pavadinimas} analizė ===")
            logger.info(f"Eilučių skaičius: {df.shape[0]}")
            logger.info(f"Stulpelių skaičius: {df.shape[1]}")
            logger.info(f"Stulpelių pavadinimai: {df.columns.tolist()}")
            logger.info(f"Trūkstamos reikšmės: {df.isnull().sum().to_dict()}")
        
        return rezultatai
    
    def analizuoti_duomenu_pasiskirstyma(self, failas, stulpeliai=None, issaugoti=False, issaugojimo_direktorija='grafikai'):
       
        if failas not in self.duomenys:
            logger.error(f"Failas {failas} nerastas duomenyse.")
            return None
        
        df = self.duomenys[failas]
        
        if stulpeliai is None:
            stulpeliai = df.columns.tolist()
        
        rezultatai = {}
        
        # Sukuriame išsaugojimo direktoriją, jei reikia
        if issaugoti:
            os.makedirs(issaugojimo_direktorija, exist_ok=True)
        
        # Analizuojame kiekvieną stulpelį
        for stulpelis in stulpeliai:
            if stulpelis not in df.columns:
                logger.warning(f"Stulpelis {stulpelis} nerastas faile {failas}.")
                continue
            
            # Nustatome duomenų tipą
            if df[stulpelis].dtype.kind in 'ifc':  # skaičiai (integer, float, complex)
                # Sukuriame histogramą
                plt.figure(figsize=(10, 6))
                sns.histplot(df[stulpelis].dropna(), kde=True)
                plt.title(f"{failas} - {stulpelis} pasiskirstymas")
                plt.tight_layout()
                
                if issaugoti:
                    plt.savefig(os.path.join(issaugojimo_direktorija, f"{failas}_{stulpelis}_histograma.png"))
                    plt.close()
                else:
                    plt.show()
                
                # Skaičiuojame aprašomąją statistiką
                statistika = df[stulpelis].describe().to_dict()
                rezultatai[stulpelis] = {
                    'tipas': 'skaitinis',
                    'statistika': statistika,
                    'pasiskirstymas': 'histograma',
                    'grafiko_kelias': os.path.join(issaugojimo_direktorija, f"{failas}_{stulpelis}_histograma.png") if issaugoti else None
                }
            else:  # kategoriniai duomenys
                # Skaičiuojame dažnius
                reiksmes = df[stulpelis].value_counts()
                
                # Sukuriame stulpelinę diagramą (jei nedaug unikalių reikšmių)
                if len(reiksmes) <= 20:  # Ribojame, kad grafikas būtų įskaitomas
                    plt.figure(figsize=(12, 6))
                    sns.barplot(x=reiksmes.index, y=reiksmes.values)
                    plt.title(f"{failas} - {stulpelis} pasiskirstymas")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    if issaugoti:
                        plt.savefig(os.path.join(issaugojimo_direktorija, f"{failas}_{stulpelis}_barplot.png"))
                        plt.close()
                    else:
                        plt.show()
                    
                    grafiko_tipas = 'stulpelinė'
                    grafiko_kelias = os.path.join(issaugojimo_direktorija, f"{failas}_{stulpelis}_barplot.png") if issaugoti else None
                else:
                    # Jei per daug unikalių reikšmių, grafiko nekuriame
                    grafiko_tipas = 'nėra (per daug unikalių reikšmių)'
                    grafiko_kelias = None
                
                # Skaičiuojame aprašomąją statistiką
                statistika = {
                    'unikalių_reikšmių': len(reiksmes),
                    'dažniausios_reikšmės': reiksmes.head(5).to_dict() if len(reiksmes) > 0 else {}
                }
                
                rezultatai[stulpelis] = {
                    'tipas': 'kategorinis',
                    'statistika': statistika,
                    'pasiskirstymas': grafiko_tipas,
                    'grafiko_kelias': grafiko_kelias
                }
        
        return rezultatai
    
    def tirti_koreliacijas(self, failas, stulpeliai=None, issaugoti=False, issaugojimo_direktorija='grafikai'):
       
        if failas not in self.duomenys:
            logger.error(f"Failas {failas} nerastas duomenyse.")
            return None
        
        df = self.duomenys[failas]
        
        # Išrenkame tik skaitinius stulpelius
        skaitiniai_stulpeliai = df.select_dtypes(include=['number']).columns.tolist()
        
        if stulpeliai is not None:
            # Išrenkame tik tuos stulpelius, kurie yra skaitiniai ir yra pateiktame sąraše
            skaitiniai_stulpeliai = [stulpelis for stulpelis in stulpeliai if stulpelis in skaitiniai_stulpeliai]
        
        if not skaitiniai_stulpeliai:
            logger.warning(f"Faile {failas} nerasta skaitinių stulpelių analizei.")
            return None
        
        # Skaičiuojame koreliaciją
        koreliacijos = df[skaitiniai_stulpeliai].corr()
        
        # Sukuriame šilumos žemėlapį (heatmap)
        plt.figure(figsize=(12, 10))
        sns.heatmap(koreliacijos, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title(f"{failas} - Koreliacijos tarp skaitinių stulpelių")
        plt.tight_layout()
        
        if issaugoti:
            os.makedirs(issaugojimo_direktorija, exist_ok=True)
            plt.savefig(os.path.join(issaugojimo_direktorija, f"{failas}_koreliacijos.png"))
            plt.close()
        else:
            plt.show()
        
        return koreliacijos
    
    def tirti_santykius(self, stulpelis1, stulpelis2, failas=None, issaugoti=False, issaugojimo_direktorija='grafikai'):
      
      
        # Nustatome, kuriame faile ieškoti stulpelių
        df = None
        
        if failas is not None and failas in self.duomenys:
            df = self.duomenys[failas]
            if stulpelis1 not in df.columns or stulpelis2 not in df.columns:
                logger.error(f"Stulpeliai {stulpelis1} ir/arba {stulpelis2} nerasti faile {failas}.")
                return None
        else:
            # Ieškome abiejų stulpelių visuose failuose
            for pav, d in self.duomenys.items():
                if stulpelis1 in d.columns and stulpelis2 in d.columns:
                    df = d
                    failas = pav
                    break
            
            if df is None:
                logger.error(f"Stulpeliai {stulpelis1} ir {stulpelis2} nerasti jokiame faile.")
                return None
        
        # Patikriname, ar abu stulpeliai yra skaitiniai
        if df[stulpelis1].dtype.kind not in 'ifc' or df[stulpelis2].dtype.kind not in 'ifc':
            logger.warning(f"Bent vienas iš stulpelių {stulpelis1} arba {stulpelis2} nėra skaitinis.")
            # Galima tęsti ir su kategoriniu stulpeliu, bet tada reikėtų kitokio grafiko
        
        # Sukuriame scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[stulpelis1], y=df[stulpelis2])
        plt.title(f"{failas} - {stulpelis1} vs {stulpelis2}")
        plt.xlabel(stulpelis1)
        plt.ylabel(stulpelis2)
        plt.tight_layout()
        
        if issaugoti:
            os.makedirs(issaugojimo_direktorija, exist_ok=True)
            plt.savefig(os.path.join(issaugojimo_direktorija, f"{failas}_{stulpelis1}_vs_{stulpelis2}.png"))
            plt.close()
        else:
            plt.show()
        
        # Skaičiuojame koreliacijos koeficientą
        from scipy.stats import pearsonr
        try:
            koef, p_reiksme = pearsonr(df[stulpelis1].dropna(), df[stulpelis2].dropna())
            logger.info(f"Koreliacijos koeficientas tarp {stulpelis1} ir {stulpelis2}: {koef:.4f} (p={p_reiksme:.4f})")
            return koef, p_reiksme
        except Exception as e:
            logger.error(f"Klaida skaičiuojant koreliaciją: {e}")
            return None

if __name__ == "__main__":
    # Pavyzdys, kaip naudoti klasę
    analizatorius = DuomenuAnalizatorius()
    duomenys = analizatorius.nuskaityti_duomenis()
    
    if duomenys:
        analizatorius.analizuoti_duomenis()
        
        # Analizuojame pasiskirstymą
        analizatorius.analizuoti_duomenu_pasiskirstyma('seima_sugeneruota', ['santuokos_trukme', 'amzius_vyras', 'amzius_moteris'])
        
        # Tiriame koreliacijas
        analizatorius.tirti_koreliacijas('seima_sugeneruota')
        
        # Tiriame santykius tarp dviejų stulpelių
        analizatorius.tirti_santykius('amzius_vyras', 'amzius_moteris', 'seima_sugeneruota')