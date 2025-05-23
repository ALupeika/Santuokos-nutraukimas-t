"""
Konfigūracijos nustatymai

Šiame faile saugomi programos konfigūracijos parametrai
"""

import os
import secrets

class Config:
    """
    Pagrindiniai konfigūracijos nustatymai
    """
    # Flask konfigūracija
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(16)
    
    # Duomenų bazės konfigūracija
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///santuoku_bylos.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Kiti nustatymai
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB maksimalus failo dydis
    
    # Modelių direktorijos
    MODELIU_DIREKTORIJA = 'modeliai'
    DUOMENU_DIREKTORIJA = 'duomenys'
    
    # Mašininio mokymosi parametrai
    ATSITIKTINIS_SEED = 42
    VALIDAVIMO_DALIS = 0.15
    TESTAVIMO_DALIS = 0.15
    
    # Neuroninio tinklo parametrai
    BATCH_DYDIS = 32
    EPOCHU_SKAICIUS = 100
    KANTRYBE = 10  # Early stopping
    
    # Optuna parametrai
    OPTIMIZAVIMO_BANDYMU_SKAICIUS = 50