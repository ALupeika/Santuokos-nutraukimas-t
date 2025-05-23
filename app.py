"""
SANTUOKŲ BYLŲ SPRENDIMŲ PROGNOZAVIMO SISTEMA

Pagrindinis programos failas, kuris paleidžia Flask serverį 
ir sujungia visus sistemos komponentus.


"""

import os
import logging
from flask import Flask, render_template, redirect, url_for, flash, request
from flask_login import LoginManager, login_required, current_user
from flask_migrate import Migrate

# Importuojame konfigūracijos nustatymus
from config import Config

# Importuojame duomenų bazės modelį
from web.db_modelis import db, User

# Importuojame visus reikiamus maršrutus
from web.routes import registruoti_visus_marsrutus
from web.vartotojai import vartotoju_bp
from datetime import datetime

# Sukuriame logger objektą
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("programa.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def patikrinti_ir_treniruoti_modelius():
    """
    Patikrina, ar modeliai egzistuoja, ir jei ne, juos treniruoja
    """
    import os
    import logging
    from config import Config
    
    logger = logging.getLogger(__name__)
    
    # Tikriname, ar egzistuoja bent vienas modelis
    modeliu_failai = os.listdir(Config.MODELIU_DIREKTORIJA) if os.path.exists(Config.MODELIU_DIREKTORIJA) else []
    modeliu_failai = [f for f in modeliu_failai if f.endswith('.joblib')]
    
    if len(modeliu_failai) == 0:
        # Jei modelių nėra, paleiskime treniravimo procesą
        logger.warning("Modeliai nerasti. Pradedamas automatinis treniravimas...")
        
        # Importuojame ir paleidžiame treniravimo funkciją
        try:
            from treniruoti import main as treniruoti_modelius
            treniruoti_modelius()
            logger.info("Modelių treniravimas baigtas!")
        except Exception as e:
            logger.error(f"Klaida treniruojant modelius: {e}")
    else:
        logger.info(f"Rasta {len(modeliu_failai)} modelių failų. Treniravimas nereikalingas.")

def sukurti_app(testavimo_rezimas=False):
    """
    Sukuria ir sukonfigūruoja Flask aplikaciją
    
    Args:
        testavimo_rezimas (bool): Ar aplikacija veiks testavimo režimu
        
    Returns:
        Flask: Sukonfigūruota Flask aplikacija
    """
    # Inicializuojame Flask aplikaciją
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Įkeliame konfigūraciją
    app.config.from_object(Config)
    
    # Jei testavimo režimas, naudojame in-memory SQLite duomenų bazę
    if testavimo_rezimas:
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['TESTING'] = True
    
    # Inicializuojame duomenų bazę
    db.init_app(app)
    
    # Inicializuojame login manager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'vartotojai.prisijungimas'
    
     
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Inicializuojame migracijas
    migrate = Migrate(app, db)
    
    # Registruojame endpoint'us
    registruoti_visus_marsrutus(app)
    
    # Registruojame blueprints
    app.register_blueprint(vartotoju_bp)
    
    # Sukuriame pagrindinį maršrutą
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Klaidos apdorojimas
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500
    
    @app.context_processor
    def bendras_kontekstas():
        return {'now': datetime.now()}
    
    return app

# Jei paleidžiame šį failą tiesiogiai
if __name__ == '__main__':
    # Sukuriame aplikaciją
    app = sukurti_app()
    
    # Sukuriame direktorijas, jei jų nėra
    for direktorija in ['duomenys', 'modeliai']:
        os.makedirs(direktorija, exist_ok=True)

    patikrinti_ir_treniruoti_modelius()

    # Sukuriame duomenų bazės lenteles, jei jų nėra
    with app.app_context():
        db.create_all()
    
    # Paleidžiame serverį
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)