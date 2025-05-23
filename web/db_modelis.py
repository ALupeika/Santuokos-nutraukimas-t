"""
Duomenų bazės modeliai

Šis modulis apima duomenų bazės modelius, kurie naudojami Flask aplikacijoje.
"""
import uuid
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Inicializuojame SQLAlchemy objektą
db = SQLAlchemy()

class User(db.Model, UserMixin):
    """
    Vartotojo modelis
    """
    __tablename__ = 'vartotojai'
    
    id = db.Column(db.Integer, primary_key=True)
    vardas = db.Column(db.String(100), nullable=False)
    pavarde = db.Column(db.String(100), nullable=False)
    el_pastas = db.Column(db.String(120), unique=True, nullable=False)
    slaptazodis_hash = db.Column(db.String(200), nullable=False)
    roles = db.Column(db.String(30), default='vartotojas')  # administratorius, vartotojas
    sukurimo_data = db.Column(db.DateTime, default=datetime.utcnow)
    paskutinis_prisijungimas = db.Column(db.DateTime)
    
    # Ryšys su bylomis
    bylos = db.relationship('Byla', backref='vartotojas', lazy=True)
    
    def set_slaptazodis(self, slaptazodis):
        """
        Nustato vartotojo slaptažodį (saugo tik hash)
        
        Args:
            slaptazodis (str): Slaptažodis
        """
        self.slaptazodis_hash = generate_password_hash(slaptazodis)
    
    def check_slaptazodis(self, slaptazodis):
        """
        Patikrina, ar slaptažodis teisingas
        
        Args:
            slaptazodis (str): Tikrinamas slaptažodis
            
        Returns:
            bool: True, jei slaptažodis teisingas
        """
        return check_password_hash(self.slaptazodis_hash, slaptazodis)
    
    def yra_administratorius(self):
        """
        Patikrina, ar vartotojas turi administratoriaus teises
        
        Returns:
            bool: True, jei vartotojas yra administratorius
        """
        return self.roles == 'administratorius'
    
    def __repr__(self):
        return f'<Vartotojas {self.vardas} {self.pavarde}>'

class Byla(db.Model):
    """
    Bylos modelis
    """
    __tablename__ = 'bylos'

    id = db.Column(db.Integer, primary_key=True)

    # Automatinis unikalus bylos identifikatorius
    byla_id = db.Column(db.String(50), unique=True, nullable=False, default=lambda: f"BYLA-{uuid.uuid4().hex[:8].upper()}")

    # Šeimos duomenys
    santuokos_trukme = db.Column(db.Integer)
    amzius_vyras = db.Column(db.Integer)
    amzius_moteris = db.Column(db.Integer)
    negyvena_kartu_menesiai = db.Column(db.Integer)
    pajamos_vyras = db.Column(db.Float)
    pajamos_moteris = db.Column(db.Float)
    vaiku_skaicius = db.Column(db.Integer)
    nutraukimo_budas = db.Column(db.String(100))

    # Turto duomenys
    bendro_turto_verte = db.Column(db.Float)
    asmeninio_turto_verte_vyras = db.Column(db.Float)
    asmeninio_turto_verte_moteris = db.Column(db.Float)
    padalijimas_proc_vyras = db.Column(db.Float)
    padalijimas_proc_moteris = db.Column(db.Float)
    turto_vyras_eur = db.Column(db.Float)
    turto_moteris_eur = db.Column(db.Float)
    turto_vyras_po_asmeniniu_lesu = db.Column(db.Float)
    turto_moteris_po_asmeniniu_lesu = db.Column(db.Float)

    # Prievolių duomenys
    bendros_prievoles = db.Column(db.Float)
    asmenines_prievoles_vyras = db.Column(db.Float)
    asmenines_prievoles_moteris = db.Column(db.Float)
    prievoles_vyras = db.Column(db.Float)
    prievoles_moteris = db.Column(db.Float)

    # Proceso duomenys
    soc_problemos = db.Column(db.String(100))
    bylinejimosi_islaidos = db.Column(db.Float)

    # Vartotojo ID, kuris sukūrė bylą
    vartotojo_id = db.Column(db.Integer, db.ForeignKey('vartotojai.id'), nullable=False)

    # Sukūrimo datos
    sukurimo_data = db.Column(db.DateTime, default=datetime.utcnow)
    atnaujinimo_data = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Ryšys su vaikais ir prognozėmis
    vaikai = db.relationship('Vaikas', backref='byla', lazy=True)
    prognozes = db.relationship('Prognoze', backref='byla', lazy=True)

    def __repr__(self):
        return f'<Byla {self.byla_id}>'
    

class Vaikas(db.Model):
    """
    Vaiko modelis
    """
    __tablename__ = 'vaikai'
    
    id = db.Column(db.Integer, primary_key=True)
    vaiko_nr = db.Column(db.Integer, nullable=False)  # vaiko numeris byloje
    
    # Vaiko duomenys
    amzius = db.Column(db.Integer)
    emocinis_rysys_mama = db.Column(db.String(50))
    emocinis_rysys_tevas = db.Column(db.String(50))
    poreikiai = db.Column(db.Float)
    gyvenamoji_vieta = db.Column(db.String(50))
    bendravimo_tvarka = db.Column(db.Text)
    islaikymas = db.Column(db.Float)
    islaikymo_iskolinimas = db.Column(db.Float)
    
    # Bylos ID, kuriai priklauso vaikas
    bylos_id = db.Column(db.Integer, db.ForeignKey('bylos.id'), nullable=False)
    
    def __repr__(self):
        return f'<Vaikas {self.vaiko_nr}, Byla ID: {self.bylos_id}>'

class Prognoze(db.Model):
    """
    Prognozės modelis
    """
    __tablename__ = 'prognozes'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Bendroji dalis
    prognozes_tipas = db.Column(db.String(100), nullable=False)  # pvz., turto_padalijimas, gyvenamoji_vieta
    prognozes_reiksme = db.Column(db.String(255))  # prognozės rezultatas
    tikimybe = db.Column(db.Float)  # prognozės tikimybė arba patikimumas (0-1)
    modelio_pavadinimas = db.Column(db.String(100))  # modelio, kuris atliko prognozę, pavadinimas
    
    # Prognozės data
    sukurimo_data = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Bylos ID, kuriai priklauso prognozė
    bylos_id = db.Column(db.Integer, db.ForeignKey('bylos.id'), nullable=False)
    
    # Vaiko ID, jei prognozė susijusi su vaiku
    vaiko_id = db.Column(db.Integer, db.ForeignKey('vaikai.id'))
    
    def __repr__(self):
        return f'<Prognozė {self.prognozes_tipas}, Byla ID: {self.bylos_id}>'