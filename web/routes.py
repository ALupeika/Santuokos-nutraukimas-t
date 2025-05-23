"""
Flask maršrutai

Šis modulis apima visus Flask aplikacijos maršrutus, išskyrus vartotojų autentikacijos maršrutus,
kurie yra vartotojai.py modulyje.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from web.db_modelis import db, Byla, Vaikas, Prognoze
from web.forms import BylosForm, VaikoForm, PrognozesUzklausa
from duomenu_apdorojimas.duomenu_analizatorius import DuomenuAnalizatorius
from duomenu_apdorojimas.duomenu_paruosimas import DuomenuParuosejas
import uuid

logger = logging.getLogger(__name__)

# Sukuriame Blueprint objektą
main_bp = Blueprint('main', __name__)

def registruoti_visus_marsrutus(app):
    """
    Registruoja visus maršrutus Flask aplikacijai
    
    Args:
        app: Flask aplikacijos objektas
    """
    app.register_blueprint(main_bp)
    # Čia galima registruoti kitus Blueprint objektus

# Pagalbinė funkcija failų įkėlimui
def leistinas_failas(filename):
    """
    Patikrina, ar failo tipas yra leistinas
    
    Args:
        filename (str): Failo pavadinimas
    
    Returns:
        bool: True, jei failo tipas leistinas
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """
    Vartotojo valdymo skydelis
    """
    # Gauname vartotojo bylas
    vartotojo_bylos = Byla.query.filter_by(vartotojo_id=current_user.id).all()
    
    # Rūšiuojame bylas pagal sukūrimo datą (naujausios viršuje)
    vartotojo_bylos = sorted(vartotojo_bylos, key=lambda x: x.sukurimo_data, reverse=True)
    
    # Apskaičiuojame statistikas
    bylu_skaicius = len(vartotojo_bylos)
    
    return render_template(
        'dashboard.html',
        title='Valdymo skydelis',
        bylos=vartotojo_bylos,
        bylu_skaicius=bylu_skaicius
    )

@main_bp.route('/bylos/nauja', methods=['GET', 'POST'])
@login_required
def sukurti_byla():
    forma = BylosForm()

    if forma.validate_on_submit():
        # Sukuriame pagrindinę bylą
        byla = Byla(
            santuokos_trukme=forma.santuokos_trukme.data,
            amzius_vyras=forma.amzius_vyras.data,
            amzius_moteris=forma.amzius_moteris.data,
            negyvena_kartu_menesiai=forma.negyvena_kartu_menesiai.data,
            pajamos_vyras=forma.pajamos_vyras.data,
            pajamos_moteris=forma.pajamos_moteris.data,
            vaiku_skaicius=forma.vaiku_skaicius.data,
            nutraukimo_budas=forma.nutraukimo_budas.data,
            bendro_turto_verte=forma.bendro_turto_verte.data,
            asmeninio_turto_verte_vyras=forma.asmeninio_turto_verte_vyras.data,
            asmeninio_turto_verte_moteris=forma.asmeninio_turto_verte_moteris.data,
            bendros_prievoles=forma.bendros_prievoles.data,
            asmenines_prievoles_vyras=forma.asmenines_prievoles_vyras.data,
            asmenines_prievoles_moteris=forma.asmenines_prievoles_moteris.data,
            turto_vyras_po_asmeniniu_lesu=forma.turto_vyras_po_asmeniniu_lesu.data,
            turto_moteris_po_asmeniniu_lesu=forma.turto_moteris_po_asmeniniu_lesu.data,
            soc_problemos=forma.soc_problemos.data,
            vartotojo_id=current_user.id
        )
        db.session.add(byla)
        db.session.commit()

        # Vaikų duomenų apdorojimas
        for i in range(byla.vaiku_skaicius):
            amzius = request.form.get(f'vaikas_{i}_amzius')
            emoc_mama = request.form.get(f'vaikas_{i}_emocinis_mama')
            emoc_tevas = request.form.get(f'vaikas_{i}_emocinis_tevas')
            poreikiai = request.form.get(f'vaikas_{i}_poreikiai')

            vaikas = Vaikas(
                vaiko_nr=i + 1,
                amzius=amzius,
                emocinis_rysys_mama=emoc_mama,
                emocinis_rysys_tevas=emoc_tevas,
                poreikiai=poreikiai,
                bylos_id=byla.id
            )
            db.session.add(vaikas)

        db.session.commit()

        # Prognozės pagal sukurtą bylą
        atlikti_visas_prognozes_su_vaidmeniu(byla)

        flash('Byla sėkmingai sukurta ir prognozės atliktos.', 'success')
        return redirect(url_for('main.perziureti_byla', bylos_id=byla.id))

    return render_template('bylos_forma.html', title='Nauja byla', forma=forma)




@main_bp.route('/bylos/<int:bylos_id>/vaikai', methods=['GET', 'POST'])
@login_required
def prideti_vaikus(bylos_id):
    """
    Vaikų pridėjimas prie bylos
    
    Args:
        bylos_id (int): Bylos ID
    """
    byla = Byla.query.get_or_404(bylos_id)
    
    # Patikriname, ar byla priklauso prisijungusiam vartotojui
    if byla.vartotojo_id != current_user.id:
        flash('Jūs neturite teisės redaguoti šios bylos.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    forma = VaikoForm()
    
    if forma.validate_on_submit():
        # Sukuriame naują vaiką
        naujas_vaikas = Vaikas(
            vaiko_nr=forma.vaiko_nr.data,
            amzius=forma.amzius.data,
            emocinis_rysys_mama=forma.emocinis_rysys_mama.data,
            emocinis_rysys_tevas=forma.emocinis_rysys_tevas.data,
            poreikiai=forma.poreikiai.data,
            bylos_id=byla.id
        )
        
        # Išsaugome vaiką
        db.session.add(naujas_vaikas)
        db.session.commit()
        
        flash(f'Vaikas {naujas_vaikas.vaiko_nr} sėkmingai pridėtas!', 'success')
        
        # Jei reikia pridėti daugiau vaikų, grįžtame į tą patį puslapį
        if 'prideti_dar' in request.form:
            return redirect(url_for('main.prideti_vaikus', bylos_id=byla.id))
        
        # Jei baigėme pridėti vaikus, nukreipiame į bylos peržiūros puslapį
        return redirect(url_for('main.perziureti_byla', bylos_id=byla.id))
    
    return render_template(
        'vaiko_forma.html',
        title='Pridėti vaiką',
        forma=forma,
        byla=byla
    )

@main_bp.route('/bylos/<int:bylos_id>')
@login_required
def perziureti_byla(bylos_id):
    """
    Bylos peržiūra
    
    Args:
        bylos_id (int): Bylos ID
    """
    byla = Byla.query.get_or_404(bylos_id)
    
    # Patikriname, ar byla priklauso prisijungusiam vartotojui
    if byla.vartotojo_id != current_user.id:
        flash('Jūs neturite teisės peržiūrėti šios bylos.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    # Gauname bylos vaikus
    vaikai = Vaikas.query.filter_by(bylos_id=byla.id).all()
    
    # Gauname bylos prognozes
    prognozes = Prognoze.query.filter_by(bylos_id=byla.id).all()
    
    return render_template(
        'prognozavimas.html',
        title=f'Byla {byla.byla_id}',
        byla=byla,
        vaikai=vaikai,
        prognozes=prognozes
    )

@main_bp.route('/bylos/<int:bylos_id>/redaguoti', methods=['GET', 'POST'])
@login_required
def redaguoti_byla(bylos_id):
    """
    Bylos redagavimas
    
    Args:
        bylos_id (int): Bylos ID
    """
    byla = Byla.query.get_or_404(bylos_id)
    
    # Patikriname, ar byla priklauso prisijungusiam vartotojui
    if byla.vartotojo_id != current_user.id:
        flash('Jūs neturite teisės redaguoti šios bylos.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    forma = BylosForm(obj=byla)
    
    if forma.validate_on_submit():
        # Atnaujiname bylos duomenis
        forma.populate_obj(byla)
        
        # Išsaugome pakeitimus
        db.session.commit()
        
        flash(f'Byla {byla.byla_id} sėkmingai atnaujinta!', 'success')
        return redirect(url_for('main.perziureti_byla', bylos_id=byla.id))
    
    return render_template(
        'bylos_forma.html',
        title=f'Redaguoti bylą {byla.byla_id}',
        forma=forma,
        redagavimas=True
    )

@main_bp.route('/bylos/<int:bylos_id>/vaikai/<int:vaiko_id>/redaguoti', methods=['GET', 'POST'])
@login_required
def redaguoti_vaika(bylos_id, vaiko_id):
    """
    Vaiko duomenų redagavimas
    
    Args:
        bylos_id (int): Bylos ID
        vaiko_id (int): Vaiko ID
    """
    byla = Byla.query.get_or_404(bylos_id)
    vaikas = Vaikas.query.get_or_404(vaiko_id)
    
    # Patikriname, ar byla priklauso prisijungusiam vartotojui
    if byla.vartotojo_id != current_user.id:
        flash('Jūs neturite teisės redaguoti šios bylos.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    # Patikriname, ar vaikas priklauso šiai bylai
    if vaikas.bylos_id != byla.id:
        flash('Vaikas nepriklauso šiai bylai.', 'danger')
        return redirect(url_for('main.perziureti_byla', bylos_id=byla.id))
    
    forma = VaikoForm(obj=vaikas)
    
    if forma.validate_on_submit():
        # Atnaujiname vaiko duomenis
        forma.populate_obj(vaikas)
        
        # Išsaugome pakeitimus
        db.session.commit()
        
        flash(f'Vaiko {vaikas.vaiko_nr} duomenys sėkmingai atnaujinti!', 'success')
        return redirect(url_for('main.perziureti_byla', bylos_id=byla.id))
    
    return render_template(
        'vaiko_forma.html',
        title=f'Redaguoti vaiką {vaikas.vaiko_nr}',
        forma=forma,
        byla=byla,
        redagavimas=True
    )

@main_bp.route('/bylos/<int:bylos_id>/prognozuoti', methods=['GET', 'POST'])
@login_required
def prognozuoti(bylos_id):
    byla = Byla.query.get_or_404(bylos_id)

    if byla.vartotojo_id != current_user.id:
        flash('Jūs neturite teisės peržiūrėti šios bylos.', 'danger')
        return redirect(url_for('main.dashboard'))

    vaikai = Vaikas.query.filter_by(bylos_id=byla.id).all()
    prognozes = Prognoze.query.filter_by(bylos_id=byla.id).all()

    forma = PrognozesUzklausa()
    forma.vaiko_id.choices = [(0, 'Nepriklauso nuo vaiko')] + [(v.id, f"Vaikas {v.vaiko_nr}") for v in vaikai]

    if forma.validate_on_submit():
        try:
            prognozes_tipas = forma.prognozes_tipas.data
            vaiko_id = forma.vaiko_id.data if forma.vaiko_id.data > 0 else None
            rezultatas, tikimybe, modelio_pavadinimas = atlikti_prognoze(prognozes_tipas, byla, vaiko_id)

            prognoze = Prognoze(
                prognozes_tipas=prognozes_tipas,
                prognozes_reiksme=str(rezultatas),
                tikimybe=tikimybe,
                modelio_pavadinimas=modelio_pavadinimas,
                bylos_id=byla.id,
                vaiko_id=vaiko_id
            )

            db.session.add(prognoze)
            db.session.commit()

            flash(f'Prognozė "{prognozes_tipas}" sėkmingai atlikta!', 'success')
            return redirect(url_for('main.prognozuoti', bylos_id=byla.id))
        except Exception as e:
            flash(f'Klaida atliekant prognozę: {str(e)}', 'danger')

    return render_template(
        'prognozavimas.html',
        title='Atlikti prognozę',
        forma=forma,
        byla=byla,
        vaikai=vaikai,
        prognozes=prognozes
    )


def atlikti_prognoze(prognozes_tipas, byla, vaiko_id=None):
    """
    Atlieka prognozę pagal nurodytą tipą
    
    Args:
        prognozes_tipas (str): Prognozės tipas
        byla (Byla): Bylos objektas
        vaiko_id (int, optional): Vaiko ID, jei prognozė susijusi su vaiku
    
    Returns:
        tuple: (prognozes_rezultatas, tikimybe, modelio_pavadinimas)
    """
    # Čia būtų naudojami ištreniruoti modeliai ir atliekama prognozė
    # Šiuo metu grąžiname fiktyvius duomenis
    
    # Užkrauname atitinkamą modelį
    modelio_pavadinimas = ""
    
    try:
        # Paruošiame duomenis prognozei
        duomenys = paruosti_duomenis_prognozei(prognozes_tipas, byla, vaiko_id)
        
        if prognozes_tipas == 'gyvenamoji_vieta':
            # Importuojame modelį gyvenamosios vietos prognozavimui
            from prognozavimas.vaiko_gyv_vieta import prognozuoti_gyvenamaja_vieta
            
            rezultatas, tikimybe = prognozuoti_gyvenamaja_vieta(duomenys)
            modelio_pavadinimas = "RandomForestClassifier"
        
        elif prognozes_tipas == 'islaikymas':
            # Importuojame modelį išlaikymo prognozavimui
            from prognozavimas.islaikymas import prognozuoti_islaikyma
            
            rezultatas, tikimybe = prognozuoti_islaikyma(duomenys)
            modelio_pavadinimas = "GradientBoostingRegressor"
        
        elif prognozes_tipas == 'bendravimo_tvarka':
            # Importuojame modelį bendravimo tvarkos prognozavimui
            from prognozavimas.bendravimo_tvarka import prognozuoti_bendravimo_tvarka
            
            rezultatas, tikimybe = prognozuoti_bendravimo_tvarka(duomenys)
            modelio_pavadinimas = "Neuroninis tinklas"
        
        elif prognozes_tipas == 'turto_padalijimas':
            # Importuojame modelį turto padalijimo prognozavimui
            from prognozavimas.turto_padalijimas import prognozuoti_turto_padalijima
            
            rezultatas, tikimybe = prognozuoti_turto_padalijima(duomenys)
            modelio_pavadinimas = "RandomForestRegressor"
        
        elif prognozes_tipas == 'prievoles':
            # Importuojame modelį prievolių paskirstymo prognozavimui
            from prognozavimas.prievoles import prognozuoti_prievoles
            
            rezultatas, tikimybe = prognozuoti_prievoles(duomenys)
            modelio_pavadinimas = "LinearRegression"
        
        elif prognozes_tipas == 'bylinejimosi_islaidos':
            # Importuojame modelį bylinėjimosi išlaidų prognozavimui
            from prognozavimas.bylinejimosi_islaidos import prognozuoti_bylinejimosi_islaidas
            
            rezultatas, tikimybe = prognozuoti_bylinejimosi_islaidas(duomenys)
            modelio_pavadinimas = "Ridge"
        
        else:
            raise ValueError(f"Nežinomas prognozės tipas: {prognozes_tipas}")
        
        return rezultatas, tikimybe, modelio_pavadinimas
    
    except ImportError:
        # Jei modelis dar neimplementuotas, grąžiname fiktyvius duomenis
        logger.warning(f"Modelis '{prognozes_tipas}' neimplementuotas, grąžinami fiktyvūs duomenys")
        
        # Fiktyvūs duomenys skirtingiems prognozių tipams
        if prognozes_tipas == 'gyvenamoji_vieta':
            return 'mama', 0.85, 'RandomForestClassifier (fiktyvus)'
        elif prognozes_tipas == 'islaikymas':
            return 350.0, 0.78, 'GradientBoostingRegressor (fiktyvus)'
        elif prognozes_tipas == 'bendravimo_tvarka':
            return 'Kas antrą savaitgalį', 0.72, 'Neuroninis tinklas (fiktyvus)'
        elif prognozes_tipas == 'turto_padalijimas':
            return {'vyras': 45, 'moteris': 55}, 0.81, 'RandomForestRegressor (fiktyvus)'
        elif prognozes_tipas == 'prievoles':
            return {'vyras': 60, 'moteris': 40}, 0.76, 'LinearRegression (fiktyvus)'
        elif prognozes_tipas == 'bylinejimosi_islaidos':
            return 1200.0, 0.68, 'Ridge (fiktyvus)'
        else:
            return 'Nežinoma', 0.5, 'Nėra modelio'

def paruosti_duomenis_prognozei(prognozes_tipas, byla, vaiko_id=None):
    """
    Paruošia duomenis prognozei
    
    Args:
        prognozes_tipas (str): Prognozės tipas
        byla (Byla): Bylos objektas
        vaiko_id (int, optional): Vaiko ID, jei prognozė susijusi su vaiku
    
    Returns:
        dict: Duomenys, paruošti prognozei
    """
    # Paruošiame duomenis iš bylos
    duomenys = {
        'santuokos_trukme': byla.santuokos_trukme,
        'amzius_vyras': byla.amzius_vyras,
        'amzius_moteris': byla.amzius_moteris,
        'negyvena_kartu_menesiai': byla.negyvena_kartu_menesiai,
        'pajamos_vyras': byla.pajamos_vyras,
        'pajamos_moteris': byla.pajamos_moteris,
        'vaiku_skaicius': byla.vaiku_skaicius,
        'nutraukimo_budas': byla.nutraukimo_budas,
        'bendro_turto_verte': byla.bendro_turto_verte,
        'asmeninio_turto_verte_vyras': byla.asmeninio_turto_verte_vyras,
        'asmeninio_turto_verte_moteris': byla.asmeninio_turto_verte_moteris,
        'bendros_prievoles': byla.bendros_prievoles,
        'asmenines_prievoles_vyras': byla.asmenines_prievoles_vyras,
        'asmenines_prievoles_moteris': byla.asmenines_prievoles_moteris,
        'soc_problemos': byla.soc_problemos
    }
    
    # Jei prognozė susijusi su vaiku, pridedame vaiko duomenis
    if vaiko_id:
        vaikas = Vaikas.query.get(vaiko_id)
        if vaikas:
            duomenys.update({
                'vaiko_amzius': vaikas.amzius,
                'emocinis_rysys_mama': vaikas.emocinis_rysys_mama,
                'emocinis_rysys_tevas': vaikas.emocinis_rysys_tevas,
                'poreikiai': vaikas.poreikiai,
                'gyvenamoji_vieta': vaikas.gyvenamoji_vieta,
                'bendravimo_tvarka': vaikas.bendravimo_tvarka,
                'islaikymas': vaikas.islaikymas,
                'islaikymo_iskolinimas': vaikas.islaikymo_iskolinimas
            })
    
    return duomenys

@main_bp.route('/duomenu-analize')
@login_required
def duomenu_analize():
    """
    Duomenų analizės puslapis
    """
    return render_template(
        'duomenu_analize.html',
        title='Duomenų analizė'
    )

@main_bp.route('/api/duomenu-statistika')
@login_required
def duomenu_statistika():
    """
    API funkcija, kuri grąžina duomenų statistiką
    
    Returns:
        json: Duomenų statistika
    """
    # Inicializuojame duomenų analizatorių
    analizatorius = DuomenuAnalizatorius()
    
    try:
        # Nuskaitome duomenis
        analizatorius.nuskaityti_duomenis()
        
        # Atliekame analizę
        statistika = analizatorius.analizuoti_duomenis()
        
        # Konvertuojame statistiką į tinkamą JSON formatą
        rezultatai = {}
        
        for pavadinimas, stat in statistika.items():
            rezultatai[pavadinimas] = {
                'eilučių_skaičius': stat['eilučių_skaičius'],
                'stulpelių_skaičius': stat['stulpelių_skaičius'],
                'stulpelių_pavadinimai': stat['stulpelių_pavadinimai'],
                'trūkstamos_reikšmės': stat['trūkstamos_reikšmės']
            }
            
            # Pridedame kategorinių stulpelių duomenis (tik pirmuosius 10 elementų)
            kategoriniai = {}
            for stulpelis, reiksmes in stat['kategorinių_stulpelių_unikalios_reikšmės'].items():
                kategoriniai[stulpelis] = dict(list(reiksmes.items())[:10])
            
            rezultatai[pavadinimas]['kategoriniai_stulpeliai'] = kategoriniai
        
        return jsonify(rezultatai)
    
    except Exception as e:
        logger.error(f"Klaida analizuojant duomenis: {e}")
        return jsonify({'klaida': str(e)}), 500

@main_bp.route('/api/duomenu-koreliacijos/<string:failas>')
@login_required
def duomenu_koreliacijos(failas):
    """
    API funkcija, kuri grąžina duomenų koreliacijas
    
    Args:
        failas (str): Failo pavadinimas
    
    Returns:
        json: Koreliacijų matrica
    """
    # Inicializuojame duomenų analizatorių
    analizatorius = DuomenuAnalizatorius()
    
    try:
        # Nuskaitome duomenis
        analizatorius.nuskaityti_duomenis()
        
        # Atliekame koreliacijos analizę
        koreliacijos = analizatorius.tirti_koreliacijas(failas)
        
        # Konvertuojame į tinkamą formatą
        koreliacijos_dict = koreliacijos.to_dict()
        
        return jsonify(koreliacijos_dict)
    
    except Exception as e:
        logger.error(f"Klaida analizuojant koreliacijas: {e}")
        return jsonify({'klaida': str(e)}), 500

@main_bp.route('/api/modeliu-metrikos')
@login_required
def modeliu_metrikos():
    """
    API funkcija, kuri grąžina modelių metrikas
    
    Returns:
        json: Modelių metrikos
    """
    try:
        # Metrikų kelias
        metriku_direktorija = os.path.join(Config.MODELIU_DIREKTORIJA, 'metrikos')
        
        # Patikriname, ar egzistuoja metrikų direktorija
        if not os.path.exists(metriku_direktorija):
            logger.warning("Modelių metrikų direktorija nerasta. Kuriame naują.")
            os.makedirs(metriku_direktorija, exist_ok=True)

        # Sukuriame rezultatų žodyną
        rezultatai = {}
        
        # Bandome nuskaityti visas modelių metrikas
        for prognozes_tipas in ['gyvenamoji_vieta', 'islaikymas', 'bendravimo_tvarka', 
                               'turto_padalijimas', 'prievoles', 'bylinejimosi_islaidos']:
            
            metriku_failas = os.path.join(metriku_direktorija, f"{prognozes_tipas}_metrikos.json")
            
            # Patikriname, ar egzistuoja metrikų failas
            if os.path.exists(metriku_failas):
                # Nuskaitome metrikų failą
                with open(metriku_failas, 'r', encoding='utf-8') as f:
                    rezultatai[prognozes_tipas] = json.load(f)
            else:
                # Jei nėra metrikų failo, bandome įvertinti modelius
                modeliu_metrikos = ivertinti_modelius(prognozes_tipas)
                
                if modeliu_metrikos:
                    rezultatai[prognozes_tipas] = modeliu_metrikos
                    
                    # Išsaugome metrikas faile
                    with open(metriku_failas, 'w', encoding='utf-8') as f:
                        json.dump(modeliu_metrikos, f, ensure_ascii=False, indent=2)
                else:
                    # Jei nepavyko įvertinti modelių, naudojame fiktyvius duomenis
                    logger.warning(f"Nerandamos metrikos modeliui '{prognozes_tipas}'. Naudojami fiktyvūs duomenys.")
                    
                    # Įtraukiame fiktyvius duomenis pagal prognozės tipą
                    if prognozes_tipas == 'gyvenamoji_vieta':
                        rezultatai[prognozes_tipas] = {
                            'RandomForestClassifier': {
                                'accuracy': 0.85,
                                'precision': 0.83,
                                'recall': 0.86,
                                'f1': 0.845,
                                'pastaba': 'Fiktyvūs duomenys'
                            },
                            'GradientBoostingClassifier': {
                                'accuracy': 0.82,
                                'precision': 0.81,
                                'recall': 0.84,
                                'f1': 0.825,
                                'pastaba': 'Fiktyvūs duomenys'
                            },
                            'Neuroninis tinklas': {
                                'accuracy': 0.87,
                                'precision': 0.86,
                                'recall': 0.87,
                                'f1': 0.865,
                                'pastaba': 'Fiktyvūs duomenys'
                            }
                        }
                    elif prognozes_tipas in ['islaikymas', 'turto_padalijimas', 'prievoles', 'bylinejimosi_islaidos']:
                        # Regresijos modelių fiktyvūs duomenys
                        rezultatai[prognozes_tipas] = {
                            'LinearRegression': {
                                'MSE': 1250.0,
                                'RMSE': 35.36,
                                'MAE': 28.5,
                                'R2': 0.78,
                                'pastaba': 'Fiktyvūs duomenys'
                            },
                            'RandomForestRegressor': {
                                'MSE': 980.0,
                                'RMSE': 31.3,
                                'MAE': 24.7,
                                'R2': 0.82,
                                'pastaba': 'Fiktyvūs duomenys'
                            },
                            'Neuroninis tinklas': {
                                'MSE': 1050.0,
                                'RMSE': 32.4,
                                'MAE': 26.3,
                                'R2': 0.81,
                                'pastaba': 'Fiktyvūs duomenys'
                            }
                        }
                    elif prognozes_tipas == 'bendravimo_tvarka':
                        # Klasifikacijos modelių fiktyvūs duomenys
                        rezultatai[prognozes_tipas] = {
                            'RandomForestClassifier': {
                                'accuracy': 0.76,
                                'precision': 0.75,
                                'recall': 0.74,
                                'f1': 0.745,
                                'pastaba': 'Fiktyvūs duomenys'
                            },
                            'LogisticRegression': {
                                'accuracy': 0.71,
                                'precision': 0.70,
                                'recall': 0.72,
                                'f1': 0.71,
                                'pastaba': 'Fiktyvūs duomenys'
                            },
                            'Neuroninis tinklas': {
                                'accuracy': 0.79,
                                'precision': 0.78,
                                'recall': 0.77,
                                'f1': 0.775,
                                'pastaba': 'Fiktyvūs duomenys'
                            }
                        }
        
        return jsonify(rezultatai)
    
    except Exception as e:
        logger.error(f"Klaida gaunant modelių metrikas: {e}")
        return jsonify({'klaida': str(e)}), 500

def atlikti_visas_prognozes_su_vaidmeniu(byla):
    prognozes_be_vaiku = ['turto_padalijimas', 'prievoles', 'bylinejimosi_islaidos']
    for tipas in prognozes_be_vaiku:
        try:
            rez, tikimybe, modelis = atlikti_prognoze(tipas, byla)
            prog = Prognoze(
                prognozes_tipas=tipas,
                prognozes_reiksme=str(rez),
                tikimybe=tikimybe,
                modelio_pavadinimas=modelis,
                bylos_id=byla.id
            )
            db.session.add(prog)
        except Exception as e:
            print(f"Klaida prognozuojant {tipas}: {e}")

    vaikai = Vaikas.query.filter_by(bylos_id=byla.id).all()
    prognozes_vaikams = ['gyvenamoji_vieta', 'islaikymas', 'bendravimo_tvarka']
    for vaikas in vaikai:
        for tipas in prognozes_vaikams:
            try:
                rez, tikimybe, modelis = atlikti_prognoze(tipas, byla, vaikas.id)
                prog = Prognoze(
                    prognozes_tipas=tipas,
                    prognozes_reiksme=str(rez),
                    tikimybe=tikimybe,
                    modelio_pavadinimas=modelis,
                    bylos_id=byla.id,
                    vaiko_id=vaikas.id
                )
                db.session.add(prog)
            except Exception as e:
                print(f"Klaida prognozuojant {tipas} vaikui {vaikas.id}: {e}")

    db.session.commit()

def ivertinti_modelius(prognozes_tipas):
    """
    Įvertina modelius pagal nurodytą prognozės tipą
    
    Args:
        prognozes_tipas (str): Prognozės tipas
    
    Returns:
        dict: Modelių metrikos
    """
    try:
        # Modelių direktorija
        modeliu_direktorija = Config.MODELIU_DIREKTORIJA
        
        # Tikriname, kokie modeliai yra prieinami šiam prognozės tipui
        modeliu_failai = [f for f in os.listdir(modeliu_direktorija) 
                        if f.startswith(prognozes_tipas) and f.endswith('.joblib')]
        
        if not modeliu_failai:
            logger.warning(f"Nerasti modeliai tipui '{prognozes_tipas}'")
            return None
        
        # Atskiriame test duomenis vertinimui
        from duomenu_apdorojimas.duomenu_paruosimas import DuomenuParuosejas
        
        # Inicializuojame duomenų paruošėją
        paruosejas = DuomenuParuosejas()
        X_test, y_test = paruosejas.gauti_testavimo_duomenis(prognozes_tipas)
        
        if X_test is None or y_test is None:
            logger.warning(f"Nepavyko gauti testavimo duomenų tipui '{prognozes_tipas}'")
            return None
        
        # Rezultatų žodynas
        metrikos = {}
        
        # Įvertiname kiekvieną modelį
        for modelio_failas in modeliu_failai:
            try:
                # Įkrauname modelį
                modelio_kelias = os.path.join(modeliu_direktorija, modelio_failas)
                modelio_objektas = joblib.load(modelio_kelias)
                
                modelis = modelio_objektas.get('modelis')
                meta_info = modelio_objektas.get('meta_info', {})
                
                if modelis is None:
                    logger.warning(f"Nepavyko įkrauti modelio iš {modelio_failas}")
                    continue
                
                # Modelio pavadinimas
                modelio_pavadinimas = meta_info.get('modelio_klase', 
                                                 os.path.basename(modelio_failas).split('_')[1])
                
                # Įvertiname modelį pagal prognozės tipą
                if prognozes_tipas == 'gyvenamoji_vieta' or prognozes_tipas == 'bendravimo_tvarka':
                    # Klasifikacijos modeliai
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    # Prognozes
                    y_pred = modelis.predict(X_test)
                    
                    # Metrikos
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    metrikos[modelio_pavadinimas] = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1)
                    }
                else:
                    # Regresijos modeliai
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    # Prognozes
                    y_pred = modelis.predict(X_test)
                    
                    # Metrikos
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    metrikos[modelio_pavadinimas] = {
                        'MSE': float(mse),
                        'RMSE': float(rmse),
                        'MAE': float(mae),
                        'R2': float(r2)
                    }
            
            except Exception as e:
                logger.error(f"Klaida įvertinant modelį {modelio_failas}: {e}")
                continue
        
        return metrikos if metrikos else None
    
    except Exception as e:
        logger.error(f"Klaida vertinant modelius: {e}")
        return None

@main_bp.route('/profilis', methods=['GET', 'POST'])
@login_required
def profilis():
    from web.forms import ProfilioForma  # jei forma yra ten
    form = ProfilioForma()

    if form.validate_on_submit():
        # čia galima atnaujinti user info: current_user.username = form.username.data ...
        flash("Pakeitimai išsaugoti", "success")
        return redirect(url_for('main.profilis'))

    # Užpildom laukus esama info (čia tik rodymui)
    form.username.data = current_user.vardas  # arba current_user.username, jei toks atributas egzistuoja
    form.email.data = current_user.el_pastas

    # Gauname vartotojo bylas
    bylos = current_user.bylos if hasattr(current_user, 'bylos') else []

    return render_template(
        "profilis.html",
        form=form,
        bylos=bylos,
        profile_image=url_for('static', filename='img/default.png')  # arba custom
    )

 