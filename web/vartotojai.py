from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_user, logout_user
from web.db_modelis import User
from web.forms import RegistracijosForma, PrisijungimoForma
from datetime import datetime
from web.db_modelis import db


vartotoju_bp = Blueprint('vartotojai', __name__)

@vartotoju_bp.route('/registracija', methods=['GET', 'POST'])
def registracija():
    form = RegistracijosForma()

    if form.validate_on_submit():
        # Patikrinam ar el. paštas jau naudojamas
        egzistuojantis = User.query.filter_by(el_pastas=form.el_pastas.data).first()
        if egzistuojantis:
            flash('Toks el. pašto adresas jau naudojamas.', 'danger')
            return render_template('registracija.html', form=form)

        # Sukuriam vartotoją
        naujas_vartotojas = User(
            vardas=form.vardas.data,
            pavarde=form.pavarde.data,
            el_pastas=form.el_pastas.data,
        )
        naujas_vartotojas.set_slaptazodis(form.slaptazodis.data)

        db.session.add(naujas_vartotojas)
        db.session.commit()

        flash('Registracija sėkminga! Galite prisijungti.', 'success')
        login_user(naujas_vartotojas)
        return redirect(url_for('index'))

    return render_template('registracija.html', form=form)

@vartotoju_bp.route('/prisijungimas', methods=['GET', 'POST'])
def prisijungimas():
    forma = PrisijungimoForma()
    if forma.validate_on_submit():
        vartotojas = User.query.filter_by(el_pastas=forma.el_pastas.data).first()
        if vartotojas and vartotojas.check_slaptazodis(forma.slaptazodis.data):
            login_user(vartotojas, remember=forma.prisiminti.data)
            flash("Sėkmingai prisijungėte!", "success")
            return redirect(url_for('main.profilis'))  # ČIA SVARBIAUSIA
        flash("Neteisingi prisijungimo duomenys.", "danger")
    return render_template('prisijungimas.html', title="Prisijungimas", form=forma, now=datetime.now())

@vartotoju_bp.route('/atsijungti')
def atsijungti():
    logout_user()
    flash("Atsijungėte.")
    return redirect(url_for('index'))
