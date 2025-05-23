"""
Formų modulis

Šis modulis apima visas formų klases, naudojamas Flask aplikacijoje.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField, SelectField, FileField
from wtforms import IntegerField, FloatField, HiddenField
from wtforms.validators import DataRequired, Email, EqualTo, Length, NumberRange, Optional

class RegistracijosForma(FlaskForm):
    vardas = StringField('Vardas', validators=[DataRequired()])
    pavarde = StringField('Pavardė', validators=[DataRequired()])
    el_pastas = StringField('El. paštas', validators=[DataRequired(), Email()])
    slaptazodis = PasswordField('Slaptažodis', validators=[DataRequired()])
    patikrinti_slaptazodi = PasswordField('Pakartoti slaptažodį', validators=[
        DataRequired(), EqualTo('slaptazodis', message='Slaptažodžiai turi sutapti')
    ])
    submit = SubmitField('Registruotis')


class PrisijungimoForma(FlaskForm):
    """
    Vartotojo prisijungimo forma
    """
    el_pastas = StringField('El. paštas', validators=[DataRequired(), Email()])
    slaptazodis = PasswordField('Slaptažodis', validators=[DataRequired()])
    prisiminti = BooleanField('Prisiminti mane')
    submit = SubmitField('Prisijungti')

class VartotojoProfilisForma(FlaskForm):
    """
    Vartotojo profilio redagavimo forma
    """
    vardas = StringField('Vardas', validators=[DataRequired(), Length(min=2, max=100)])
    pavarde = StringField('Pavardė', validators=[DataRequired(), Length(min=2, max=100)])
    esamas_slaptazodis = PasswordField('Esamas slaptažodis', validators=[Optional()])
    naujas_slaptazodis = PasswordField('Naujas slaptažodis', validators=[Optional(), Length(min=6)])
    patvirtinti_slaptazodi = PasswordField('Patvirtinti naują slaptažodį', 
                                         validators=[Optional(), EqualTo('naujas_slaptazodis')])
    roles = SelectField('Rolė', choices=[('vartotojas', 'Vartotojas'), ('administratorius', 'Administratorius')])
    submit = SubmitField('Atnaujinti')

class BylosForm(FlaskForm):
    """
    Bylos kūrimo ir redagavimo forma
    """
    
    # Šeimos duomenys
    santuokos_trukme = IntegerField('Santuokos trukmė (metais)', validators=[DataRequired(), NumberRange(min=0)])
    amzius_vyras = IntegerField('Vyro amžius', validators=[DataRequired(), NumberRange(min=18, max=120)])
    amzius_moteris = IntegerField('Moters amžius', validators=[DataRequired(), NumberRange(min=18, max=120)])
    negyvena_kartu_menesiai = IntegerField('Negyvena kartu (mėnesiais)', validators=[DataRequired(), NumberRange(min=0)])
    pajamos_vyras = FloatField('Vyro pajamos (EUR)', validators=[DataRequired(), NumberRange(min=0)])
    pajamos_moteris = FloatField('Moters pajamos (EUR)', validators=[DataRequired(), NumberRange(min=0)])
    vaiku_skaicius = IntegerField('Vaikų skaičius', validators=[DataRequired(), NumberRange(min=0)])
    nutraukimo_budas = SelectField('Nutraukimo būdas', choices=[
        ('bendru sutarimu', 'Bendru sutarimu'),
        ('ginčo teisena', 'Ginčo teisena')
    ])
    
    # Turto duomenys
    bendro_turto_verte = FloatField('Bendro turto vertė (EUR)', validators=[DataRequired(), NumberRange(min=0)])
    asmeninio_turto_verte_vyras = FloatField('Asmeninio turto vertė: vyras (EUR)', validators=[DataRequired(), NumberRange(min=0)])
    asmeninio_turto_verte_moteris = FloatField('Asmeninio turto vertė: moteris (EUR)', validators=[DataRequired(), NumberRange(min=0)])
    
    # Prievolių duomenys
    bendros_prievoles = FloatField('Bendros prievolės (EUR)', validators=[DataRequired(), NumberRange(min=0)])
    asmenines_prievoles_vyras = FloatField('Asmeninės prievolės: vyras (EUR)', validators=[DataRequired(), NumberRange(min=0)])
    asmenines_prievoles_moteris = FloatField('Asmeninės prievolės: moteris (EUR)', validators=[DataRequired(), NumberRange(min=0)])
    turto_vyras_po_asmeniniu_lesu = FloatField(
                            'Vyrui atitenkantis turtas po asmeninių lėšų įskaičiavimo',
                            validators=[Optional(), NumberRange(min=0)]
)

    turto_moteris_po_asmeniniu_lesu = FloatField(
                            'Moteriai atitenkantis turtas po asmeninių lėšų įskaičiavimo',
                            validators=[Optional(), NumberRange(min=0)]
)
    
    # Proceso duomenys
    soc_problemos = SelectField('Socialinės problemos', choices=[
        ('nėra', 'Nėra'),
        ('alkoholizmas', 'Alkoholizmas'),
        ('smurtas', 'Smurtas'),
        ('lošimai', 'Lošimai'),
        ('kita', 'Kita')
    ])
    
    submit = SubmitField('Išsaugoti')

class VaikoForm(FlaskForm):
    """
    Vaiko duomenų forma
    """
    vaiko_nr = IntegerField('Vaiko numeris', validators=[DataRequired(), NumberRange(min=1)])
    amzius = IntegerField('Vaiko amžius', validators=[DataRequired(), NumberRange(min=0, max=18)])
    emocinis_rysys_mama = SelectField('Emocinis ryšys su mama', choices=[
        ('silpnas', 'Silpnas'),
        ('vidutinis', 'Vidutinis'),
        ('stiprus', 'Stiprus')
    ])
    emocinis_rysys_tevas = SelectField('Emocinis ryšys su tėvu', choices=[
        ('silpnas', 'Silpnas'),
        ('vidutinis', 'Vidutinis'),
        ('stiprus', 'Stiprus')
    ])
    poreikiai = FloatField('Vaiko poreikiai (EUR)', validators=[DataRequired(), NumberRange(min=0)])
    gyvenamoji_vieta = SelectField('Gyvenamoji vieta', choices=[
        ('mama', 'Su mama'),
        ('tevas', 'Su tėvu'),
        ('lygiomis', 'Lygiomis dalimis'),
        ('kita', 'Kita')
    ])
    bendravimo_tvarka = SelectField('Bendravimo tvarka', choices=[
        ('konkreti tvarka, nustatant bendravimą atostogų metu, šventėmis', 'Konkreti tvarka, nustatant bendravimą atostogų metu, šventėmis'),
        ('kas antrą savaitgalį', 'Kas antrą savaitgalį'),
        ('neribota bendravimo tvarka', 'Neribota bendravimo tvarka'),
        ('kita', 'Kita')
    ])
    islaikymas = FloatField('Išlaikymas (EUR)', validators=[Optional(), NumberRange(min=0)])
    islaikymo_iskolinimas = FloatField('Išlaikymo įsiskolinimas (EUR)', validators=[Optional(), NumberRange(min=0)])
    
    submit = SubmitField('Išsaugoti')
    prideti_dar = SubmitField('Išsaugoti ir pridėti dar vieną vaiką')

class PrognozesUzklausa(FlaskForm):
    """
    Prognozės užklausos forma
    """
    prognozes_tipas = SelectField('Prognozės tipas', choices=[
        ('gyvenamoji_vieta', 'Vaiko gyvenamoji vieta'),
        ('islaikymas', 'Išlaikymo vaikui priteisimas'),
        ('bendravimo_tvarka', 'Bendravimo tvarkos nustatymas'),
        ('islaikymo_iskolinimas', 'Išlaikymo įsiskolinimo priteisimas'),
        ('turto_padalijimas', 'Turto padalijimas'),
        ('prievoles', 'Prievolių paskirstymas'),
        ('bylinejimosi_islaidos', 'Bylinėjimosi išlaidų paskirstymas')
    ])
    vaiko_id = SelectField('Vaikas', coerce=int, validators=[Optional()])
    submit = SubmitField('Atlikti prognozę')

class ProfilioForma(FlaskForm):
    username = StringField("Vartotojo vardas", validators=[DataRequired()])
    email = StringField("El. paštas", validators=[DataRequired(), Email()])
    picture = FileField("Profilio nuotrauka")
    submit = SubmitField("Išsaugoti")