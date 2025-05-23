"""
Neuroninio tinklo modulis

Šis modulis implementuoja neuroninius tinklus naudojant Keras biblioteką.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import Config

logger = logging.getLogger(__name__)

def sukurti_regresijos_tinkla(ivesties_dydis, hidden_layers=[64, 32], activation='relu', 
                             learning_rate=0.001, dropout_rate=0.2):
    """
    Sukuria regresijos neuroninį tinklą
    
    Args:
        ivesties_dydis (int): Įvesties sluoksnio dydis (požymių skaičius)
        hidden_layers (list): Paslėptų sluoksnių neuronų skaičius
        activation (str): Aktyvavimo funkcija
        learning_rate (float): Mokymosi tempas
        dropout_rate (float): Dropout sluoksnio koeficientas
    
    Returns:
        keras.Model: Sukonfigūruotas neuroninis tinklas
    """
    # Kuriame modelį su norima struktūra
    model = keras.Sequential()
    
    # Įvesties sluoksnis
    model.add(layers.Input(shape=(ivesties_dydis,)))
    
    # Paslėpti sluoksniai
    for units in hidden_layers:
        model.add(layers.Dense(
            units=units,
            activation=activation,
            kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    # Išvesties sluoksnis (regresija - vienas neuronas be aktyvavimo funkcijos)
    model.add(layers.Dense(1))
    
    # Kompiliuojame modelį
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    logger.info(f"Sukurtas regresijos neuroninis tinklas su {len(hidden_layers)} paslėptais sluoksniais")
    model.summary(print_fn=logger.info)
    
    return model

def sukurti_klasifikavimo_tinkla(ivesties_dydis, klasiu_skaicius, hidden_layers=[64, 32], 
                                activation='relu', learning_rate=0.001, dropout_rate=0.2):
    """
    Sukuria klasifikavimo neuroninį tinklą
    
    Args:
        ivesties_dydis (int): Įvesties sluoksnio dydis (požymių skaičius)
        klasiu_skaicius (int): Išvesties klasių skaičius
        hidden_layers (list): Paslėptų sluoksnių neuronų skaičius
        activation (str): Aktyvavimo funkcija
        learning_rate (float): Mokymosi tempas
        dropout_rate (float): Dropout sluoksnio koeficientas
    
    Returns:
        keras.Model: Sukonfigūruotas neuroninis tinklas
    """
    # Kuriame modelį su norima struktūra
    model = keras.Sequential()
    
    # Įvesties sluoksnis
    model.add(layers.Input(shape=(ivesties_dydis,)))
    
    # Paslėpti sluoksniai
    for units in hidden_layers:
        model.add(layers.Dense(
            units=units,
            activation=activation,
            kernel_regularizer=regularizers.l2(0.001)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    # Išvesties sluoksnis - klasifikavimo atveju priklauso nuo klasių skaičiaus
    if klasiu_skaicius == 2:
        # Binarinė klasifikacija - vienas išvesties neuronas su sigmoid aktyvavimo funkcija
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        # Daugiaklasė klasifikacija - klasių_skaičius neuronų su softmax aktyvavimo funkcija
        model.add(layers.Dense(klasiu_skaicius, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    # Kompiliuojame modelį
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    logger.info(f"Sukurtas klasifikavimo neuroninis tinklas su {len(hidden_layers)} paslėptais sluoksniais")
    model.summary(print_fn=logger.info)
    
    return model

def treniruoti_tinkla(model, X_train, y_train, X_val, y_val, epochos=100, batch_dydis=32,
                     ankstyvasis_sustabdymas=True, modelio_kelias=None):
    """
    Treniruoja neuroninį tinklą
    
    Args:
        model (keras.Model): Neuroninis tinklas
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        epochos (int): Treniravimo epochų skaičius
        batch_dydis (int): Batch dydis
        ankstyvasis_sustabdymas (bool): Ar naudoti ankstyvąjį sustabdymą
        modelio_kelias (str): Kelias, kur išsaugoti geriausią modelį
    
    Returns:
        tuple: (history, modelis) - treniravimo istorija ir ištreniruotas modelis
    """
    callbacks = []
    
    # Nustatome ankstyvojo sustabdymo sąlygas
    if ankstyvasis_sustabdymas:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=Config.KANTRYBE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    # Automatinis mokymosi tempo mažinimas, kai nebegerėja rezultatai
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Jei nurodytas kelias, išsaugome geriausią modelį
    if modelio_kelias:
        model_checkpoint = ModelCheckpoint(
            filepath=modelio_kelias,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(model_checkpoint)
    
    # Treniruojame modelį
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochos,
        batch_size=batch_dydis,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Neuroninis tinklas sėkmingai ištreniruotas")
    
    return history, model

def atvaizduoti_treniravimo_istorija(history, metrikos=['loss', 'val_loss'], title="Mokymosi kreivė"):
    """
    Atvaizduoja neuroninio tinklo treniravimo istoriją
    
    Args:
        history (keras.callbacks.History): Treniravimo istorija
        metrikos (list): Metrikų sąrašas, kurias norima atvaizduoti
        title (str): Grafiko pavadinimas
    
    Returns:
        matplotlib.figure.Figure: Sukurtas grafikas
    """
    plt.figure(figsize=(12, 6))
    
    for metrika in metrikos:
        if metrika in history.history:
            plt.plot(history.history[metrika], label=metrika)
    
    plt.title(title)
    plt.xlabel('Epocha')
    plt.ylabel('Reikšmė')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def ivertinti_regresijos_tinkla(model, X_test, y_test):
    """
    Įvertina regresijos neuroninį tinklą
    
    Args:
        model (keras.Model): Neuroninis tinklas
        X_test: Testavimo duomenų požymiai
        y_test: Testavimo duomenų tikslo kintamasis
    
    Returns:
        dict: Įvertinimo metrikos
    """
    # Atliekame prognozes
    y_pred = model.predict(X_test)
    
    # Apskaičiuojame metrikas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Sudarome rezultatų žodyną
    rezultatai = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    logger.info(f"Neuroninio tinklo regresijos įvertinimas: "
               f"MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
    return rezultatai

def ivertinti_klasifikavimo_tinkla(model, X_test, y_test, klasiu_skaicius):
    """
    Įvertina klasifikavimo neuroninį tinklą
    
    Args:
        model (keras.Model): Neuroninis tinklas
        X_test: Testavimo duomenų požymiai
        y_test: Testavimo duomenų tikslo kintamasis
        klasiu_skaicius (int): Klasių skaičius
    
    Returns:
        dict: Įvertinimo metrikos
    """
    # Atliekame prognozes
    if klasiu_skaicius == 2:
        # Binarinė klasifikacija
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    else:
        # Daugiaklasė klasifikacija
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Apskaičiuojame metrikas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Sudarome rezultatų žodyną
    rezultatai = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    logger.info(f"Neuroninio tinklo klasifikavimo įvertinimas: "
               f"accuracy={accuracy:.4f}, precision={precision:.4f}, "
               f"recall={recall:.4f}, f1={f1:.4f}")
    
    return rezultatai

def issaugoti_tinkla(model, modelio_pavadinimas, direktorija=None):
    """
    Išsaugo neuroninį tinklą
    
    Args:
        model (keras.Model): Neuroninis tinklas
        modelio_pavadinimas (str): Modelio pavadinimas
        direktorija (str): Direktorija, kurioje išsaugoti modelį
    
    Returns:
        str: Pilnas kelias iki išsaugoto modelio
    """
    import os
    from datetime import datetime
    
    if direktorija is None:
        direktorija = Config.MODELIU_DIREKTORIJA
    
    # Sukuriame direktoriją, jei jos nėra
    os.makedirs(direktorija, exist_ok=True)
    
    # Sukuriame failo pavadinimą
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failo_pavadinimas = f"{modelio_pavadinimas}_{timestamp}"
    pilnas_kelias = os.path.join(direktorija, failo_pavadinimas)
    
    # Išsaugome modelį
    model.save(pilnas_kelias)
    logger.info(f"Neuroninis tinklas išsaugotas: {pilnas_kelias}")
    
    return pilnas_kelias

def ikrauti_tinkla(modelio_kelias):
    """
    Įkrauna neuroninį tinklą
    
    Args:
        modelio_kelias (str): Kelias iki modelio
    
    Returns:
        keras.Model: Įkrautas neuroninis tinklas
    """
    try:
        model = keras.models.load_model(modelio_kelias)
        logger.info(f"Neuroninis tinklas įkrautas iš: {modelio_kelias}")
        return model
    except Exception as e:
        logger.error(f"Klaida įkraunant neuroninį tinklą: {e}")
        return None

def sukurti_bendravimo_tvarkos_nt(X_train, y_train, X_val, y_val, unikalios_klases):
    """
    Sukuria ir treniruoja neuroninį tinklą bendravimo tvarkos nustatymui
    
    Args:
        X_train: Treniravimo duomenų požymiai
        y_train: Treniravimo duomenų tikslo kintamasis
        X_val: Validavimo duomenų požymiai
        y_val: Validavimo duomenų tikslo kintamasis
        unikalios_klases (list): Unikalių klasių sąrašas
    
    Returns:
        tuple: (model, history) - ištreniruotas modelis ir treniravimo istorija
    """
    # Nustatome įvesties dydį
    ivesties_dydis = X_train.shape[1]
    
    # Nustatome klasių skaičių
    klasiu_skaicius = len(unikalios_klases)
    
    logger.info(f"Kurimas neuroninis tinklas bendravimo tvarkos nustatymui. Klasių skaičius: {klasiu_skaicius}")
    
    # Sukuriame neuroninį tinklą su didesne architektūra, nes uždavinys gali būti sudėtingas
    model = sukurti_klasifikavimo_tinkla(
        ivesties_dydis=ivesties_dydis,
        klasiu_skaicius=klasiu_skaicius,
        hidden_layers=[128, 64, 32],  # Daugiau neuronų ir sluoksnių
        activation='relu',
        learning_rate=0.001,
        dropout_rate=0.3  # Didesnis dropout, kad išvengtume permokymo
    )
    
    # Treniruojame modelį
    history, model = treniruoti_tinkla(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochos=Config.EPOCHU_SKAICIUS,
        batch_dydis=Config.BATCH_DYDIS,
        ankstyvasis_sustabdymas=True
    )
    
    return model, history 