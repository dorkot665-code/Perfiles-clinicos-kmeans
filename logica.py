"""Lógica de la aplicación (independiente de Streamlit para poder probarla).
Carga el MISMO pipeline guardado por el notebook final y asigna un paciente a un perfil."""
import json
import os
import pandas as pd
from sklearn.pipeline import Pipeline

AQUI = os.path.dirname(os.path.abspath(__file__))
VARIABLES = ['age', 'resting_bp_s', 'cholesterol', 'max_heart_rate', 'oldpeak',
             'sex', 'exercise_angina', 'fasting_blood_sugar']
BINARIAS = ['sex', 'exercise_angina', 'fasting_blood_sugar']


def cargar_modelo():
    modelo = joblib.load(os.path.join(AQUI, 'modelo_clustering.joblib'))
    with open(os.path.join(AQUI, 'metadatos_modelo.json'), encoding='utf-8') as f:
        meta = json.load(f)
    return modelo, meta


def validar_registro(reg, meta):
    """Devuelve la lista de problemas encontrados (vacía si el registro es válido)."""
    errores = []
    for v in VARIABLES:
        if v not in reg or reg[v] is None or reg[v] == '':
            errores.append('Falta el campo obligatorio: %s.' % v)
            continue
        try:
            x = float(reg[v])
        except (TypeError, ValueError):
            errores.append('%s debe ser numérico.' % v)
            continue
        if v in BINARIAS:
            if x not in (0.0, 1.0):
                errores.append('%s debe ser 0 o 1.' % v)
        else:
            lo, hi = meta['rangos_observados'][v]
            if v == 'cholesterol' and x == 0:
                continue  # 0 = colesterol no disponible; el pipeline lo imputa con la mediana
            if not (lo <= x <= hi):
                errores.append('%s = %s está fuera del rango observado en los datos (%s a %s).' % (v, x, lo, hi))
    return errores


def asignar(reg, modelo, meta):
    errores = validar_registro(reg, meta)
    if errores:
        raise ValueError(' '.join(errores))
    fila = pd.DataFrame([{v: float(reg[v]) for v in VARIABLES}])[VARIABLES]
    cluster = int(modelo.predict(fila)[0])
    previo = Pipeline(modelo.steps[:-1])
    distancias = modelo.named_steps['kmeans'].transform(previo.transform(fila))[0]
    orden = distancias.argsort()
    return {
        'cluster': cluster,
        'perfil': meta['nombres_cluster'][str(cluster)],
        'distancia_propio': float(distancias[orden[0]]),
        'segundo_cluster': int(orden[1]),
        'distancia_segundo': float(distancias[orden[1]]),
        'tamano_cluster': meta['tamanos'][str(cluster)],
        'colesterol_imputado': float(reg['cholesterol']) == 0.0,
    }
