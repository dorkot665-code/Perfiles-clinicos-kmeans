"""Aplicación de Streamlit: asigna un paciente a uno de los cuatro perfiles clínicos del modelo K-means final."""
import pandas as pd
import streamlit as st
from logica import cargar_modelo, asignar

st.set_page_config(page_title='Perfiles clínicos – K-means', page_icon='🫀', layout='centered')


@st.cache_resource
def _cargar():
    return cargar_modelo()


modelo, meta = _cargar()
R = meta['rangos_observados']

st.title('Perfiles clínicos de pacientes cardíacos')
st.caption('Proyecto de aula de Minería de Datos · K-means (k = %d) sobre %d pacientes · Hernando David Montero Salazar'
           % (meta['k'], meta['n_pacientes']))
st.warning('Esta herramienta es **académica**. Los perfiles describen patrones estadísticos del conjunto de datos; '
           'no son un diagnóstico, no miden riesgo y **no sustituyen una valoración profesional**.')

with st.form('formulario'):
    c1, c2 = st.columns(2)
    with c1:
        edad = st.number_input('Edad (años)', min_value=int(R['age'][0]), max_value=int(R['age'][1]), value=55, step=1)
        pa = st.number_input('Presión arterial en reposo (mm Hg)', min_value=int(R['resting_bp_s'][0]),
                             max_value=int(R['resting_bp_s'][1]), value=130, step=1)
        fc = st.number_input('Frecuencia cardíaca máxima (lpm)', min_value=int(R['max_heart_rate'][0]),
                             max_value=int(R['max_heart_rate'][1]), value=130, step=1)
        oldpeak = st.number_input('Depresión del ST (oldpeak)', min_value=float(R['oldpeak'][0]),
                                  max_value=float(R['oldpeak'][1]), value=1.0, step=0.1)
    with c2:
        sin_col = st.checkbox('No dispongo del colesterol (se usará la mediana del conjunto)')
        colesterol = st.number_input('Colesterol (mg/dl)', min_value=int(R['cholesterol'][0]),
                                     max_value=int(R['cholesterol'][1]), value=240, step=1, disabled=sin_col)
        sexo = st.selectbox('Sexo', ['Hombre', 'Mujer'])
        angina = st.selectbox('Angina inducida por ejercicio', ['No', 'Sí'])
        glucemia = st.selectbox('Glucemia en ayunas alta', ['No', 'Sí'])
    enviar = st.form_submit_button('Asignar perfil')

if enviar:
    registro = {
        'age': edad, 'resting_bp_s': pa, 'cholesterol': 0 if sin_col else colesterol,
        'max_heart_rate': fc, 'oldpeak': oldpeak,
        'sex': 1 if sexo == 'Hombre' else 0,
        'exercise_angina': 1 if angina == 'Sí' else 0,
        'fasting_blood_sugar': 1 if glucemia == 'Sí' else 0,
    }
    try:
        r = asignar(registro, modelo, meta)
    except ValueError as e:
        st.error('No se pudo procesar el registro: %s' % e)
    else:
        st.subheader('Clúster %d: %s' % (r['cluster'], r['perfil']))
        st.write('Este grupo reúne a %d de los %d pacientes del conjunto de datos.' % (r['tamano_cluster'], meta['n_pacientes']))
        if r['colesterol_imputado']:
            st.info('El colesterol no se informó: el modelo usó la mediana del conjunto (237 mg/dl).')
        margen = r['distancia_segundo'] - r['distancia_propio']
        st.write('Distancia al centro de su grupo: **%.2f**; al siguiente grupo más cercano (clúster %d): **%.2f**.'
                 % (r['distancia_propio'], r['segundo_cluster'], r['distancia_segundo']))
        if margen < 0.5:
            st.warning('El paciente queda casi a la misma distancia de dos grupos (diferencia %.2f). '
                       'La separación entre grupos es débil (silueta global %.3f), así que esta asignación es poco concluyente.'
                       % (margen, meta['silueta_global']))
        prom = pd.DataFrame(meta['perfil_promedios']).T
        prom.index = ['Clúster %s' % i for i in prom.index]
        st.write('Promedios de cada grupo en el conjunto de datos (para comparar):')
        st.dataframe(prom)
        
"""Lógica de la aplicación (independiente de Streamlit para poder probarla).
Carga el MISMO pipeline guardado por el notebook final y asigna un paciente a un perfil."""
import json
import os
import joblib
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

"""Pruebas de la lógica de la aplicación (se ejecutan con: python pruebas_app.py)."""
from logica import cargar_modelo, asignar, validar_registro

modelo, meta = cargar_modelo()
R = meta['rangos_observados']
base = dict(age=55, resting_bp_s=130, cholesterol=240, max_heart_rate=130, oldpeak=1.0, sex=1, exercise_angina=1, fasting_blood_sugar=0)
filas = []

def anotar(nombre, ok, detalle):
    filas.append((nombre, 'CUMPLE' if ok else 'FALLA', detalle)); 

# 1. Entrada válida
r = asignar(base, modelo, meta)
anotar('Entrada válida', r['cluster'] == 1, 'clúster %d (%s)' % (r['cluster'], r['perfil']))

# 2. Valores límite (exactos) y justo fuera del rango
inf = dict(age=R['age'][0], resting_bp_s=R['resting_bp_s'][0], cholesterol=R['cholesterol'][0], max_heart_rate=R['max_heart_rate'][0], oldpeak=R['oldpeak'][0], sex=0, exercise_angina=0, fasting_blood_sugar=0)
sup = dict(age=R['age'][1], resting_bp_s=R['resting_bp_s'][1], cholesterol=R['cholesterol'][1], max_heart_rate=R['max_heart_rate'][1], oldpeak=R['oldpeak'][1], sex=1, exercise_angina=1, fasting_blood_sugar=1)
ri, rs = asignar(inf, modelo, meta), asignar(sup, modelo, meta)
anotar('Límite inferior', True, 'procesa; clúster %d' % ri['cluster'])
anotar('Límite superior', True, 'procesa; clúster %d' % rs['cluster'])
for campo, valor in [('age', 27), ('age', 78), ('resting_bp_s', 79), ('max_heart_rate', 203), ('oldpeak', 6.3)]:
    e = validar_registro({**base, campo: valor}, meta)
    anotar('Fuera de rango: %s=%s' % (campo, valor), len(e) == 1, e[0] if e else 'NO advirtió')

# 3. Entrada incompleta o inválida
e = validar_registro({k: v for k, v in base.items() if k != 'age'}, meta); anotar('Campo omitido (age)', len(e) == 1, e[0] if e else 'NO advirtió')
e = validar_registro({**base, 'age': 'abc'}, meta); anotar('Texto en campo numérico', len(e) == 1, e[0] if e else 'NO advirtió')
e = validar_registro({**base, 'sex': 2}, meta); anotar('Binaria con valor 2', len(e) == 1, e[0] if e else 'NO advirtió')
try:
    asignar({**base, 'age': 200}, modelo, meta); anotar('asignar() con dato inválido', False, 'no lanzó error')
except ValueError as ex:
    anotar('asignar() con dato inválido', True, 'lanza ValueError controlado (la app lo muestra como mensaje)')

# 4. Colesterol no disponible = mediana
a, b = asignar({**base, 'cholesterol': 0}, modelo, meta), asignar({**base, 'cholesterol': 237}, modelo, meta)
anotar('Colesterol 0 equivale a mediana 237', a['cluster'] == b['cluster'] and abs(a['distancia_propio'] - b['distancia_propio']) < 1e-9, 'clúster %d en ambos; distancia %.4f' % (a['cluster'], a['distancia_propio']))

# 5. Consistencia con el notebook (filas 0, 100 y 500 del conjunto depurado; Tabla 19 del notebook: 2, 1, 1)
reales = {0: (dict(age=40, resting_bp_s=140, cholesterol=289, max_heart_rate=172, oldpeak=0.0, sex=1, exercise_angina=0, fasting_blood_sugar=0), 2),
          100: (dict(age=65, resting_bp_s=130, cholesterol=275, max_heart_rate=115, oldpeak=1.0, sex=1, exercise_angina=1, fasting_blood_sugar=0), 1),
          500: (dict(age=63, resting_bp_s=130, cholesterol=308, max_heart_rate=138, oldpeak=2.0, sex=1, exercise_angina=1, fasting_blood_sugar=0), 1)}
for fila, (reg, esperado) in reales.items():
    c = asignar(reg, modelo, meta)['cluster']
    anotar('Consistencia notebook/app, fila %d' % fila, c == esperado, 'app=%d, notebook=%d' % (c, esperado))

print('%-48s %-8s %s' % ('Prueba', 'Estado', 'Detalle'))
for f in filas: print('%-48s %-8s %s' % f)
print('\nResumen: %d de %d pruebas cumplen.' % (sum(f[1] == 'CUMPLE' for f in filas), len(filas)))
