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
