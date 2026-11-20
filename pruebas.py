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
