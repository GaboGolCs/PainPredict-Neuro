# -*- coding: utf-8 -*-
"""
SCRIPT 1: VALIDACIÓN DE RUTAS
Descripción: Recorre el archivo ZIP de bioseñales, verifica que cada CSV
             sea legible y exporta las rutas válidas a un archivo rutas.csv
"""

import pandas as pd
import zipfile

# Archivo que maneja todos los datos (Punto de búsqueda). OJO hay que revisar la ruta para cada caso
ARCHIVO_ZIP = '/content/drive/Shareddrives/GPI/biosignals_filtered.zip'

# Lista vacía donde se irán guardando las rutas válidas
rutas = []

# Abre el ZIP en modo lectura
with zipfile.ZipFile(ARCHIVO_ZIP, 'r') as zip_ref:
    archivos = [f for f in zip_ref.namelist() if f.endswith('.csv')]
    print(f'Total archivos encontrados: {len(archivos)}')

    for file in archivos:
        try:
            with zip_ref.open(file) as f:
                pd.read_csv(f)
                rutas.append({'ruta': file})
        except Exception as e:
            print(f'❌ {file} — Error: {e}')

df_rutas = pd.DataFrame(rutas)
df_rutas.to_csv('rutas.csv', index=False)
print(f'[OK] rutas.csv exportado con {len(df_rutas)} rutas válidas.')


def mapeo_de_rutas(csv):
    df = pd.read_csv(csv)
    with zipfile.ZipFile(ARCHIVO_ZIP, 'r') as zip_ref:
        nombres_zip = zip_ref.namelist()
        for file in df['ruta']:
            if any(file in ruta for ruta in nombres_zip):
                print('CSV Ruta absoluta:', file)

# RECUERDA MANTENER LA RUTA SIEMPRE ACTUALIZADA
csv = '/content/rutas.csv'
mapeo_de_rutas(csv)
