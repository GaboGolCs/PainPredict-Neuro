# -*- coding: utf-8 -*-
"""
SCRIPT 2: VALIDACIÓN DE TEMPERATURA GROUND TRUTH
Descripción: Lee los archivos de temperatura desde el ZIP, hace merge con
             samples.csv y valida que clase 4 coincida con picos >= 48°C
             y clase 0 con temperatura base entre 30-34°C.
             Exporta el resultado a reporte_validacion_termica.csv
"""

import pandas as pd
import zipfile
import os

# --- CONFIGURACIÓN DE RUTAS ---
path_samples = '/content/drive/Shareddrives/GPI/samples.csv'
zip_path = '/content/drive/Shareddrives/GPI/temperature.zip'
path_salida_csv = '/content/reporte_validacion_termica.csv'


def ejecutar_validacion_y_exportar(ruta_salida):
    print("Iniciando extracción al vuelo, cruce de datos y exportación...")

    # --- 2. Carga del archivo principal (Ground Truth) ---
    try:
        df_samples = pd.read_csv(path_samples, sep='\t')
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo en {path_samples}")
        return None

    # --- 3. Lectura directa desde el archivo ZIP ---
    stats_list = []

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            archivos_csv = [f for f in zip_ref.namelist() if f.endswith('_temp.csv') and '__MACOSX' not in f]

            if not archivos_csv:
                print("Error: No se encontraron archivos de temperatura dentro del ZIP.")
                return None

            for file_path in archivos_csv:
                filename = os.path.basename(file_path)
                sample_name = filename.replace('_temp.csv', '')

                with zip_ref.open(file_path) as f:
                    try:
                        df_t = pd.read_csv(f, sep='\t', on_bad_lines='skip')
                        df_t.columns = df_t.columns.str.strip().str.lower()

                        if 'temperature' not in df_t.columns:
                            continue

                        stats_list.append({
                            'sample_name': sample_name,
                            'max_temp': df_t['temperature'].max(),
                            'mean_temp': df_t['temperature'].mean()
                        })
                    except Exception:
                        continue

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo ZIP en {zip_path}")
        return None

    df_temp_stats = pd.DataFrame(stats_list)

    # --- 4. El Cruce de Datos (Merge) ---
    df_merged = pd.merge(df_samples, df_temp_stats, on='sample_name', how='inner')

    # --- 5. Validación de los resultados ---
    df_merged['valid_pico_c4'] = (df_merged['class_id'] == 4) & (df_merged['max_temp'] >= 48.0)
    df_merged['valid_base_c0'] = (df_merged['class_id'] == 0) & (df_merged['mean_temp'].between(30.0, 34.0))

    # --- 6. Mostrar resultados en consola ---
    total_c4 = (df_merged['class_id'] == 4).sum()
    total_c0 = (df_merged['class_id'] == 0).sum()
    picos_confirmados = df_merged['valid_pico_c4'].sum()
    bases_confirmadas = df_merged['valid_base_c0'].sum()

    print("\n" + "=" * 50)
    print("REPORTE DE VALIDACIÓN")
    print("=" * 50)
    print(f"Total de muestras procesadas con éxito: {len(df_merged)}")
    print("-" * 50)
    print(f"CLASE 4 (Dolor Máximo - aprox 50°C):")
    print(f"  > Picos validados correctamente: {picos_confirmados} de {total_c4}")
    print("-" * 50)
    print(f"CLASE 0 (Base - aprox 32°C):")
    print(f"  > Bases validadas correctamente: {bases_confirmadas} de {total_c0}")
    print("=" * 50)

    # --- 7. EXPORTACIÓN DEL ARCHIVO CSV ---
    resultado_final = df_merged[df_merged['class_id'].isin([0, 4])].copy()

    try:
        resultado_final.to_csv(ruta_salida, index=False)
        print(f"\n[+] ÉXITO: Archivo CSV generado en: {ruta_salida}")
    except Exception as e:
        print(f"\n[-] ERROR AL EXPORTAR: {e}")

    return resultado_final


# =====================================================================
# EJECUCIÓN DEL SCRIPT
# =====================================================================
if __name__ == "__main__":
    df_validacion = ejecutar_validacion_y_exportar(path_salida_csv)
