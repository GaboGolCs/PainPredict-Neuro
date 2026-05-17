# -*- coding: utf-8 -*-
"""
SCRIPT 2: VALIDACIÓN DE TEMPERATURA GROUND TRUTH (VERSIÓN MEMORIA)
Descripción: Extrae, consolida y valida señales térmicas desde archivos ZIP
             sin realizar escrituras intermedias en disco.
"""

import zipfile
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN DE RUTAS ---
zip_path = '/content/drive/Shareddrives/GPI/temperature.zip'
base_colab_path = '/content'


def validate_temperature_ground_truth(zip_target, peak_threshold=50.0, base_temp_range=(30.0, 34.0)):
    """
    Analiza archivos de temperatura dentro de un ZIP para validar integridad de datos.

    Args:
        zip_target (str): Ruta al archivo comprimido.
        peak_threshold (float): Umbral para clasificar picos térmicos (Clase 5).
        base_temp_range (tuple): Rango esperado para temperatura base (Clase 4).
    """
    all_dfs = []

    print("\n" + "=" * 70)
    print("INICIANDO PROTOCOLO DE EXTRACCIÓN DIRECTA EN MEMORIA")
    print("=" * 70)

    try:
        # FASE 1: INFILTRACIÓN Y EXTRACCIÓN
        with zipfile.ZipFile(zip_target, 'r') as zip_ref:

            archivos_csv = [f for f in zip_ref.namelist() if f.endswith('.csv') and '__MACOSX' not in f]

            if not archivos_csv:
                raise ValueError("No se detectaron archivos CSV en el objetivo.")

            print(f"[+] Se detectaron {len(archivos_csv)} archivos de señales. Procesando al vuelo...")

            for file_path in archivos_csv:
                try:
                    partes_ruta = file_path.split('/')
                    subject_name = partes_ruta[-2] if len(partes_ruta) >= 2 else "Sujeto_Desconocido"

                    with zip_ref.open(file_path) as f:
                        df = pd.read_csv(f, sep='\t')
                        df['subject'] = subject_name
                        all_dfs.append(df)

                except Exception as exc:
                    print(f"  [ERROR] Falla de lectura en sector {file_path} - {exc}")

    except FileNotFoundError:
        raise ValueError(f"CRÍTICO: No se encontró el archivo ZIP en {zip_target}.")

    if not all_dfs:
        raise ValueError("Fallo en la consolidación. Bases de datos vacías.")

    temp_data = pd.concat(all_dfs, ignore_index=True)
    print(f"✓ Extracción completada. Total de registros: {len(temp_data)}\n")

    # =================================================================
    # FASE 2: ANÁLISIS DE SEÑAL Y VENTANEO TEMPORAL
    # =================================================================

    if 'time' not in temp_data.columns or 'temperature' not in temp_data.columns:
        raise ValueError("Faltan columnas vitales ('time' o 'temperature').")

    if pd.api.types.is_numeric_dtype(temp_data['time']):
        temp_data['time'] = pd.to_timedelta(temp_data['time'], unit='us')
    else:
        temp_data['time'] = pd.to_datetime(temp_data['time'])

    temp_data = temp_data.sort_values('time').drop_duplicates(subset=['time'])

    tmin = temp_data['time'].min()
    tmax = temp_data['time'].max()

    start_sec = tmin.total_seconds() if hasattr(tmin, 'total_seconds') else 0
    end_sec = tmax.total_seconds() if hasattr(tmax, 'total_seconds') else 1
    edges = np.linspace(start_sec, end_sec, 11)

    windows = []
    for i in range(10):
        start = pd.to_timedelta(edges[i], unit='s')
        end = pd.to_timedelta(edges[i + 1], unit='s')

        seg = temp_data[(temp_data['time'] >= start) & (temp_data['time'] <= end)]

        if seg.empty:
            continue

        max_temp = seg['temperature'].max()
        mean_temp = seg['temperature'].mean()
        min_temp = seg['temperature'].min()

        class_id = 5 if max_temp >= peak_threshold else 4

        windows.append({
            'start_ts': start,
            'end_ts': end,
            'class_id': int(class_id),
            'max_temp': float(max_temp),
            'mean_temp': float(mean_temp),
            'min_temp': float(min_temp)
        })

    result_df = pd.DataFrame(windows)

    # FASE 3: PROTOCOLO DE VALIDACIÓN
    def validate_class(row):
        """Verifica si los valores térmicos corresponden a la clase asignada."""
        if row['class_id'] == 5:
            return row['max_temp'] >= peak_threshold
        elif row['class_id'] == 4:
            return base_temp_range[0] <= row['mean_temp'] <= base_temp_range[1]
        return False

    result_df['is_valid'] = result_df.apply(validate_class, axis=1)

    def _format_time(val):
        if pd.isna(val): return ''
        if isinstance(val, (pd.Timedelta, np.timedelta64)):
            td = pd.to_timedelta(val)
            ts = td.total_seconds()
            return f"{int(ts//3600):02d}:{int((ts%3600)//60):02d}:{ts%60:06.3f}"
        return str(val)

    result_df['start_ts'] = result_df['start_ts'].apply(_format_time)
    result_df['end_ts'] = result_df['end_ts'].apply(_format_time)

    print("\n" + "=" * 70)
    print("REPORTE FINAL: VALIDACIÓN DE TEMPERATURA - GROUND TRUTH")
    print("=" * 70)
    print(f"\nVentanas válidas: {result_df['is_valid'].sum()}/{len(result_df)}")
    print(f"Picos detectados: {(result_df['class_id'] == 5).sum()}")
    print("\n" + result_df.to_string(index=False))

    return result_df


# =====================================================================
# EJECUCIÓN DEL SCRIPT
# =====================================================================
if __name__ == "__main__":
    try:
        resultados = validate_temperature_ground_truth(
            zip_target=zip_path,
            peak_threshold=50.0,
            base_temp_range=(30.0, 34.0)
        )

        out_file = f'{base_colab_path}/temperature_validation_test_out.csv'
        resultados.to_csv(out_file, index=False)
        print(f"\n[EXITO] Misión cumplida. Resultados en: {out_file}")

    except Exception as e:
        print(f"\n[ALERTA CRÍTICA] Misión abortada: {e}")
