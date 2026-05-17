# -*- coding: utf-8 -*-
"""
SCRIPT 4: PROCESAMIENTO - EXTRACCIÓN DE CARACTERÍSTICAS Y CRUCE DE DATOS (ÍTEM 3)
Descripción: Cruza las coincidencias encontradas en el reporte de temperatura
             con los datos de dolor. Rellena vacíos con 0.
             Genera el archivo final: dataset_features_biovid.csv
"""

import sys
sys.path.append('/content/drive/Shareddrives/GPI/')

import pandas as pd
import zipfile
import os
from tqdm import tqdm
import signal_processor as sp

# --- 1. Rutas de archivos ---
ZIP_SEÑALES     = '/content/drive/Shareddrives/GPI/biosignals_filtered.zip'
CSV_SAMPLES     = '/content/drive/Shareddrives/GPI/samples.csv'
CSV_TEMPERATURA = '/content/reporte_validacion_termica.csv'
CSV_SALIDA      = '/content/dataset_features_biovid.csv'


def motor_definitivo(batch_size=100):
    print("\n" + "=" * 70)
    print("INICIANDO PROCESAMIENTO: ÍTEM 3 (BIOSEÑALES + TEMPERATURA)")
    print("=" * 70)

    # --- 2. CARGA Y COMBINACIÓN DE DATOS BASE ---
    try:
        df_maestro = pd.read_csv(CSV_SAMPLES, sep='\t')
        print(f"[+] Archivo base (samples.csv) cargado: {len(df_maestro)} registros.")

        if os.path.exists(CSV_TEMPERATURA):
            df_temp = pd.read_csv(CSV_TEMPERATURA)
            columnas_temp = ['sample_name', 'max_temp', 'mean_temp']
            df_temp = df_temp[columnas_temp]
            df_maestro = pd.merge(df_maestro, df_temp, on='sample_name', how='left')
            print(f"[+] Datos de temperatura combinados exitosamente.")
        else:
            print(f"[-] ADVERTENCIA: No se encontró {CSV_TEMPERATURA}. Se procederá sin datos térmicos.")

    except Exception as e:
        print(f"[ERROR CRÍTICO] Error al cargar los datos base: {e}")
        return

    total_archivos = len(df_maestro)
    header_written = os.path.exists(CSV_SALIDA)
    lote_resultados = []

    # --- 3. LECTURA Y PROCESAMIENTO DE SEÑALES ---
    try:
        with zipfile.ZipFile(ZIP_SEÑALES, 'r') as zip_ref:
            archivos_en_zip = zip_ref.namelist()

            for index, row in tqdm(df_maestro.iterrows(), total=total_archivos, desc="Procesando Señales"):

                subject_name = row['subject_name']
                sample_name = row['sample_name']
                ruta_interna = f"biosignals_filtered/{subject_name}/{sample_name}_bio.csv"

                info_paciente = row.to_dict()

                if ruta_interna not in archivos_en_zip:
                    archivos_coincidentes = [f for f in archivos_en_zip if f.endswith(f"{sample_name}_bio.csv")]
                    if archivos_coincidentes:
                        ruta_interna = archivos_coincidentes[0]
                    else:
                        info_paciente['error_procesamiento'] = "Archivo físico no encontrado"
                        lote_resultados.append(info_paciente)
                        continue

                try:
                    with zip_ref.open(ruta_interna) as f:
                        df_signal = pd.read_csv(f, sep='\t')

                    # ECG
                    if 'ecg' in df_signal.columns:
                        feats_ecg = sp.extract_ecg_features(df_signal['ecg'].values)
                        info_paciente.update(feats_ecg)

                    # GSR
                    if 'gsr' in df_signal.columns:
                        feats_gsr = sp.extract_gsr_features(df_signal['gsr'].values)
                        info_paciente.update(feats_gsr)

                    # EMG (3 músculos)
                    columnas_emg = ['emg_trapezius', 'emg_corrugator', 'emg_zygomaticus']
                    for col_emg in columnas_emg:
                        if col_emg in df_signal.columns:
                            feats_emg = sp.extract_emg_features(df_signal[col_emg].values)
                            feats_renombradas = {f"{col_emg}_{k.replace('emg_', '')}": v for k, v in feats_emg.items()}
                            info_paciente.update(feats_renombradas)

                except Exception as e:
                    info_paciente['error_procesamiento'] = f"Falla de cálculo: {str(e)}"

                lote_resultados.append(info_paciente)

                # --- 4. GUARDADO PROGRESIVO POR LOTES (CON RELLENO DE VACÍOS) ---
                if len(lote_resultados) >= batch_size:
                    df_lote = pd.DataFrame(lote_resultados)
                    df_lote = df_lote.fillna(0)
                    df_lote.to_csv(CSV_SALIDA, mode='a', header=not header_written, index=False)
                    header_written = True
                    lote_resultados = []

            if lote_resultados:
                df_lote = pd.DataFrame(lote_resultados)
                df_lote = df_lote.fillna(0)
                df_lote.to_csv(CSV_SALIDA, mode='a', header=not header_written, index=False)

    except FileNotFoundError:
        print(f"[ERROR CRÍTICO] Archivo ZIP no encontrado en: {ZIP_SEÑALES}")
        return

    # --- 5. VALIDACIÓN Y RESUMEN FINAL ---
    print("\n" + "=" * 70)
    print(f"[ÉXITO] Procesamiento del Ítem 3 completado.")
    print(f"Dataset final guardado en: {CSV_SALIDA}")
    print("=" * 70)

    df_verificacion = pd.read_csv(CSV_SALIDA)
    print(f"Registros Totales: {len(df_verificacion)}")
    print(f"Métricas extraídas:\n {list(df_verificacion.columns)}")


# =====================================================================
# EJECUCIÓN DEL SCRIPT
# =====================================================================
if __name__ == "__main__":
    motor_definitivo(batch_size=100)
