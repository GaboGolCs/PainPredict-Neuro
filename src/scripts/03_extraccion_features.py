# -*- coding: utf-8 -*-
"""
SCRIPT 3: EXTRACCIÓN DE CARACTERÍSTICAS (ÍTEM 3) - Sin Temperature
Descripción: Bucle iterativo sobre rutas validadas, lectura en memoria
             desde el archivo ZIP, extracción de features (ECG, GSR, EMG)
             y guardado progresivo en disco para evitar el colapso de RAM.
"""

import sys
sys.path.append('/content/drive/Shareddrives/GPI/')

import pandas as pd
import zipfile
import os
from tqdm import tqdm
import signal_processor as sp

# --- CONFIGURACIÓN DE RUTAS ---
ZIP_SEÑALES = '/content/drive/Shareddrives/GPI/biosignals_filtered.zip'
CSV_RUTAS   = '/content/drive/Shareddrives/GPI/rutas.csv'
CSV_SALIDA  = '/content/features_consolidadas.csv'


def motor_principal_extraccion(zip_path, csv_rutas, csv_salida, batch_size=100):
    print("\n" + "=" * 70)
    print("INICIANDO PROCESO DE EXTRACCIÓN Y GUARDADO POR LOTES")
    print("=" * 70)

    try:
        df_maestro = pd.read_csv(csv_rutas)
        total_archivos = len(df_maestro)
        print(f"[-] Total de archivos a procesar: {total_archivos} desde {csv_rutas}.")
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo de rutas en: {csv_rutas}")
        return

    header_written = os.path.exists(csv_salida)
    lote_resultados = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        for index, row in tqdm(df_maestro.iterrows(), total=total_archivos, desc="Procesando Lotes"):
            ruta_interna = row['ruta']
            info_paciente = row.to_dict()

            try:
                with zip_ref.open(ruta_interna) as f:
                    df_signal = pd.read_csv(f, sep='\t')  # CAMBIO CRÍTICO: sep='\t'

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
                info_paciente['error_procesamiento'] = str(e)

            lote_resultados.append(info_paciente)

            if len(lote_resultados) >= batch_size:
                df_lote = pd.DataFrame(lote_resultados)
                df_lote.to_csv(csv_salida, mode='a', header=not header_written, index=False)
                header_written = True
                lote_resultados = []

        if lote_resultados:
            df_lote = pd.DataFrame(lote_resultados)
            df_lote.to_csv(csv_salida, mode='a', header=not header_written, index=False)

    print("\n" + "=" * 70)
    print(f"[ÉXITO] Proceso finalizado. Datos consolidados guardados en: {csv_salida}")
    print("=" * 70)


# =====================================================================
# EJECUCIÓN DEL SCRIPT
# =====================================================================
if __name__ == "__main__":
    motor_principal_extraccion(
        zip_path=ZIP_SEÑALES,
        csv_rutas=CSV_RUTAS,
        csv_salida=CSV_SALIDA,
        batch_size=100
    )
