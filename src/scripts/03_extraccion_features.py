# -*- coding: utf-8 -*-
"""
SCRIPT 3: EXTRACCIÓN DE CARACTERÍSTICAS (ÍTEM 3)
Descripción: Bucle iterativo sobre rutas validadas, lectura en memoria
             desde el archivo ZIP, extracción de features (usando el
             módulo signal_processor) y guardado progresivo en disco
             para evitar el colapso de la memoria RAM.
"""

import sys
# Agregamos la ruta al path para poder importar signal_processor.py
sys.path.append('/content/drive/Shareddrives/GPI/')

import pandas as pd
import zipfile
import os
from tqdm import tqdm  # Barra de progreso visual
import signal_processor as sp  # Módulo de procesamiento de bioseñales

# --- CONFIGURACIÓN DE RUTAS ---
ZIP_SEÑALES = '/content/drive/Shareddrives/GPI/biosignals_filtered.zip'
CSV_RUTAS = '/content/drive/Shareddrives/GPI/rutas.csv'  # Archivo generado en el paso anterior
CSV_SALIDA = '/content/features_consolidadas.csv'


def motor_principal_extraccion(zip_path, csv_rutas, csv_salida, batch_size=100):
    """
    Ejecuta el procesamiento masivo de bioseñales leyendo directamente desde un ZIP.
    """
    print("\n" + "=" * 70)
    print("INICIANDO PROCESO DE EXTRACCIÓN Y GUARDADO POR LOTES")
    print("=" * 70)

    # 1. Carga del dataframe con las rutas válidas
    try:
        df_maestro = pd.read_csv(csv_rutas)
        total_archivos = len(df_maestro)
        print(f"[-] Total de archivos a procesar: {total_archivos} desde {csv_rutas}.")
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo de rutas en: {csv_rutas}")
        return

    # Variable para controlar si ya se escribieron los encabezados en el CSV final
    header_written = os.path.exists(csv_salida)
    lote_resultados = []

    # 2. Lectura del archivo ZIP (se abre una sola vez para mejorar el rendimiento)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        for index, row in tqdm(df_maestro.iterrows(), total=total_archivos, desc="Procesando Lotes"):
            ruta_interna = row['ruta']
            info_paciente = row.to_dict()

            try:
                with zip_ref.open(ruta_interna) as f:
                    # NOTA: Verificar si el separador del CSV es ',' o '\t'
                    df_signal = pd.read_csv(f)

                # --- EXTRACCIÓN DE CARACTERÍSTICAS ---
                # Importante: Verificar que los nombres de las columnas coincidan ('EMG', 'ECG', 'EDA')

                if 'EMG' in df_signal.columns:
                    feats_emg = sp.extract_emg_features(df_signal['EMG'].values)
                    info_paciente.update(feats_emg)

                if 'ECG' in df_signal.columns:
                    feats_ecg = sp.extract_ecg_features(df_signal['ECG'].values)
                    info_paciente.update(feats_ecg)

                if 'EDA' in df_signal.columns:
                    feats_gsr = sp.extract_gsr_features(df_signal['EDA'].values)
                    info_paciente.update(feats_gsr)

            except Exception as e:
                info_paciente['error_procesamiento'] = str(e)

            lote_resultados.append(info_paciente)

            # 3. GUARDADO POR LOTES (Para liberar memoria RAM en Colab)
            if len(lote_resultados) >= batch_size:
                df_lote = pd.DataFrame(lote_resultados)
                # mode='a' (append) permite añadir filas al final del archivo existente
                df_lote.to_csv(csv_salida, mode='a', header=not header_written, index=False)
                header_written = True
                lote_resultados = []

        # 4. Guardar los registros restantes que no completaron un lote exacto
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
