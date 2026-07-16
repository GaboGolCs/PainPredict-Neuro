"""
Validación de temperatura para ventanas de clase.

Comprueba que:
  - class_id = 5 (dolor máximo) coincida con picos de temperatura (~50°C)
  - class_id = 4 (base) coincida con rangos base (~32°C)

Uso:
    from funcionTemp import validate_temperature_ground_truth
    result = validate_temperature_ground_truth(base_path='/ruta/proyecto')
"""

from pathlib import Path
import pandas as pd
import numpy as np
import datetime


def validate_temperature_ground_truth(base_path, peak_threshold=50.0, base_temp_range=(30.0, 34.0)):
    """
    Valida Ground Truth de temperatura: Lee archivos de la carpeta temperature,
    crea ventanas de clase y comprueba que coincidan con picos (class_id=5, ~50°C)
    y base (class_id=4, ~32°C).

    Args:
        base_path: Ruta base del proyecto (que contiene data/temperature/)
        peak_threshold: Umbral de temperatura para picos (°C), por defecto 50°C
        base_temp_range: Rango de temperatura para base (min, max), por defecto (30°C, 34°C)

    Returns:
        DataFrame con columnas:
          - start_ts, end_ts: Rango temporal de la ventana
          - class_id: Clase asignada (4=base, 5=pico)
          - max_temp, mean_temp, min_temp: Estadísticas de temperatura
          - is_valid: True si coincide con el patrón esperado (class_id matchea con temperatura)
    """
    base_path = Path(base_path)
    temp_dir = base_path / 'data' / 'temperature'

    # 1. Cargar archivos de temperatura
    print(f"Cargando archivos de temperatura desde {temp_dir}...")
    all_dfs = []

    for subject_dir in sorted(temp_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject_name = subject_dir.name
        csv_files = sorted(subject_dir.glob('*.csv'))

        if not csv_files:
            continue

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, sep='\t')
                df['subject'] = subject_name
                all_dfs.append(df)
                print(f"  [OK] {subject_name}/{csv_file.name} ({len(df)} registros)")
            except Exception as exc:
                print(f"  [ERROR] {subject_name}/{csv_file.name} - {exc}")

    if not all_dfs:
        raise ValueError(f"No CSV files found in {temp_dir}")

    temp_data = pd.concat(all_dfs, ignore_index=True)
    print(f"✓ Total de registros de temperatura: {len(temp_data)}\n")

    # 2. Crear ventanas de clase basadas en rangos de temperatura
    # Asumimos columnas: 'time' (timestamp) y 'temperature' (valor)
    if 'time' not in temp_data.columns or 'temperature' not in temp_data.columns:
        raise ValueError("CSV de temperatura debe tener columnas 'time' y 'temperature'")

    # Preparar tiempo (convertir a Timedelta si es numérico, asumir microsegundos)
    if pd.api.types.is_numeric_dtype(temp_data['time']):
        temp_data['time'] = pd.to_timedelta(temp_data['time'], unit='us')
    else:
        temp_data['time'] = pd.to_datetime(temp_data['time'])

    temp_data = temp_data.sort_values('time').drop_duplicates(subset=['time'])

    # Crear 10 ventanas equidistantes
    tmin = temp_data['time'].min()
    tmax = temp_data['time'].max()
    edges = np.linspace(tmin.total_seconds() if hasattr(tmin, 'total_seconds') else 0,
                        tmax.total_seconds() if hasattr(tmax, 'total_seconds') else 1,
                        11)

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

        # Asignar clase según temperatura
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

    # 3. Validar correspondencia clase-temperatura
    def validate_class(row):
        """Verifica que la clase coincida con el patrón de temperatura esperado."""
        if row['class_id'] == 5:  # Pico esperado (~50°C)
            return row['max_temp'] >= peak_threshold
        elif row['class_id'] == 4:  # Base esperada (~32°C)
            return base_temp_range[0] <= row['mean_temp'] <= base_temp_range[1]
        return False

    result_df['is_valid'] = result_df.apply(validate_class, axis=1)

    # Formatear tiempos para legibilidad
    def _format_time(val):
        if pd.isna(val):
            return ''
        if isinstance(val, (pd.Timedelta, np.timedelta64)):
            td = pd.to_timedelta(val)
            total_seconds = td.total_seconds()
            hrs = int(total_seconds // 3600)
            mins = int((total_seconds % 3600) // 60)
            secs = total_seconds % 60
            return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"
        return str(val)

    result_df['start_ts'] = result_df['start_ts'].apply(_format_time)
    result_df['end_ts'] = result_df['end_ts'].apply(_format_time)

    # Resumen
    valid_count = result_df['is_valid'].sum()
    print("\n" + "=" * 70)
    print("VALIDACIÓN DE TEMPERATURA - GROUND TRUTH")
    print("=" * 70)
    print(f"\nTotal ventanas: {len(result_df)}")
    print(f"Ventanas válidas (clase matchea): {valid_count}/{len(result_df)}")
    print(f"\nClass 5 (Pico ~{peak_threshold}°C): {(result_df['class_id'] == 5).sum()} ventanas")
    print(f"Class 4 (Base {base_temp_range[0]}-{base_temp_range[1]}°C): {(result_df['class_id'] == 4).sum()} ventanas")
    print("\n" + result_df.to_string(index=False))
    print("\n" + "=" * 70 + "\n")

    return result_df



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validación de temperatura - Ground Truth')
    parser.add_argument('--base', required=True, help='Ruta base del proyecto')
    parser.add_argument('--out', default='temperature_validation_out.csv', help='Archivo de salida')
    parser.add_argument('--peak-threshold', type=float, default=50.0, help='Umbral de pico (°C)')
    parser.add_argument('--base-range', type=float, nargs=2, default=[30.0, 34.0], help='Rango de base (min max)')
    
    args = parser.parse_args()

    result = validate_temperature_ground_truth(
        base_path=args.base,
        peak_threshold=args.peak_threshold,
        base_temp_range=tuple(args.base_range)
    )
    
    result.to_csv(args.out, index=False)
    print(f"\n[OK] Validación guardada en: {args.out}")

