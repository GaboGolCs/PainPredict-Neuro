"""
Script de prueba para validación de temperatura - Ground Truth
"""

from pathlib import Path
from funcionTemp import validate_temperature_ground_truth


def main():
    """Ejecuta la validación de temperatura."""
    base = Path(__file__).resolve().parents[1]

    print("=" * 70)
    print("VALIDACIÓN DE TEMPERATURA - GROUND TRUTH")
    print("=" * 70 + "\n")

    # Ejecutar validación
    result_df = validate_temperature_ground_truth(
        base_path=base,
        peak_threshold=50.0,
        base_temp_range=(30.0, 34.0)
    )

    # Guardar resultados
    out = base / 'temperature_validation_test_out.csv'
    result_df.to_csv(out, index=False)
    print(f"[OK] Resultados guardados en: {out}\n")


if __name__ == '__main__':
    main()
