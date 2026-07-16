"""
signal_processor.py
====================
Módulo de extracción de características (features) para bioseñales.
Procesa ventanas de 5.5 segundos de señales EMG, ECG y GSR.

Autor  : Gabriel Luciano Martínez Pérez — Grupo 2
Fecha  : Mayo 2025
Versión: 2.0.0

Dependencias:
    - numpy
    - scipy
    - neurokit2
"""

import numpy as np
import scipy.integrate as integrate
import neurokit2 as nk

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ──────────────────────────────────────────────────────────────────────────────

SAMPLING_RATE: int = 512          # Hz — frecuencia de muestreo del sistema
WINDOW_SECONDS: float = 5.5       # Duración de la ventana de análisis (segundos)
SECONDS_PER_MINUTE: int = 60      # Constante para conversión BPM


# ──────────────────────────────────────────────────────────────────────────────
# COMPLEJIDAD — Entropía Aproximada (ApEn) y Lempel-Ziv (LZC)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_complexity(signal: np.ndarray) -> dict:
    """
    Calcula medidas de complejidad/irregularidad de una señal.

    Utiliza la Entropía Aproximada (ApEn) para cuantificar la imprevisibilidad
    de la serie temporal, y la Complejidad de Lempel-Ziv (LZC) para cuantificar
    la aleatoriedad estructural de la señal binarizada internamente por NeuroKit2.

    Parámetros
    ----------
    signal : np.ndarray
        Array 1-D con las muestras de la señal original (sin rectificar).

    Retorna
    -------
    dict con las siguientes claves:
        apen (float) : Entropía Aproximada. 0.0 si ocurre un error.
        lzc  (float) : Complejidad de Lempel-Ziv. 0.0 si ocurre un error.

    Notas
    -----
    Ambas métricas son sensibles a ventanas muy cortas o señales planas;
    cualquier excepción interna de NeuroKit2 se captura y se retornan ceros
    para no interrumpir el pipeline de extracción.
    """
    try:
        apen, _ = nk.entropy_approximate(signal)
        lzc, _ = nk.complexity_lempelziv(signal)
        return {"apen": float(apen), "lzc": float(lzc)}
    except Exception:
        return {"apen": 0.0, "lzc": 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# EMG — Electromiografía Facial (Corrugador, Cigomático, Trapecio)
# ──────────────────────────────────────────────────────────────────────────────

def extract_emg_features(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> dict:
    """
    Extrae características estadísticas y de complejidad de una señal EMG.

    Las señales EMG oscilan alrededor de cero (señales bipolares), por lo que
    se aplica valor absoluto antes de calcular los features de amplitud,
    garantizando que se trabaje sobre la envolvente de activación muscular.
    Las métricas de cruces por cero y complejidad se calculan sobre la señal
    original (no rectificada), ya que dependen del signo y la dinámica cruda
    de la señal.

    Parámetros
    ----------
    signal : np.ndarray
        Array 1-D con las muestras de la señal EMG en microvolts (µV).
        Se espera una ventana de ~5.5 s, es decir ~2816 muestras a 512 Hz.
    sampling_rate : int, opcional
        Frecuencia de muestreo en Hz. Por defecto 512 Hz.

    Retorna
    -------
    dict con las siguientes claves:
        emg_max   (float) : Pico máximo de amplitud rectificada [µV].
        emg_mean  (float) : Media de la señal rectificada [µV].
        emg_std   (float) : Desviación estándar de la señal rectificada [µV].
        emg_auc   (float) : Área bajo la curva — energía total [µV·s].
                            Calculada con integración trapezoidal sobre el
                            eje temporal real (muestras / sampling_rate).
        emg_zcr   (float)   : Número de cruces por cero (cambios de signo entre
                            muestras consecutivas) en la señal original.
                            Refleja la frecuencia de oscilación de la señal.
        emg_apen  (float) : Entropía Aproximada de la señal original.
        emg_lzc   (float) : Complejidad de Lempel-Ziv de la señal original.

    Excepciones
    -----------
    ValueError : Si `signal` está vacío o no es un array numérico 1-D.

    Ejemplo
    -------
    >>> import numpy as np
    >>> signal = np.random.randn(2816) * 50   # señal ficticia a 512 Hz
    >>> feats = extract_emg_features(signal)
    >>> print(feats)
    {'emg_max': ..., 'emg_mean': ..., 'emg_std': ..., 'emg_auc': ...,
     'emg_zcr': ..., 'emg_apen': ..., 'emg_lzc': ...}
    """
    # ── Validación de entrada ──────────────────────────────────────────────
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(
            "extract_emg_features: 'signal' debe ser un array 1-D no vacío."
        )

    # ── Rectificación (valor absoluto) ────────────────────────────────────
    rectified = np.abs(signal)

    # ── Eje temporal en segundos ───────────────────────────────────────────
    time_axis = np.arange(len(rectified)) / sampling_rate

    # ── Cálculo de features de amplitud (v1.0.0) ──────────────────────────
    emg_max  = float(np.max(rectified))
    emg_mean = float(np.mean(rectified))
    emg_std  = float(np.std(rectified, ddof=0))
    emg_auc  = float(integrate.trapezoid(rectified, time_axis))

    # ── Cruces por cero (sobre la señal original, sin rectificar) ─────────
    # Fórmula usada en la plantilla oficial del profesor
    emg_zcr = float(((signal[:-1] * signal[1:]) < 0).sum())

    # ── Complejidad (ApEn, LZC) sobre la señal original ───────────────────
    complexity = _extract_complexity(signal)

    return {
        "emg_max":  emg_max,
        "emg_mean": emg_mean,
        "emg_std":  emg_std,
        "emg_auc":  emg_auc,
        "emg_zcr":  emg_zcr,
        "emg_apen": complexity["apen"],
        "emg_lzc":  complexity["lzc"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# ECG — Electrocardiografía (Corazón)
# ──────────────────────────────────────────────────────────────────────────────

def extract_ecg_features(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    window_seconds: float = WINDOW_SECONDS
) -> dict:
    """
    Extrae características de frecuencia cardíaca, variabilidad y complejidad
    a partir de una señal ECG.

    Utiliza NeuroKit2 para detectar los picos R del complejo QRS. A partir de
    los intervalos R-R (tiempo entre latidos consecutivos) se derivan métricas
    de variabilidad cardíaca (RMSSD, rango R-R), además de la frecuencia
    cardíaca media extrapolada a BPM y medidas de complejidad de la señal cruda.

    Parámetros
    ----------
    signal : np.ndarray
        Array 1-D con las muestras de la señal ECG en milivoltios (mV).
        Se espera una ventana de ~5.5 s, es decir ~2816 muestras a 512 Hz.
    sampling_rate : int, opcional
        Frecuencia de muestreo en Hz. Por defecto 512 Hz.
    window_seconds : float, opcional
        Duración real de la ventana en segundos. Por defecto 5.5 s.

    Retorna
    -------
    dict con las siguientes claves:
        ecg_bpm      (float) : Frecuencia cardíaca media extrapolada a BPM.
                               Si no se detectan picos R, retorna 0.0.
        ecg_rmssd    (float) : Raíz cuadrada de la media de las diferencias al
                               cuadrado entre intervalos R-R consecutivos
                               (en segundos). Mide la variabilidad de corto
                               plazo asociada al tono parasimpático.
                               0.0 si hay menos de 2 latidos o menos de
                               2 intervalos R-R.
        ecg_rr_range (float) : Diferencia entre el intervalo R-R máximo y el
                               mínimo (en segundos). Mide la dispersión total
                               de la variabilidad cardíaca. 0.0 si hay menos
                               de 2 latidos.
        ecg_apen     (float) : Entropía Aproximada de la señal ECG original.
        ecg_lzc      (float) : Complejidad de Lempel-Ziv de la señal ECG
                               original.

    Excepciones
    -----------
    ValueError : Si `signal` está vacío o no es un array numérico 1-D.

    Nota
    ----
    La extrapolación de BPM es: BPM = (n_latidos / window_seconds) * 60

    Ejemplo
    -------
    >>> import numpy as np
    >>> signal = np.random.randn(2816) * 0.5
    >>> feats = extract_ecg_features(signal)
    >>> print(feats)
    {'ecg_bpm': ..., 'ecg_rmssd': ..., 'ecg_rr_range': ...,
     'ecg_apen': ..., 'ecg_lzc': ...}
    """
    # ── Validación de entrada ──────────────────────────────────────────────
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(
            "extract_ecg_features: 'signal' debe ser un array 1-D no vacío."
        )

    # ── Detección de picos R con NeuroKit2 ────────────────────────────────
    try:
        _, r_peaks_info = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
        r_peak_indices = r_peaks_info.get("ECG_R_Peaks", np.array([]))
        r_peak_indices = np.asarray(r_peak_indices)
        r_peak_indices = r_peak_indices[r_peak_indices > 0]
        n_beats = int(len(r_peak_indices))
    except Exception:
        # Si NeuroKit2 falla (señal muy corta, plana, ruidosa, etc.) → 0 latidos
        r_peak_indices = np.array([])
        n_beats = 0

    # ── Extrapolación a BPM ───────────────────────────────────────────────
    if n_beats == 0 or window_seconds <= 0:
        ecg_bpm = 0.0
    else:
        ecg_bpm = float((n_beats / window_seconds) * SECONDS_PER_MINUTE)

    # ── Intervalos R-R (en segundos) y variabilidad ───────────────────────
    if n_beats < 2:
        ecg_rmssd = 0.0
        ecg_rr_range = 0.0
    else:
        rr_intervals = np.diff(r_peak_indices) / sampling_rate

        # RMSSD requiere al menos 2 intervalos R-R (3 latidos)
        if len(rr_intervals) < 2:
            ecg_rmssd = 0.0
        else:
            diff_rr = np.diff(rr_intervals)
            ecg_rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))

        ecg_rr_range = float(np.max(rr_intervals) - np.min(rr_intervals))

    # ── Complejidad (ApEn, LZC) sobre la señal ECG original ───────────────
    complexity = _extract_complexity(signal)

    return {
        "ecg_bpm":      ecg_bpm,
        "ecg_rmssd":    ecg_rmssd,
        "ecg_rr_range": ecg_rr_range,
        "ecg_apen":     complexity["apen"],
        "ecg_lzc":      complexity["lzc"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# GSR — Respuesta Galvánica de la Piel / Micro-sudoración (EDA)
# ──────────────────────────────────────────────────────────────────────────────

def extract_gsr_features(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE
) -> dict:
    """
    Extrae características de activación simpática y complejidad a partir de
    una señal GSR/EDA.

    Utiliza NeuroKit2 para detectar los picos SCR (Skin Conductance Response)
    sobre el componente fásico de la señal. Además del conteo y amplitud de
    picos, se calcula el rango total de la señal, la pendiente de su tramo
    final (tendencia de activación tardía) y medidas de complejidad/entropía.
    Incluye manejo de errores robusto para ventanas donde no se detectan picos
    (respuesta basal baja o ausente).

    Parámetros
    ----------
    signal : np.ndarray
        Array 1-D con las muestras de la señal GSR/EDA en microsiemens (µS).
        Se espera una ventana de ~5.5 s, es decir ~2816 muestras a 512 Hz.
    sampling_rate : int, opcional
        Frecuencia de muestreo en Hz. Por defecto 512 Hz.

    Retorna
    -------
    dict con las siguientes claves:
        gsr_peaks_count    (int)   : Número total de picos SCR detectados
                                     en la ventana. 0 si no hay picos.
        gsr_max_amplitude  (float) : Amplitud máxima del pico SCR más alto
                                     [µS]. 0.0 si no hay picos.
        gsr_min_max_diff   (float) : Diferencia entre el valor máximo y mínimo
                                     de la señal completa [µS]. Mide el rango
                                     dinámico total de la respuesta.
        gsr_slope_late     (float) : Pendiente (regresión lineal de orden 1)
                                     calculada sobre los últimos 2.75 s de la
                                     ventana [µS/s]. Indica si la activación
                                     sudomotora sube o baja hacia el final de
                                     la ventana. 0.0 si la señal es más corta
                                     que ese tramo.
        gsr_apen           (float) : Entropía Aproximada de la señal GSR
                                     original.
        gsr_lzc            (float) : Complejidad de Lempel-Ziv de la señal
                                     GSR original.

    Excepciones
    -----------
    ValueError : Si `signal` está vacío o no es un array numérico 1-D.

    Nota Senior
    -----------
    NeuroKit2 puede no detectar picos cuando la ventana es muy corta o la
    actividad sudomotora es mínima. En esos casos se retornan ceros para no
    interrumpir el pipeline de procesamiento del Grupo 1.
    """
    # ── Validación de entrada ──────────────────────────────────────────────
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(
            "extract_gsr_features: 'signal' debe ser un array 1-D no vacío."
        )

    # ── Detección de picos EDA/GSR con NeuroKit2 ──────────────────────────
    try:
        eda_signals, eda_info = nk.eda_peaks(
            signal, sampling_rate=sampling_rate
        )

        # SCR_Height contiene la amplitud real de cada pico detectado.
        # NeuroKit2 coloca 0.0 en muestras que no son pico.
        # SCR_Amplitude (amplitud corregida por recuperación) también es válida,
        # pero SCR_Height es más robusta en ventanas cortas.
        amplitude_col = None
        for col in ("SCR_Height", "SCR_Amplitude"):
            if col in eda_signals.columns:
                amplitude_col = col
                break

        if amplitude_col is not None:
            amplitudes = eda_signals[amplitude_col]
            valid_amplitudes = amplitudes[amplitudes > 0]
            gsr_peaks_count   = int(len(valid_amplitudes))
            gsr_max_amplitude = float(valid_amplitudes.max()) if gsr_peaks_count > 0 else 0.0
        else:
            gsr_peaks_count   = 0
            gsr_max_amplitude = 0.0

    except Exception:
        # Manejo de errores: señal plana, muy corta, o error interno de NK2
        gsr_peaks_count   = 0
        gsr_max_amplitude = 0.0

    # ── Rango dinámico total de la señal ──────────────────────────────────
    gsr_min_max_diff = float(np.max(signal) - np.min(signal))

    # ── Pendiente del tramo final (últimos 2.75 s) ────────────────────────
    n_late_samples = int(2.75 * sampling_rate)

    if len(signal) < n_late_samples:
        gsr_slope_late = 0.0
    else:
        segment = signal[-n_late_samples:]
        x_axis = np.arange(n_late_samples)
        gsr_slope_late = float(np.polyfit(x_axis, segment, 1)[0])

    # ── Complejidad (ApEn, LZC) sobre la señal GSR original ───────────────
    complexity = _extract_complexity(signal)

    return {
        "gsr_peaks_count":   gsr_peaks_count,
        "gsr_max_amplitude": gsr_max_amplitude,
        "gsr_min_max_diff":  gsr_min_max_diff,
        "gsr_slope_late":    gsr_slope_late,
        "gsr_apen":          complexity["apen"],
        "gsr_lzc":           complexity["lzc"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE DE PRUEBA RÁPIDA (se ejecuta solo si se llama el script directamente)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  signal_processor.py — Test de funcionamiento (v2.0.0)")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    n_samples = int(SAMPLING_RATE * WINDOW_SECONDS)   # 2816 muestras

    # ── Test EMG ──────────────────────────────────────────────────────────
    emg_signal = rng.normal(0, 50, n_samples)          # µV, media ≈ 0
    emg_feats  = extract_emg_features(emg_signal)
    print("\n[EMG] Señal sintética (Gaussiana, µ=0, σ=50 µV)")
    for k, v in emg_feats.items():
        if isinstance(v, int):
            print(f"  {k:<12} = {v}")
        else:
            print(f"  {k:<12} = {v:.4f}")

    # ── Test ECG ──────────────────────────────────────────────────────────
    try:
        ecg_signal = nk.ecg_simulate(
            duration=5,                  # ecg_simulate requiere duración entera
            sampling_rate=SAMPLING_RATE,
            heart_rate=72,
            random_state=42
        )
        ecg_feats = extract_ecg_features(ecg_signal)
        print("\n[ECG] Señal simulada (72 BPM esperados)")
        for k, v in ecg_feats.items():
            print(f"  {k:<12} = {v:.4f}")
    except Exception as e:
        print(f"\n[ECG] No se pudo simular señal: {e}")

    # ── Test GSR ──────────────────────────────────────────────────────────
    try:
        gsr_signal = nk.eda_simulate(
            duration=5,                  # eda_simulate requiere duración entera
            sampling_rate=SAMPLING_RATE,
            scr_number=3,
            random_state=42
        )
        gsr_feats = extract_gsr_features(gsr_signal)
        print("\n[GSR] Señal simulada (3 SCR esperados)")
        for k, v in gsr_feats.items():
            if isinstance(v, int):
                print(f"  {k:<16} = {v}")
            else:
                print(f"  {k:<16} = {v:.4f}")
    except Exception as e:
        print(f"\n[GSR] No se pudo simular señal: {e}")

    print("\n" + "=" * 60)
    print("  Todas las funciones operativas.")
    print("=" * 60)