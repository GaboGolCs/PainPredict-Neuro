PainPredict-Neuro

Predicción de dolor agudo a partir de bioseñales (ECG, EMG, GSR) y EEG, usando los
datasets BioVid HeatPain y Biosemi ds005293 (95 By BP). El proyecto abarca
todo el pipeline: carga de datos cruda, extracción y selección de características,
normalización, análisis exploratorio, balanceo de clases y modelamiento
(XGBoost y EEGNet) con seguimiento de experimentos en MLflow/DagsHub.


Proyecto desarrollado de forma colaborativa por distintos grupos/integrantes como
parte de las actividades del curso (Hitos 1–4). Varios notebooks conservan el
nombre de la actividad y del/los autores tal como fueron entregados.

🎯 Objetivo del proyecto

Construir modelos capaces de predecir el nivel de dolor de una persona a partir
de señales fisiológicas, comparando dos fuentes de datos con naturalezas distintas:


BioVid HeatPain: bioseñales periféricas (ECG, GSR, EMG) con estímulo de calor,
con dolor como variable binaria/categórica (BL1 a PA4).
Biosemi ds005293: EEG de 63 canales con estímulo láser, con dolor como escala
ordinal NRS (2, 4, 6, 8).


Ambos datasets se mantienen separados (no se combinan directamente) debido a
incompatibilidades de muestreo, modalidad y tipo de etiqueta — ver
data_registry.json para el detalle.

📁 Estructura del repositorio
Training-PainPredict-Neuro/
├── docs/
│   ├── PIA2 - PainPredict_Neuro Perfil Proyecto PIA v1 GPI 2026-I 16Marzo2026.pdf
│   └── readme.txt
├── README.md
├── requirements.txt
├── src/
│   ├── data_loaders/
│   │   ├── biosemi_loader.py
│   │   ├── biovid_loader.py
│   │   ├── data_registry.json
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── TEST_FUNCIONAMIENTO.py
│   │   └── unified_interface.py
│   ├── model_training/
│   │   ├── final_autonomic/
│   │   │   └── Modelamiento_XGBoost_Biovid.ipynb
│   │   ├── final_eegnet_training/
│   │   ├── instrucciones.txt
│   │   ├── old_autonomic_tries/
│   │   │   ├── xgboost_Intento1.ipynb
│   │   │   └── xgboost_refinado_eeg.ipynb
│   │   └── old_eegnet_training/
│   │       └── Eegnet_training_local_g4.py
│   └── notebooks/
│       ├── autonomic_notebooks/
│       │   ├── autonomic_Selection/
│       │   │   └── selection_biovid_elite.ipynb
│       │   ├── autonomic_statistics/
│       │   │   ├── act_cuerpo_desbalance_clases.ipynb
│       │   │   ├── Actividad 4 - Normalizacion por Sujeto.ipynb
│       │   │   ├── actividad_anali_cardiaco.ipynb
│       │   │   ├── actividad_cuerpo_corr.ipynb
│       │   │   └── actividad_reproducibilidad.ipynb
│       │   ├── old_autonomic_deletion/
│       │   │   └── caracteristicas_VIP_body.ipynb
│       │   └── old_autonomic_extraction/
│       │       ├── signal_processor.py
│       │       └── temp_validators/
│       │           ├── funcionTemp.py
│       │           └── test_validate_Tem.py
│       ├── eeg_notebooks/
│       │   ├── eeg_old_normalizations/
│       │   │   ├── Actividad 5 Normalización por Sujeto – Señales del Cerebro (Grupos EEG).ipynb
│       │   │   └── Orquestación_y_Normalización_Z_score.ipynb
│       │   ├── eeg_old_selection/
│       │   │   └── 04_seleccion_caracteristicas_eeg.ipynb
│       │   └── eeg_statistics/
│       │       ├── Actividad_2_-_Análisis_Espectral_por_Condición_(Dolor_NRS_2,_4,_6,_8).ipynb
│       │       ├── Actividad_3_Generación_de_Mapas_Topográficos_2D_de_la_Corteza_(EEG).ipynb
│       │       ├── actividad_5A_eeg_desbalance.ipynb
│       │       └── Matriz_de_Correlación_para_Características_EEG.ipynb
│       └── readme.txt
└── readme.txt



📊 Datasets

BioVid HeatPain

EspecificaciónValorCanales5 (GSR, ECG, EMG_trapezius, EMG_corrugator, EMG_zygomaticus)Frecuencia de muestreo512 HzFormatoCSV (tabulación)CondicionesBL1 (baseline), PA1–PA4 (niveles de dolor crecientes)Inventario maestroPartD/samples.csvSujetos aprox.87

Biosemi ds005293 (95 By BP)

EspecificaciónValorCanales63 (sistema 10-20)Frecuencia de muestreo1000 HzFormatoBrainVision (.vhdr, .vmrk, .eeg) / .fif (epochs)Escala de dolorNRS ordinal — 2, 4, 6, 8 (eventos 32–35)EstimulaciónLáser (dorso de la mano)Sujetos95, 6 sesiones c/u, 80 trials por sesión

⚠️ Incompatibilidades críticas entre datasets


Modalidad: bioseñales periféricas (5 ch) vs. EEG (63 ch)
Formato: CSV vs. BrainVision
Muestreo: 512 Hz vs. 1000 Hz
Dolor: binario/categórico vs. ordinal (NRS)
Duración: 100–500 s vs. ~2200 s por sesión


Recomendación: mantener pipelines y loaders independientes por dataset; si se
requiere combinar información, hacerlo mediante ensemble (fusión tardía de
predicciones), no a nivel de features crudas.


🧩 Sistema de carga de datos (src/data_loaders/)

Interfaz unificada para acceder a ambos datasets sin romper sus formatos originales.

python# Opción 1: Factory function
from src.data_loaders import get_loader

biovid = get_loader("biovid")
subjects = biovid.list_subjects()

# Opción 2: Registry (recomendado)
from src.data_loaders import DatasetRegistry

registry = DatasetRegistry()
biovid = registry.get_loader("biovid")
biosemi = registry.get_loader("biosemi")
registry.print_summary()


BioVidLoader: lista sujetos, metadata, carga bioseñales crudas o filtradas por
condición.
BiosemiLoader: lista sujetos/sesiones, metadata, archivos EEG por sesión
(carga real de señal delegada a mne.io.read_raw_brainvision).
DatasetRegistry: capa central que expone ambos loaders (lazy init), compara
especificaciones y reporta incompatibilidades desde data_registry.json.


Prueba de humo del sistema completo: TEST_FUNCIONAMIENTO.py.


🔬 Procesamiento de señales

signal_processor.py (v2.0.0)

Extracción de características sobre ventanas de 5.5 segundos de señal cruda a
512 Hz, apoyado en neurokit2:


extract_ecg_features() — features cardíacas (BPM, variabilidad, etc.)
extract_gsr_features() — picos y amplitud de respuesta galvánica
extract_emg_features() — actividad muscular (trapecio, corrugador, cigomático)
_extract_complexity() — Entropía Aproximada (ApEn) y Complejidad de
Lempel-Ziv (LZC)


Orquestación_y_Normalización_Z_score.ipynb

Pipeline de orquestación masiva: extrae features de todos los sujetos/condiciones
desde un .zip de bioseñales filtradas, aplica signal_processor y normaliza
Z-score por sujeto de forma vectorizada con groupby().transform(), generando
el dataset final (dataset_features_biovid_normalizado_V2.csv).

funcionTemp.py + test_validate_Tem.py

Validación de ground truth: comprueba que las ventanas de class_id = 5
(dolor máximo) coincidan con picos de temperatura (~50 °C) y que class_id = 4
(baseline) coincida con el rango de reposo (~30–34 °C). Se usa como control de
calidad sobre el etiquetado del dataset.


📓 Notebooks — Actividades por etapa

NotebookDatasetEtapaDescripciónActividad_2_Analisis_Espectral...Biosemi (EEG)ExploraciónAnálisis espectral por condición de dolor (NRS 2/4/6/8): desincronización de banda Alpha parietal y evolución de banda Theta.Actividad_3_Mapas_Topograficos_2D...Biosemi (EEG)ExploraciónGeneración de mapas topográficos 2D de la corteza usando mne.04_seleccion_caracteristicas_eegBiosemi (EEG)Selección de featuresPipeline de 2 filtros: eliminación de multicolinealidad (Pearson) + ranking por Información Mutua, con heatmap y ranking VIP.Matriz_de_Correlacion_para_Caracteristicas_EEGBiosemi (EEG)ExploraciónMatriz de correlación entre bandas/características EEG.Actividad_4_Normalizacion_por_SujetoBioVidNormalizaciónResta de línea base (BL1) por sujeto para eliminar variabilidad biológica; recalcula matriz de correlación y valida matemáticamente la normalización.Actividad_5_Normalizacion_por_Sujeto_EEGBiosemi (EEG)NormalizaciónAnálogo a Actividad 4 pero para señales EEG por grupo.act_cuerpo_desbalance_clasesBioVidBalanceo de clasesCarga de dataset_features_biovid.csv, análisis y visualización de desbalance entre clases de dolor.actividad_5A_eeg_desbalanceBiosemi (EEG)Balanceo de clasesPorcentaje de clases (NRS) con value_counts(normalize=True), countplot y exportación de figura.actividad_anali_cardiacoBioVidExploraciónAnálisis específico de la señal ECG (frecuencia cardíaca / BPM).actividad_cuerpo_corrBioVidExploraciónMatriz/análisis de correlación entre features de bioseñales periféricas.actividad_reproducibilidadBioVidExploraciónAnálisis de reproducibilidad intra-sujeto: tendencia individual de EMG corrugador a lo largo de los niveles de dolor (BL1–PA4) en una muestra de 5 sujetos.caracteristicas_VIP_bodyBioVidSelección de featuresRanking de features "VIP" del cuerpo (bioseñales) vía Mutual Information.selection_biovid_eliteBioVidSelección de features (avanzado)Optimización del filtro de multicolinealidad/umbrales de correlación, benchmarking de impacto sobre el modelo (RandomForest) con StratifiedGroupKFold.

🤖 Notebooks — Modelamiento

NotebookDatasetModeloDescripciónxgboost_Intento1BioVidXGBoostPrimer intento: XGBClassifier multiclase (5 clases, BL1–PA4) con GroupKFold (5 folds), pesos balanceados, tracking en MLflow (DagsHub). Macro F1 ≈ 0.25–0.30.xgboost_refinado_eegBiosemi (EEG)XGBoost + SMOTETomekVersión refinada sobre EEG: feature engineering (ratio Theta/Beta por región), pipeline imblearn (SMOTETomek + XGBClassifier) dentro de GroupKFold/GridSearchCV, evaluación out-of-fold, tracking en MLflow.Modelamiento_XGBoost_BiovidBioVidXGBoost + Optuna + SHAPPipeline completo: optimización de hiperparámetros con Optuna, entrenamiento final multiclase, exportación de resultados a JSON y explicabilidad global/local con SHAP.Eegnet_training_local_g4.pyBiosemi (EEG)EEGNet v2 (PyTorch)Entrenamiento local (GPU) de red convolucional EEGNet (Temporal Conv → Depthwise Conv → Separable Conv → Classifier) para clasificar dolor NRS 2/4/6/8 a partir de epochs .fif. Validación con GroupKFold por sujeto, CrossEntropyLoss ponderada, métricas (F1 macro, accuracy, MCC, AUC-ROC), matriz de confusión y tracking en MLflow.


⚙️ Instalación

bashgit clone <url-del-repositorio>
cd PainPredict-Neuro
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt


Los notebooks (.ipynb) fueron desarrollados en Google Colab, que ya incluye
numpy, pandas, matplotlib, seaborn y scikit-learn. Al ejecutarlos
localmente o en otro entorno, usa requirements.txt para instalar todo lo
necesario (incluye xgboost, optuna, shap, imbalanced-learn, torch,
mne, neurokit2 y mlflow).



Configuración de MLflow / DagsHub

Varios notebooks y scripts registran experimentos en MLflow apuntando a un
repositorio remoto de DagsHub:

pythonos.environ["MLFLOW_TRACKING_URI"]      = "https://dagshub.com/<usuario>/PainPredict-Neuro.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "<usuario>"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<token-personal>"

⚠️ Importante: el token es personal e intransferible. Nunca lo subas al
repositorio; usa variables de entorno o un .env (agregado a .gitignore) y
elimínalo del notebook antes de hacer commit.


📐 Convenciones del equipo

Reglas ya definidas por el equipo (instrucciones.txt, readme.txt):


Nomenclatura de notebooks de modelos: nombremodelo_grupo_intento
(ej. xgboost_g2_1).
Nomenclatura general de notebooks: numeración secuencial (01_nombre,
02_nombre, etc.).
Antes de subir a GitHub: limpiar los outputs pesados de los notebooks
para no saturar el repositorio.
Documentación: cada celda de código debe explicarse con Markdown.
Cada dataset mantiene su loader independiente; no forzar un esquema común
entre BioVid y Biosemi (ver incompatibilidades).



🧪 Tests


TEST_FUNCIONAMIENTO.py — valida carga de loaders, listado de sujetos, registry,
lectura de metadata y datos de ambos datasets end-to-end.
test_validate_Tem.py — valida la correspondencia entre class_id y las
ventanas de temperatura (ground truth).


bashpython TEST_FUNCIONAMIENTO.py
python test_validate_Tem.py


📦 Dependencias principales

Ver requirements.txt para el listado completo con
versiones. Librerías clave por área:

Datos y análisis: pandas, numpy, scipy
Visualización: matplotlib, seaborn
ML clásico: scikit-learn, xgboost, imbalanced-learn, optuna, shap
Deep Learning: torch
Señales EEG / bioseñales: mne, neurokit2
MLOps: mlflow
Utilidades: tqdm
