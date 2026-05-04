# Data Loaders: Índice Centralizado

Sistema de carga de datos unificado para BioVid y Biosemi manteniendo integridad de cada formato.

## 📁 Estructura

```
src/data_loaders/
├── __init__.py              # Exports principales
├── data_registry.json       # Catálogo centralizado
├── biovid_loader.py         # Loader para BioVid (CSV)
├── biosemi_loader.py        # Loader para Biosemi (Brainvision)
├── unified_interface.py     # Interfaz agnóstica + Registry
└── README.md               # Este archivo
```

## 🚀 Uso Rápido

### Opción 1: Factory Function (Simple)

```python
from src.data_loaders import get_loader

# Cargar BioVid
biovid = get_loader("biovid")
subjects = biovid.list_subjects()

# Cargar Biosemi
biosemi = get_loader("biosemi")
subjects = biosemi.list_subjects()
```

### Opción 2: Registry (Recomendado)

```python
from src.data_loaders import DatasetRegistry

registry = DatasetRegistry()

# Acceso a loaders
biovid = registry.get_loader("biovid")
biosemi = registry.get_loader("biosemi")

# Comparación de datasets
comparison = registry.compare_datasets()

# Resumen formateado
registry.print_summary()
```

## 📊 Arquitectura

### BioVidLoader

Carga biosignales CSV de BioVid:

```python
loader = get_loader("biovid")

# Listar sujetos
subjects = loader.list_subjects()

# Cargar metadata
metadata = loader.get_subject_metadata("071309_w_21")

# Cargar todos los datos de un sujeto
data = loader.load_subject_raw("071309_w_21", modality="raw")
# Retorna: {
#   "metadata": {...},
#   "conditions": {"BL1": [...], "PA1": [...], ...},
#   "sampling_rate": 512,
#   "channels": ["GSR", "ECG", "EMG_trapezius", ...],
# }

# Cargar una condición específica
condition_data = loader.load_subject_condition("071309_w_21", "PA1")
```

**Especificaciones:**
- **Canales:** 5 (GSR, ECG, EMG_trapezius, EMG_corrugator, EMG_zygomaticus)
- **Muestreo:** 512 Hz
- **Formato:** CSV (tabulación)
- **Unidad de tiempo:** Microsegundos
- **Conditions:** BL1, PA1, PA2, PA3, PA4

### BiosemiLoader

Carga datos EEG en formato Brainvision de Biosemi:

```python
loader = get_loader("biosemi")

# Listar sujetos
subjects = loader.list_subjects()

# Cargar metadata
metadata = loader.get_subject_metadata("sub-001")

# Listar sesiones
sessions = loader.list_sessions("sub-001")

# Listar archivos EEG
eeg_files = loader.list_eeg_files("sub-001", "ses-1")

# Cargar info de sesión (sin datos EEG)
session_info = loader.load_session_info("sub-001", "ses-1")

# Para cargar datos EEG reales, use mne-python:
import mne
raw = mne.io.read_raw_brainvision("path_to_vhdr_file")
```

**Especificaciones:**
- **Canales:** 63 (10-20 standard: Fp1, AF3, AF7, ..., Oz)
- **Muestreo:** 1000 Hz
- **Formato:** Brainvision (.vhdr, .vmrk, .eeg)
- **Unidad de tiempo:** Segundos
- **Sesiones:** 6 por sujeto
- **Trials:** 80 por sesión

## ⚠️ Incompatibilidades Críticas

| Aspecto | BioVid | Biosemi |
|---------|--------|---------|
| Modalidad | Biosignales (5 canales) | EEG (63 canales) |
| Formato | CSV | Brainvision |
| Muestreo | 512 Hz | 1000 Hz |
| Dolor | Binario | Ordinal NRS |

**Recomendación:** Mantener datasets separados; usar ensemble si se necesita combinación.

## 📝 data_registry.json

Catálogo centralizado con metadata de ambos datasets:

```json
{
  "datasets": {
    "biovid": {
      "name": "BioVid HeatPain Dataset",
      "type": "biosignals",
      "specifications": { ... }
    },
    "biosemi": {
      "name": "Biosemi ds005293",
      "type": "EEG",
      "specifications": { ... }
    }
  },
  "incompatibilities": { ... }
}
```

## 🔧 Extensibilidad

Para añadir un nuevo dataset:

1. Crear `xxxxx_loader.py` en `src/data_loaders/`
2. Implementar interfaz estándar (métodos `list_subjects()`, `load_*()`)
3. Actualizar `data_registry.json` con metadata
4. Registrar en `__init__.py`

Ejemplo:

```python
# src/data_loaders/seed_loader.py
class SEEDLoader:
    def __init__(self, root_path=None):
        ...
    
    def list_subjects(self):
        ...
    
    def load_subject_raw(self, subject_id, **kwargs):
        ...
```

```python
# src/data_loaders/__init__.py
from .seed_loader import SEEDLoader

def get_loader(dataset_name):
    ...
    elif dataset_name == "seed":
        return SEEDLoader()
```

## 📖 Ejemplos Completos

Ver [EXAMPLE_USAGE.py](../EXAMPLE_USAGE.py) en la raíz del proyecto.

## 🎯 Casos de Uso

### 1. Exploración de datos

```python
registry = DatasetRegistry()
registry.print_summary()
```

### 2. Preprocesamiento independiente

```python
biovid = get_loader("biovid")
biosemi = get_loader("biosemi")

# Procesar cada uno con su pipeline específico
for subject in biovid.list_subjects():
    data = biovid.load_subject_raw(subject)
    # pipeline_biovid(data)

for subject in biosemi.list_subjects():
    # pipeline_biosemi(subject)
```

### 3. Ensemble Learning

```python
# Entrenar modelos separados
model_biovid = train_model(biovid_data)
model_biosemi = train_model(biosemi_data)

# Combinar predicciones (late fusion)
predictions = ensemble([model_biovid, model_biosemi])
```

## 📦 Dependencias

```
pandas
numpy
pathlib (builtin)
json (builtin)
```

Para cargar datos EEG de Biosemi:
```
mne-python (opcional, para raw data)
```

## ✅ Validación

Los loaders validan:
- Existencia de archivos maestros (samples.csv, participants.tsv)
- Existencia de directorios de datos
- Formato de IDs de sujetos

Lanzan excepciones específicas si algo falta.

## 📞 Troubleshooting

**Error: "samples.csv no encontrado"**
- Verificar ruta en `data/raw/Biovid HeatPain Dataset/PartD/`
- Confirmar que no fue eliminado accidentalmente

**Error: "participants.tsv no encontrado"**
- Verificar ruta en `data/raw/Dataset_ _142 by Biosemi_.../ds005293-main/`

**Error: "Sujeto no encontrado"**
- Usar `loader.list_subjects()` para ver IDs válidos
- Verificar formato del ID (ej: "071309_w_21" para BioVid)

---

**Última actualización:** 2026-05-03  
**Status:** ✅ Production Ready
