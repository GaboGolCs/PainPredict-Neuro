# PainPredict-Neuro

**Proyecto de Gestión de Proyectos Informáticos (GPI-2) - UTEM 2026**

Sistema de evaluación automatizada del dolor mediante el procesamiento de señales EEG y biomédicas utilizando Inteligencia Artificial.

---

Descripción del Proyecto
Este proyecto busca desarrollar modelos de aprendizaje profundo (Deep Learning) capaces de clasificar niveles de dolor de forma objetiva, utilizando señales electroencefalográficas de diversas poblaciones (adultos y neonatos). El objetivo final es proporcionar una herramienta de apoyo clínico para pacientes no verbales.

Estructura del Repositorio

- `data/`: (Local/Drive) Carpeta para datasets (BioVid, SEED, OpenNeuro). _No se sincroniza con GitHub_.
- `notebooks/`: Experimentos y análisis exploratorio en Google Colab.
- `src/`: Código fuente modular (preprocesamiento, extracción de características, modelos).
- `docs/`: Documentación del proyecto (PIA2, manuales técnicos).

Reglas de Colaboración
Para mantener el orden con 30 integrantes, es obligatorio:

1. **Ramas:** `tipo/grupo-descripción` (ej: `feat/g3-filtro-eeg`).
2. **Commits:** `tipo: descripción` (ej: `feat: carga de dataset SEED`).
3. **Pull Requests:** Todo cambio a `main` debe ser vía PR y requiere 1 aprobación.

Guía de Inicio Rápido (Google Colab)
Para trabajar de forma estandarizada, todos seguiremos este flujo:

1. **Montar Drive:** Sube los datasets a una carpeta compartida en tu Google Drive.
2. **Abrir Notebook:** Abre cualquier archivo en `notebooks/` y selecciona "Abrir en Colab".
3. **Instalar Dependencias:** Ejecuta siempre la celda inicial:
   ```python
   !pip install -r requirements.txt
   ```
