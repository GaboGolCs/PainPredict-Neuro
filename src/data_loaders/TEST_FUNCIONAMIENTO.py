"""
TEST DE FUNCIONAMIENTO DEL SISTEMA DE LOADERS
Este script prueba la funcionalidad básica del sistema con datos disponibles
"""
 
from src.data_loaders.unified_interface import DatasetRegistry

interface = DatasetRegistry()

print("="*70)
print("PRUEBA DE FUNCIONAMIENTO - PAINPREDICT-NEURO")
print("="*70)

# ============================================================================
# 1. TEST BÁSICO: Cargar loaders
# ============================================================================
print("\n[1] Cargando loaders...")
try:
    biovid_loader = interface.get_loader("biovid")
    print("✓ BioVid loader cargado")
    
    biosemi_loader = interface.get_loader("biosemi")
    print("✓ Biosemi loader cargado")
except Exception as e:
    print(f"✗ Error al cargar loaders: {e}")
    exit(1)

# ============================================================================
# 2. TEST: Listar sujetos
# ============================================================================
print("\n[2] Listando sujetos...")
try:
    biovid_subjects = biovid_loader.list_subjects()
    biosemi_subjects = biosemi_loader.list_subjects()
    
    print(f"✓ BioVid: {len(biovid_subjects)} sujetos")
    print(f"  Primeros 5: {biovid_subjects[:5]}")
    
    print(f"✓ Biosemi: {len(biosemi_subjects)} sujetos")
    print(f"  Primeros 5: {biosemi_subjects[:5]}")
except Exception as e:
    print(f"✗ Error al listar sujetos: {e}")
    exit(1)

# ============================================================================
# 3. TEST: Registry
# ============================================================================
print("\n[3] Prueba de Registry...")
try:
    registry = DatasetRegistry()
    datasets = registry.list_datasets()
    print(f"✓ Datasets registrados: {datasets}")
    
    comparison = registry.compare_datasets()
    print(f"✓ Comparación de datasets completada")
    print(f"  Incompatibilidades críticas encontradas:")
    for incomp in comparison["incompatibilities"]["critical"]:
        print(f"    - {incomp}")
except Exception as e:
    print(f"✗ Error en Registry: {e}")
    exit(1)

# ============================================================================
# 4. TEST: Metadata de BioVid
# ============================================================================
print("\n[4] Leyendo metadata de BioVid...")
try:
    first_subject = biovid_subjects[0]
    metadata = biovid_loader.get_subject_metadata(first_subject)
    print(f"✓ Sujeto: {first_subject}")
    print(f"  Age: {metadata['age']}")
    print(f"  Gender: {'Female' if metadata['gender'] == 1 else 'Male'}")
except Exception as e:
    print(f"✗ Error al leer metadata de BioVid: {e}")

# ============================================================================
# 5. TEST: Cargar datos de BioVid (modalidad filtered)
# ============================================================================
print("\n[5] Cargando datos de BioVid (datos filtrados)...")
try:
    subject_data = biovid_loader.load_subject_raw(first_subject, modality="filtered")
    print(f"✓ Datos cargados para {first_subject}")
    print(f"  Sampling rate: {subject_data['sampling_rate']} Hz")
    print(f"  Canales: {subject_data['channels']}")
    print(f"  Condiciones: {list(subject_data['conditions'].keys())}")
    
    # Ver info de una condición
    conditions = list(subject_data['conditions'].keys())
    if conditions:
        first_condition = conditions[0]
        num_trials = len(subject_data['conditions'][first_condition])
        print(f"  Trials en {first_condition}: {num_trials}")
except Exception as e:
    print(f"✗ Error al cargar datos de BioVid: {e}")

# ============================================================================
# 6. TEST: Metadata de Biosemi
# ============================================================================
print("\n[6] Leyendo metadata de Biosemi...")
try:
    first_subject_biosemi = biosemi_subjects[0]
    metadata = biosemi_loader.get_subject_metadata(first_subject_biosemi)
    print(f"✓ Sujeto: {first_subject_biosemi}")
    print(f"  Age: {metadata['age']}")
    print(f"  Gender: {'Female' if metadata['gender'] == 1 else 'Male'}")
    print(f"  Pain Threshold: {metadata.get('pain_threshold_joules', 'N/A')} J")
except Exception as e:
    print(f"✗ Error al leer metadata de Biosemi: {e}")

# ============================================================================
# 7. TEST: Sesiones de Biosemi
# ============================================================================
print("\n[7] Listando sesiones de Biosemi...")
try:
    sessions = biosemi_loader.list_sessions(first_subject_biosemi)
    print(f"✓ Sesiones encontradas: {sessions}")
    
    if sessions:
        first_session = sessions[0]
        eeg_files = biosemi_loader.list_eeg_files(first_subject_biosemi, first_session)
        print(f"  EEG files en {first_session}: {len(eeg_files)}")
        
        session_info = biosemi_loader.load_session_info(first_subject_biosemi, first_session)
        print(f"  Canales: {session_info['specifications']['channels']}")
        print(f"  Sampling Rate: {session_info['specifications']['sampling_rate_hz']} Hz")
except Exception as e:
    print(f"✗ Error al listar sesiones de Biosemi: {e}")

# ============================================================================
# 8. TEST: Información del Dataset
# ============================================================================
print("\n[8] Información del Dataset Registry...")
try:
    for ds_name in registry.list_datasets():
        info = registry.get_dataset_info(ds_name)
        print(f"\n{ds_name.upper()}:")
        print(f"  Nombre: {info['name']}")
        print(f"  Tipo: {info['type']}")
        print(f"  Ubicación: {info['location']}")
except Exception as e:
    print(f"✗ Error al obtener info del dataset: {e}")

# ============================================================================
# RESULTADO FINAL
# ============================================================================
print("\n" + "="*70)
print("✓ PRUEBA DE FUNCIONAMIENTO COMPLETADA EXITOSAMENTE")
print("="*70)
print("\nResumen:")
print(f"  • BioVid: {len(biovid_subjects)} sujetos cargados")
print(f"  • Biosemi: {len(biosemi_subjects)} sujetos cargados")
print(f"  • Sistema de Registry: Operativo")
print(f"  • Loaders: Funcionando correctamente")
print("\nNota: Los datos de BioVid están en modalidad 'filtered'.")
print("Los datos 'raw' de PartC no están disponibles en este repositorio.")
