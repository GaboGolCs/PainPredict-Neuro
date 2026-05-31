"""
BioVid HeatPain Dataset Loader

Carga biosignales en formato CSV de BioVid manteniendo estructura original.
Utiliza PartD/samples.csv como inventario maestro.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple


class BioVidLoader:
    """Loader para BioVid HeatPain Dataset"""
    
    def __init__(self, root_path: Optional[str] = None):
        """
        Inicializa el loader de BioVid.
        
        Args:
            root_path: Ruta a la carpeta raíz del dataset. Si es None, busca en data/raw/
        """
        if root_path is None:
            root_path = Path(__file__).parent.parent.parent / "data" / "raw" #/ "Biovid HeatPain Dataset"
        
        self.root_path = Path(root_path)
        self.samples_csv = self.root_path / "PartD" / "samples.csv"
        self.biosignals_raw = self.root_path / "PartC" / "biosignals_raw"
        self.biosignals_filtered = self.root_path / "PartA" / "biosignals_filtered"
        
        # Validar que exista el inventario
        if not self.samples_csv.exists():
            raise FileNotFoundError(f"Archivo samples.csv no encontrado en {self.samples_csv}")
        
        # Cargar inventario
        self._load_inventory()
    
    def _load_inventory(self):
        """Carga el inventario maestro de PartD/samples.csv"""
        self.inventory = pd.read_csv(self.samples_csv, sep="\t")
        self.subjects = self.inventory["subject_name"].unique()
    
    def list_subjects(self) -> List[str]:
        """Retorna lista de todos los sujetos disponibles"""
        return sorted(self.subjects)
    
    def get_subject_metadata(self, subject_id: str) -> Dict:
        """
        Obtiene metadata de un sujeto (age, gender, etc.)
        
        Args:
            subject_id: ID del sujeto (ej: "071309_w_21")
        
        Returns:
            Dict con metadata del sujeto
        """
        subject_data = self.inventory[self.inventory["subject_name"] == subject_id]
        
        if subject_data.empty:
            raise ValueError(f"Sujeto no encontrado: {subject_id}")
        
        row = subject_data.iloc[0]
        return {
            "subject_id": row["subject_id"],
            "subject_name": row["subject_name"],
            "age": row["age"],
            "gender": row["gender"],  # 0=M, 1=F
        }
    
    def load_subject_raw(self, subject_id: str, modality: str = "raw") -> Dict:
        """
        Carga todos los archivos de biosignales de un sujeto.
        
        Args:
            subject_id: ID del sujeto (ej: "071309_w_21")
            modality: "raw" (PartC) o "filtered" (PartA)
        
        Returns:
            Dict con:
            - metadata: info del sujeto
            - conditions: Dict[condition_name -> list of DataFrames]
            - sampling_rate: 512 Hz
        """
        metadata = self.get_subject_metadata(subject_id)
        
        # Seleccionar ruta según modalidad
        if modality == "raw":
            data_path = self.biosignals_raw / subject_id
        elif modality == "filtered":
            data_path = self.biosignals_filtered / subject_id
        else:
            raise ValueError(f"Modalidad desconocida: {modality}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"No hay datos para {subject_id} en {modality}")
        
        # Agrupar por condición (BL1, PA1, PA2, PA3, PA4)
        conditions = {}
        for file in sorted(data_path.glob("*.csv")):
            # Parsear nombre: 071309_w_21-BL1-081_bio.csv
            parts = file.stem.split("-")
            condition = parts[1]  # BL1, PA1, PA2, etc.
            
            if condition not in conditions:
                conditions[condition] = []
            
            # Cargar CSV (tiempo en microsegundos, valores separados por tabulación)
            df = pd.read_csv(file, sep="\t")
            conditions[condition].append(df)
        
        return {
            "metadata": metadata,
            "conditions": conditions,
            "modality": modality,
            "sampling_rate": 512,
            "channels": ["GSR", "ECG", "EMG_trapezius", "EMG_corrugator", "EMG_zygomaticus"],
            "time_unit": "microseconds",
        }
    
    def load_subject_condition(self, subject_id: str, condition: str, modality: str = "raw") -> Dict:
        """
        Carga solo una condición de un sujeto.
        
        Args:
            subject_id: ID del sujeto
            condition: Condición (BL1, PA1, PA2, PA3, PA4)
            modality: "raw" o "filtered"
        
        Returns:
            Dict con señales y metadata
        """
        data = self.load_subject_raw(subject_id, modality)
        
        if condition not in data["conditions"]:
            raise ValueError(f"Condición no encontrada: {condition}")
        
        return {
            "metadata": data["metadata"],
            "condition": condition,
            "trials": data["conditions"][condition],
            "sampling_rate": data["sampling_rate"],
            "channels": data["channels"],
        }
    
    def get_summary(self) -> Dict:
        """Retorna resumen del dataset"""
        return {
            "dataset": "BioVid HeatPain",
            "total_subjects": len(self.subjects),
            "channels": 5,
            "channel_names": ["GSR", "ECG", "EMG_trapezius", "EMG_corrugator", "EMG_zygomaticus"],
            "sampling_rate_hz": 512,
            "file_format": "CSV",
            "inventory_file": str(self.samples_csv),
        }
