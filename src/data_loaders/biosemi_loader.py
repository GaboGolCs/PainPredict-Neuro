"""
Biosemi ds005293 Dataset Loader

Carga datos EEG en formato Brainvision de Biosemi.
Utiliza mne-python para parsing de archivos.
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple


class BiosemiLoader:
    """Loader para Biosemi ds005293 (95 By BP) Dataset"""
    
    def __init__(self, root_path: Optional[str] = None):
        """
        Inicializa el loader de Biosemi.
        
        Args:
            root_path: Ruta a ds005293-main. Si es None, busca en data/raw/
        """
        if root_path is None:
            root_path = Path(__file__).parent.parent.parent / "data" / "raw" / "ds005293-main"#\
                       #"Dataset_ _142 by Biosemi_ (ds005292)" / "ds005293-main"
        
        self.root_path = Path(root_path)
        self.participants_tsv = self.root_path / "participants.tsv"
        self.participants_json = self.root_path / "participants.json"
        self.events_json = self.root_path / "task-95ByBP_events.json"
        
        # Validar que exista inventario
        if not self.participants_tsv.exists():
            raise FileNotFoundError(f"participants.tsv no encontrado en {self.participants_tsv}")
        
        # Cargar metadata
        self._load_inventory()
    
    def _load_inventory(self):
        """Carga inventario de participants.tsv"""
        self.inventory = pd.read_csv(self.participants_tsv, sep="\t")
        self.subjects = [s for s in self.inventory["participant_id"].unique() 
                        if s.startswith("sub-")]
        
        # Cargar descripción de campos si existe
        if self.participants_json.exists():
            with open(self.participants_json, "r") as f:
                self.field_descriptions = json.load(f)
        else:
            self.field_descriptions = {}
        
        # Cargar codificación de eventos
        if self.events_json.exists():
            with open(self.events_json, "r") as f:
                self.event_codes = json.load(f)
        else:
            self.event_codes = {}
    
    def list_subjects(self) -> List[str]:
        """Retorna lista de todos los sujetos disponibles"""
        return sorted(self.subjects)
    
    def get_subject_metadata(self, subject_id: str) -> Dict:
        """
        Obtiene metadata de un sujeto.
        
        Args:
            subject_id: ID del sujeto (ej: "sub-001")
        
        Returns:
            Dict con metadata
        """
        # Estandarizar ID
        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{int(subject_id):03d}"
        
        subject_data = self.inventory[self.inventory["participant_id"] == subject_id]
        
        if subject_data.empty:
            raise ValueError(f"Sujeto no encontrado: {subject_id}")
        
        row = subject_data.iloc[0]
        return {
            "participant_id": row["participant_id"],
            "id": row.get("ID"),
            "pre_id": row.get("Pre_ID"),
            "age": row["Age"],
            "gender": row["Gender"],  # M or F
            "pain_threshold_joules": row.get("Pain_Threshod"),
        }
    
    def list_sessions(self, subject_id: str) -> List[str]:
        """
        Lista sesiones disponibles para un sujeto.
        
        Args:
            subject_id: ID del sujeto
        
        Returns:
            Lista de sesiones (ses-1, ses-2, etc.)
        """
        # Estandarizar ID
        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{int(subject_id):03d}"
        
        subject_path = self.root_path / subject_id
        if not subject_path.exists():
            return []
        
        sessions = [d.name for d in subject_path.iterdir() if d.name.startswith("ses-")]
        return sorted(sessions)
    
    def list_eeg_files(self, subject_id: str, session_id: str) -> List[Dict]:
        """
        Lista archivos EEG de una sesión.
        
        Args:
            subject_id: ID del sujeto
            session_id: ID de sesión (ej: "ses-1")
        
        Returns:
            Lista de dicts con info de archivos
        """
        # Estandarizar IDs
        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{int(subject_id):03d}"
        if not session_id.startswith("ses-"):
            session_id = f"ses-{int(session_id)}"
        
        eeg_path = self.root_path / subject_id / session_id / "eeg"
        
        if not eeg_path.exists():
            return []
        
        files = []
        for file in sorted(eeg_path.glob("*eeg")):
            files.append({
                "filename": file.name,
                "path": str(file),
                "type": "Brainvision EEG data",
            })
        
        return files
    
    def load_session_info(self, subject_id: str, session_id: str) -> Dict:
        """
        Carga información de una sesión (sin EEG, solo metadata).
        
        Nota: Para cargar datos EEG reales, use mne.io.read_raw_brainvision()
        
        Args:
            subject_id: ID del sujeto
            session_id: ID de sesión
        
        Returns:
            Dict con metadata de sesión
        """
        metadata = self.get_subject_metadata(subject_id)
        eeg_files = self.list_eeg_files(subject_id, session_id)
        
        return {
            "metadata": metadata,
            "session": session_id,
            "eeg_files": eeg_files,
            "specifications": {
                "channels": 63,
                "sampling_rate_hz": 1000,
                "file_format": "Brainvision",
                "trials_per_session": 80,
                "conditions": 8,
            },
            "event_codes": self.event_codes,
            "note": "Para cargar datos EEG, use: mne.io.read_raw_brainvision(vhdr_file)"
        }
    
    def get_summary(self) -> Dict:
        """Retorna resumen del dataset"""
        return {
            "dataset": "Biosemi ds005293 (95 By BP)",
            "total_subjects": len(self.subjects),
            "sessions_per_subject": 6,
            "channels": 63,
            "channel_system": "10-20 standard",
            "sampling_rate_hz": 1000,
            "file_format": "Brainvision (vhdr, vmrk, eeg)",
            "bids_version": "1.1.1",
            "inventory_file": str(self.participants_tsv),
            "stimulation_type": "Laser (L1-L4)",
            "pain_levels_nrs": [2, 4, 6, 8],
        }
