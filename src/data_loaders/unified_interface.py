"""
Unified Interface for Dataset Access

Proporciona acceso centralizado a ambos datasets manteniendo sus particularidades.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from .biovid_loader import BioVidLoader
from .biosemi_loader import BiosemiLoader


class DatasetRegistry:
    """
    Registro centralizado para acceso unificado a todos los datasets.
    Mantiene loaders independientes para cada modalidad.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Inicializa el registro centralizado.
        
        Args:
            registry_path: Ruta al archivo data_registry.json. Si es None, busca en el módulo.
        """
        if registry_path is None:
            registry_path = Path(__file__).parent / "data_registry.json"
        
        self.registry_path = Path(registry_path)
        
        # Cargar registry
        with open(self.registry_path, "r") as f:
            self.registry = json.load(f)
        
        # Inicializar loaders
        self.loaders = {
            "biovid": None,  # Se carga lazy
            "biosemi": None,
        }
        self._initialized = {
            "biovid": False,
            "biosemi": False,
        }
    
    def get_loader(self, dataset_name: str):
        """
        Obtiene el loader para un dataset específico (lazy initialization).
        
        Args:
            dataset_name: "biovid" o "biosemi"
        
        Returns:
            Loader instance
        """
        dataset_name = dataset_name.lower().strip()
        
        if dataset_name not in self.loaders:
            raise ValueError(f"Dataset desconocido: {dataset_name}")
        
        # Inicializar lazy
        if not self._initialized[dataset_name]:
            if dataset_name == "biovid":
                self.loaders["biovid"] = BioVidLoader()
            elif dataset_name == "biosemi":
                self.loaders["biosemi"] = BiosemiLoader()
            self._initialized[dataset_name] = True
        
        return self.loaders[dataset_name]
    
    def list_datasets(self) -> List[str]:
        """Retorna lista de datasets disponibles"""
        return list(self.registry["datasets"].keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Obtiene información básica de un dataset.
        
        Args:
            dataset_name: "biovid" o "biosemi"
        
        Returns:
            Dict con información del dataset
        """
        dataset_name = dataset_name.lower().strip()
        
        if dataset_name not in self.registry["datasets"]:
            raise ValueError(f"Dataset desconocido: {dataset_name}")
        
        return self.registry["datasets"][dataset_name]
    
    def get_incompatibilities(self) -> Dict:
        """Retorna información sobre incompatibilidades entre datasets"""
        return self.registry.get("incompatibilities", {})
    
    def get_loader_summary(self, dataset_name: str) -> Dict:
        """
        Obtiene resumen del loader (consulta al loader mismo).
        
        Args:
            dataset_name: "biovid" o "biosemi"
        
        Returns:
            Dict con resumen del dataset
        """
        loader = self.get_loader(dataset_name)
        return loader.get_summary()
    
    def list_all_subjects(self, dataset_name: str) -> List[str]:
        """
        Lista todos los sujetos de un dataset.
        
        Args:
            dataset_name: "biovid" o "biosemi"
        
        Returns:
            Lista de IDs de sujetos
        """
        loader = self.get_loader(dataset_name)
        return loader.list_subjects()
    
    def compare_datasets(self) -> Dict:
        """
        Retorna comparación de ambos datasets.
        
        Returns:
            Dict con tabla comparativa
        """
        return {
            "biovid": self.get_loader_summary("biovid"),
            "biosemi": self.get_loader_summary("biosemi"),
            "incompatibilities": self.get_incompatibilities(),
            "recommendation": "Usar datasets separados con loaders específicos"
        }
    
    def print_summary(self):
        """Imprime resumen de datasets disponibles"""
        print("\n" + "="*70)
        print("PAINPREDICT-NEURO: DATASET REGISTRY")
        print("="*70)
        
        for ds_name in self.list_datasets():
            info = self.get_dataset_info(ds_name)
            print(f"\n📊 {info['name']}")
            print(f"   Type: {info['type']}")
            print(f"   Location: {info['location']}")
            print(f"   Channels: {info['specifications']['channels']}")
            print(f"   Sampling Rate: {info['specifications']['sampling_rate_hz']} Hz")
            print(f"   Subjects: {info.get('approx_subjects', info.get('subjects', 'N/A'))}")


# Convenience module-level functions
_default_registry = None


def get_registry() -> DatasetRegistry:
    """Obtiene instancia global del registro (singleton pattern)"""
    global _default_registry
    if _default_registry is None:
        _default_registry = DatasetRegistry()
    return _default_registry


def show_datasets():
    """Imprime info de datasets disponibles"""
    registry = get_registry()
    registry.print_summary()
