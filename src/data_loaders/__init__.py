"""
Data loaders for PainPredict-Neuro datasets.

Proporciona acceso unificado a BioVid y Biosemi manteniendo sus formatos originales.
"""

from .biovid_loader import BioVidLoader
from .biosemi_loader import BiosemiLoader
from .unified_interface import DatasetRegistry

__all__ = ["BioVidLoader", "BiosemiLoader", "DatasetRegistry", "get_loader"]


def get_loader(dataset_name: str):
    """
    Factory function para obtener el loader correcto.
    
    Args:
        dataset_name: "biovid" o "biosemi"
    
    Returns:
        Loader instance (BioVidLoader o BiosemiLoader)
    
    Raises:
        ValueError: si dataset_name no es válido
    """
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name == "biovid":
        return BioVidLoader()
    elif dataset_name == "biosemi":
        return BiosemiLoader()
    else:
        raise ValueError(f"Dataset desconocido: {dataset_name}. Use 'biovid' o 'biosemi'")
