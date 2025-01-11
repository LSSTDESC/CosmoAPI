import importlib
import sys
from typing import Dict, List, Any, Type
sys.path.append("..")
from not_implemented import not_implemented_message

_DESC_SCENARIOS = {"LSST_Y10_SOURCE_BIN_COLLECTION", "LSST_Y10_LENS_BIN_COLLECTION",
                   "LSST_Y1_LENS_BIN_COLLECTION", "LSST_Y1_SOURCE_BIN_COLLECTION",}

def _load_nz(yaml_data: Dict[str, Any]) -> List[Any]:
    try: 
        nz_type = yaml_data["nz_type"]
    except KeyError:
        raise ValueError("nz_type not found in 2pt yaml section")

    if nz_type in _DESC_SCENARIOS:
        return _load_nz_from_module(nz_type).generate()
    else:
        raise NotImplementedError(not_implemented_message)

def load_all_nz(yaml_data: Dict[str, Any]) -> List[Any]:
    nzs = []
    for probe, propr in yaml_data['probes'].items():
        if 'nz_type' in propr:
            #print(propr['nz_type'])
            nzs += _load_nz(propr)
    return nzs

def _load_nz_from_module(nz_type: str) -> Type:
    # Define the module path
    module_path = "firecrown.generators.inferred_galaxy_zdist"

    try:
        # Dynamically import the module and the object
        module = importlib.import_module(module_path)
        nz_type_class = getattr(module, nz_type)
        return nz_type_class
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"'{nz_type}' not found in module {module_path}: {e}")
