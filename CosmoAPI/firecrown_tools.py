import importlib
import yaml
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap
from firecrown.utils import base_model_from_yaml
from firecrown.ccl_factory import (
    CCLFactory,
    CAMBExtraParams,
    PoweSpecAmplitudeParameter,
)
from pydantic import BaseModel

from CosmoAPI.not_implemented import not_implemented_message

def load_systematics_factory(probe_systematics: dict) -> BaseModel: 
    """
    Dynamically load a class based on the systematics 'type' specified in the YAML file.

    Args:
        systematics_type (str): The 'type' field from the YAML specifying which factory to use.

    Returns:
        The loaded class from the firecrown library.
    """
    # Define base module path based on firecrown's library structure
    base_module = "firecrown.likelihood"

    # Mapping of known factories to their submodules
    type_to_submodule = {
        'WeakLensingFactory': 'weak_lensing',
        'NumberCountsFactory': 'number_counts',
        # Add other mappings as needed, or consider an automatic lookup if patterns are consistent
    }

    systematics_type = probe_systematics['type']
    # Get the submodule for the type
    submodule = type_to_submodule.get(systematics_type)

    if submodule is None:
        print(not_implemented_message)
        raise ImportError(f"Unknown systematics type: {systematics_type}")

    # Construct the full module path
    module_path = f"{base_module}.{submodule}"

    try:
        # Dynamically import the module
        module = importlib.import_module(module_path)
        # Get the class from the module
        factory_class = getattr(module, systematics_type)
        # copy the systematics dictionary
        systematics_yaml = probe_systematics.copy()
        # remove the type key
        del systematics_yaml['type']
        # instantiate the factory
        factory = base_model_from_yaml(factory_class, yaml.dump(systematics_yaml))
        return factory
    except ImportError as e:
        raise ImportError(f"Could not import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class '{systematics_type}' not found in module {module_path}: {e}")

def build_modeling_tools(config: dict) -> ModelingTools:
    """
    Create a ModelingTools object from the configuration.

    Args:
        config (dict): Dictionary containing cosmology and
        systematics parameters.

    Returns:
        ModelingTools: Modeling tools object with cosmology parameters.
    """
    cosmo_config = config["cosmology"]
    #factory_config = config["firecrown_parameters"]

    if "Omega_m" in cosmo_config.keys():
        cosmo_config["Omega_c"] = (
            cosmo_config["Omega_m"] - cosmo_config["Omega_b"]
        )
        del cosmo_config["Omega_m"]

    if "m_nu" in cosmo_config.keys():
        mass_split = (
            cosmo_config["mass_split"]
            if "mass_split" in cosmo_config.keys()
            else "normal"
        )
        cosmo_config["mass_split"] = mass_split

    if "A_s" in cosmo_config.keys() and "sigma8" not in cosmo_config.keys():
        psa_param = PoweSpecAmplitudeParameter.AS
    elif "sigma8" in cosmo_config.keys() and "A_s" not in cosmo_config.keys():
        psa_param = PoweSpecAmplitudeParameter.SIGMA8
    else:
        raise ValueError("The amplitude parameter must be"
                         "either A_s or sigma8")

    if "extra_parameters" in cosmo_config.keys():
        _tools = ModelingTools(
            ccl_factory=CCLFactory(
                require_nonlinear_pk=True,
                mass_split=cosmo_config["mass_split"],
                amplitude_parameter=psa_param,
                camb_extra_params=CAMBExtraParams(
                    **cosmo_config["extra_parameters"]["camb"]
                ),
            )
        )
    else:
        _tools = ModelingTools(
            ccl_factory=CCLFactory(
                require_nonlinear_pk=True,
                amplitude_parameter=psa_param,
                mass_split=cosmo_config["mass_split"],
            )
        )

    return _tools