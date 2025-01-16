import importlib
import yaml
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap
from firecrown.updatable import UpdatableCollection, get_default_params
from firecrown.utils import base_model_from_yaml
from firecrown.ccl_factory import (
    CCLFactory,
    CAMBExtraParams,
    PoweSpecAmplitudeParameter,
)
from pydantic import BaseModel

from CosmoAPI.not_implemented import not_implemented_message

def extract_per_bin_systematics(yaml_data: dict) -> dict:
    """
    Extracts the per_bin_systematics parameters from the YAML data 
    and stores them in a dictionary with the naming convention 
    {probe_name}_{index}_{systematic_name}.

    Args:
        yaml_data (dict): Parsed YAML data in dictionary format.

    Returns:
        dict: Dictionary with the extracted per_bin_systematics parameters.
    """
    result = {}

    probes = yaml_data.get('probes', {})
    for probe_name, probe_data in probes.items():
        systematics = probe_data.get('systematics', {})
        per_bin_systematics = systematics.get('per_bin_systematics', [])

        for systematic in per_bin_systematics:
            systematic_type = systematic.get('type')
            for key, values in systematic.items():
                if key != 'type':
                    for idx, value in enumerate(values):
                        result[f"{probe_name}_{idx}_{key}"] = float(value)

    return result

def update_missing_keys(my_values: dict, default_values: dict) -> dict:
    """
    Update my_values dictionary by adding keys and values from default_values
    that are missing in my_values.

    Args:
        my_values (dict): The dictionary to be updated.
        default_values (dict): The dictionary containing default keys and values.

    Returns:
        dict: The updated my_values dictionary.
    """
    for key, value in default_values.items():
        if key not in my_values:
            my_values[key] = value

    return my_values

def build_firecrown_params_map_and_tools(yaml_data: dict,
                                        firecrown_updateble: UpdatableCollection,
                                        update_tools: bool = True
                                        ) -> ParamsMap:
    """
    Build a ParamsMap object from the YAML data and the UpdatableCollection.

    Args:
        yaml_data (dict): Parsed YAML data in dictionary format.
        firecrown_updateble (UpdatableCollection): UpdatableCollection object.
    """

    # build modelling tools
    tools = build_modeling_tools(yaml_data)

    # Extract the per_bin_systematics parameters
    per_bin_systematics = extract_per_bin_systematics(yaml_data)

    # cosmology dictionary
    cosmology = yaml_data.get('cosmology', {}).copy()

    # deletes the extra parameters from the cosmology dictionary
    _ = cosmology.pop('extra_parameters', {})
    combined_dict = {**cosmology, **per_bin_systematics}

    # gets the default parameters from firecrown_updateble and tools
    default_values = get_default_params(tools, firecrown_updateble)

    # update the combined_dict with the default_values
    combined_dict = update_missing_keys(combined_dict, default_values)

    firecrown_param_maps = ParamsMap(combined_dict)

    if update_tools:
        tools.update(firecrown_param_maps)
        tools.prepare()
    return firecrown_param_maps, tools


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

        # Clean up per_bin_systematics
        if 'per_bin_systematics' in systematics_yaml:
            for item in systematics_yaml['per_bin_systematics']:
                keys_to_remove = [key for key in item if key != 'type']
                for key in keys_to_remove:
                    del item[key]

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
            cosmo_config["extra_parameters"]["mass_split"]
            if "mass_split" in cosmo_config.keys()
            else "normal"
        )
        cosmo_config["extra_parameters"]["mass_split"] = mass_split

    if "A_s" in cosmo_config.keys() and "sigma8" not in cosmo_config.keys():
        psa_param = PoweSpecAmplitudeParameter.AS
    elif "sigma8" in cosmo_config.keys() and "A_s" not in cosmo_config.keys():
        psa_param = PoweSpecAmplitudeParameter.SIGMA8
    else:
        raise ValueError("The amplitude parameter must be"
                         "either A_s or sigma8")

    if "camb" in cosmo_config['extra_parameters'].keys():
        _tools = ModelingTools(
            ccl_factory=CCLFactory(
                require_nonlinear_pk=True,
                mass_split=cosmo_config["extra_parameters"]["mass_split"],
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
                mass_split=cosmo_config["extra_parameters"]["mass_split"],
            )
        )

    return _tools