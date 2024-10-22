import sys
import importlib
import yaml
from firecrown.utils import base_model_from_yaml

from nz_loader import load_nz
sys.path.append("..")
from not_implemented import not_implemented_message

def load_metadata_function_class(function_name):
    """
    Dynamically load a class based on the 'function' name specified in the YAML file.
    FIXME: Change the docstrings
    Args:
        function_name (str): The name of the function specified in the YAML.

    Returns:
        The loaded class based on the function name.
    """
    # Assume functions are part of a module like 'firecrown.functions'
    base_module = "firecrown.metadata_functions"
    
    try:
        # Dynamically import the module
        module = importlib.import_module(base_module)
        # Get the function class from the module
        function_class = getattr(module, function_name)
        return function_class
    except ImportError as e:
        raise ImportError(f"Could not import module {base_module}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class '{function_name}' not found in module {base_module}: {e}")

def load_systematics_factory(probe_systematics):
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

def process_probes(yaml_data):
    """
    Process the probes from the YAML data, check if 'function' is the same across probes with 'nz_type',
    and dynamically load the corresponding function classes.

    Args:
        yaml_data (dict): Parsed YAML data in dictionary format.

    Returns:
        A dictionary containing the dynamically loaded function classes for each probe.
    """
    probes_data = yaml_data.get('probes', {})
    
    # Variables to track the function consistency
    nz_type_probes = []
    function_name = None
    
    function_classes = {}
    
    # Iterate over each probe in the YAML data
    for probe_name, probe_data in probes_data.items():
        nz_type = probe_data.get('nz_type')
        probe_function = probe_data.get('function')
        
        # If the probe has 'nz_type', we need to check the function
        if nz_type:
            nz_type_probes.append(probe_name)
            
            # If it's the first nz_type probe, set the expected function name
            if function_name is None:
                function_name = probe_function
            else:
                # If another nz_type probe has a different function, raise an error
                if probe_function != function_name:
                    raise ValueError(f"Probes '{nz_type_probes[0]}' and '{probe_name}' have different 'function' values: '{function_name}' != '{probe_function}'")
            
            # Dynamically load the function class
            if probe_function:
                try:
                    loaded_function = load_metadata_function_class(probe_function)
                    function_classes[probe_name] = loaded_function
                except (ImportError, AttributeError) as e:
                    raise ImportError(f"Error loading function for probe '{probe_name}': {e}")
    
    # If nz_type_probes is non-empty, it confirms nz_type presence and function consistency
    if nz_type_probes:
        print(f"All nz_type probes have the same function: {function_name}")

    return loaded_function