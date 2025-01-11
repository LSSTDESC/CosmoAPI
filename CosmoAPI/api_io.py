import yaml
import importlib

def load_yaml_file(yaml_file: str) -> dict:
    """
    Load the YAML configuration file.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed YAML data.
    """
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_per_bin_systematics(yaml_data: dict) -> dict:
    """
    Extracts the per_bin_systematics parameters from the YAML data and stores them in a dictionary
    with the naming convention {probe_name}_{index}_{systematic_name}.

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
                        result[f"{probe_name}_{idx}_{key}"] = value

    return result

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
