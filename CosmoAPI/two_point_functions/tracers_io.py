from typing import Dict, Tuple, Any, List, Type

from CosmoAPI.not_implemented import not_implemented_message
from CosmoAPI.api_io import load_metadata_function_class


def process_probes_load_2pt(yaml_data: Dict[str, Any]) -> Tuple[Type, List[str]]:
    """
    Process the probes from the YAML data, check if 'function' 
    is the same across probes with 'nz_type',
    and dynamically load the corresponding function classes.

    Args:
        yaml_data (dict): Parsed YAML data in dictionary format.

    Returns:
        A tuple containing the dynamically loaded function class and a list of probe names with 'nz_type'.
    """
    probes_data = yaml_data.get('probes', {})

    # Variables to track the function consistency
    nz_type_probes = []
    probe_dict = dict()
    function_name = None

    function_classes = {}

    # Iterate over each probe in the YAML data
    for probe_name, probe_data in probes_data.items():
        nz_type = probe_data.get('nz_type')
        probe_function = probe_data.get('function')
        probe_type = probe_data.get('systematics', {}).get('type')

        # If the probe has 'nz_type', we need to check the function
        if nz_type:
            nz_type_probes.append(probe_name)
            probe_dict[probe_name] = probe_type

            # If it's the first nz_type probe, set the expected function name
            if function_name is None:
                function_name = probe_function
            else:
                # If another nz_type probe has a different function, raise an error
                if probe_function != function_name:
                    raise ValueError(
                        f"Probes '{nz_type_probes[0]}' and '{probe_name}' have different "
                        f"'function' values: '{function_name}' != '{probe_function}'"
                    )

                # Dynamically load the function class
                if probe_function:
                    try:
                        loaded_function = load_metadata_function_class(probe_function)
                        function_classes[probe_name] = loaded_function
                    except (ImportError, AttributeError) as e:
                        raise ImportError(
                            f"Error loading function for probe '{probe_name}': {e}"
                        )

    # If nz_type_probes is non-empty, it confirms nz_type presence and function consistency
    if nz_type_probes:
        print(f"All nz_type probes have the same function: {function_name}")

    return loaded_function, probe_dict