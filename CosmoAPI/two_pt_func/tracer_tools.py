import numpy as np
from typing import Dict, Tuple, Any, List, Type
from firecrown.metadata_types import TwoPointXY
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

def generate_ell_theta_arrays(yaml_data: Dict[str, Any], dtype: Type[float] = float) -> Dict[str, np.ndarray]:
    """
    Generate linear or logarithmic arrays based on the configuration
    for all probe combinations.

    Args:
        yaml_data (dict): Parsed YAML data in dictionary format.
        dtype (Type[float]): The data type of the generated arrays.

    Returns:
        Dict[str, np.ndarray]: Dictionary of generated arrays based on the scale_cuts configuration
                               for each probe combination.
    """
    scale_arrays = {}

    probe_combinations = yaml_data.get('probe_combinations', {})
    for probe_combination, config in probe_combinations.items():
        scale_cuts = config.get('scale_cuts', {})

        array_type = scale_cuts.get('type')
        min_val = scale_cuts.get('min')
        max_val = scale_cuts.get('max')
        nbins = scale_cuts.get('nbins')

        if array_type == 'log':
            scale_arrays[probe_combination] = np.unique(np.logspace(np.log10(min_val), np.log10(max_val), nbins).astype(dtype))
        elif array_type == 'linear':
            scale_arrays[probe_combination] = np.linspace(min_val, max_val, nbins).astype(dtype)
        else:
            raise ValueError(f"Unknown array type: {array_type}")

    return scale_arrays

def build_twopointxy_combinations(config_yaml: dict, _distribution_list: list,) -> dict:

    _all_two_point_combinations = {k: [] for k in config_yaml["probe_combinations"].keys()}
    _tracer_combinations = config_yaml["probe_combinations"]

    bin_names_nz = sorted([sample.bin_name for sample in _distribution_list])


    # Check from the names of the probes if these make sense:
    for k in _all_two_point_combinations.keys():
        tracer1, tracer2 = k.split('_')
        probes = config_yaml['probes'].keys()
        if tracer1 not in probes or tracer2 not in probes:
            raise ValueError(f"Tracer combination {k} has tracers {tracer1} "
                             f"or {tracer2} not in probes {probes}")

    for tracer in _tracer_combinations.keys():
        bin_combinations = _tracer_combinations[tracer]["bin_combinations"]

        if isinstance(bin_combinations, str):
            if bin_combinations == 'all':
                # Generate all combinations
                max_bin = max(int(b.split('_')[-1]) for b in bin_names_nz if tracer.split('_')[0] in b)
                bin_combinations = [(i, j) for i in range(max_bin + 1) for j in range(i, max_bin + 1)]
                # change the bin_combinations to the numerical values in the original dictionary
                config_yaml["probe_combinations"][tracer]["bin_combinations"] = bin_combinations

            elif bin_combinations == 'autos':
                # Generate auto combinations (x = y)
                bins = [int(b.split('_')[-1]) for b in bin_names_nz if tracer.split('_')[0] in b]
                bin_combinations = [[x, x] for x in bins]
                # change the bin_combinations to the numerical values in the original dictionary
                config_yaml["probe_combinations"][tracer]["bin_combinations"] = bin_combinations
            else:
                raise ValueError(f"Unknown bin_combinations value: {bin_combinations}")
        for comb in bin_combinations:
            x = f"{tracer.split("_")[0]}_{str(comb[0])}"
            y = f"{tracer.split("_")[1]}_{str(comb[1])}"
            x_measurement = None
            y_measurement = None
            for sample in _distribution_list:
                if sample.bin_name == x:
                    x_dist = sample
                    x_measurement = next(iter(x_dist.measurements))
                if sample.bin_name == y:
                    y_dist = sample
                    y_measurement = next(iter(y_dist.measurements))
            if x_measurement is not None and y_measurement is not None:
                _all_two_point_combinations[tracer].append(
                    TwoPointXY(x=x_dist, y=y_dist,
                               x_measurement=x_measurement,
                               y_measurement=y_measurement))
    return _all_two_point_combinations