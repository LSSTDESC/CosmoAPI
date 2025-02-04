import sys
import yaml
import numpy as np
import importlib
import firecrown
from firecrown.metadata_functions import make_all_photoz_bin_combinations
import firecrown.likelihood.two_point as tp
from firecrown.utils import base_model_from_yaml


from .nz_loader import load_all_nz
sys.path.append("..")
from not_implemented import not_implemented_message
from api_io import load_metadata_function_class


def generate_ell_theta_array_from_yaml(yaml_data, type_key, dtype=float):
    """
    Generate a linear or logarithmic array based on the  configuration in the YAML data.
    
    Args:
        yaml_data (dict): Parsed YAML data in dictionary format.
        
    Returns:
        np.ndarray: Generated array based on the ell_bins configuration.
    """
    # calling thix x because it could be ell_bins or theta_bins
    x_array = yaml_data.get(type_key, {})

    array_type = x_array.get('type')
    min_val = x_array.get('min')
    max_val = x_array.get('max')
    nbins = x_array.get('nbins')

    if array_type == 'log':
        return np.unique(np.logspace(np.log10(min_val), np.log10(max_val), nbins).astype(dtype))
    elif array_type == 'linear':
        return np.linspace(min_val, max_val, nbins).astype(dtype)
    else:
        raise ValueError(f"Unknown array type: {array_type}")

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

def process_probes_load_2pt(yaml_data):
    """
    Process the probes from the YAML data, check if 'function' 
    is the same across probes with 'nz_type',
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

    return loaded_function, nz_type_probes

def generate_two_point_metadata(yaml_data, two_point_function, two_pt_probes, 
                                two_point_bins):
    """
    Generate the metadata for the two-point functions based on the YAML data.

    Parameters
    ----------
    yaml_data : dict
        Parsed YAML data in dictionary format.
    two_point_function :  firecrown.metadata_type
        The class for the two-point function.
    two_pt_probes : list
        List of probes for the two-point functions.
    two_point_bins : list
        List of two-point bins.

    Returns
    -------
    list
        List of metadata objects for the two-point functions.
    """
    # check if real or harmonic space function
    if two_point_function is firecrown.metadata_types.TwoPointHarmonic:
        xtype = 'ell_bins'
        ells_list = []
        for p in two_pt_probes:
            ells_list.append(generate_ell_theta_array_from_yaml(yaml_data['probes'][p], xtype))
        all_two_point_metadata = [two_point_function(XY=ij, ells=ell) 
                                 for ij, ell in zip(two_point_bins, ells_list)]
    elif two_point_function is firecrown.metadata_types.TwoPointReal:
        #FIXME: this is breaking for some strange reason
        print(not_implemented_message)
        raise NotImplementedError("Real space two-point functions not implemented")
        xtype = 'theta_bins'
        theta_list = []
        for p in two_pt_probes:
            theta_list.append(generate_ell_theta_array_from_yaml(yaml_data['probes'][p], xtype))
        all_two_point_metadata = [two_point_function(XY=ij, thetas=theta) 
                                 for ij, theta in zip(two_point_bins, theta_list)]
    else:
        raise ValueError("Unknown TwoPointFunction type")
    return all_two_point_metadata

def prepare_2pt_functions(yaml_data):
    # here we call this X because we do not know if it is ell_bins or theta_bins
    two_point_function, two_pt_probes = process_probes_load_2pt(yaml_data)

    if len(two_pt_probes) > 2:
        print(not_implemented_message)
        raise NotImplementedError("More than 2 2pt probes not implemented")

    # loads the nz_type probes
    nzs = load_all_nz(yaml_data)

    # make all the bin combinations:
    all_two_point_bins = make_all_photoz_bin_combinations(nzs)

    # load all the systematics for all probes:
    probes = yaml_data.get("probes", [])
    for p in two_pt_probes:
        type_factory = probes[p]['systematics'].get('type')
        if type_factory == 'WeakLensingFactory':
            wlfact = load_systematics_factory(yaml_data['probes'][p]['systematics'])
        elif type_factory == 'NumberCountsFactory':
            ncfact = load_systematics_factory(yaml_data['probes'][p]['systematics'])
        else:
            raise ValueError(f"Unknown systematics type: {type_factory} for probe {p}")

    # generate the metadata for the two-point functions
    all_two_point_metadata = generate_two_point_metadata(yaml_data,
                                                         two_point_function,
                                                         two_pt_probes,
                                                         all_two_point_bins)

    # prepare all the two point functions:
    all_two_point_functions = tp.TwoPoint.from_metadata(
        metadata_seq=all_two_point_metadata,
        wl_factory=wlfact,
        nc_factory=ncfact,
    )

    return all_two_point_functions, all_two_point_metadata
