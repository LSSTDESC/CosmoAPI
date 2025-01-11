import numpy as np
from typing import Dict, Tuple, List, Any

import firecrown
from firecrown.metadata_functions import make_all_photoz_bin_combinations
import firecrown.likelihood.two_point as tp
from firecrown.utils import base_model_from_yaml
from firecrown.updatable import UpdatableCollection

from CosmoAPI.two_pt_func.nz_loader import load_all_redshift_distr
from CosmoAPI.two_pt_func.tracer_tools import process_probes_load_2pt
from CosmoAPI.two_pt_func.tracer_tools import generate_ell_theta_array
from CosmoAPI.not_implemented import not_implemented_message
from CosmoAPI.firecrown_tools import load_systematics_factory

#FIXME: Add type hints
def _generate_two_point_metadata(yaml_data: dict, two_point_function, two_pt_probes, 
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
            ells_list.append(generate_ell_theta_array(yaml_data['probes'][p], xtype))
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

def prepare_2pt_functions(yaml_data: dict) -> Tuple[UpdatableCollection, List[Any]]:
    """
    Prepare the two-point functions based on the YAML data.

    Parameters
    ----------
    yaml_data : dict
        Parsed YAML data in dictionary format.
    
    Returns
    -------
    tuple
        A tuple containing the two-point functions and the metadata 
        for the two-point functions.
    """
    # here we call this X because we do not know if it is ell_bins or theta_bins
    two_point_function, two_pt_probes = process_probes_load_2pt(yaml_data)

    if len(two_pt_probes) > 2:
        print(not_implemented_message)
        raise NotImplementedError("More than 2 2pt probes not implemented")

    # loads the nz_type probes
    nzs = load_all_redshift_distr(yaml_data)

    # make all the bin combinations:
    all_two_point_bins = make_all_photoz_bin_combinations(nzs)

    # load all the systematics for all probes:
    probes = yaml_data.get("probes", [])
    for p in two_pt_probes.keys():
        type_factory = probes[p]['systematics'].get('type')
        if type_factory == 'WeakLensingFactory':
            wlfact = load_systematics_factory(yaml_data['probes'][p]['systematics'])
        elif type_factory == 'NumberCountsFactory':
            ncfact = load_systematics_factory(yaml_data['probes'][p]['systematics'])
        else:
            raise ValueError(f"Unknown systematics type: {type_factory} for probe {p}")

    # generate the metadata for the two-point functions
    all_two_point_metadata = _generate_two_point_metadata(yaml_data,
                                                         two_point_function,
                                                         two_pt_probes.keys(),
                                                         all_two_point_bins)

    # prepare all the two point functions:
    all_two_point_functions = tp.TwoPoint.from_metadata(
        metadata_seq=all_two_point_metadata,
        wl_factory=wlfact,
        nc_factory=ncfact,
    )

    return all_two_point_functions, all_two_point_metadata
