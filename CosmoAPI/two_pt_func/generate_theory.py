import numpy as np
from typing import Dict, Tuple, List, Any, Type

import firecrown
from firecrown.metadata_functions import make_all_photoz_bin_combinations
import firecrown.likelihood.two_point as tp
from firecrown.utils import base_model_from_yaml
from firecrown.updatable import UpdatableCollection

from CosmoAPI.two_pt_func.nz_loader import load_all_redshift_distr
from CosmoAPI.two_pt_func.tracer_tools import (
    process_probes_load_2pt,
    generate_ell_theta_arrays,
    build_twopointxy_combinations
)
from CosmoAPI.not_implemented import not_implemented_message
from CosmoAPI.firecrown_tools import load_systematics_factory

def _generate_two_point_metadata(yaml_data: dict,
                                 two_point_function: Type,
                                 tomographic_bins: dict,
                                 scales: dict) -> List:
    """
    Generate the metadata for the two-point functions based on the YAML data.

    Parameters
    ----------
    yaml_data : dict
        Parsed YAML data in dictionary format.
    two_point_function :  firecrown.metadata_type
        The class for the two-point function.
    tomographic_bins : dict
        The tomographic bins for each probe.
    scales : dict
        A dictionary containing the scales for each probe combination

    Returns
    -------
    list
        List of metadata objects for the two-point functions.
    """
    # gets the tomographic bins for each probe:
    probes_tomo_bins = build_twopointxy_combinations(yaml_data, tomographic_bins)

     # construct the metadata for the two-point functions
    two_point_metadata_list = []
    for tracers in yaml_data['probe_combinations'].keys():
        for i in range(len(yaml_data['probe_combinations'][tracers]['bin_combinations'])):
            xy = probes_tomo_bins[tracers][i]
            #print(xy)
            if two_point_function is firecrown.metadata_types.TwoPointHarmonic:
                two_point_metadata_list.append(
                    two_point_function(XY=xy, ells=scales[tracers])
                    )
            elif two_point_function is firecrown.metadata_types.TwoPointReal:
                two_point_metadata_list.append(
                    two_point_function(XY=xy, thetas=scales[tracers])
                    )
            else:
                raise ValueError(f"Unknown two-point function type: {two_point_function}")
    return two_point_metadata_list

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
    # load the basics
    two_point_function, two_pt_probes = process_probes_load_2pt(yaml_data)

    if len(two_pt_probes) > 2:
        print(not_implemented_message)
        raise NotImplementedError("More than 2 2pt probes not implemented")

    # loads the nz_type probes
    nzs = load_all_redshift_distr(yaml_data)

    # scale arrays for each probe combination
    scales = generate_ell_theta_arrays(yaml_data)

    # load all the systematics for all probes:
    # FIXME: Need to generalise for more than 2 probes!
    probes = yaml_data.get("probes", [])
    for p in two_pt_probes.keys():
        print(f"Loading systematics for probe {p}")
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
                                                         nzs, scales,)

    # prepare all the two point functions:
    all_two_point_functions = tp.TwoPoint.from_metadata(
        metadata_seq=all_two_point_metadata,
        wl_factory=wlfact,
        nc_factory=ncfact,
    )

    return all_two_point_functions, all_two_point_metadata
