import getpass
import datetime
import sacc
import firecrown
import firecrown.likelihood.two_point as tp
from typing import Dict, Tuple, List, Any, Type
from firecrown.metadata_functions import InferredGalaxyZDist
from firecrown.updatable import UpdatableCollection
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap
from firecrown.metadata_types import TwoPointHarmonic, TwoPointReal

from CosmoAPI import __version__ as version
from CosmoAPI.two_pt_func.nz_loader import load_all_redshift_distr
from CosmoAPI.two_pt_func.tracer_tools import (
    process_probes_load_2pt,
    generate_ell_theta_arrays,
    build_twopointxy_combinations
)
from CosmoAPI.not_implemented import not_implemented_message
from CosmoAPI.firecrown_tools import load_systematics_factory

def _generate_two_point_metadata(
        yaml_data: dict,
        two_point_function: Type,
        tomographic_bins: dict,
        scales: dict
) -> List:
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

def prepare_2pt_functions(
        yaml_data: dict,
        tomo_z_bins: List[InferredGalaxyZDist],
) -> Tuple[UpdatableCollection, List[Any]]:
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
    #nzs = load_all_redshift_distr(yaml_data)

    # scale arrays for each probe combination
    scales = generate_ell_theta_arrays(yaml_data)

    # load all the systematics for all probes:
    # FIXME: Need to generalise for more than 2 probes!
    # Firecrown does not support more than 2 probes yet: 
    # https://github.com/LSSTDESC/firecrown/issues/480
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
                                                         tomo_z_bins, scales,)

    # prepare all the two point functions:
    all_two_point_functions = tp.TwoPoint.from_metadata(
        metadata_seq=all_two_point_metadata,
        wl_factory=wlfact,
        nc_factory=ncfact,
    )

    return all_two_point_functions, all_two_point_metadata

def construct_sacc(
        yaml_data: dict,
        tomo_z_bins: List[InferredGalaxyZDist],
        _tools: ModelingTools,
        _two_point_functions: UpdatableCollection,
        _two_point_metadata: List[Type],
        _params_maps: ParamsMap,
) -> sacc.Sacc:
    """
    Construct a sacc object based on the modeling tools, two-point functions, and parameter maps.

    Parameters
    ----------
    _tools : ModelingTools
        The modeling tools object.
    _two_point_functions : UpdatableCollection
        The two-point functions object.
    _params_maps : firecrown.ParamsMap
        The parameter maps object.

    Returns
    -------
    sacc.Sacc
        The sacc object.
    """

    # updates the tools with the parameter maps
    _two_point_functions.update(_params_maps)

    # instantiates the sacc object
    sacc_data = sacc.Sacc()
    sacc_data.metadata["time_created"] = datetime.datetime.now().isoformat()
    sacc_data.metadata["info"] = f"Synthetic sacc constructed by CosmoAPI v{version} by user {getpass.getuser()}"
    sacc_data.metadata["run_name"] = yaml_data['general']['run_name']

    # adds the redshift distributions to the sacc object
    for sample in tomo_z_bins:
        z_arr = sample.z
        dndz = sample.dndz
        sacc_tracer = sample.bin_name
        quantity = sample.measurements
        if next(iter(quantity)).name == "COUNTS":
            sacc_data.add_tracer(
                "NZ", sacc_tracer, quantity="galaxy_density", z=z_arr, nz=dndz
            )
        if next(iter(quantity)).name == "SHEAR_E":
            sacc_data.add_tracer(
                "NZ", sacc_tracer, quantity="galaxy_shear", z=z_arr, nz=dndz
            )

    for i,tp in enumerate(_two_point_functions):
        tracer0 = tp.sacc_tracers.name1
        tracer1 = tp.sacc_tracers.name2
        if type(_two_point_metadata[i]) == TwoPointHarmonic:
            _ells = tp.ells
            c_ell = tp.compute_theory_vector(_tools)
            galaxy_type = tp.sacc_data_type
            sacc_data.add_ell_cl(galaxy_type, tracer0, tracer1, _ells, c_ell)
        elif type(_two_point_metadata[i]) == TwoPointReal:
            _thetas = tp.thetas
            xi = tp.compute_theory_vector(_tools)
            galaxy_type = tp.sacc_data_type
            sacc_data.add_theta_xi(galaxy_type, tracer0, tracer1, _thetas, xi)
        else:
            raise ValueError(f"Unknown two-point function type: {type(_two_point_metadata[i])}")

    return sacc_data
