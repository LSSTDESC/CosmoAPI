"""The Sacc generator using Firecrown infrastructure."""
import numpy as np
import yaml
import sacc
import pyccl as ccl
#from tjpcov.covariance_gaussian_fsky import FourierGaussianFsky
from firecrown.generators.inferred_galaxy_zdist import LinearGrid1D, ZDistLSSTSRD, Y1_LENS_BINS, Y1_SOURCE_BINS, Y10_LENS_BINS, Y10_SOURCE_BINS 
from firecrown.metadata_types import Galaxies, InferredGalaxyZDist
from firecrown.metadata_types import TwoPointXY, TwoPointHarmonic
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
from firecrown.ccl_factory import CCLFactory
from firecrown.utils import base_model_from_yaml
from firecrown.likelihood.two_point import TwoPoint
from firecrown.parameters import ParamsMap
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import get_default_params
from augur.utils.cov_utils import TJPCovGaus
import time

def load_yaml(config_path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed YAML data.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_SRD_distribution_binned(z: np.ndarray, tracer_name: str, year: str) -> list[InferredGalaxyZDist]:
    """Get the binned dNdz distributions for lens or sources tracers from the SRD Y1 or Y10.

    Args:
        z (np.ndarray): Redshift array.
        tracer_name (str): Name of the tracer

    Returns:
        list: List of InferredGalaxyZDist objects.
    """
    if 'lens' in tracer_name:
        if year == '1':
            zdist = ZDistLSSTSRD.year_1_lens(use_autoknot=True, autoknots_reltol=1.0e-5)
            bin_edges = Y1_LENS_BINS['edges']
            sigma_z = Y1_LENS_BINS['sigma_z']
            measurements = {Galaxies.COUNTS}
        elif year == '10':
            zdist = ZDistLSSTSRD.year_10_lens(use_autoknot=True, autoknots_reltol=1.0e-5)
            bin_edges = Y10_LENS_BINS['edges']
            sigma_z = Y10_LENS_BINS['sigma_z']
            measurements = {Galaxies.COUNTS}
    elif 'src' or 'source' in tracer_name:
        if year == '1':
            zdist = ZDistLSSTSRD.year_1_source(use_autoknot=True, autoknots_reltol=1.0e-5)
            bin_edges = Y1_SOURCE_BINS['edges']
            sigma_z = Y1_SOURCE_BINS['sigma_z']
            measurements = {Galaxies.SHEAR_E}
        elif year == '10':
            zdist = ZDistLSSTSRD.year_10_source(use_autoknot=True, autoknots_reltol=1.0e-5)
            bin_edges = Y10_SOURCE_BINS['edges']
            sigma_z = Y10_SOURCE_BINS['sigma_z']
            measurements = {Galaxies.SHEAR_E}
    dndz_binned = []
    for i in range(len(bin_edges)-1):
        dndz_binned.append(zdist.binned_distribution(zpl=bin_edges[i], zpu=bin_edges[i+1], sigma_z=sigma_z, z=z, name=f'{tracer_name}{i}', measurements=measurements))
    return dndz_binned

def build_distribution_binned(z: np.ndarray, distribution_path: str, tracer_name: str, measurements: Galaxies) -> list[InferredGalaxyZDist]:
    """Inport the binned dNdz distributions for lens or sources to build the InferredGalaxyZDist objects.

    Args:
        z (np.ndarray): Redshift array.
        distribution_path (str): Path to the distribution file.
        tracer_name (str): Name of the tracer.
        measurements (Galaxies): Measurement type.

    Returns:
        list: List of InferredGalaxyZDist objects.
    """
    dndz_binned = np.loadtxt(distribution_path).T
    infzdist = []
    for i in range(len(dndz_binned)):
        infzdist_binned = InferredGalaxyZDist(
            bin_name=f"{tracer_name}{i}",
            z=z,
            dndz=dndz_binned[i],
            measurements={measurements},
            
        )
        infzdist.append(infzdist_binned)
    return infzdist


def build_twopointxy_combinations(distribution_list: list[InferredGalaxyZDist], combinations: list) -> list[TwoPointXY]:
    """Create all possible two-point combinations of tracers for the analysis using TwoPointXY objects.

    Args:
        distribution_list (list): List of distribution objects.
        combinations (list): List of combinations.

    Returns:
        list: List of TwoPointXY objects.
    """
    all_two_point_combinations = []
    for comb in combinations:
        x = comb['x']
        y = comb['y']
        for sample in distribution_list:
            if sample.bin_name == x:
                x_dist = sample
                x_measurement = next(iter(x_dist.measurements))
            if sample.bin_name == y:
                y_dist = sample
                y_measurement = next(iter(y_dist.measurements))
        all_two_point_combinations.append(
            TwoPointXY(x=x_dist, y=y_dist, x_measurement=x_measurement, y_measurement=y_measurement)
        )
    return all_two_point_combinations


def build_sacc_file(tools: ModelingTools, distribution_list: list[InferredGalaxyZDist], all_two_points_functions: list[TwoPoint]) -> sacc.Sacc:
    """Create the Sacc object witohut the covariance matrix using firecrown infrastructure.

    Args:
        tools (ModelingTools): Modeling tools object.
        distribution_list (list): List of distribution objects.
        all_two_points_functions (list): List of TwoPoint objects.

    Returns:
        sacc.Sacc: Sacc object.
    """
    sacc_data = sacc.Sacc()
    # Add the tracers to the Sacc object
    for sample in distribution_list:
        z = sample.z
        dndz = sample.dndz
        sacc_tracer = sample.bin_name
        quantity = sample.measurements
        if next(iter(quantity)).name == 'COUNTS':
            sacc_data.add_tracer('NZ', sacc_tracer, quantity='galaxy_density', z=z, nz=dndz)
        if next(iter(quantity)).name == 'SHEAR_E':
            sacc_data.add_tracer('NZ', sacc_tracer, quantity='galaxy_shear', z=z, nz=dndz)

    # Add the two-point functions to the Sacc object
    for tw in all_two_points_functions:
        tracer0 = tw.sacc_tracers.name1
        tracer1 = tw.sacc_tracers.name2
        ells = tw.ells
        C_ell = tw.compute_theory_vector(tools)
        galaxy_type = tw.sacc_data_type
        sacc_data.add_ell_cl(galaxy_type, tracer0, tracer1, ells, C_ell)
    return sacc_data


def build_covariance_matrix(tools: ModelingTools, sacc_data: sacc.Sacc, config: dict, ells_edges: np.ndarray) -> np.ndarray:
    """Create the a gaussian fsky covariance matrix for multiple surveys using tjpcov.

    Args:
        tools (ModelingTools): Modeling tools object.
        sacc_data (sacc.Sacc): Sacc object.
        config (dict): Configuration dictionary.
        ell_edges (np.ndarray): Array of ell edges.

    Returns:
        np.ndarray: Covariance matrix.
    """
    tjpcov_config = {'tjpcov': {'cosmo': tools.ccl_cosmo, 'sacc_file': sacc_data}}
    for tracer_name in sacc_data.tracers:
        if tracer_name.startswith('src'):
            src_config = config['analysis_choices']['surveys_choices']['lsst']['tracers']['src']
            tjpcov_config['tjpcov'].update({
                f'Ngal_{tracer_name}': src_config['ngal'][tracer_name],
                f'sigma_z_{tracer_name}': src_config['sigma_z'],
                f'sigma_e_{tracer_name}': src_config['sigma_e'],
            })
            tjpcov_config['tjpcov']['IA'] = src_config['ia']
        if tracer_name.startswith('spec'):
            spec_config = config['analysis_choices']['surveys_choices']['desi']['tracers']['spec']
            tjpcov_config['tjpcov'].update({
                f'Ngal_{tracer_name}': spec_config['ngal'][tracer_name],
                f'sigma_z_{tracer_name}': spec_config['sigma_z'],
                f'bias_{tracer_name}': spec_config['bias'][tracer_name]
            })
        elif tracer_name.startswith('lens'):
            lens_config = config['analysis_choices']['surveys_choices']['lsst']['tracers']['lens']
            tjpcov_config['tjpcov'].update({
                f'Ngal_{tracer_name}': lens_config['ngal'][tracer_name],
                f'sigma_z_{tracer_name}': lens_config['sigma_z'],
                f'bias_{tracer_name}': lens_config['bias'][tracer_name]
            })

    # create three FourierGaussianFsky for each fsky value of the surveys to build the covariance blocks based on the tracers
    tjpcov_config['GaussianFsky'] = {'fsky': config['analysis_choices']['surveys_choices']['lsst']['fsky']}
    tjpcov_config['tjpcov']['binning_info'] = {'ell_edges': ells_edges}
    print(tjpcov_config)

    cov_calc = TJPCovGaus(tjpcov_config)
    # build the covariance matrix based on the tracers
    tracers_comb = sacc_data.get_tracer_combinations()
    ndata = len(sacc_data.mean)
    matrix = np.zeros((ndata, ndata))
    for i, trs1 in enumerate(tracers_comb):
        ii = sacc_data.indices(tracers=trs1)
        for trs2 in tracers_comb[i:]:
            print(trs1, trs2)
            jj = sacc_data.indices(tracers=trs2)
            #print(ii, jj)
            ii_all, jj_all = np.meshgrid(ii, jj, indexing='ij')
            cov_blocks = cov_calc.get_covariance_block(trs1, trs2, include_b_modes=False)
            matrix[ii_all, jj_all] = cov_blocks[:len(ii), :len(jj)]
            matrix[jj_all.T, ii_all.T] = cov_blocks[:len(ii), :len(jj)].T            
    return matrix

if __name__ == "__main__":
    """Build to generate Sacc object to be read for Firecrown later."""
    # time the execution
    start = time.time()

    # chose the analysis and the type of galaxies
    analysis = '3x2pt'
    type_galaxies = ''
    print(f'You are generating the sacc file for the {analysis} analysis with {type_galaxies} galaxies')

    # Load the configuration file
    config_file = load_yaml(f'./config_yamls/config_{analysis}_{type_galaxies}.yaml')
    config_analysis = config_file['analysis_choices']
  
    # load the systematics and cosmological parameters from the config
    config_params = config_file['firecrown_parameters']

    # Create the ParamsMap object with the systematics parameters needed
    params = ParamsMap(config_params)

    # Create the ModelingTools object
    tools = ModelingTools(ccl_factory=CCLFactory(require_nonlinear_pk=True))

    # Update the ModelingTools and the TwoPoint functions objects with the systematics and cosmological parameters
    tools.update(params)
    tools.prepare()
    cosmo = tools.ccl_cosmo

    # Define the redshift array
    config_z = config_analysis['z_array']
    z = LinearGrid1D(start=config_z['z_start'], end=config_z['z_stop'], num=config_z['z_number'])
    z_arr = z.generate()

    # generate the binned dNdz distributions for lens and source galaxies using the firecrown infrastructure
    lsst_y1_lens_binned = get_SRD_distribution_binned(z_arr, tracer_name='lens', year='1')
    lsst_y1_src_binned = get_SRD_distribution_binned(z_arr, tracer_name='src', year='1')
    
    distribution_list = lsst_y1_lens_binned + lsst_y1_src_binned

    # Create the TwoPointXY objects
    config_tracer_combinations = config_file['tracer_combinations']
    all_two_point_combinations = build_twopointxy_combinations(distribution_list, config_tracer_combinations)
    
    # Create the ell edges and the ell center
    config_ell = config_analysis['ell_array']
    ells_edges = np.geomspace(config_ell['ell_start'], config_ell['ell_stop'], config_ell['ell_bins'], endpoint=True)
    ells = np.sqrt(ells_edges[:-1] * ells_edges[1:]) #geometric average

    #####################################################################################################################################################
    # Create the TwoPointHarmonic objects
    #FIXME: put this in a function perhaps. For now, it is a temporary hack to use scale
    all_two_points_cells = []
    for xy in all_two_point_combinations:
        x_tracer = xy.x.bin_name
        y_tracer = xy.y.bin_name
        for comb in config_file['tracer_combinations']:
            if comb['x'] == x_tracer and comb['y'] == y_tracer:
                if 'lmax' in comb.keys():
                    ells_cut = ells[ells <= comb['lmax']]
                    #print(f'ell cut for {x_tracer} and {y_tracer} is {comb["lmax"]}')
                elif 'kmax' in comb.keys():
                    if 'lens' in x_tracer and 'lens' in y_tracer:
                        kmax = comb['kmax']
                        z_avg1 = np.average(xy.x.z, weights=xy.x.dndz/np.sum(xy.x.dndz))
                        z_avg2 = np.average(xy.y.z, weights=xy.y.dndz/np.sum(xy.y.dndz))
                        a = np.array([1./(1+z_avg1), 1./(1+z_avg2)])
                        scale_cut = np.min((kmax*ccl.comoving_radial_distance(cosmo, a))-0.5)
                    else:
                        kmax = comb['kmax']
                        z_avg = np.average(xy.x.z, weights=xy.x.dndz/np.sum(xy.x.dndz))
                        a = 1./(1+z_avg)
                        scale_cut = np.min((kmax*ccl.comoving_radial_distance(cosmo, a))-0.5)
                    ells_cut = ells[ells <= scale_cut].astype(int)
                    #print(f'ell cut for {x_tracer} and {y_tracer} is {scale_cut}')
                else:
                    print('No ell cut')
                    ells_cut = ells.astype(int)
                #ells_cut = ells.astype(int) #FIXME: this is a temporary hack to use not cuts ell values. To do it, just unconment.
        all_two_points_cells.append(TwoPointHarmonic(XY=xy, ells=ells_cut))
    ###################################################################################################################################################

    # Create the weak lensing and number counts factories from the configuration file
    config_factories = config_file['firecrown_factories']
    ncf_config = str(config_factories['nc_factory'])
    ncf = base_model_from_yaml(nc.NumberCountsFactory, ncf_config)
    wlf_config = str(config_factories['wl_factory'])
    wlf = base_model_from_yaml(wl.WeakLensingFactory, wlf_config)

    # Create the TwoPoint objects from the metadata
    all_two_points_functions = TwoPoint.from_metadata(
        metadata_seq=all_two_points_cells,
        wl_factory=wlf,
        nc_factory=ncf,
    )
    # Update the TwoPoint objects with the parameters from the configuration file
    all_two_points_functions.update(params)

    # Build the Sacc object
    sacc_data = build_sacc_file(tools, distribution_list, all_two_points_functions)
    #sacc_data.save_fits(f"./sacc_files/lsst_y1_{analysis}_{type_galaxies}.sacc", overwrite=True)

    # Build the covariance matrix
    cov_matrix = build_covariance_matrix(tools, sacc_data, config_file, ells_edges)
    
    # Add the covariance matrix to the Sacc object and save it
    sacc_data.add_covariance(cov_matrix)
    sacc_data.save_fits(f"./sacc_files/lsst_y1_{analysis}_{type_galaxies}.sacc", overwrite=True)
    end = time.time()
    print('Now your sacc is ready to be used in the Firecrown analysis')
    print(f'The execution time was: {int((end-start)//60)} min {int((end-start)% 60)} s')
    
