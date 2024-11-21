#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sacc generator script for the 3x2pt LSST analysis.

This Python script generates a data vector for the 3x2pt LSST analysis
using the Firecrown infrastructure and tjpcov/augur packages to generate
the covariance matrix.

The script performs the following tasks:
1. Reads a configuration file in YAML format containing all the necessary
   information to generate the full data vector.
2. Loads cosmological parameters and prepares modeling tools.
3. Generates binned redshift distributions for lens and source tracers based
   on the LSST SRD or from an external file.
4. Creates all defined two-point combinations of tracers for the analysis.
5. Constructs TwoPointHarmonic objects from the TwoPointXY objects, applying
   scale cuts.
6. Builds a Sacc object with the C_ell's and the redshift distribution.
7. Create a Gaussian fsky covariance matrix using the tjpcov package under
   the ells cut from Augur.
8. Adds the covariance matrix to the Sacc object and saves it to a SACC file.

The script is designed to be run as a standalone program and requires the
following dependencies and their respective versions:
- numpy     (version: 1.26.4)
- pyccl     (version: 3.0.2)
- yaml      (version: 6.0.2)
- sacc      (version: 0.16)
- firecrown (version: 1.8.0a0 e229d35cc20686e215f2a7b077fd48d610815789)
- augur     (version: 0.5.0)
- tjpcov    (version: 0.4.1 30d0859)

The script can be run from the command line as follows:
$ python sacc_generator.py

Configuration file:
The configuration file is a YAML file that contains the following sections:
- analysis_choices: Contains the cosmological parameters, redshift array,
  ell array, and the surveys choices.
- firecrown_parameters: Contains the firecrown parameters.
- tracer_combinations: Contains the list of tracer combinations for the
  analysis and the scale cuts for each combination.
- firecrown_factories: Contains the factories for the number counts
  and weak lensing tracers. Follow the structure of the configuration file
  to generate the data vector.
"""

import datetime
import sys
import numpy as np
import pyccl as ccl
import yaml
import sacc
from firecrown.generators.inferred_galaxy_zdist import (
    LinearGrid1D,
    ZDistLSSTSRD,
    Y1_LENS_BINS,
    Y1_SOURCE_BINS,
    Y10_LENS_BINS,
    Y10_SOURCE_BINS,
)
from firecrown.metadata_types import (
    TwoPointXY,
    TwoPointHarmonic,
    Galaxies,
    InferredGalaxyZDist,
)
from firecrown.ccl_factory import (
    CCLFactory,
    CAMBExtraParams,
    PoweSpecAmplitudeParameter,
)
from firecrown.modeling_tools import ModelingTools
from firecrown.utils import base_model_from_yaml
import firecrown.likelihood.number_counts as nc
import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.two_point as tp
from firecrown.parameters import ParamsMap
from augur.utils.cov_utils import TJPCovGaus


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


def build_modeling_tools(config: dict) -> ModelingTools:
    """
    Create a ModelingTools object from the configuration.

    Args:
        config (dict): Dictionary containing cosmology and
        systematics parameters.

    Returns:
        ModelingTools: Modeling tools object with cosmology parameters.
    """
    cosmo_config = config["analysis_choices"]["cosmo"]
    factory_config = config["firecrown_parameters"]

    if "Omega_m" in factory_config.keys():
        factory_config["Omega_c"] = (
            factory_config["Omega_m"] - factory_config["Omega_b"]
        )
        del factory_config["Omega_m"]

    if "m_nu" in factory_config.keys():
        omega_nu = factory_config["m_nu"] / (
            93.14 * factory_config["h"] ** 2
        )  # Approximation
        factory_config["Omega_c"] -= omega_nu
        mass_split = (
            cosmo_config["mass_split"]
            if "mass_split" in cosmo_config.keys()
            else "normal"
        )
        cosmo_config["mass_split"] = mass_split

    if "A_s" in cosmo_config.keys() and "sigma8" not in cosmo_config.keys():
        psa_param = PoweSpecAmplitudeParameter.AS
    elif "sigma8" in cosmo_config.keys() and "A_s" not in cosmo_config.keys():
        psa_param = PoweSpecAmplitudeParameter.SIGMA8
    else:
        raise ValueError("The amplitude parameter must be"
                         "either A_s or sigma8")

    if "extra_parameters" in cosmo_config.keys():
        _tools = ModelingTools(
            ccl_factory=CCLFactory(
                require_nonlinear_pk=True,
                mass_split=cosmo_config["mass_split"],
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
                mass_split=cosmo_config["mass_split"],
            )
        )

    firecrown_parameters = ParamsMap(factory_config)
    _tools.update(firecrown_parameters)
    _tools.prepare()

    return _tools


def get_redshift_disribution(config: dict) -> list:
    """
    Get the redshift distribution defined on the configuration file.

    Args:
        config (dict): Configuration file

    Returns:
        The binned redshift distribution
    """
    lens_distribution = config["redshift_distribution"]["lens"]
    src_distribution = config["redshift_distribution"]["src"]

    # Define the redshift array
    config_z = config["analysis_choices"]["z_array"]
    z_ = LinearGrid1D(
        start=config_z["z_start"],
        end=config_z["z_stop"],
        num=config_z["z_number"]
    )
    z_array = z_.generate()

    if lens_distribution == "SRD_Y1":
        lens_binned = get_srd_distribution_binned(z_array, tracer_name="lens",
                                                  year="1")
    elif lens_distribution == "SRD_Y10":
        lens_binned = get_srd_distribution_binned(z_array, tracer_name="lens",
                                                  year="10")
    else:
        if not src_distribution.endswith(".txt"):
            sys.exit(
                "You must provide a file path for lens or use the SRD_Y1"
                "or SRD_Y10 by default."
            )
        lens_binned = build_distribution_binned(
            lens_distribution, "lens", Galaxies.COUNTS
        )

    if src_distribution == "SRD_Y1":
        src_binned = get_srd_distribution_binned(z_array, tracer_name="src",
                                                 year="1")
    elif src_distribution == "SRD_Y10":
        src_binned = get_srd_distribution_binned(z_array, tracer_name="src",
                                                 year="10")
    else:
        if not src_distribution.endswith(".txt"):
            sys.exit(
                "You must provide a file path for lens or use the SRD_Y1"
                "or SRD_Y10 by default."
            )
        src_binned = build_distribution_binned(
            src_distribution, "src", Galaxies.SHEAR_E
        )

    # Create the list of inferred galaxy z distributions
    _distribution = lens_binned + src_binned

    return _distribution


def build_distribution_binned(
    distribution_path: str, tracer_name: str, measurements: Galaxies
) -> list[InferredGalaxyZDist]:
    """
    Import the binned distribution to build the InferredGalaxyZDist objects.

    Args:
        distribution_path (str): Path to the file containing the binned dndz.
        tracer_name (str): Name of the tracer.
        measurements (Galaxies): Measurements to associate with each inferred
                                 distribution.

    Returns:
        list: A list of InferredGalaxyZDist objects constructed from the input
        data.
    """
    dndz_binned = np.loadtxt(distribution_path)

    if dndz_binned.shape[0] < 2:
        raise ValueError(
            "Input distribution must have at least one redshift bin "
            "and one dndz value."
        )

    z_array = dndz_binned[0]
    dndz_distributions = dndz_binned[1:]

    infzdist = [
        InferredGalaxyZDist(
            bin_name=f"{tracer_name}{i}",
            z=z_array,
            dndz=dndz,
            measurements={measurements},
        )
        for i, dndz in enumerate(dndz_distributions)
    ]

    return infzdist


def get_srd_distribution_binned(z: np.ndarray, tracer_name: str,
                                year: str) -> list:
    """
    Get the binned distribution for lens/source tracer from the SRD Y1 or Y10.

    Args:
        z (np.ndarray): Redshift array.
        tracer_name (str): Name of the tracer (e.g., 'lens' or 'source').
        year (str): Year of the survey ('1' for Y1, '10' for Y10).

    Returns:
        list: List of InferredGalaxyZDist objects representing the binned
        dNdz distributions.
    """
    if "lens" in tracer_name:
        if year == "1":
            zdist = ZDistLSSTSRD.year_1_lens(use_autoknot=True,
                                             autoknots_reltol=1.0e-5)
            bin_edges = Y1_LENS_BINS["edges"]
            sigma_z = Y1_LENS_BINS["sigma_z"]
            measurements = {Galaxies.COUNTS}

        elif year == "10":
            zdist = ZDistLSSTSRD.year_10_lens(
                use_autoknot=True, autoknots_reltol=1.0e-5
            )
            bin_edges = Y10_LENS_BINS["edges"]
            sigma_z = Y10_LENS_BINS["sigma_z"]
            measurements = {Galaxies.COUNTS}

    elif "src" in tracer_name or "source" in tracer_name:
        if year == "1":
            zdist = ZDistLSSTSRD.year_1_source(
                use_autoknot=True, autoknots_reltol=1.0e-5
            )
            bin_edges = Y1_SOURCE_BINS["edges"]
            sigma_z = Y1_SOURCE_BINS["sigma_z"]
            measurements = {Galaxies.SHEAR_E}

        elif year == "10":
            zdist = ZDistLSSTSRD.year_10_source(
                use_autoknot=True, autoknots_reltol=1.0e-5
            )
            bin_edges = Y10_SOURCE_BINS["edges"]
            sigma_z = Y10_SOURCE_BINS["sigma_z"]
            measurements = {Galaxies.SHEAR_E}

    dndz_binned = [
        zdist.binned_distribution(
            zpl=bin_edges[i],
            zpu=bin_edges[i + 1],
            sigma_z=sigma_z,
            z=z,
            name=f"{tracer_name}{i}",
            measurements=measurements,
        )
        for i in range(len(bin_edges) - 1)
    ]

    return dndz_binned


def build_twopointxy_combinations(
    _distribution_list: list, _tracer_combinations: list
) -> dict:
    """
    Create the two-point combinations of tracers using TwoPointXY objects.

    Args:
        _distribution_list (list): List of distribution objects.
        _tracer_combinations (list): List of combinations.

    Returns:
        dict: Dictionary with TwoPointXY objects for each tracer
    """
    _all_two_point_combinations = {"lens_lens": [], "lens_src": [],
                                   "src_src": []}
    for tracer in _tracer_combinations.keys():
        for comb in _tracer_combinations[tracer]["combinations"]:
            x = tracer.split("_")[0] + str(comb[0])
            y = tracer.split("_")[1] + str(comb[1])
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


def build_metadata_cells(
    config: dict,
    _two_point_comb: list,
    _cosmo: ccl.Cosmology,
    _ells: np.ndarray,
) -> list:
    """
    Create the TwoPointHarmonic objects from the TwoPointXY.

    This function apply the scale cuts define in the configuration file.
    Args:
        config (dict): Configuration file
        _two_point_comb (list): List of TwoPointXY objects.
        _cosmo (ccl.Cosmology): Cosmology object.
        _ells (np.ndarray): Array of ells.

    Returns:
        list: List of TwoPointHarmonic objects.
    """
    _tracer_combinations = config["tracer_combinations"]
    kmax = _tracer_combinations["lens_lens"]["kmax"]
    lmax = _tracer_combinations["src_src"]["lmax"]
    two_points_cells = []
    for tracer in _tracer_combinations.keys():
        for i in range(len(_tracer_combinations[tracer]["combinations"])):
            xy = _two_point_comb[tracer][i]
            if tracer == "lens_lens":
                z_avg1 = np.average(xy.x.z,
                                    weights=xy.x.dndz / np.sum(xy.x.dndz))
                z_avg2 = np.average(xy.y.z,
                                    weights=xy.y.dndz / np.sum(xy.y.dndz))
                a = np.array([1. / (1 + z_avg1), 1. / (1 + z_avg2)])
                scale_cut = np.min(
                    (kmax * ccl.comoving_radial_distance(_cosmo, a)) - 0.5
                    )
                ells_cut = _ells[_ells <= scale_cut].astype(np.int32)
            elif tracer == "lens_src":
                z_avg = np.average(xy.x.z,
                                   weights=xy.x.dndz / np.sum(xy.x.dndz))
                a = 1. / (1 + z_avg)
                scale_cut = np.min(
                    (kmax * ccl.comoving_radial_distance(_cosmo, a)) - 0.5
                    )
                ells_cut = _ells[_ells <= scale_cut].astype(np.int32)
            elif tracer == "src_src":
                ells_cut = _ells[_ells <= lmax].astype(np.int32)
            else:
                sys.exit("Tracer name must be 'lens_lens',"
                         "'src_src' or 'lens_src'.")
            two_points_cells.append(TwoPointHarmonic(XY=xy, ells=ells_cut))

    return two_points_cells


def build_sacc_file(
    _tools: ModelingTools, _distribution_list: list, _two_point_functions: list
) -> sacc.Sacc:
    """
    Create the sacc object with the computed Cell's.

    Args:
        _tools (ModelingTools): Modeling tools object with cosmology.
        _distribution_list (list): List of distribution objects.
        _two_point_functions (list): List of TwoPoint objects.

    Returns:
        sacc.Sacc: Sacc object.
    """
    # Initialize sacc object
    _sacc_data = sacc.Sacc()
    _sacc_data.metadata["start"] = datetime.datetime.now().isoformat()
    _sacc_data.metadata["info"] = "Mock data vector and covariance matrix"

    # Adding tracers to the sacc object from _distribution_list
    for sample in _distribution_list:
        z_arr = sample.z
        dndz = sample.dndz
        sacc_tracer = sample.bin_name
        quantity = sample.measurements

        if next(iter(quantity)).name == "COUNTS":
            _sacc_data.add_tracer(
                "NZ", sacc_tracer, quantity="galaxy_density", z=z_arr, nz=dndz
            )
        if next(iter(quantity)).name == "SHEAR_E":
            _sacc_data.add_tracer(
                "NZ", sacc_tracer, quantity="galaxy_shear", z=z_arr, nz=dndz
            )

    # Adding c_ells to the sacc file from two_point_functions
    for tw in _two_point_functions:
        tracer0 = tw.sacc_tracers.name1
        tracer1 = tw.sacc_tracers.name2

        _ells = tw.ells
        c_ell = tw.compute_theory_vector(_tools)
        galaxy_type = tw.sacc_data_type
        _sacc_data.add_ell_cl(galaxy_type, tracer0, tracer1, _ells, c_ell)

    return _sacc_data


def build_cov_dict(
    _tools: ModelingTools,
    _sacc_data: sacc.Sacc,
    _config: dict,
    _ells_edges: np.ndarray,
) -> dict:
    """
    Create a dictionary to be read by the tjpcov package.

    Args:
        _tools (ModelingTools): Modeling tools object.
        _sacc_data (sacc.Sacc): Sacc object.
        _config (dict): Configuration dictionary.
        _ells_edges (np.ndarray): Array of ell edges.

    Returns:
        dict: Dictionary to be read by the tjpcov package.
    """
    config = _config["analysis_choices"]
    # Create the dictionary to be read by the tjpcov package
    tjpcov_config = {"tjpcov": {"cosmo": _tools.ccl_cosmo,
                                "sacc_file": _sacc_data}}

    for tracer_name in _sacc_data.tracers:
        if tracer_name.startswith("lens"):
            lens_config = config["surveys_choices"]["tracers"]["lens"]
            tjpcov_config["tjpcov"].update(
                {
                    f"Ngal_{tracer_name}": lens_config["ngal"][tracer_name],
                    f"bias_{tracer_name}": lens_config["bias"][tracer_name],
                }
            )

        if tracer_name.startswith("src"):
            src_config = config["surveys_choices"]["tracers"]["src"]
            tjpcov_config["tjpcov"].update(
                {
                    f"Ngal_{tracer_name}": src_config["ngal"][tracer_name],
                    f"sigma_e_{tracer_name}": src_config["sigma_e"],
                    "IA": src_config["ia"],
                }
            )

    # Create three FourierGaussianFsky for each fsky value of the surveys to
    # build the covariance blocks based on the tracers.
    tjpcov_config["GaussianFsky"] = {"fsky": config["surveys_choices"]["fsky"]}
    tjpcov_config["tjpcov"]["binning_info"] = {"ell_edges": _ells_edges}
    print("\n", tjpcov_config, "\n")
    return tjpcov_config


def build_covariance_matrix(
    _tools: ModelingTools,
    _sacc_data: sacc.Sacc,
    _config: dict,
    _ells_edges: np.ndarray,
) -> np.ndarray:
    """
    Create a Gaussian fsky covariance matrix using tjpcov.

    Args:
        _tools (ModelingTools): Modeling tools object.
        _sacc_data (sacc.Sacc): Sacc object.
        _config (dict): Configuration dictionary.
        _ells_edges (np.ndarray): Array of ell edges.

    Returns:
        np.ndarray: Covariance matrix.
    """
    # Create the dictionary to be read by the tjpcov package
    tjpcov_config = build_cov_dict(_tools, _sacc_data, _config, _ells_edges)
    cov_calc = TJPCovGaus(tjpcov_config)

    # Build the covariance matrix based on the tracers
    tracers_comb = _sacc_data.get_tracer_combinations()
    ndata = len(_sacc_data.mean)
    cov_matrix = np.zeros((ndata, ndata))

    for i, trs1 in enumerate(tracers_comb):
        ii = _sacc_data.indices(tracers=trs1)

        for trs2 in tracers_comb[i:]:
            print(trs1, trs2)
            jj = _sacc_data.indices(tracers=trs2)
            ii_all, jj_all = np.meshgrid(ii, jj, indexing="ij")

            cov_blocks = cov_calc.get_covariance_block(
                trs1, trs2, include_b_modes=False
            )
            cov_matrix[ii_all, jj_all] = cov_blocks[: len(ii), : len(jj)]
            cov_matrix[jj_all.T, ii_all.T] = cov_blocks[: len(ii), : len(jj)].T

    return cov_matrix


if __name__ == "__main__":
    # Load the configuration file
    config_file = load_yaml_file("./config_yamls/config_3x2pt_.yaml")

    # Define the cosmology
    tools = build_modeling_tools(config_file)
    ccl_cosmo = tools.ccl_cosmo
    ccl_cosmo.compute_nonlin_power()

    # Create the ParamsMap object with the firecrown parameters needed
    firecrown_params = ParamsMap(config_file["firecrown_parameters"])

    distribution_list = get_redshift_disribution(config_file)

    # Create the TwoPointXY objects of the combinations from the config file
    config_tracer_combinations = config_file["tracer_combinations"]
    all_two_point_combinations = build_twopointxy_combinations(
        distribution_list, config_tracer_combinations
    )

    # Create the ell bins
    config_ell = config_file["analysis_choices"]["ell_array"]
    ells_edges = np.unique(
        np.geomspace(
            config_ell["ell_start"],
            config_ell["ell_stop"],
            config_ell["ell_bins"] + 1,
            endpoint=True,
        )
    ).astype(np.int32)
    # Linear average exclusive.
    ells = 0.5 * (ells_edges[:-1] + ells_edges[1:])

    # Create the weak lensing and number counts factories from the
    # configuration file
    config_factories = config_file["firecrown_factories"]
    NCF_CONFIG = str(config_factories["nc_factory"])
    ncf = base_model_from_yaml(nc.NumberCountsFactory, NCF_CONFIG)

    WLF_CONFIG = str(config_factories["wl_factory"])
    wlf = base_model_from_yaml(wl.WeakLensingFactory, WLF_CONFIG)

    # Create the TwoPointHarmonic with scale cuts
    all_two_points_cells = build_metadata_cells(
        config_file, all_two_point_combinations, ccl_cosmo, ells
    )

    # Create the TwoPoint objects from the metadata
    all_two_points_functions = tp.TwoPoint.from_metadata(
        metadata_seq=all_two_points_cells,
        wl_factory=wlf,
        nc_factory=ncf,
    )

    # Update the TwoPoint objects with the systematics parameters
    all_two_points_functions.update(firecrown_params)

    # Build the Sacc object
    sacc_data = build_sacc_file(tools, distribution_list,
                                all_two_points_functions)

    # Build the covariance matrix
    covariance_matrix = build_covariance_matrix(
        tools, sacc_data, config_file, ells_edges
    )

    # Add the covariance matrix to the Sacc object and save it
    sacc_data.add_covariance(covariance_matrix)

    # Save the Sacc object
    sacc_data.metadata["stop"] = datetime.datetime.now().isoformat()
    sacc_data.save_fits(
        f"./sacc_files/{config_file['sacc_name']}",
        overwrite=True)

    print("Sacc file was completed and saved")
    print("Start time: ", sacc_data.metadata["start"])
    print("End time: ", sacc_data.metadata["stop"])
