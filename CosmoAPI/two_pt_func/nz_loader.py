import numpy as np
from typing import Type
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
    TwoPointReal,
    Galaxies,
    InferredGalaxyZDist,
)

from CosmoAPI.two_pt_func.tracer_tools import process_probes_load_2pt
from CosmoAPI.not_implemented import not_implemented_message


def load_all_redshift_distr(yaml_data: dict) -> list:
    """
    Loads all the redshift distributions from the configuration file.

    Args:
        yaml_data (dict): Parsed YAML data in dictionary format.

    Returns:
        list: List of redshift distributions.
    """
    # gets the 2pt function type and a dict of probes and probe types
    two_pt_function, probes_dict = process_probes_load_2pt(yaml_data)
    nzs = []
    for probe, prob_type in probes_dict.items():
        _distr = _get_redshift_disribution(yaml_data, probe,
                                          prob_type, two_pt_function)
        nzs += _distr
    return nzs

def _get_redshift_disribution(config: dict, probe_name: str, probe_type: str,
                             two_pt_type: Type) -> list:
    """
    Get the redshift distribution defined on the configuration file.

    Args:
        config (dict): Configuration file
        probe_name (str): Name of the probe.
        probe_type (str): Type of the probe.
        two_pt_type (Type): Type of the two point function.

    Returns:
        The binned redshift distribution
    """
    try:
        _nz_type = config['probes'][probe_name]['nz_type']
    except KeyError:
        raise KeyError(f"Probe '{probe_name}' not a Nz Tracer!")

    try:
        config_z = config["probes"][probe_name]["z_array"]
        z_ = LinearGrid1D(
            start=config_z["z_min"],
            end=config_z["z_max"],
            num=config_z["z_number"]
        )
    except KeyError:
        print("No z_array provided. Using default redshift array from 0.0001 to 3.5")
        z_ = LinearGrid1D(start=0.0001, end=3.5, num=1000)

    # generate the redshift array
    z_array = z_.generate()

    if _nz_type == "SRD_Y1":
        nz_binned = _get_srd_distribution_binned(z_array, tracer_name=probe_name,
                                                 tracer_type=probe_type,
                                                 function_type=two_pt_type,
                                                 year="1")
    elif _nz_type == "SRD_Y10":
        nz_binned = _get_srd_distribution_binned(z_array, tracer_name=probe_name,
                                                 tracer_type=probe_type,
                                                 function_type=two_pt_type,
                                                 year="1")
    elif _nz_type == "file":
        nz_file = config['probes'][probe_name]['nz_file']
        nz_binned = _build_distribution_binned(nz_file, probe_name,
                                               probe_type, two_pt_type)
    elif _nz_type == "sacc":
        raise NotImplementedError(
            "sacc support not implemented yet" + not_implemented_message
        )
    else:
        raise ValueError(
            "Unknown nz_type, valid options are 'SRD_Y1', 'SRD_Y10', 'file' and 'sacc'"
        )

    return nz_binned

def _build_distribution_binned(distribution_path: str,
                              tracer_name: str,
                              tracer_type: str,
                              two_pt_type: Type) -> list[InferredGalaxyZDist]:
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
    try:
        dndz_binned = np.loadtxt(distribution_path)
    except FileNotFoundError: #FIXME: Need to change this part!
        raise FileNotFoundError(f"File '{distribution_path}' not found")
    except (UnicodeDecodeError, ValueError):
        raise ValueError(f"Invalid file format for '{distribution_path}'")

    if dndz_binned.shape[0] < 2:
        raise ValueError(
            "Input distribution must have at least one redshift bin "
            "and one dndz value."
        )

    z_array = dndz_binned[0]
    dndz_distributions = dndz_binned[1:]

    if "NumberCountsFactory" in tracer_type:
        measurements = {Galaxies.COUNTS}
    elif "WeakLensingFactory" in tracer_type:
        if two_pt_type == TwoPointHarmonic:
            measurements = {Galaxies.SHEAR_E}
        elif two_pt_type == TwoPointReal:
            measurements = {Galaxies.SHEAR_PLUS, Galaxies.SHEAR_MINUS}
        else:
            raise ValueError("Unknown TwoPointFunction type")
    else:
        raise ValueError(f"Unknown tracer type: {tracer_type}")

    infzdist = [
        InferredGalaxyZDist(
            bin_name=f"{tracer_name}{i}",
            z=z_array,
            dndz=dndz,
            measurements=measurements,
        )
        for i, dndz in enumerate(dndz_distributions)
    ]

    return infzdist

def _get_srd_distribution_binned(z: np.ndarray, tracer_name: str,
                                tracer_type: str, function_type: str,
                                year: str) -> list:
    """
    Get the binned distribution for lens/source tracer from the SRD Y1 or Y10.

    Args:
        z (np.ndarray): Redshift array.
        tracer_name (str): Name of the tracer (e.g., 'lens' or 'source' or anything).
        tracer_type (str): Type of the tracer 
            (e.g. NumberCountsFactory or WeakLensingFactory ).
        function_type (str): Type of the function (e.g., harmonic or real space).
        year (str): Year of the survey ('1' for Y1, '10' for Y10).

    Returns:
        list: List of InferredGalaxyZDist objects representing the binned
        dNdz distributions.
    """
    if "NumberCountsFactory" in tracer_type:
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

    elif "WeakLensingFactory" in tracer_type:
        if function_type == TwoPointHarmonic:
            measure = {Galaxies.SHEAR_E}
        elif function_type ==  TwoPointReal:
            measure = {Galaxies.SHEAR_PLUS, Galaxies.SHEAR_MINUS}
        else:
            raise ValueError("Unknown TwoPointFunction type")
        if year == "1":
            zdist = ZDistLSSTSRD.year_1_source(
                use_autoknot=True, autoknots_reltol=1.0e-5
            )
            bin_edges = Y1_SOURCE_BINS["edges"]
            sigma_z = Y1_SOURCE_BINS["sigma_z"]
            measurements = measure

        elif year == "10":
            zdist = ZDistLSSTSRD.year_10_source(
                use_autoknot=True, autoknots_reltol=1.0e-5
            )
            bin_edges = Y10_SOURCE_BINS["edges"]
            sigma_z = Y10_SOURCE_BINS["sigma_z"]
            measurements = measure

    dndz_binned = [
        zdist.binned_distribution(
            zpl=bin_edges[i],
            zpu=bin_edges[i + 1],
            sigma_z=sigma_z,
            z=z,
            name=f"{tracer_name}_{i}",
            measurements=measurements,
        )
        for i in range(len(bin_edges) - 1)
    ]

    return dndz_binned