from firecrown.ccl_factory import (
    CCLFactory,
    CAMBExtraParams,
    PoweSpecAmplitudeParameter,
)
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap

def build_modeling_tools(config: dict) -> ModelingTools:
    """
    Create a ModelingTools object from the configuration.

    Args:
        config (dict): Dictionary containing cosmology and
        systematics parameters.

    Returns:
        ModelingTools: Modeling tools object with cosmology parameters.
    """
    cosmo_config = config["cosmology"]
    factory_config = config["firecrown_parameters"]

    if "Omega_m" in factory_config.keys():
        factory_config["Omega_c"] = (
            factory_config["Omega_m"] - factory_config["Omega_b"]
        )
        del factory_config["Omega_m"]

    if "m_nu" in factory_config.keys():
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