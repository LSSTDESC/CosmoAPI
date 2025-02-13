import os
import yaml
import logging
import logging.config
import importlib


def load_yaml_file(yaml_file: str) -> dict:
    """
    Load the YAML configuration file.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed YAML data.
    """
    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    # add the file name to the yaml_data
    yaml_data["general"]['config_file'] = yaml_file
    return yaml_data

def load_metadata_function_class(function_name):
    """
    Dynamically load a class based on the 'function' name specified in the YAML file.
    FIXME: Change the docstrings
    Args:
        function_name (str): The name of the function specified in the YAML.

    Returns:
        The loaded class based on the function name.
    """
    # Assume functions are part of a module like 'firecrown.functions'
    base_module = "firecrown.metadata_functions"
    
    try:
        # Dynamically import the module
        module = importlib.import_module(base_module)
        # Get the function class from the module
        function_class = getattr(module, function_name)
        return function_class
    except ImportError as e:
        raise ImportError(f"Could not import module {base_module}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Class '{function_name}' not found in module {base_module}: {e}")

def create_output_directory(output_dir: str) -> None:
    """
    Create the output directory if it does not exist.

    # FIXME: later on we need to create the outputs based on
        the main that is called and the data products that are generated

    Args:
        output_dir (str): Path to the output directory.
    """
    # Convert the relative path to an absolute path
    absolute_output_dir = os.path.abspath(output_dir)

    if not os.path.exists(absolute_output_dir):
        os.makedirs(absolute_output_dir)
    return absolute_output_dir

def setup_logging(config: dict={}, default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration"""
    logging_config = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'level': 'DEBUG',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'detailed',
                'level': 'DEBUG',
                'filename': 'app.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
            },
        },
        'loggers': {
            'CosmoAPI': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
        },
    }

    # Override the logging level from the config
    if not config:
        log_level = config.get('general', {}).get('verbose_level', default_level)
        logging_config['loggers']['CosmoAPI']['level'] = log_level
        logging_config['root']['level'] = log_level

    value = os.getenv(env_key, None)
    if value and os.path.exists(value):
        with open(value, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.config.dictConfig(logging_config)

def set_log_level(level):
    """
    Dynamically sets the log level for all loggers in the package.
    """
    logging.getLogger('CosmoAPI').setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.info("Log level changed to %s", level)

setup_logging()
logger = logging.getLogger('CosmoAPI')
