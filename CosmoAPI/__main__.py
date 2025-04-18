import argparse
import logging
import logging.config
from .api_io import load_yaml_file, logger, set_log_level
from . import __version__
from .two_pt_func.generate_theory import generate_sacc_theory_vector
from .not_implemented import not_implemented_message

banner = rf"""

 　　　　　/)───―ヘ       _____             _ 
 　　　＿／　　　　＼    /  __ \           (_)
 　 ／　　　　●　　　●  | /   \/__ _ _ __  _ 
 　｜　　　　　　　▼　| | |    / _` | '_ \| |
 　｜　　　　　　　亠ノ | \__/\ (_| | |_) | |
 　 U￣U￣￣￣￣U￣U 　  \____/\__,_| .__/|_|
 　　　　　　　　　　　　　　　　　 | |      
 　　　　　　　　　　　　　　　　　 |_|  

                   - Capi stands for CAPIVARA -
   Cosmology API for Validation, Analysis, and Research Applications 
        DESC's "Press Enter for Cosmology" Pipeline Interface
                       Version {__version__}
"""

def generate_sacc(config):
    # Functionality for generating data vector
    logger.info(f"Generating Synthetic Data Vectors ")
    generate_sacc_theory_vector(config, save_sacc=True)


def gen_covariance(config): 
    # Functionality for generating covariance
    print(not_implemented_message)

def forecast(config):
    # Functionality for forecast
    print(not_implemented_message)

def main():
    parser = argparse.ArgumentParser(
        prog="CosmoAPI",
        description="CosmoAPI: Cosmology Analysis Pipeline Interface"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # gen_datavec subcommand
    parser_datavec = subparsers.add_parser(
        'generate_sacc',
        help="Generate a synthetic datavector given the configuration choices"
    )
    parser_datavec.add_argument(
        'config_file',
        type=str,
        help="Path to the YAML configuration file"
    )
    parser_datavec.add_argument(
        '--verbose',
        action='store_true', 
        help="Enable verbose output for debugging purposes"
    )

    # gen_covariance subcommand
    parser_covariance = subparsers.add_parser(
        'gen_covariance',
        help="Generate covariance matrix from configuration"
    )
    parser_covariance.add_argument(
        'config_file',
        type=str,
        help="Path to the YAML configuration file"
    )

    # forecast subcommand
    parser_forecast = subparsers.add_parser(
        'forecast',
        help="Run forecast analysis from configuration"
    )
    parser_forecast.add_argument(
        'config_file',
        type=str,
        help="Path to the YAML configuration file"
    )

    args = parser.parse_args()

    # Check if a command is provided; if not, show help and available commands
    if args.command is None:
        parser.print_help()
        return

    # Load the YAML configuration file
    config = load_yaml_file(args.config_file)

    _log_level = config['general'].get('verbose_level', 'INFO').upper()
    if args.verbose:
        _log_level = 'DEBUG'
    if _log_level != "INFO":
        set_log_level(_log_level)

    logger.info(banner)
    logger.info(f"Loaded YAML configuration file: {args.config_file}")
    logger.debug(f"Configuration data: {config}")

    # Call the appropriate function based on the command
    if args.command == 'generate_sacc':
        generate_sacc(config)
    elif args.command == 'gen_covariance':
        gen_covariance(config)
    elif args.command == 'forecast':
        forecast(config)

if __name__ == "__main__":
    main()