import argparse
from typing import Dict, Any

from .api_io import load_yaml_file
from .not_implemented import not_implemented_message

def gen_datavec(config: Dict[str, Any], verbose: bool = False) -> None:
    # Functionality for generating data vector
    if verbose:
        print("Verbose mode enabled.")
    print("Generating data vector with config:", config)

def gen_covariance(config: Dict[str, Any]) -> None:
    # Functionality for generating covariance
    print(not_implemented_message)

def forecast(config: Dict[str, Any]) -> None:
    # Functionality for forecast
    print(not_implemented_message)

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="CosmoAPI",
        description="CosmoAPI: Cosmology Analysis Pipeline Interface"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # gen_datavec subcommand
    parser_datavec = subparsers.add_parser(
        'gen_datavec',
        help="Generate data vector from configuration"
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

    # Call the appropriate function based on the command
    if args.command == 'gen_datavec':
        gen_datavec(config, verbose=args.verbose)
    elif args.command == 'gen_covariance':
        gen_covariance(config)
    elif args.command == 'forecast':
        forecast(config)

if __name__ == "__main__":
    main()
