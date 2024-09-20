import yaml

def load_yaml_file(file_path):
    """Helper function to load a YAML file"""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)