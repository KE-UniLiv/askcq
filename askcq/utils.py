"""
Collects utility functions for the xaniml package.
"""

import yaml


def get_key(service, config_file="api_config.yml"):
    """
    Get the API key for the given service from a config file in YAML format. For
    example, `get_key("openai")` will return the OpenAI API key.
    """
    # Read the YML file first and get the right entry
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config[service]["key"]

# Generate a hash from the LLM config
def generate_hash(config):
    return hash(frozenset(config.items()))
