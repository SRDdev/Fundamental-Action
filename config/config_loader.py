"""
Author: Shreyas Dixit
This file loads and updates configurations.
"""
import yaml

class LoadConfig:
    def __init__(self, config_path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str): Path to the configuration YAML file.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str): Key for the configuration setting.
            default: Default value if the key is not found.
        
        Returns:
            The value associated with the key, or the default value.
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        return value

    def update(self, updates):
        """
        Update the configuration with the provided key-value pairs.
        
        Args:
            updates (dict): A dictionary containing configuration updates. 
                            Keys should be in dot-separated format (e.g., 'TrajectoryNet.training.learning_rate').
        """
        for key, value in updates.items():
            keys = key.split('.')
            config_section = self.config
            for k in keys[:-1]:
                if k not in config_section:
                    raise KeyError(f"Key '{k}' not found in configuration.")
                config_section = config_section[k]
            if keys[-1] not in config_section:
                raise KeyError(f"Key '{keys[-1]}' not found in configuration.")
            config_section[keys[-1]] = value
