from .config import config_file, save_config
import json

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    save_config(config_file)
