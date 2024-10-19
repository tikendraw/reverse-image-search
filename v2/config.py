from configparser import ConfigParser
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from json import JSONDecodeError
import os


config_dir = os.path.expanduser('~/.reverse_image_search')
config_dir = Path(config_dir)
config_dir.mkdir(parents=True, exist_ok=True)



EMBEDDING_DIR = config_dir/ 'local_embeddings'

@dataclass
class Config:
    folders_embedded: list[str] 
    batch_size: int = 16
    num_similar_images: int = 20
    n_cols: int = 5

config_file = config_dir / 'config.json'

def save_config(
    path:str|Path=config_file,
    folder_embedded: list[str]=None, 
    batch_size: int=16, 
    num_similar_images: int=10, 
    n_cols: int=5,
    config:Config=None,
    ):
    if folder_embedded is None:
        folder_embedded = []
    
    if not config:
        config = Config(folders_embedded=folder_embedded, batch_size=batch_size, num_similar_images=num_similar_images, n_cols=n_cols)
    data_dict = asdict(config)

    with open(path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

def read_config(path:str|Path=config_file):
    with open(path, 'r') as json_file:
            data = json.load(json_file)
    return data

def load_config(path:str|Path=config_file) -> dict:
    try: 
        data = read_config(path)
    except (FileNotFoundError, JSONDecodeError):
        save_config(path)
        data = read_config(path)
        
    return data