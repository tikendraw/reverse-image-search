from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Config:
    folders_embedded: list[str] 
    batch_size: int = 16
    num_similar_images: int = 20
    n_cols: int = 5

config_file = Path(__file__).parent.parent / 'config.json'

def save_config(
    path:str|Path=config_file,
    folder_embedded: list[str]=None, 
    batch_size: int=16, 
    num_similar_images: int=10, 
    n_cols: int=5
    ):
    if folder_embedded is None:
        folder_embedded = []
    config = Config(folders_embedded=folder_embedded, batch_size=batch_size, num_similar_images=num_similar_images, n_cols=n_cols)
    data_dict = asdict(config)

    with open(path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

