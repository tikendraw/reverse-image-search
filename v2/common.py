import json
import logging
from functools import cache, partial
from pathlib import Path

import click
import streamlit as st
from PIL import Image
from tqdm import tqdm

from v2.config import EMBEDDING_DIR, config_file
from v2.config import load_config as l_config
from v2.embed_model import EfficientNetEmbeddingFunction
from v2.embedding_store import EmbeddingStore, list_images

from .utils import list_items_in_dir

INCLUDE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", )


@cache
def load_embed_store():
    embedding_model = EfficientNetEmbeddingFunction()
    db = EmbeddingStore(save_dir=str(EMBEDDING_DIR.absolute()), embedding_model=embedding_model)
    return db

def create_embeddings(db:EmbeddingStore, image_dir, recursive, config) -> list[str]:
    '''
    Creates embeddings for images in a directory.
    return list of directories where images are (to update the config)
    '''
    if image_dir not in config["folders_embedded"]:
        image_paths = list_images(image_dir, recursive=recursive)
        
        # Batching logic (if needed)
        batch_size = config.get("batch_size", 16) 
        num_batches = (len(image_paths) + batch_size - 1) // batch_size

        if "streamlit" in str(st.__path__):
            progress_bar = st.progress(0)
        else:
            progress_bar = None

        for i in tqdm(range(num_batches), desc="Creating Embeddings"):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(image_paths))
            batch_paths = image_paths[start:end]
            db.update_images(image_paths=batch_paths)
            if progress_bar:
                progress_bar.progress((i + 1) / num_batches)


        if not "streamlit" in str(st.__path__):
            click.echo(f"Embeddings created for images in '{image_dir}' and config updated.")
    else:
        image_paths=[]
        if not "streamlit" in str(st.__path__):
            click.echo(f"Embeddings already exist for '{image_dir}'. Use 'embed update' to update them.")
            
    embedded_dirs = [str(Path(i).parent.absolute()) for i in image_paths]
    return list(set(embedded_dirs))



def update_embeddings(db:EmbeddingStore, dir_path, recursive, config) -> list[str]:
    ''' updates embeddings for images in a directory
    return list of directories where images are (to update the config)
    '''
    if dir_path in config["folders_embedded"]:
        db.update_images(dir_path=dir_path, recursive=recursive)
        if not "streamlit" in str(st.__path__):
            click.echo(f"Embeddings updated for images in '{dir_path}'.")
            
    else:
        if not "streamlit" in str(st.__path__):
            click.echo(f"Embeddings do not exist for '{dir_path}'. Use 'embed create' to create them.")
    
    # Get all image paths in directory to update config
    image_paths = list_images(dir_path, recursive=recursive)
    embedded_dirs = [str(Path(p).parent.absolute()) for p in image_paths]
    return list(set(embedded_dirs))



def delete_embeddings(db:EmbeddingStore, dir_path, recursive, config) -> list[str]:
    """deletes embeddings for images in a directory, returns embedding deleted directories"""
    deleted_image_paths=[]

    if str(dir_path).lower().strip() == 'delete_all_embeddings':
        db.delete_collection()
        deleted_image_paths=config["folders_embedded"]
            
    else:
        if dir_path in config["folders_embedded"]:
            db.delete_images(dir_path=dir_path, recursive=recursive)
            deleted_image_paths.extend(list_images(dir_path, recursive=recursive))
        
            
    embedded_dirs = [str(Path(i).parent.absolute()) for i in deleted_image_paths]
    return list(set(embedded_dirs))


def get_similar_images(db:EmbeddingStore, image_paths, num_results):
    if not db.collection.count() > 0:
        if not "streamlit" in str(st.__path__):
            click.echo("Error: No embeddings found. Please create embeddings first.")
        return None

    if image_paths:
        try: 
            return db.get_similar_images(image_paths, k=num_results)
        except ValueError:
            logging.error("Failed to generate Embeddings.")
            return {}
    else:
        if not "streamlit" in str(st.__path__):
            click.echo("Please provide at least one image path.")
        return None

def load_config()-> dict:
    return l_config(config_file)

def save_config(config) -> None:
    with open(config_file, "w") as f:
        json.dump(config, f)


def show_images2(x: list, num_columns: int = 7):
    cols = st.columns(num_columns)
    x:list[Path] = [Path(i) for i in x]

    for num, i in enumerate(x, 1):
        try: 
            img = Image.open(i)

            with cols[(num - 1) % num_columns]:
                st.image(img, caption=i.name, use_container_width=True)
        except FileNotFoundError as e:
            st.error(f"Error: File not found: {i}, update the embeddings.")