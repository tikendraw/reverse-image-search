import click
from pathlib import Path
from PIL import Image
import os
import tempfile
import json
from .utils import show_image_in_terminal
from v2.embed_model import EfficientNetEmbeddingFunction
from v2.embedding_store import EmbeddingStore
from functools import cache
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {}
    
@cache
def load_embed_store():
    embedding_model = EfficientNetEmbeddingFunction()
    db = EmbeddingStore("local_embeddings", embedding_model)
    return db

@click.group()
def cli_func():
    pass

@click.command()
@click.argument('image_paths', type=click.Path(exists=True), nargs=-1 )
@click.option('--num_results', '-n', default=config.get("num_similar_images", 5), help='Number of similar images to retrieve')
@click.option('--show_image', type=click.BOOL, default=True, help='Show the main image')
def search(image_paths, num_results, show_image):
    """
    Performs a reverse image search.
    """
    db = load_embed_store()


    if not db.collection.count() > 0:
        click.echo("Error: No embeddings found. Please create embeddings first.")
        return
    
    if image_paths:

        result = db.get_n_similar_images(image_paths, k=num_results)

        for i, similar_image_paths in enumerate(result["uris"]):
            image_path = str(Path(image_paths[i]).absolute())

            clickable_link =  f"file://{image_path}"
            click.echo(f"Main image: {click.style(clickable_link, fg='blue', underline=True)}")

            if show_image:
                show_image_in_terminal(image_path)
            
            click.echo(f"Similar images:")
            for i, similar_image_path in enumerate(similar_image_paths,1):
                clickable_link =  f"file://{similar_image_path}"
                click.echo(f"{i}.  {click.style(clickable_link, fg='blue', underline=True)}")
            
            print("\n","-"*3, "\n")
    else:
        click.echo("Please provide at least one image path using the '-i' option.")


@click.group()
def embed():
    """
    Manage embeddings.
    """
    pass

@embed.command()
@click.option('--dir_path', '-d', type=click.Path(exists=True), required=True, help='Path to the directory containing images')
def create(dir_path):
    """
    Creates embeddings for images in a directory.
    """
    db = load_embed_store()

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {"folders_embedded": []}

    if dir_path not in config["folders_embedded"]:
        
        db.add_images(image_dir=dir_path)
        config["folders_embedded"].append(dir_path)

        with open("config.json", "w") as f:
            json.dump(config, f)

        click.echo(f"Embeddings created for images in '{dir_path}' and config updated.")
    else:
        click.echo(f"Embeddings already exist for '{dir_path}'. Use 'embed update' to update them.")

@embed.command()
@click.option('--dir_path', '-d', type=click.Path(exists=True), required=True, help='Path to the directory containing images')
def update(dir_path):
    """
    Updates embeddings for images in a directory.
    """
    db = load_embed_store()

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {"folders_embedded": []}

    if dir_path in config["folders_embedded"]:
        db.update_images(image_dir=dir_path)
        click.echo(f"Embeddings updated for images in '{dir_path}'.")
    else:
        click.echo(f"Embeddings do not exist for '{dir_path}'. Use 'embed create' to create them.")

@embed.command()
@click.option('--dir_path', '-d', type=click.Path(exists=True), required=True, help='Path to the directory containing images')
def delete(dir_path):
    """
    Deletes embeddings for images in a directory.
    """
    db = load_embed_store()

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {"folders_embedded": []}

    if dir_path in config["folders_embedded"]:
        
        if not str(dir_path).lower().strip()=='delete_all_embeddings':
            db.delete_embeddings(image_dir=dir_path)
            
            # update the config
            config["folders_embedded"].remove(dir_path)
            with open("config.json", "w") as f:
                json.dump(config, f)
        else:
            db.delete_collection()
            
            # update the config
            config["folders_embedded"] = []
            with open("config.json", "w") as f:
                json.dump(config, f)
                
        click.echo(f"Deleting embeddings for '{dir_path}'!")
    else:
        click.echo(f"Embeddings do not exist for '{dir_path}'.")

@click.command()
def show_embedded_folders():
    """
    Shows the list of folders for which embeddings have been created.
    """
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {"folders_embedded": []}

    if config["folders_embedded"]:
        click.echo("Embedded folders:")
        for folder in config["folders_embedded"]:
            click.echo(f"- {folder}")
    else:
        click.echo("No folders have been embedded yet.")

@click.command()
def show_configs():
    """
    Shows the current configuration.
    """
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}

    click.echo(json.dumps(config, indent=4))

@click.command()
@click.option('--batch_size', '-b', type=int, default=config.get("batch_size"), help='Batch size for creating embeddings')
@click.option('--num_similar_images', '-n', type=int, default=config.get("num_similar_images"), help='Number of similar images to retrieve during search')
@click.option('--n_cols', '-c', type=int, default=config.get("n_cols"), help='Number of columns for displaying images in the Streamlit app')
def change_configs(batch_size, num_similar_images, n_cols):
    """
    Changes the configuration options.
    """
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}

    if batch_size:
        config["batch_size"] = batch_size
    if num_similar_images:
        config["num_similar_images"] = num_similar_images
    if n_cols:
        config["n_cols"] = n_cols

    with open("config.json", "w") as f:
        json.dump(config, f)

    click.echo("Configuration updated successfully.")


cli_func.add_command(search)
cli_func.add_command(embed)
cli_func.add_command(show_embedded_folders)
cli_func.add_command(show_configs)
cli_func.add_command(change_configs)


if __name__ == '__main__':
    cli_func()
