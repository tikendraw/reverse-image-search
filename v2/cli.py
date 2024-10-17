import click
from pathlib import Path
from PIL import Image
import os
import tempfile
import json

from v2.embed_model import EfficientNetEmbeddingFunction
from v2.embedding_store import EmbeddingStore

try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    config = {}


@click.group()
def cli_func():
    pass


@click.command()
@click.option('--image_paths', '-i', type=click.Path(exists=True), multiple=True, help='Path to the image file')
@click.option('--num_results', '-n', default=config.get("num_similar_images", 5), help='Number of similar images to retrieve')
def search(image_paths, num_results):
    """
    Performs a reverse image search.
    """
    embedding_model = EfficientNetEmbeddingFunction()
    db = EmbeddingStore("local_embeddings", embedding_model)

    if not db.collection.count() > 0:
        click.echo("Error: No embeddings found. Please create embeddings first.")
        return

    if image_paths:

        result = db.get_n_similar_images(image_paths, k=num_results)


        click.echo("Similar images:")
        for i, similar_image_paths in enumerate(result["uris"]):
            click.echo(f"For image '{image_paths[i]}':")
            for similar_image_path in similar_image_paths:
                click.echo(f"- {similar_image_path}")

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
    embedding_model = EfficientNetEmbeddingFunction()
    db = EmbeddingStore("local_embeddings", embedding_model)

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
    embedding_model = EfficientNetEmbeddingFunction()
    db = EmbeddingStore("local_embeddings", embedding_model)

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
    embedding_model = EfficientNetEmbeddingFunction()
    db = EmbeddingStore("local_embeddings", embedding_model)

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {"folders_embedded": []}

    if dir_path in config["folders_embedded"]:
        # TODO: Implement logic to delete embeddings for a specific directory.
        # This might involve iterating through the image cache and deleting
        # corresponding entries from the ChromaDB collection.
        click.echo(f"Deleting embeddings for '{dir_path}' is not yet implemented.")
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
