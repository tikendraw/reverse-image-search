from functools import partial
from pathlib import Path

import click

from v2.common import (
    create_embeddings_cli,
    delete_embeddings,
    get_similar_images_cli,
    list_images,
    load_config,
    load_embed_store,
    save_config,
    update_embeddings_cli,
)

from .utils import show_image_in_terminal

INCLUDE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", )

list_images = partial(list_images, include_extensions=INCLUDE_IMAGE_EXTENSIONS)

config = load_config()

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

    result = get_similar_images_cli(db, image_paths, num_results)

    if result:
        for i, similar_image_paths in enumerate(result["uris"]):
            image_path = str(Path(image_paths[i]).absolute())

            clickable_link =  f"file://{image_path}"
            click.echo(f"Main image: {click.style(clickable_link, fg='blue', underline=True)}")

            if show_image:
                show_image_in_terminal(image_path)
            
            click.echo("Similar images:")
            for i, similar_image_path in enumerate(similar_image_paths,1):
                clickable_link =  f"file://{similar_image_path}"
                click.echo(f"{i}.  {click.style(clickable_link, fg='blue', underline=True)}")
            
            print("\n","-"*3, "\n")
    else:
        click.echo("Failed to generate Embeddings.")
        

@click.group()
def embed():
    """
    Manage embeddings.
    """
    pass

@embed.command()
@click.option('--dir_path', '-d', type=click.Path(exists=True), required=True, help='Path to the directory containing images')
@click.option('--recursive', '-r', type=click.BOOL, default=True, help='search recursively in  directory')
def create(dir_path, recursive):
    """
    Creates embeddings for images in a directory.
    """
    db = load_embed_store()
    created_paths = create_embeddings_cli(db, dir_path, recursive, config)
    config["folders_embedded"].extend(created_paths)
    config["folders_embedded"] = list(set(config["folders_embedded"]))
    save_config(config)

@embed.command()
@click.option('--dir_path', '-d', type=click.Path(exists=True), required=True, help='Path to the directory containing images')
@click.option('--recursive', '-r', type=click.BOOL, default=False, help='update embeddings recursively in  directory')
def update(dir_path, recursive):
    """
    Updates embeddings for images in a directory.
    """
    db = load_embed_store()
    updated_paths = update_embeddings_cli(db, dir_path, recursive, config)
    
    if updated_paths:
        config["folders_embedded"].extend(updated_paths)
        config["folders_embedded"] = list(set(config["folders_embedded"]))
        save_config(config)


@embed.command()
@click.option('--dir_path', '-d', type=click.Path(exists=True), required=True, help='Path to the directory containing images')
@click.option('--recursive', '-r', type=click.BOOL, default=False, help='delete embeddings recursively')
def delete(dir_path, recursive):
    """
    Deletes embeddings for images in a directory.
    """
    db = load_embed_store()
    deleted_paths = delete_embeddings(db, dir_path, recursive, config)
    
    if deleted_paths:
        config["folders_embedded"] = [i for i in config["folders_embedded"] if i not in deleted_paths]
        
    save_config(config)

@click.command()
def show_embedded_folders():
    """
    Shows the list of folders for which embeddings have been created.
    """
    
    if config["folders_embedded"]:
        click.echo("Embedded folders:")
        for folder in config["folders_embedded"]:
            click.echo(folder)
    else:
        click.echo("No folders have been embedded.")



@click.command()
def show_configs():
    """
    Shows the current configuration.
    """
    for i in config:
        click.echo(f"{i}: {config[i]}")


@click.command()
@click.option('--batch_size', '-b', type=int, default=config.get("batch_size"), help='Batch size for creating embeddings')
@click.option('--num_similar_images', '-n', type=int, default=config.get("num_similar_images"), help='Number of similar images to retrieve during search')
@click.option('--n_cols', '-c', type=int, default=config.get("n_cols"), help='Number of columns for displaying images in the Streamlit app')
def change_configs(batch_size, num_similar_images, n_cols):
    """
    Changes the configuration options.
    """
    config['batch_size']=batch_size
    config['num_similar_images']=num_similar_images
    config['n_cols']=n_cols
    save_config(config)



cli_func.add_command(search)
cli_func.add_command(embed)
cli_func.add_command(show_embedded_folders)
cli_func.add_command(show_configs)
cli_func.add_command(change_configs)


if __name__ == '__main__':
    cli_func()
