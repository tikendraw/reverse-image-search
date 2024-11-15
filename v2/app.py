import os
import tempfile
from functools import partial

import pyperclip
import streamlit as st
from PIL import Image

from v2.common import (
    create_embeddings,
    delete_embeddings,
    get_similar_images,
    list_images,
    load_config,
    load_embed_store,
    save_config,
    show_images2,
    update_embeddings,
)

INCLUDE_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", )

list_images = partial(list_images, include_extensions=INCLUDE_IMAGE_EXTENSIONS)


def main():
    config:dict = load_config()

    db = load_embed_store()

    st.set_page_config(layout="wide")
    st.title("Reverse Image Search")

    with st.sidebar:
        st.header("Embeddings")

        image_dir = st.text_input(
            "Enter directory path containing images for embeddings:"
        )
        
        if config["folders_embedded"]:
            with st.expander(f"Embedded folders({len(config['folders_embedded'])})"):
                for folder in config["folders_embedded"]:
                    st.write(folder)

        recursive = st.checkbox("Recursive", value=True)
        
        if st.button("Create Embeddings"):
            print('creating embeddings')
            if image_dir and os.path.isdir(image_dir):
                created_paths = create_embeddings(db, image_dir, recursive, config)
                
                if image_dir not in config["folders_embedded"]:
                    config["folders_embedded"].extend(created_paths)

                st.success("Embeddings created successfully!")
            else:
                st.error("Please enter a valid directory path.")

        if st.button("Update Embeddings"):
            print('updating embeddings')
            if image_dir and os.path.isdir(image_dir):
                updated_path=update_embeddings(db, image_dir, recursive, config)
            
            else:
                for path in config["folders_embedded"]:
                    updated_path=update_embeddings(db, path, recursive, config)
            
            if updated_path:
                config["folders_embedded"].extend(updated_path)
                config["folders_embedded"] = list(set(config["folders_embedded"]))
                    
            st.success("Embeddings updated successfully!")

        if st.button("Delete Embeddings"):
            print('deleting embeddings')
            deleted_paths = delete_embeddings(db, image_dir, recursive, config)
            db.setup()
            
            if image_dir:
                if image_dir.lower().strip() == 'delete_all_embeddings':
                    config["folders_embedded"] = []
                else:
                    config["folders_embedded"] = [i for i in config["folders_embedded"] if i not in deleted_paths]
                
                save_config(config)
            
                st.success("Embeddings deleted successfully!")

        st.header("Search Settings")
        num_similar_images = st.number_input(
            "Number of similar images to find:",
            min_value=1,
            max_value=100,
            value=config.get("num_similar_images", 20),
            key="num_similar_images_input",
        )
        batch_size = st.number_input(
            "Batch size for creating embeddings:",
            min_value=1,
            value=config.get("batch_size", 16),
            key="batch_size_input",
        )
        n_cols = st.number_input(
            "Number of columns for displaying images:",
            min_value=1,
            value=config.get("n_cols", 5),
            key="n_cols_input",
        )

        config['batch_size']=batch_size
        config['num_similar_images']=num_similar_images
        config['n_cols']=n_cols

        print('config: ', config)
        save_config(config)
        config = load_config()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if not db.collection.count() > 0:
            st.error("Create Embeddings first")
            return

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            with Image.open(temp_file_path) as image:
                results = get_similar_images(
                    db, [temp_file_path], config.get("num_similar_images", 20)
                )

                if results:
                    similar_images = results["uris"][0]

                    _, l, _ = st.columns([2, 2, 2])

                    with l:
                        st.image(image, caption="Main Image", use_column_width=True, width=200)

                    st.subheader("Similar Images")

                    fig = show_images2(similar_images, n_cols)

                    if st.button('Copy paths to clipboard'):    
                        pyperclip.copy(similar_images)
                        st.success('Text copied successfully!')

        finally:
            try:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.close(os.open(temp_file_path, os.O_RDONLY))  
                    os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")


if __name__ == "__main__":
    main()