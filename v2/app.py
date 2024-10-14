import streamlit as st
from PIL import Image
import os
import tempfile
from tqdm import tqdm
import json
from pathlib import Path

from PIL import Image
import numpy as np
from embed_model import EfficientNetEmbeddingFunction
from embedding_store import EmbeddingStore
import os
import tempfile
from tqdm import tqdm


def show_images2(x: list, num_columns: int = 7):
    cols = st.columns(num_columns)
    x = [Path(i) for i in x]

    for num, i in enumerate(x, 1):
        img = Image.open(i)

        with cols[(num - 1) % num_columns]:
            st.image(img, caption=i.name, use_column_width=True)


def main():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {"folders_embedded": [], "batch_size": 16, "num_similar_images": 20, "n_cols":5}


    db = EmbeddingStore("local_embeddings", embedding_model=EfficientNetEmbeddingFunction())

    st.set_page_config(layout="wide")
    st.title("Reverse Image Search")

    with st.sidebar:
        st.header("Embeddings")

        image_dir = st.text_input(
            "Enter directory path containing images for embeddings:"
        )
        st.write("Embedded Directories:")
        for dir in config["folders_embedded"]:
            st.write(f"- {dir}")

        if st.button("Create Embeddings"):
            if image_dir and os.path.isdir(image_dir):
                if image_dir in config["folders_embedded"]:
                    st.warning(
                        "Embeddings already exist for this directory. Use 'Update Embeddings' to refresh."
                    )
                else:
                    image_paths = [
                        os.path.join(image_dir, f)
                        for f in os.listdir(image_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ]

                    # Batching logic
                    batch_size = config.get("batch_size", 16)
                    num_batches = (len(image_paths) + batch_size - 1) // batch_size

                    progress_bar = st.progress(0)
                    for i in tqdm(range(num_batches)):
                        start = i * batch_size
                        end = min((i + 1) * batch_size, len(image_paths))
                        batch_paths = image_paths[start:end]
                        db.add_images(batch_paths)
                        progress_bar.progress((i + 1) / num_batches)

                    st.success("Embeddings created successfully!")
                    config["folders_embedded"].append(image_dir)

            else:
                st.error("Please enter a valid directory path.")

        if st.button("Update Embeddings"):
            for path in config["folders_embedded"]:
                db.update_images(image_dir=path)
            st.success("Embeddings updated successfully!")

        if st.button("Delete Embeddings"):
            db.delete_collection()
            db.setup()
            config["folders_embedded"] = []
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

        config["num_similar_images"] = num_similar_images
        config["batch_size"] = batch_size
        config['n_cols'] = n_cols


        with open("config.json", "w") as f:
            json.dump(config, f)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        if not db.collection.count() > 0:
            st.error("Create Embeddings first")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        image = Image.open(temp_file_path)

        results = db.get_n_similar_images(
            [temp_file_path], k=config.get("num_similar_images", 20)
        )
        similar_images = results["uris"][0]
        # similarity_value = results['distances'][0]

        _, l, _ = st.columns([2, 2, 2])

        with l:
            st.image(image, caption="Main Image", use_column_width=True)

        st.subheader("Similar Images")

        fig = show_images2(similar_images, n_cols)

        # Clean up temporary file
        os.unlink(temp_file_path)


if __name__ == "__main__":
    main()
