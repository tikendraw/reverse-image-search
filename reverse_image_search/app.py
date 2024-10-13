import streamlit as st
from pathlib import Path
from PIL import Image
from model import ImageSimilarity
import os
from utils import get_image_paths, show_images, show_images2, show_images3


st.set_page_config(layout="wide")

star = """
<script async defer src="https://buttons.github.io/buttons.js"></script>

<!-- Place this tag where you want the button to render. -->
<a class="github-button"
   href="https://github.com/tikendraw/reverse-image-search"
   data-icon="octicon-star"
   data-size="large"
   data-show-count="true"
   aria-label="Star ntkme/github-buttons on GitHub">Star</a>"""


def main():
    st.title("Image Similarity App")

    st.write("Current Directory: ", Path(os.getcwd()))

    # Input form
    col1, col2, col12 = st.columns([5, 3, 5])
    col3, col4, col5, _ = st.columns([3, 3, 3, 3])

    with col1:
        img_dir = st.text_input(
            "Enter the Absolute directory path where to search:",
            placeholder="e.g. /home/username/some-folder-name",
        )

    with col2:
        st.write("Upload image or Paste full filepath to Image")
        img_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    with col12:
        img_path2 = st.text_input(
            "Absolute filepath of Image you want to compare.",
            placeholder="e.g. /home/username/picture/hello.jpg",
        )

    with col3:
        similar_images = st.number_input(
            "Number of similar images to display:",
            min_value=-1,
            max_value=50,
            value=7,
            step=1,
        )

    with col4:
        save_model = st.checkbox(
            "Save Model",
            value=False,
            help="Save the model for faster loads, check if you search in same folder again and again",
        )
        recursive = st.checkbox(
            "Recursive",
            value=False,
            help="Search recursively for images in child folders",
        )
        submit_button = st.button("Find Similar Images")

    with col5:
        st.components.v1.html(star, width=None, height=None, scrolling=False)
    # Find similar images on button click
    if submit_button:
        if os.path.isdir(img_dir) and (img_path or img_path2):
            img_path = img_path2 if img_path2 else img_path

            total_images = len(get_image_paths(Path(img_dir), recursive=recursive))

            st.markdown(f"## {total_images} Images found.")
            similar_images = min(similar_images, total_images)

            main_image = Image.open(img_path)
            image_similarity = ImageSimilarity(
                img_dir=Path(img_dir), recursive=recursive, save_model=save_model
            )

            with st.spinner("Wait for it... Model training"):
                (
                    similar_image_paths,
                    similarity_value,
                ) = image_similarity.find_similar_images2(main_image, k=similar_images)

            # Display main image
            _, l, _ = st.columns([4, 2, 4])

            with l:
                st.subheader("Main Image")
                st.image(main_image, caption="Main Image", use_column_width=True)

            # Display similar images horizontally
            st.subheader("Similar Images")

            fig = show_images2(similar_image_paths, similarity_value)

            if save_model:
                image_similarity.save_image_dict()
        else:
            st.error("Please enter the directory path and image/image path.")


if __name__ == "__main__":
    main()
