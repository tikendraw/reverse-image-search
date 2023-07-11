from streamlit.runtime.scriptrunner import add_script_run_ctx
import streamlit as st
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import concurrent.futures


def get_image_paths(directory_path: Path, recursive: bool = False) -> list:
    image_extensions = [".jpg", ".jpeg", ".png"]  # Add more extensions if needed
    image_paths = []

    for file_path in directory_path.iterdir():
        if file_path.is_file() and (file_path.suffix.lower() in image_extensions):
            image_paths.append(str(file_path.absolute()))

        elif recursive and file_path.is_dir():
            image_paths.extend(get_image_paths(file_path, recursive))

    return image_paths


def show_images(x: list, similar: list = None, figsize=None):
    n_plots = len(x)

    if figsize is None:
        figsize = (20, int(n_plots // 5) * 4) if n_plots > 4 else (20, 5)

    fig, axes = plt.subplots((n_plots // 5) + 1, 5, figsize=figsize)
    axes = axes.flatten()

    x = [Path(i) for i in x]

    for num, i in enumerate(x, 1):
        img = plt.imread(i)
        axes[num - 1].imshow(img)
        title = (
            f"{i.name}\n({100 * similar[num - 1]:.2f}%)"
            if similar is not None
            else i.name
        )
        axes[num - 1].set_title(title)
        _ = axes[num - 1].axis(False)

    plt.tight_layout()
    return fig


def show_images2(x: list, similar: list = None):
    n_plots = len(x)
    num_columns = 7
    num_rows = n_plots // num_columns  # Ceiling division

    # Create the columns for image display
    cols = st.columns(num_columns)

    x = [Path(i) for i in x]

    for num, i in enumerate(x, 1):
        img = Image.open(i)
        title = (
            f"{i.name}\n({100 * similar[num - 1]:.2f}%)"
            if similar is not None
            else i.name
        )

        # Display image and title in the appropriate column
        with cols[(num - 1) % num_columns]:
            st.image(img, caption=title, use_column_width=True)


def show_images3(x: list, similar: list = None):
    n_plots = len(x)
    num_columns = 5
    num_rows = (n_plots // num_columns) + (
        n_plots % num_columns > 0
    )  # Ceiling division

    # Create the columns for image display
    cols = st.columns(num_columns)

    x = [Path(i) for i in x]

    def process_image(index, path, similarity):
        img = Image.open(path)
        title = (
            f"{path.name}\n({100 * similarity:.2f}%)"
            if similar is not None
            else path.name
        )

        # Display image and title in the appropriate column
        with cols[index]:
            st.image(img, caption=title, use_column_width=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for num, (path, similarity) in enumerate(zip(x, similar or []), 1):
            executor.submit(process_image, (num - 1) % num_columns, path, similarity)

            for t in executor._threads:
                add_script_run_ctx(t)

        # concurrent.futures.wait(futures)


if __name__ == "__main__":
    print(__file__, " running...")
