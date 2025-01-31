import chafa
from chafa.loader import Loader
from PIL import Image
import tempfile
import os
from pathlib import Path
from typing import List
from pathlib import Path
from typing import List, Tuple, Optional

# Define default image extensions
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ) #".gif", ".bmp", ".tiff")


def list_items_in_dir(
    dir: str,
    include_extensions: Optional[Tuple[str]] = IMAGE_EXTENSIONS,
    exclude_extensions: Optional[Tuple[str]] = None,
    recursive: bool = True,
) -> List[str]:
    """Return a list of item paths in a directory with valid extensions.

    Args:
        dir (str): The directory to search.
        include_extensions (tuple): File extensions to include (default: IMAGE_EXTENSIONS).
        exclude_extensions (tuple): File extensions to exclude (default: None).
        recursive (bool): Whether to search recursively (default: True).

    Returns:
        List[str]: List of file paths.
    """

    if not os.path.isdir(dir):
        raise ValueError(f"Invalid directory path: {dir}")

    # Set the glob pattern based on the recursive flag
    pattern = "**/*" if recursive else "*"

    # Ensure include and exclude extensions are in lowercase for comparison
    include_extensions = tuple(ext.lower() for ext in include_extensions)
    exclude_extensions = (
        tuple(ext.lower() for ext in exclude_extensions) if exclude_extensions else ()
    )

    return [
        str(p.absolute())
        for p in Path(dir).glob(pattern)
        if p.is_file()
        and p.suffix.lower() in include_extensions
        and p.suffix.lower() not in exclude_extensions
    ]


def list_images_in_dir(
    dir: str,
    filter_callable: Optional[callable] = None,
    include_extensions=IMAGE_EXTENSIONS,
    exclude_extensions: Optional[Tuple[str]] = None,
    recursive: bool = True,
) -> List[str]:
    """Return a list of image paths in a directory that pass the filter callable.

    Args:
        dir (str): The directory to search.
        filter_callable (callable): Optional function that takes a path and returns bool.
        exclude_extensions (tuple): Image extensions to exclude (default: None).
        recursive (bool): Whether to search recursively (default: True).

    Returns:
        List[str]: List of filtered image file paths.
    """
    # Get all image files using the existing function
    image_files = list_items_in_dir(
        dir,
        include_extensions=include_extensions,
        exclude_extensions=exclude_extensions,
        recursive=recursive
    )

    # If no filter callable is provided, return all images
    if filter_callable is None:
        return image_files

    # Apply the filter callable to each image path
    return [
        img_path for img_path in image_files
        if filter_callable(img_path)
    ]


def filter_image_by_size(
    image_path: str,
    min_width: int = 0,
    min_height: int = 0
) -> bool:
    """Filter images based on minimum dimensions and checks if file is not empty.
    
    Args:
        image_path (str): Path to the image file
        min_width (int): Minimum width required (default: 0)
        min_height (int): Minimum height required (default: 0)
        
    Returns:
        bool: True if image meets criteria, False otherwise
    """
    # Check if file exists and is not empty
    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        return False
        
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return width >= min_width and height >= min_height
    except Exception:
        return False


def show_image_in_terminal(image_path):
    image_path = Path(image_path)
    img = Image.open(image_path)
    width, height = img.size

    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Perform center cropping
    cropped_img = img.crop((left, top, right, bottom))
    temp_filename = f"temp_cropped_{image_path.stem}.jpg"
    cropped_img.save(temp_filename)
    image = Loader(temp_filename)

    # Create config
    config = chafa.CanvasConfig()

    config.height = 30
    config.width = 70

    # Create canvas
    canvas = chafa.Canvas(config)

    # Draw to the canvas
    canvas.draw_all_pixels(
        image.pixel_type,
        image.get_pixels(),
        image.width,
        image.height,
        image.rowstride,
    )

    # print output
    output = canvas.print().decode()

    print(output)

    # Delete the temporary file
    os.remove(temp_filename)
