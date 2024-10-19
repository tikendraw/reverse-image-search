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
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")


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
