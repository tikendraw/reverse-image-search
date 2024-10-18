import chafa
from chafa.loader import Loader
from PIL import Image
import tempfile
import os
from pathlib import Path


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

    config.height = 30 #image.height//5
    config.width  = 70 #image.width//5

    # Create canvas
    canvas = chafa.Canvas(config)


    # Draw to the canvas
    canvas.draw_all_pixels(
        image.pixel_type,
        image.get_pixels(),
        image.width, image.height,
        image.rowstride
    )

    # print output
    output = canvas.print().decode()

    print(output)



    # Delete the temporary file
    os.remove(temp_filename)
