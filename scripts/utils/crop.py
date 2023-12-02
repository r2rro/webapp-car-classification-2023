from PIL import Image
import numpy as np

def crop(filename: str, target_width: int, target_height:int):
    """
    Load an image from a file, crop and resize it to the specified dimensions.

    Parameters:
        filename (str): The path to the image file.
        target_width (int): The desired width of the output image.
        target_height (int): The desired height of the output image.

    Returns:
        np.ndarray: The cropped and resized image as a NumPy array.
    """
    image = Image.open(filename)
    original_width, original_height = image.size

    aspect_ratio = original_width / float(original_height)

    target_aspect_ratio = target_width / float(target_height)

    if aspect_ratio > target_aspect_ratio:
        new_width = int(target_aspect_ratio * original_height)
        offset = (original_width - new_width) / 2
        crop_box = (offset, 0, original_width - offset, original_height)
    else:
        new_height = int(original_width / target_aspect_ratio)
        offset = (original_height - new_height) / 2
        crop_box = (0, offset, original_width, original_height - offset)


    cropped_and_resized_img = image.crop(crop_box).resize((target_width, target_height), Image.ANTIALIAS)

    result_array = np.array(cropped_and_resized_img)

    return result_array