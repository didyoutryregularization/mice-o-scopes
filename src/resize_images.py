import PIL
import os

def resize_images(image_path: str, output_path:str, resolution: tuple):
    """
    Resize all images in a directory and save them to a new directory.
    Args:
        image_path (str): Path to the directory containing images to resize.
        output_path (str): Path to the directory to save the resized images.
        resolution (tuple): Target resolution as (width, height).
    """

    for file in os.listdir(image_path):
        image = PIL.Image.open(f"{image_path}/{file}")
        image = image.resize(resolution, PIL.Image.LANCZOS)
        image.save(f"{output_path}/{file}")

    print("Images resized successfully!")
